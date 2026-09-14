"""Gazebo-native adaptation of the GP-Navigation baseline.

The algorithm follows Leininger et al.'s public ICRA 2024 implementation:
a sparse Gaussian process models local elevation and uncertainty, geometric
terrain measures form a traversability grid, and RRT* selects a collision-free
local path.  ROS messages and the differential-drive waypoint follower are
replaced by NumPy inputs and an Ackermann pure-pursuit follower so the planner
can run against this repository's existing sensors and vehicle.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

from autonomy.meridian_drive.core import Route, VehicleModel, wrap_angle


@dataclass(frozen=True)
class GPNavigationConfig:
    """Parameters corresponding to the upstream GP and RRT* configuration."""

    resolution: float = 0.25
    radius: float = 5.0
    inducing_points: int = 160
    lengthscale: float = 0.7
    rational_quadratic_alpha: float = 10.0
    observation_noise: float = 0.04
    map_period_s: float = 0.5
    step_height_critical: float = 0.30
    flatness_critical: float = 0.436
    slope_critical: float = 0.436
    step_height_weight: float = 0.4
    flatness_weight: float = 0.4
    slope_weight: float = 0.2
    uncertainty_factor: float = 1.4
    uncertainty_safe_radius: float = 2.0
    step_len: float = 0.5
    iter_max: int = 1000
    # The upstream environments use 0.3. Their README explicitly marks this
    # parameter environment-dependent; 0.6 retains a useful connected free
    # region on this simulator's substantially rougher hill-country surface.
    traversability_limit: float = 0.6
    neighbor_radius: float = 1.0
    goal_bias: float = 0.10
    replan_period_s: float = 0.5
    path_lookahead_m: float = 0.8
    target_speed: float = 1.7
    speed_max: float = 2.2
    horizon: int = 60
    dt: float = 0.05
    steer_cmd_slew_rad_s: float = 3.0

    def __post_init__(self) -> None:
        if self.resolution <= 0.0 or self.radius <= self.resolution:
            raise ValueError("GP map resolution and radius must be positive")
        if self.inducing_points < 8:
            raise ValueError("at least eight inducing points are required")
        if self.step_len <= 0.0 or self.neighbor_radius < self.step_len:
            raise ValueError("RRT* neighbor radius must include one step")
        if not 0.0 <= self.traversability_limit <= 1.0:
            raise ValueError("traversability limit must be between zero and one")


@dataclass(frozen=True)
class TraversabilityGrid:
    """One robot-centred GP terrain estimate in the frame of its source scan."""

    elevation: np.ndarray
    uncertainty: np.ndarray
    slope: np.ndarray
    step_height: np.ndarray
    flatness: np.ndarray
    traversability: np.ndarray
    center_x: float
    center_y: float
    yaw: float
    resolution: float
    radius: float
    stamp_s: float = 0.0

    @property
    def shape(self) -> tuple[int, int]:
        return self.traversability.shape

    def sample(
        self, world_x: np.ndarray, world_y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Nearest-neighbour sample of traversability and map validity."""
        return self.sample_layer(self.traversability, world_x, world_y, 1.0)

    def sample_layer(
        self,
        layer: np.ndarray,
        world_x: np.ndarray,
        world_y: np.ndarray,
        invalid_value: float | bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample another layer expressed on the same robot-centred grid."""
        x = np.asarray(world_x, dtype=np.float64)
        y = np.asarray(world_y, dtype=np.float64)
        dx, dy = x - self.center_x, y - self.center_y
        cosine, sine = math.cos(self.yaw), math.sin(self.yaw)
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        col = np.rint((local_x + self.radius) / self.resolution).astype(np.int64)
        row = np.rint((local_y + self.radius) / self.resolution).astype(np.int64)
        valid = (
            (row >= 0)
            & (row < self.shape[0])
            & (col >= 0)
            & (col < self.shape[1])
            & (local_x * local_x + local_y * local_y <= self.radius * self.radius)
        )
        values = np.full(x.shape, invalid_value, dtype=np.asarray(layer).dtype)
        values[valid] = layer[row[valid], col[valid]]
        return values, valid

    def world_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the world position of every grid-cell centre."""
        rows, cols = self.shape
        local_x = -self.radius + np.arange(cols) * self.resolution
        local_y = -self.radius + np.arange(rows) * self.resolution
        grid_x, grid_y = np.meshgrid(local_x, local_y)
        cosine, sine = math.cos(self.yaw), math.sin(self.yaw)
        world_x = self.center_x + cosine * grid_x - sine * grid_y
        world_y = self.center_y + sine * grid_x + cosine * grid_y
        return world_x, world_y


class SparseGPTerrainMapper:
    """Fit a bounded subset-of-data GP to one local 3D LiDAR sweep."""

    def __init__(self, config: GPNavigationConfig) -> None:
        self.config = config

    def _kernel(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        squared = np.sum(
            (first[:, None, :] - second[None, :, :]) ** 2, axis=2
        )
        alpha = self.config.rational_quadratic_alpha
        scale = 2.0 * alpha * self.config.lengthscale**2
        return (1.0 + squared / scale) ** (-alpha)

    def _downsample(self, xy: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Keep the lowest return per cell, then uniformly choose inducing data."""
        cfg = self.config
        bins = np.floor((xy + cfg.radius) / cfg.resolution).astype(np.int64)
        width = int(math.ceil(2.0 * cfg.radius / cfg.resolution)) + 1
        key = bins[:, 1] * width + bins[:, 0]
        order = np.lexsort((z, key))
        sorted_key = key[order]
        first = np.r_[True, sorted_key[1:] != sorted_key[:-1]]
        chosen = order[first]
        if chosen.size > cfg.inducing_points:
            selection = np.linspace(
                0, chosen.size - 1, cfg.inducing_points, dtype=np.int64
            )
            chosen = chosen[selection]
        return xy[chosen], z[chosen]

    @staticmethod
    def _window_peak_to_peak(values: np.ndarray, size: int = 5) -> np.ndarray:
        half = size // 2
        padded = np.pad(values, half, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
        return np.ptp(windows, axis=(-2, -1))

    @staticmethod
    def _window_mean(values: np.ndarray, size: int = 5) -> np.ndarray:
        half = size // 2
        padded = np.pad(values, half, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
        return np.mean(windows, axis=(-2, -1))

    def build(
        self,
        xyz_world: np.ndarray,
        pose: tuple[float, float, float],
        stamp_s: float = 0.0,
    ) -> TraversabilityGrid | None:
        """Return a local GP map, or ``None`` when a sweep has too little data."""
        points = np.asarray(xyz_world, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("point cloud must have shape (N, 3)")
        finite = np.all(np.isfinite(points), axis=1)
        points = points[finite]
        if len(points) < 8:
            return None

        center_x, center_y, yaw = pose
        cosine, sine = math.cos(yaw), math.sin(yaw)
        dx, dy = points[:, 0] - center_x, points[:, 1] - center_y
        local_xy = np.column_stack(
            (cosine * dx + sine * dy, -sine * dx + cosine * dy)
        )
        inside = np.sum(local_xy * local_xy, axis=1) <= self.config.radius**2
        local_xy, heights = local_xy[inside], points[inside, 2]
        if len(local_xy) < 8:
            return None
        train_xy, train_z = self._downsample(local_xy, heights)
        if len(train_xy) < 8:
            return None

        constant_mean = float(np.median(train_z))
        centered_z = train_z - constant_mean
        covariance = self._kernel(train_xy, train_xy)
        covariance.flat[:: len(covariance) + 1] += self.config.observation_noise
        try:
            factor = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            covariance.flat[:: len(covariance) + 1] += 1e-5
            try:
                factor = np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError:
                return None
        alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, centered_z))

        coordinates = np.arange(
            -self.config.radius,
            self.config.radius + self.config.resolution * 0.5,
            self.config.resolution,
            dtype=np.float64,
        )
        grid_x, grid_y = np.meshgrid(coordinates, coordinates)
        query = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        cross_covariance = self._kernel(query, train_xy)
        elevation = (constant_mean + cross_covariance @ alpha).reshape(grid_x.shape)
        solved = np.linalg.solve(factor, cross_covariance.T)
        uncertainty = np.maximum(0.0, 1.0 - np.sum(solved * solved, axis=0)).reshape(
            grid_x.shape
        )

        gradient_y, gradient_x = np.gradient(elevation, self.config.resolution)
        slope = np.arctan(np.hypot(gradient_x, gradient_y))
        step_height = self._window_peak_to_peak(elevation)
        local_mean = self._window_mean(elevation)
        flat_y, flat_x = np.gradient(local_mean, self.config.resolution)
        flatness = np.arctan(np.hypot(flat_x, flat_y))
        raw = (
            self.config.step_height_weight
            * step_height
            / self.config.step_height_critical
            + self.config.flatness_weight
            * flatness
            / self.config.flatness_critical
            + self.config.slope_weight * slope / self.config.slope_critical
        )
        minimum, maximum = float(np.min(raw)), float(np.max(raw))
        if maximum - minimum <= 1e-12:
            traversability = np.zeros_like(raw)
        else:
            traversability = (raw - minimum) / (maximum - minimum)

        mean_uncertainty = float(np.mean(uncertainty))
        high_uncertainty = uncertainty >= mean_uncertainty * self.config.uncertainty_factor
        near_robot = grid_x * grid_x + grid_y * grid_y <= (
            self.config.uncertainty_safe_radius**2
        )
        traversability = np.where(high_uncertainty & ~near_robot, 1.0, traversability)
        outside_circle = grid_x * grid_x + grid_y * grid_y > self.config.radius**2
        traversability[outside_circle] = 1.0

        return TraversabilityGrid(
            elevation=elevation,
            uncertainty=uncertainty,
            slope=slope,
            step_height=step_height,
            flatness=flatness,
            traversability=traversability,
            center_x=center_x,
            center_y=center_y,
            yaw=yaw,
            resolution=self.config.resolution,
            radius=self.config.radius,
            stamp_s=stamp_s,
        )


@dataclass
class _RRTNode:
    x: float
    y: float
    parent: int | None
    cost: float


class RRTStarPlanner:
    """Local RRT* with the upstream traversability rejection rule."""

    _FOOTPRINT = np.asarray(
        [
            (0.0, 0.0),
            (-0.225, 0.0),
            (0.225, 0.0),
            (0.0, -0.225),
            (0.0, 0.225),
            (-0.16, -0.16),
            (-0.16, 0.16),
            (0.16, -0.16),
            (0.16, 0.16),
        ],
        dtype=np.float64,
    )

    def __init__(self, config: GPNavigationConfig, seed: int = 7) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)

    def _edge_is_clear(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        grid: TraversabilityGrid,
        map_stack: object | None,
        collision_grid: np.ndarray | None = None,
    ) -> bool:
        distance = math.dist(start, end)
        samples = max(2, int(math.ceil(distance / (grid.resolution * 0.5))) + 1)
        center_x = np.linspace(start[0], end[0], samples)
        center_y = np.linspace(start[1], end[1], samples)
        query_x = center_x[:, None] + self._FOOTPRINT[None, :, 0]
        query_y = center_y[:, None] + self._FOOTPRINT[None, :, 1]
        traversability, valid = grid.sample(query_x, query_y)
        blocked = (~valid) | (traversability > self.config.traversability_limit)
        # The upstream implementation rejects an edge once at least three cells
        # in its 3x3 footprint exceed the traversability limit.
        if np.any(np.sum(blocked, axis=1) >= 3):
            return False
        if collision_grid is not None:
            collision, _ = grid.sample_layer(
                collision_grid, center_x, center_y, True
            )
            if bool(np.any(collision)):
                return False
        elif map_stack is not None:
            # MapStack already expands each centreline sample over the rover
            # footprint. Passing the nine GP footprint samples here would
            # expand a second time, over-inflate obstacles, and multiply every
            # RRT* edge check by another factor of nine.
            _, collision = map_stack.cost(center_x, center_y)
            if bool(np.any(collision)):
                return False
        return True

    @staticmethod
    def _extract(nodes: list[_RRTNode], index: int) -> np.ndarray:
        path: list[tuple[float, float]] = []
        while True:
            node = nodes[index]
            path.append((node.x, node.y))
            if node.parent is None:
                break
            index = node.parent
        path.reverse()
        return np.asarray(path, dtype=np.float64)

    def plan(
        self,
        grid: TraversabilityGrid,
        start: tuple[float, float],
        goal: tuple[float, float],
        map_stack: object | None = None,
    ) -> np.ndarray | None:
        """Plan to the goal if local, otherwise to the map frontier toward it."""
        goal_vector = np.asarray(goal, dtype=np.float64) - np.asarray(start)
        goal_distance = float(np.linalg.norm(goal_vector))
        local_reach = self.config.radius - self.config.step_len
        if goal_distance > local_reach:
            local_goal = np.asarray(start) + goal_vector * (local_reach / goal_distance)
        else:
            local_goal = np.asarray(goal, dtype=np.float64)

        collision_grid: np.ndarray | None = None
        if map_stack is not None:
            world_x, world_y = grid.world_coordinates()
            _, collision_grid = map_stack.cost(world_x, world_y)

        if self._edge_is_clear(
            start, tuple(local_goal), grid, None, collision_grid
        ):
            return np.asarray((start, tuple(local_goal)), dtype=np.float64)

        nodes = [_RRTNode(start[0], start[1], None, 0.0)]
        start_array = np.asarray(start, dtype=np.float64)
        for _ in range(self.config.iter_max):
            if self.rng.random() < self.config.goal_bias:
                sample = local_goal
            else:
                angle = self.rng.uniform(-math.pi, math.pi)
                radius = self.config.radius * math.sqrt(self.rng.random())
                sample = start_array + radius * np.asarray(
                    (math.cos(angle), math.sin(angle))
                )
            coordinates = np.asarray([(node.x, node.y) for node in nodes])
            nearest_index = int(np.argmin(np.sum((coordinates - sample) ** 2, axis=1)))
            nearest = nodes[nearest_index]
            delta = sample - np.asarray((nearest.x, nearest.y))
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-12:
                continue
            new_xy = np.asarray((nearest.x, nearest.y)) + delta * (
                min(distance, self.config.step_len) / distance
            )
            if np.linalg.norm(new_xy - start_array) > self.config.radius:
                continue
            if not self._edge_is_clear(
                (nearest.x, nearest.y), tuple(new_xy), grid, None, collision_grid
            ):
                continue

            distances = np.linalg.norm(coordinates - new_xy, axis=1)
            near = np.flatnonzero(distances <= self.config.neighbor_radius)
            parent = nearest_index
            cost = nearest.cost + math.dist((nearest.x, nearest.y), tuple(new_xy))
            for candidate_index in near:
                candidate = nodes[int(candidate_index)]
                candidate_xy = (candidate.x, candidate.y)
                candidate_cost = candidate.cost + math.dist(candidate_xy, tuple(new_xy))
                if candidate_cost < cost and self._edge_is_clear(
                    candidate_xy, tuple(new_xy), grid, None, collision_grid
                ):
                    parent, cost = int(candidate_index), candidate_cost
            new_index = len(nodes)
            nodes.append(_RRTNode(float(new_xy[0]), float(new_xy[1]), parent, cost))

            for candidate_index in near:
                candidate = nodes[int(candidate_index)]
                candidate_xy = (candidate.x, candidate.y)
                rewired_cost = cost + math.dist(tuple(new_xy), candidate_xy)
                if rewired_cost < candidate.cost and self._edge_is_clear(
                    tuple(new_xy), candidate_xy, grid, None, collision_grid
                ):
                    candidate.parent = new_index
                    candidate.cost = rewired_cost

            if math.dist(tuple(new_xy), tuple(local_goal)) <= self.config.step_len and (
                self._edge_is_clear(
                    tuple(new_xy), tuple(local_goal), grid, None, collision_grid
                )
            ):
                nodes.append(
                    _RRTNode(
                        float(local_goal[0]),
                        float(local_goal[1]),
                        new_index,
                        cost + math.dist(tuple(new_xy), tuple(local_goal)),
                    )
                )
                return self._extract(nodes, len(nodes) - 1)

        if len(nodes) == 1:
            return None
        endpoints = np.asarray([(node.x, node.y) for node in nodes[1:]])
        nearest_to_goal = 1 + int(
            np.argmin(np.linalg.norm(endpoints - local_goal[None, :], axis=1))
        )
        return self._extract(nodes, nearest_to_goal)


class GPNavigationPlanner:
    """Planner facade compatible with the existing Gazebo autonomy loop."""

    def __init__(
        self,
        route: Route,
        config: GPNavigationConfig | None = None,
        model: VehicleModel | None = None,
        seed: int = 7,
    ) -> None:
        self.route = route
        self.config = config or GPNavigationConfig()
        self.model = model or VehicleModel()
        self.mapper = SparseGPTerrainMapper(self.config)
        self.rrt = RRTStarPlanner(self.config, seed)
        self.progress_m = 0.0
        self.nominal = np.zeros((self.config.horizon, 2), dtype=np.float64)
        self.previous = self.nominal.copy()
        self.last_trajectories: np.ndarray | None = None
        self.last_costs: np.ndarray | None = None
        self.last_path: np.ndarray | None = None
        self._grid: TraversabilityGrid | None = None
        self._grid_lock = threading.Lock()
        self._last_map_s = -math.inf
        self._ticks_since_plan = math.inf

    @property
    def traversability_grid(self) -> TraversabilityGrid | None:
        with self._grid_lock:
            return self._grid

    def update_point_cloud(
        self,
        xyz_world: np.ndarray,
        pose: tuple[float, float, float],
        stamp_s: float,
    ) -> bool:
        """Fit at most at the configured 2 Hz mapping rate."""
        if stamp_s >= self._last_map_s and (
            stamp_s - self._last_map_s < self.config.map_period_s
        ):
            return False
        grid = self.mapper.build(xyz_world, pose, stamp_s)
        if grid is None:
            return False
        with self._grid_lock:
            self._grid = grid
            self._last_map_s = stamp_s
        return True

    def _route_goal(self, state: np.ndarray) -> tuple[float, float]:
        _, progress, _ = self.route.nearest(
            np.asarray([state[0]]),
            np.asarray([state[1]]),
            start_s=max(0.0, self.progress_m - 2.0),
            end_s=min(float(self.route.distance[-1]), self.progress_m + 12.0),
        )
        self.progress_m = max(self.progress_m, float(progress[0]))
        goal_s = min(
            float(self.route.distance[-1]),
            self.progress_m + self.config.radius - self.config.step_len,
        )
        index = min(
            len(self.route.xy) - 1,
            int(np.searchsorted(self.route.distance, goal_s, side="left")),
        )
        return float(self.route.xy[index, 0]), float(self.route.xy[index, 1])

    def _control_for_path(self, state: np.ndarray, path: np.ndarray) -> np.ndarray:
        position = state[:2]
        distances = np.linalg.norm(path - position[None, :], axis=1)
        nearest = int(np.argmin(distances))
        target_index = nearest
        travelled = 0.0
        while target_index + 1 < len(path) and travelled < self.config.path_lookahead_m:
            travelled += math.dist(tuple(path[target_index]), tuple(path[target_index + 1]))
            target_index += 1
        target = path[target_index]
        target_distance = max(1e-6, float(np.linalg.norm(target - position)))
        bearing = math.atan2(target[1] - state[1], target[0] - state[0])
        heading_error = float(wrap_angle(bearing - state[2]))
        wheel_angle = math.atan2(
            2.0 * self.model.wheelbase * math.sin(heading_error), target_distance
        )
        steering = float(np.clip(wheel_angle / self.model.steer_max, -1.0, 1.0))
        turn_scale = 1.0 - 0.60 * min(1.0, abs(steering))
        speed = min(self.config.speed_max, self.config.target_speed) * turn_scale
        return np.asarray((speed, steering), dtype=np.float64)

    def _predict(self, state: np.ndarray, path: np.ndarray) -> None:
        controls = np.empty((self.config.horizon, 2), dtype=np.float64)
        states = np.empty((self.config.horizon + 1, 5), dtype=np.float64)
        states[0] = state
        steer_alpha = 1.0 - math.exp(-self.config.dt / self.model.steer_tau)
        velocity_alpha = 1.0 - math.exp(-self.config.dt / self.model.velocity_tau)
        for step in range(self.config.horizon):
            controls[step] = self._control_for_path(states[step], path)
            old = states[step]
            speed_error = controls[step, 0] - old[3]
            acceleration = float(
                np.clip(
                    velocity_alpha * speed_error / self.config.dt,
                    -self.model.brake_max,
                    self.model.acceleration_max,
                )
            )
            speed = max(
                -self.model.reverse_speed_max, old[3] + self.config.dt * acceleration
            )
            target_steer = self.model.steer_max * controls[step, 1]
            steer = old[4] + steer_alpha * (target_steer - old[4])
            yaw_rate = speed * math.tan(steer) / self.model.wheelbase
            yaw_limit = self.model.lateral_acceleration_max / max(abs(speed), 0.1)
            yaw_rate = float(np.clip(yaw_rate, -yaw_limit, yaw_limit))
            states[step + 1, 0] = old[0] + self.config.dt * speed * math.cos(old[2])
            states[step + 1, 1] = old[1] + self.config.dt * speed * math.sin(old[2])
            states[step + 1, 2] = float(wrap_angle(old[2] + self.config.dt * yaw_rate))
            states[step + 1, 3] = speed
            states[step + 1, 4] = steer
        self.previous = self.nominal.copy()
        self.nominal = controls
        self.last_trajectories = states[None, ...]
        self.last_costs = np.asarray(
            [float(np.sum(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1)))]
        )

    def _set_stationary_prediction(self, state: np.ndarray) -> None:
        """Expose a full stationary horizon to shared diagnostics while stopped."""
        self.nominal.fill(0.0)
        self.previous.fill(0.0)
        stationary = np.repeat(
            np.asarray(state, dtype=np.float64)[None, :],
            self.config.horizon + 1,
            axis=0,
        )
        self.last_trajectories = stationary[None, ...]
        self.last_costs = np.asarray([0.0])

    def command(
        self, state: np.ndarray, map_stack: object, _obstacles: np.ndarray
    ) -> np.ndarray:
        grid = self.traversability_grid
        if grid is None:
            self._set_stationary_prediction(state)
            return np.zeros(2, dtype=np.float64)

        goal = self._route_goal(state)
        self._ticks_since_plan += 1.0
        plan_ticks = max(1, int(round(self.config.replan_period_s / self.config.dt)))
        path_exhausted = self.last_path is not None and math.dist(
            tuple(state[:2]), tuple(self.last_path[-1])
        ) <= self.config.path_lookahead_m * 0.5
        if self._ticks_since_plan >= plan_ticks or path_exhausted:
            self.last_path = self.rrt.plan(
                grid, tuple(state[:2]), goal, map_stack=map_stack
            )
            self._ticks_since_plan = 0.0
        if self.last_path is None or len(self.last_path) < 2:
            self._set_stationary_prediction(state)
            return np.zeros(2, dtype=np.float64)
        self._predict(np.asarray(state, dtype=np.float64), self.last_path)
        return self.nominal[0].copy()

    def best_trajectory(self) -> np.ndarray | None:
        if self.last_trajectories is None:
            return None
        return self.last_trajectories[0]
