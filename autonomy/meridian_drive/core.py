"""Vectorized MPPI and route logic derived from Meridian Drive.

The source model is ``terrain_aware_mppi`` and ``trucksim`` in the
Meridian Drive repository. This module removes the ROS wrapper. Gazebo
transport stays in ``gazebo_node.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

X, Y, YAW, SPEED, STEER = range(5)


@dataclass(frozen=True)
class VehicleModel:
    """The velocity-command bicycle model used by Meridian Drive MPPI."""

    wheelbase: float = 0.29
    # A 60 degree limit at the inside physical wheel corresponds to this
    # virtual center-wheel angle for the rover's Ackermann geometry.
    steer_max: float = 0.7099452630375517
    steer_tau: float = 0.131
    velocity_tau: float = 0.12
    acceleration_max: float = 2.46
    brake_max: float = 2.3
    lateral_acceleration_max: float = 6.25


@dataclass(frozen=True)
class MppiConfig:
    horizon: int = 60
    samples: int = 192
    dt: float = 0.05
    temperature: float = 0.15
    target_speed: float = 1.7
    speed_max: float = 2.2
    velocity_sigma: float = 0.40
    steering_sigma: float = 0.28
    velocity_knots: int = 4
    steering_knots: int = 6
    # Meridian Drive's shipped values: path_cross_track_weight and
    # path_heading_weight in terrain_aware_mppi/config/mppi_params.yaml. Lowering
    # them does make the planner leave the line for obstacles, but it is treating
    # a symptom: the obstacle cost here is missing the look-ahead ray and
    # clearance field that give the real stack its lateral gradient.
    route_weight: float = 4.0
    heading_weight: float = 6.0
    speed_weight: float = 1.0
    terminal_progress_weight: float = 10.0
    steering_rate_weight: float = 1.0
    # Final backstop on the published wheel angle, from Meridian's
    # steer_cmd_slew_rad_s. Without it a one-cycle steering jump slams the
    # wheels to a new angle and the rover oscillates; 1.5 rad/s allows
    # 0.075 rad per 50 ms tick.
    steer_cmd_slew_rad_s: float = 1.5
    map_cost_weight: float = 3.0
    obstacle_cost: float = 80.0
    collision_cost: float = 1_000_000.0
    clearance_m: float = 0.325


@dataclass(frozen=True)
class Route:
    """A dense metric route with distance and tangent at each sample."""

    xy: np.ndarray
    distance: np.ndarray
    yaw: np.ndarray

    @classmethod
    def from_waypoints(cls, waypoints: list[tuple[float, float]], spacing: float = 0.25) -> "Route":
        if len(waypoints) < 2:
            raise ValueError("a route needs at least two waypoints")
        dense: list[tuple[float, float]] = []
        for index, (start, end) in enumerate(zip(waypoints, waypoints[1:])):
            length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
            steps = max(1, int(np.ceil(length / spacing)))
            for step in range(0 if index == 0 else 1, steps + 1):
                ratio = step / steps
                dense.append(
                    (
                        start[0] + ratio * (end[0] - start[0]),
                        start[1] + ratio * (end[1] - start[1]),
                    )
                )
        xy = np.asarray(dense, dtype=np.float64)
        segments = np.diff(xy, axis=0)
        distance = np.concatenate(([0.0], np.cumsum(np.linalg.norm(segments, axis=1))))
        derivative = np.gradient(xy, axis=0)
        yaw = np.arctan2(derivative[:, 1], derivative[:, 0])
        return cls(xy=xy, distance=distance, yaw=yaw)

    def nearest(
        self,
        x: np.ndarray,
        y: np.ndarray,
        start_s: float = 0.0,
        end_s: float = math.inf,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return distance, path distance, and tangent for query points."""
        selected = (self.distance >= start_s) & (self.distance <= end_s)
        if not np.any(selected):
            selected = np.ones(len(self.distance), dtype=bool)
        route_xy = self.xy[selected]
        route_distance = self.distance[selected]
        route_yaw = self.yaw[selected]
        shape = x.shape
        query = np.stack((x.ravel(), y.ravel()), axis=1)
        result_distance = np.empty(len(query), dtype=np.float64)
        result_s = np.empty(len(query), dtype=np.float64)
        result_yaw = np.empty(len(query), dtype=np.float64)
        # Batches bound peak memory when a long route is used.
        for start in range(0, len(query), 4096):
            batch = query[start : start + 4096]
            delta = batch[:, None, :] - route_xy[None, :, :]
            squared = np.einsum("bpi,bpi->bp", delta, delta)
            nearest = np.argmin(squared, axis=1)
            stop = start + len(batch)
            result_distance[start:stop] = np.sqrt(squared[np.arange(len(batch)), nearest])
            result_s[start:stop] = route_distance[nearest]
            result_yaw[start:stop] = route_yaw[nearest]
        return (
            result_distance.reshape(shape),
            result_s.reshape(shape),
            result_yaw.reshape(shape),
        )


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(angle), np.cos(angle))


def rollout(initial: np.ndarray, controls: np.ndarray, model: VehicleModel, dt: float) -> np.ndarray:
    """Roll out the velocity bicycle model for all sampled controls."""
    samples, horizon, _ = controls.shape
    states = np.empty((samples, horizon + 1, 5), dtype=np.float64)
    states[:, 0] = initial
    steer_alpha = 1.0 - np.exp(-dt / model.steer_tau)
    velocity_alpha = 1.0 - np.exp(-dt / model.velocity_tau)
    for step in range(horizon):
        old = states[:, step]
        command = controls[:, step]
        speed_error = command[:, 0] - old[:, SPEED]
        acceleration = np.clip(
            velocity_alpha * speed_error / dt,
            -model.brake_max,
            model.acceleration_max,
        )
        speed = np.maximum(0.0, old[:, SPEED] + dt * acceleration)
        steer_target = model.steer_max * command[:, 1]
        steer = old[:, STEER] + steer_alpha * (steer_target - old[:, STEER])
        yaw_rate = speed * np.tan(steer) / model.wheelbase
        yaw_limit = model.lateral_acceleration_max / np.maximum(speed, 0.1)
        yaw_rate = np.clip(yaw_rate, -yaw_limit, yaw_limit)
        states[:, step + 1, X] = old[:, X] + dt * speed * np.cos(old[:, YAW])
        states[:, step + 1, Y] = old[:, Y] + dt * speed * np.sin(old[:, YAW])
        states[:, step + 1, YAW] = wrap_angle(old[:, YAW] + dt * yaw_rate)
        states[:, step + 1, SPEED] = speed
        states[:, step + 1, STEER] = steer
    return states


def _smooth_noise(rng: np.random.Generator, samples: int, horizon: int, knots: int) -> np.ndarray:
    knot_x = np.linspace(0.0, horizon - 1, knots)
    full_x = np.arange(horizon)
    values = rng.standard_normal((samples, knots))
    output = np.empty((samples, horizon), dtype=np.float64)
    for index in range(samples):
        output[index] = np.interp(full_x, knot_x, values[index])
    return output


@dataclass
class MPPI:
    """A ROS-free form of the Meridian Drive receding-horizon planner."""

    route: Route
    config: MppiConfig = field(default_factory=MppiConfig)
    model: VehicleModel = field(default_factory=VehicleModel)
    seed: int = 7

    def __post_init__(self) -> None:
        self.nominal = np.zeros((self.config.horizon, 2), dtype=np.float64)
        self.previous = np.zeros(2, dtype=np.float64)
        self.rng = np.random.default_rng(self.seed)
        self.last_trajectories: np.ndarray | None = None
        self.last_costs: np.ndarray | None = None
        self.progress_m = 0.0

    def reset(self) -> None:
        self.nominal.fill(0.0)
        self.previous.fill(0.0)
        self.progress_m = 0.0

    def command(self, state: np.ndarray, map_stack: object, lidar_points: np.ndarray) -> np.ndarray:
        cfg = self.config
        _, current_progress, _ = self.route.nearest(
            np.asarray([state[X]]),
            np.asarray([state[Y]]),
            max(0.0, self.progress_m - 0.5),
            self.progress_m + 8.0,
        )
        self.progress_m = max(self.progress_m, float(current_progress[0]))
        noise = np.zeros((cfg.samples, cfg.horizon, 2), dtype=np.float64)
        pairs = cfg.samples // 2
        velocity_noise = _smooth_noise(
            self.rng, pairs, cfg.horizon, cfg.velocity_knots
        ) * cfg.velocity_sigma
        steering_noise = _smooth_noise(
            self.rng, pairs, cfg.horizon, cfg.steering_knots
        ) * cfg.steering_sigma
        # Use the same velocity history for each left/right pair. This keeps a
        # straight route exactly symmetric after the forward-only speed clamp.
        noise[:pairs, :, 0] = velocity_noise
        noise[:pairs, :, 1] = steering_noise
        noise[pairs : 2 * pairs, :, 0] = velocity_noise
        noise[pairs : 2 * pairs, :, 1] = -steering_noise
        controls = self.nominal[None, :, :] + noise
        controls[:, :, 0] = np.clip(controls[:, :, 0], 0.0, cfg.speed_max)
        controls[:, :, 1] = np.clip(controls[:, :, 1], -1.0, 1.0)

        trajectories = rollout(state, controls, self.model, cfg.dt)
        driven = trajectories[:, 1:]
        cross_track, progress, path_yaw = self.route.nearest(
            driven[:, :, X],
            driven[:, :, Y],
            max(0.0, self.progress_m - 0.5),
            self.progress_m + cfg.speed_max * cfg.horizon * cfg.dt + 5.0,
        )
        costs = cfg.route_weight * np.sum(cross_track**2, axis=1)
        costs += cfg.heading_weight * np.sum(
            1.0 - np.cos(wrap_angle(driven[:, :, YAW] - path_yaw)), axis=1
        )
        costs += cfg.speed_weight * np.sum(
            (driven[:, :, SPEED] - cfg.target_speed) ** 2, axis=1
        )
        costs -= cfg.terminal_progress_weight * progress[:, -1]
        previous = np.concatenate(
            (np.broadcast_to(self.previous, (cfg.samples, 1, 2)), controls[:, :-1]), axis=1
        )
        costs += cfg.steering_rate_weight * np.sum(
            (controls[:, :, 1] - previous[:, :, 1]) ** 2, axis=1
        )

        map_cost, map_collision = map_stack.cost(
            driven[:, :, X], driven[:, :, Y]
        )
        costs += cfg.map_cost_weight * np.sum(map_cost, axis=1)
        hard_collision = np.any(map_collision, axis=1)
        costs += hard_collision * cfg.collision_cost

        if lidar_points.size:
            points = lidar_points[-240:]
            dx = driven[:, :, X, None] - points[None, None, :, 0]
            dy = driven[:, :, Y, None] - points[None, None, :, 1]
            clearance = np.sqrt(np.min(dx * dx + dy * dy, axis=2))
            costs += cfg.obstacle_cost * np.sum(
                np.maximum(0.0, 1.0 - clearance / cfg.clearance_m) ** 2, axis=1
            )
            lidar_collision = np.any(clearance < 0.18, axis=1)
            hard_collision |= lidar_collision
            costs += lidar_collision * cfg.collision_cost

        if np.all(hard_collision):
            self.previous.fill(0.0)
            self.last_trajectories = trajectories
            self.last_costs = costs
            return np.zeros(2, dtype=np.float64)

        weights = np.exp(-(costs - np.min(costs)) / cfg.temperature)
        # A symmetric obstacle gives equally good left and right groups. Their
        # weighted mean points into the obstacle. Meridian Drive selects one
        # steering mode before averaging. Apply that rule when the population
        # contains a material collision set.
        if float(np.mean(hard_collision)) >= 0.05:
            best = int(np.argmin(costs))
            selected_steering = float(np.mean(controls[best, :, 1]))
            if abs(selected_steering) >= 0.03:
                mean_steering = np.mean(controls[:, :, 1], axis=1)
                selected = np.sign(mean_steering) == np.sign(selected_steering)
                weights = np.where(selected, weights, 0.0)
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            best = int(np.argmin(costs))
            self.nominal = controls[best].copy()
        else:
            weights /= total
            self.nominal = np.einsum("n,ntc->tc", weights, controls)
        output = self.nominal[0].copy()
        self.previous = output.copy()
        self.nominal[:-1] = self.nominal[1:]
        self.nominal[-1] = self.nominal[-2]
        self.last_trajectories = trajectories
        self.last_costs = costs
        return output

    def best_trajectory(self) -> np.ndarray | None:
        if self.last_trajectories is None or self.last_costs is None:
            return None
        return self.last_trajectories[int(np.argmin(self.last_costs)), 1:]
