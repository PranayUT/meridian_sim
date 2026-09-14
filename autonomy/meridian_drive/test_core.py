"""Focused tests for the simulator autonomy boundary."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from autonomy.meridian_drive.assistance import AssistanceManager, MappedRecovery
from autonomy.meridian_drive.core import MPPI, MppiConfig, Route, VehicleModel, rollout
from autonomy.meridian_drive.ground_mapping import SemanticMapper
from autonomy.meridian_drive.maps import (
    COMBINED_MAP_TYPE,
    OCCUPANCY_MAP_TYPE,
    SEMANTIC_MAP_TYPE,
    LocalGridMap,
    MapStack,
    TerrainMap,
    UavMap,
    load_uav_map,
)
from autonomy.meridian_drive.obstacle_grid_logic import (
    GROUND,
    SOLID,
    TALL,
    UNSEEN,
    LocalGrid,
    ProbabilisticOccupancyGrid,
    classify,
    rasterize_window,
)
from autonomy.meridian_drive.routes import load_route
from autonomy.meridian_drive.uav_ground_truth import GroundTruthUav, LabeledEllipse
from tools.run_experiment import drag_target
from tools.vegetation import Plant


class DynamicsTests(unittest.TestCase):
    def test_mapped_recovery_backs_out_then_observes_cooldown(self) -> None:
        recovery = MappedRecovery(
            duration_s=2.0,
            speed_mps=0.6,
            steering_fraction=0.65,
            cooldown_s=3.0,
            max_attempts_per_location=2,
        )
        self.assertIsNone(
            recovery.update(4.0, stalled=True, enabled=False, progress_m=0.0)
        )
        self.assertEqual(recovery.update(5.0, stalled=True, enabled=True), (-0.6, 0.65, True))
        self.assertEqual(recovery.update(6.0, stalled=False, enabled=True), (-0.6, 0.65, False))
        self.assertIsNone(recovery.update(7.0, stalled=True, enabled=True))
        self.assertIsNone(recovery.update(9.9, stalled=True, enabled=True))
        self.assertEqual(recovery.update(10.0, stalled=True, enabled=True), (-0.6, -0.65, True))
        self.assertIsNone(recovery.update(15.0, stalled=True, enabled=True))
        self.assertEqual(
            recovery.update(16.0, stalled=True, enabled=True, progress_m=2.0),
            (-0.6, 0.65, True),
        )

    def test_ackermann_inside_wheel_is_limited_to_45_degrees(self) -> None:
        model = VehicleModel()
        track_width = 0.34
        center_radius = model.wheelbase / math.tan(model.steer_max)
        inside_angle = math.atan(model.wheelbase / (center_radius - track_width / 2.0))
        self.assertAlmostEqual(math.degrees(inside_angle), 45.0, places=6)
        self.assertAlmostEqual(center_radius, 0.46, places=6)

    def test_straight_velocity_command_moves_forward(self) -> None:
        controls = np.zeros((1, 20, 2), dtype=np.float64)
        controls[:, :, 0] = 1.0
        states = rollout(np.zeros((1, 5)), controls, VehicleModel(), 0.05)
        self.assertGreater(states[0, -1, 0], 0.4)
        self.assertAlmostEqual(states[0, -1, 1], 0.0)
        self.assertAlmostEqual(states[0, -1, 2], 0.0)

    def test_route_reports_forward_progress(self) -> None:
        route = Route.from_waypoints([(0.0, 0.0), (10.0, 0.0)])
        distance, progress, yaw = route.nearest(np.asarray([5.0]), np.asarray([1.0]))
        self.assertAlmostEqual(float(distance[0]), 1.0)
        self.assertAlmostEqual(float(progress[0]), 5.0)
        self.assertAlmostEqual(float(yaw[0]), 0.0)

    def test_planner_selects_one_side_of_a_symmetric_obstacle(self) -> None:
        planner = MPPI(
            Route.from_waypoints([(0.0, 0.0), (20.0, 0.0)]),
            config=MppiConfig(samples=128, horizon=60),
        )
        obstacle = np.column_stack((np.full(9, 1.5), np.linspace(-0.4, 0.4, 9)))
        state = np.zeros(5, dtype=np.float64)
        for _ in range(3):
            command = planner.command(state, MapStack(), obstacle)
        self.assertGreater(abs(float(command[1])), 0.03)

    def test_planner_drives_around_a_mapped_obstacle_instead_of_stopping(self) -> None:
        route = Route.from_waypoints([(0.0, 0.0), (20.0, 0.0)])
        planner = MPPI(
            route,
            config=MppiConfig(samples=128, horizon=60),
            seed=7,
        )
        resolution = 0.25
        grid = np.zeros((32, 40), dtype=np.int8)
        # A one-metre body centered on the route, with open ground on both
        # sides. The old population converged to x=2.38 and stayed there.
        grid[14:19, 14:18] = SOLID
        stack = MapStack(
            ground_obstacles=LocalGridMap(
                grid, -1.0, -4.0, resolution
            )
        )
        state = np.zeros(5, dtype=np.float64)
        maximum_lateral_offset = 0.0
        for _ in range(240):
            command = planner.command(state, stack, np.empty((0, 2)))
            state = rollout(
                state, command.reshape(1, 1, 2), planner.model, planner.config.dt
            )[0, 1]
            maximum_lateral_offset = max(
                maximum_lateral_offset, abs(float(state[1]))
            )
        self.assertGreater(float(state[0]), 4.0)
        self.assertGreater(maximum_lateral_offset, 0.4)

    def test_planner_drives_out_when_current_footprint_is_already_occupied(self) -> None:
        route = Route.from_waypoints([(0.0, 0.0), (10.0, 0.0)])
        planner = MPPI(route, config=MppiConfig(samples=128, horizon=60), seed=7)
        grid = np.zeros((16, 24), dtype=np.int8)
        grid[7:9, 3:5] = SOLID
        stack = MapStack(
            ground_obstacles=LocalGridMap(grid, -1.0, -2.0, 0.25)
        )
        state = np.zeros(5, dtype=np.float64)
        _, collision = stack.cost(state[None, 0], state[None, 1])
        self.assertTrue(bool(collision[0]))
        for _ in range(40):
            command = planner.command(state, stack, np.empty((0, 2)))
            state = rollout(
                state, command.reshape(1, 1, 2), planner.model, planner.config.dt
            )[0, 1]
        self.assertGreater(float(state[0]), 0.5)

    def test_planner_can_reverse_out_of_confirmed_overlap(self) -> None:
        route = Route.from_waypoints([(0.0, 0.0), (10.0, 0.0)])
        planner = MPPI(route, config=MppiConfig(samples=128, horizon=60), seed=7)
        grid = np.zeros((24, 32), dtype=np.int8)
        # The current footprint and all forward exits are occupied, while the
        # ground behind the rover is clear. Forward-only escape cannot solve
        # this contact geometry.
        grid[10:14, 4:16] = SOLID
        stack = MapStack(
            ground_obstacles=LocalGridMap(grid, -1.0, -3.0, 0.25)
        )
        state = np.zeros(5, dtype=np.float64)
        for _ in range(30):
            command = planner.command(state, stack, np.empty((0, 2)))
            state = rollout(
                state, command.reshape(1, 1, 2), planner.model, planner.config.dt
            )[0, 1]
        self.assertLess(float(state[0]), -0.15)

    def test_route_11_projects_to_terrain_coordinates(self) -> None:
        path = Path(__file__).resolve().parents[2] / "paths" / "Route 11.kmz"
        points = load_route(path)
        self.assertEqual(len(points), 15)
        self.assertAlmostEqual(points[0][0], -8.802, places=2)
        self.assertAlmostEqual(points[0][1], -94.620, places=2)
        self.assertTrue(all(abs(x) <= 256.0 and abs(y) <= 256.0 for x, y in points))

    def test_drag_target_keeps_monotonic_progress_at_route_crossing(self) -> None:
        route = Route.from_waypoints(
            [(0.0, 0.0), (10.0, 0.0), (0.0, 0.0), (10.0, 0.0)]
        )
        x, y, _, progress = drag_target(
            route, (0.0, 0.0), 3.0, progress_m=20.0
        )
        self.assertAlmostEqual(x, 3.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(progress, 23.0)


class ClassifierTests(unittest.TestCase):
    """A body that shows its own ground under an overhang is still SOLID."""

    RES = 0.25
    W = H = 48

    def _window(self, body_x: tuple[float, float], lit_to_x: float, shadow_to_x: float):
        """Ground out to ``lit_to_x``, a body in ``body_x``, then nothing.

        ``lit_to_x`` past the far edge of the body is the bush case: the
        lidar sees under the overhang before the shadow starts.
        """
        origin = (-1.0, -3.0)
        points = []
        step = self.RES / 3.0
        for x in np.arange(origin[0], origin[0] + self.W * self.RES, step):
            for y in np.arange(-2.5, 2.5, step):
                in_body_rows = abs(y) <= 0.5
                if in_body_rows and body_x[0] <= x <= body_x[1]:
                    continue  # the body hides its own ground
                if in_body_rows and x > lit_to_x:
                    continue  # shadow: no return at all
                points.append((x, y, 0.0))
        for x in np.arange(body_x[0], body_x[1], step):
            for y in np.arange(-0.5, 0.5, step):
                for z in np.arange(0.15, 0.70, step):
                    points.append((x, y, z))
        xyz = np.asarray(points, dtype=np.float64)
        raster = rasterize_window(xyz, origin[0], origin[1], self.W, self.H, self.RES)
        return raster, origin

    def test_body_lit_beneath_its_overhang_is_still_solid(self) -> None:
        # Ground visible 0.5 m past the body's far edge, shadow beyond that.
        raster, origin = self._window((4.0, 5.0), lit_to_x=5.5, shadow_to_x=9.0)
        cls = classify(
            raster.min_z, raster.count, raster.xyz, self.RES, origin,
            sensor_xy=(0.0, 0.0), sensor_ground_z=0.0,
            body_band_lo_m=0.10, body_band_hi_m=0.75, body_band_returns=2,
            shadow_depth_m=2.0,
        )
        self.assertIn(SOLID, cls, "a body lit under its overhang read as porous")

    def test_body_with_ground_seen_well_past_it_stays_tall(self) -> None:
        # Ground seen all the way out: a genuinely porous body, still TALL.
        raster, origin = self._window((4.0, 5.0), lit_to_x=11.0, shadow_to_x=11.0)
        cls = classify(
            raster.min_z, raster.count, raster.xyz, self.RES, origin,
            sensor_xy=(0.0, 0.0), sensor_ground_z=0.0,
            body_band_lo_m=0.10, body_band_hi_m=0.75, body_band_returns=2,
            shadow_depth_m=2.0,
        )
        self.assertIn(TALL, cls)
        self.assertNotIn(SOLID, cls, "a body the rays saw straight through read as solid")

    def test_simulator_confidence_clears_direct_rays_but_not_tall(self) -> None:
        grid = ProbabilisticOccupancyGrid(
            res=0.25, window_m=1.0, observation_confidence=3.0
        )
        classes = np.full((4, 4), UNSEEN, dtype=np.int8)
        classes[0, 0] = GROUND
        classes[0, 1] = TALL
        grid.update(classes, classes != UNSEEN, 1.0)

        probability, variance, support, _ = grid.evidence_grid(1.0)
        self.assertLess(float(probability[0, 0]), 0.20)
        self.assertLess(float(variance[0, 0]), 0.04)
        self.assertGreaterEqual(float(probability[0, 1]), 0.20)
        self.assertLessEqual(float(probability[0, 1]), 0.80)
        self.assertEqual(float(support[0, 0]), 3.0)
        # Structural classification still sees only one actual sweep.
        self.assertNotEqual(int(grid.classes()[0, 1]), SOLID)


class RayClearTests(unittest.TestCase):
    """Pin the vectorised ray march: this is the mapping thread's hot loop."""

    def _grid(self) -> LocalGrid:
        grid = LocalGrid(0.25, 40.0, history=10, ground_history=100)
        grid.origin_x = -20.0
        grid.origin_y = -20.0
        return grid

    def test_ray_clears_cells_inside_the_body_band_but_not_the_endpoint(self) -> None:
        grid = self._grid()
        ground = np.zeros((grid.n, grid.n), dtype=np.float64)
        # One return 5 m out along +X, with the sensor and the ray inside the
        # collision band the whole way.
        xyz = np.asarray([[5.0, 0.0, 0.3]])
        clear = grid._ray_clear_mask(
            xyz, ground, (0.0, 0.0, 0.3), body_band_hi_m=0.45
        )
        row = int(np.floor((0.0 - grid.origin_y) / grid.res))
        first = int(np.floor((0.25 - grid.origin_x) / grid.res))
        endpoint = int(np.floor((5.0 - grid.origin_x) / grid.res))
        self.assertTrue(clear[row, first])
        self.assertFalse(clear[row, endpoint])
        self.assertEqual(int(clear.sum()), 19)

    def test_canopy_ray_does_not_clear_beneath_itself(self) -> None:
        grid = self._grid()
        ground = np.zeros((grid.n, grid.n), dtype=np.float64)
        # Same geometry, but the whole ray rides 2 m above the floor.
        high = np.asarray([[5.0, 0.0, 2.0]])
        clear = grid._ray_clear_mask(
            high, ground, (0.0, 0.0, 2.0), body_band_hi_m=0.45
        )
        self.assertEqual(int(clear.sum()), 0)

    def test_unknown_ground_and_degenerate_scans_clear_nothing(self) -> None:
        grid = self._grid()
        unknown = np.full((grid.n, grid.n), np.nan, dtype=np.float64)
        xyz = np.asarray([[5.0, 0.0, 0.3]])
        self.assertEqual(
            int(grid._ray_clear_mask(
                xyz, unknown, (0.0, 0.0, 0.3), body_band_hi_m=0.45
            ).sum()),
            0,
        )
        ground = np.zeros((grid.n, grid.n), dtype=np.float64)
        for label, points, sensor in (
            ("empty", np.empty((0, 3)), (0.0, 0.0, 0.3)),
            ("no sensor", xyz, None),
            # Returns closer than one cell have no interior to march.
            ("sub-resolution", np.asarray([[0.1, 0.0, 0.3]]), (0.0, 0.0, 0.3)),
        ):
            with self.subTest(label):
                self.assertEqual(
                    int(grid._ray_clear_mask(
                        points, ground, sensor, body_band_hi_m=0.45
                    ).sum()),
                    0,
                )


class MapTests(unittest.TestCase):
    def test_ground_uncertainty_drives_rollout_exposure_and_uav_overrides_it(self) -> None:
        variance = np.full((4, 4), 0.25, dtype=np.float32)
        stack = MapStack(
            ground_occupancy_uncertainty=LocalGridMap(variance, 0.0, 0.0, 1.0)
        )
        trajectories = np.asarray([[[0.5, 0.5], [1.5, 0.5], [2.5, 0.5], [3.5, 0.5]]])
        exposure, roi = stack.uncertainty_exposure(trajectories)
        self.assertEqual(exposure, 1.0)
        self.assertIsNotNone(roi)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.npz"
            np.savez_compressed(
                path,
                cost=np.zeros((4, 4), dtype=np.float32),
                obstacle=np.zeros((4, 4), dtype=np.float32),
                uncertainty=np.zeros((4, 4), dtype=np.float32),
                origin_xy=np.asarray((0.0, 0.0)),
                resolution=np.asarray(1.0),
            )
            stack.aerial = load_uav_map(path)
        exposure, roi = stack.uncertainty_exposure(trajectories)
        self.assertEqual(exposure, 0.0)
        self.assertIsNone(roi)

    def test_cell_maturity_does_not_transfer_across_moving_frontier(self) -> None:
        variance = np.full((2, 8), 0.25, dtype=np.float32)
        stack = MapStack(
            ground_occupancy_uncertainty=LocalGridMap(
                variance, 0.0, 0.0, 1.0
            ),
            uncertainty_maturity_s=1.0,
        )
        first = np.asarray([[[0.5, 0.5], [1.5, 0.5]]])
        replacement = np.asarray([[[4.5, 0.5], [5.5, 0.5]]])

        initial = stack.evaluate_assistance(first, None, 0.2, now_s=10.0)
        almost = stack.evaluate_assistance(first, None, 0.2, now_s=10.9)
        mature = stack.evaluate_assistance(first, None, 0.2, now_s=11.0)
        replaced = stack.evaluate_assistance(
            replacement, None, 0.2, now_s=11.1
        )

        self.assertEqual(initial.uncertainty_exposure, 0.0)
        self.assertEqual(almost.uncertainty_exposure, 0.0)
        self.assertEqual(mature.uncertainty_exposure, 1.0)
        self.assertEqual(replaced.uncertainty_exposure, 0.0)

    def test_counterfactual_does_not_resolve_immature_frontier(self) -> None:
        probability = np.full((4, 8), np.nan, dtype=np.float32)
        variance = np.full_like(probability, np.nan)
        stack = MapStack(
            ground_obstacle_probability=LocalGridMap(
                probability, 0.0, 0.0, 1.0
            ),
            ground_occupancy_uncertainty=LocalGridMap(
                variance, 0.0, 0.0, 1.0
            ),
            uncertainty_maturity_s=1.0,
        )
        trajectories = np.asarray(
            [[[0.5, 0.5], [1.5, 0.5], [2.5, 0.5], [3.5, 0.5]]]
        )
        immature = stack.evaluate_assistance(
            trajectories, (0.0, 0.0), 0.2, now_s=1.0
        )
        mature = stack.evaluate_assistance(
            trajectories, (0.0, 0.0), 0.2, now_s=2.0
        )
        self.assertFalse(immature.decision_relevant)
        self.assertIsNone(immature.roi)
        self.assertTrue(mature.decision_relevant)
        self.assertIsNotNone(mature.roi)

    def test_selected_trajectory_has_independent_uncertainty_trigger(self) -> None:
        variance = np.zeros((4, 16), dtype=np.float32)
        variance[:, :4] = 0.25
        stack = MapStack(
            ground_occupancy_uncertainty=LocalGridMap(
                variance, 0.0, 0.0, 1.0
            ),
            uncertainty_maturity_s=0.0,
        )
        selected = np.asarray(
            [[0.5, 1.5], [1.5, 1.5], [2.5, 1.5], [3.5, 1.5]]
        )
        rejected = np.asarray(
            [[10.5, 1.5], [11.5, 1.5], [12.5, 1.5], [13.5, 1.5]]
        )

        action = stack.evaluate_action_assistance(selected, 0.20)
        unrelated_action = stack.evaluate_action_assistance(rejected, 0.20)

        self.assertTrue(action.action_relevant)
        self.assertEqual(action.source, "lidar_occupancy")
        self.assertIsNotNone(action.roi)
        self.assertFalse(unrelated_action.action_relevant)
        self.assertIsNone(unrelated_action.roi)

    def test_meridian_ground_classes_reach_planner_cost(self) -> None:
        grid = np.asarray([[0, 50, 100]], dtype=np.int8)
        stack = MapStack(ground_obstacles=LocalGridMap(grid, 0.0, 0.0, 1.0))
        cost, collision = stack.cost(np.asarray([0.5, 1.5, 2.5]), np.asarray([0.5] * 3))
        self.assertEqual(cost.tolist(), [0.0, 3.0, 0.0])
        self.assertEqual(collision.tolist(), [False, False, True])

    def test_ground_occupancy_probability_is_a_soft_planner_cost(self) -> None:
        probability = np.asarray([[0.1, 0.7, np.nan]], dtype=np.float32)
        stack = MapStack(
            ground_obstacle_probability=LocalGridMap(
                probability, 0.0, 0.0, 1.0
            )
        )
        cost, collision = stack.cost(
            np.asarray([0.5, 1.5, 2.5]), np.asarray([0.5] * 3)
        )
        self.assertGreater(float(cost[1]), float(cost[0]))
        self.assertEqual(collision.tolist(), [False, False, False])

    def test_clear_uav_occupancy_supersedes_only_ground_occupancy(self) -> None:
        ground_classes = LocalGridMap(
            np.asarray([[100, 100, 100]], dtype=np.int8), 0.0, 0.0, 1.0
        )
        ground_probability = LocalGridMap(
            np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
            0.0,
            0.0,
            1.0,
        )
        semantic_obstacles = LocalGridMap(
            np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
            0.0,
            0.0,
            1.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "occupancy.npz"
            np.savez_compressed(
                path,
                obstacle=np.zeros((1, 2), dtype=np.float32),
                uncertainty=np.zeros((1, 2), dtype=np.float32),
                map_types=np.asarray((OCCUPANCY_MAP_TYPE,)),
                origin_xy=np.asarray((0.0, 0.0)),
                resolution=np.asarray(1.0),
            )
            aerial = load_uav_map(path)
        stack = MapStack(
            aerial=aerial,
            ground_obstacles=ground_classes,
            ground_obstacle_probability=ground_probability,
            ground_semantic_obstacles=semantic_obstacles,
        )

        cost, collision = stack.cost(
            np.asarray([0.5, 1.5, 2.5]), np.asarray([0.5] * 3)
        )

        # Clear occupancy truth removes matching structural and probabilistic
        # ground evidence inside its footprint. It neither clears the semantic
        # channel nor affects ground occupancy beyond the UAV map.
        self.assertEqual(cost.tolist(), [0.0, 4.0, 4.0])
        self.assertEqual(collision.tolist(), [False, True, True])

    def test_clear_uav_semantics_supersedes_only_ground_semantics(self) -> None:
        ground_classes = LocalGridMap(
            np.asarray([[0, 100, 0]], dtype=np.int8), 0.0, 0.0, 1.0
        )
        semantic_cost = LocalGridMap(
            np.asarray([[0.8, 0.8, 0.8]], dtype=np.float32),
            0.0,
            0.0,
            1.0,
        )
        semantic_obstacles = LocalGridMap(
            np.ones((1, 3), dtype=np.float32), 0.0, 0.0, 1.0
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "semantic.npz"
            np.savez_compressed(
                path,
                cost=np.zeros((1, 2), dtype=np.float32),
                obstacle=np.zeros((1, 2), dtype=np.float32),
                uncertainty=np.zeros((1, 2), dtype=np.float32),
                map_types=np.asarray((SEMANTIC_MAP_TYPE,)),
                origin_xy=np.asarray((0.0, 0.0)),
                resolution=np.asarray(1.0),
            )
            aerial = load_uav_map(path)
        stack = MapStack(
            aerial=aerial,
            ground_obstacles=ground_classes,
            ground_semantics=semantic_cost,
            ground_semantic_obstacles=semantic_obstacles,
        )

        cost, collision = stack.cost(
            np.asarray([0.5, 1.5, 2.5]), np.asarray([0.5] * 3)
        )

        # Semantic truth clears only its matching layers. The ground SOLID in
        # the occupancy channel remains a collision, as does semantic evidence
        # outside aerial coverage.
        self.assertEqual(cost.tolist(), [0.0, 0.0, 4.0])
        self.assertEqual(collision.tolist(), [False, True, True])

    def test_perfect_camera_labels_project_to_semantic_cost(self) -> None:
        mapper = SemanticMapper()
        labels = np.zeros((5, 5), dtype=np.uint8)
        depth = np.full((5, 5), np.inf, dtype=np.float32)
        labels[4, 0] = 17
        depth[4, 0] = 2.0
        mapper.set_labels(labels, 1.0)
        mapper.set_depth(depth, 1.0)
        self.assertTrue(mapper.project_if_ready((0.0, 0.0, 0.0, 0.0), 1.0))
        _, cost, _, observed, obstacle, cost_variance = mapper.render_layers(
            (0.0, 0.0), 1.0
        )
        self.assertEqual(np.count_nonzero(observed), 1)
        self.assertAlmostEqual(float(cost[np.isfinite(cost)][0]), 0.70, places=5)
        self.assertAlmostEqual(float(obstacle[np.isfinite(obstacle)][0]), 1.0)
        self.assertAlmostEqual(
            float(cost_variance[np.isfinite(cost_variance)][0]), 0.0, places=7
        )

    def test_semantic_assistance_uses_cost_variance_not_weak_support(self) -> None:
        mapper = SemanticMapper()
        probabilities = np.zeros((1, 64), dtype=np.float64)
        probabilities[0, 17] = 0.5
        probabilities[0, 31] = 0.5
        mapper.grid.update(
            np.asarray([[0.1, 0.1]]), probabilities, np.ones(1), 1.0
        )
        _, _, viewer_uncertainty, observed, _, cost_variance = (
            mapper.render_layers((0.0, 0.0), 1.0)
        )
        cell = observed > 0.0
        self.assertGreater(float(viewer_uncertainty[cell][0]), 0.8)
        expected = ((0.7**2 + 0.2**2) / 2.0 - 0.45**2) / 2.0
        self.assertAlmostEqual(float(cost_variance[cell][0]), expected, places=7)

    def test_semantic_obstacle_probability_is_a_planner_collision(self) -> None:
        cost_grid = np.asarray([[0.2, 0.6, 0.4]], dtype=np.float32)
        obstacle_grid = np.asarray([[0.0, 0.67, 0.0]], dtype=np.float32)
        stack = MapStack(
            ground_semantics=LocalGridMap(cost_grid, 0.0, 0.0, 1.0),
            ground_semantic_obstacles=LocalGridMap(
                obstacle_grid, 0.0, 0.0, 1.0
            ),
        )
        cost, collision = stack.cost(
            np.asarray([0.5, 1.5, 2.5]), np.asarray([0.5] * 3)
        )
        self.assertGreater(float(cost[1]), float(cost[0]))
        self.assertEqual(collision.tolist(), [False, True, False])

    def test_terrain_layer_rejects_steep_ground(self) -> None:
        terrain = TerrainMap(
            elevation_grid=np.zeros((3, 3)),
            slope_grid=np.asarray([[0.05, 0.20, 0.50]] * 3),
            origin_x=0.0,
            origin_y=0.0,
            resolution=1.0,
        )
        cost, collision = MapStack(terrain=terrain).cost(
            np.asarray([0.0, 1.0, 2.0]), np.asarray([1.0, 1.0, 1.0])
        )
        self.assertEqual(float(cost[0]), 0.0)
        self.assertGreater(float(cost[1]), 0.0)
        self.assertFalse(bool(collision[1]))
        self.assertTrue(bool(collision[2]))

    def test_native_map_uses_lower_left_origin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.npz"
            np.savez_compressed(
                path,
                cost=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
                obstacle=np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32),
                uncertainty=np.zeros((2, 2), dtype=np.float32),
                origin_xy=np.asarray((-1.0, 2.0)),
                resolution=np.asarray(1.0),
                sequence=np.asarray(4),
            )
            layer = load_uav_map(path)
            cost, obstacle, _, valid = layer.sample(np.asarray([-0.5]), np.asarray([2.5]))
            self.assertTrue(bool(valid[0]))
            self.assertAlmostEqual(float(cost[0]), 0.1, places=6)
            self.assertEqual(float(obstacle[0]), 0.0)
            self.assertEqual(layer.sequence, 4)

    def test_uav_obstacles_provide_soft_clearance_without_false_collision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.npz"
            np.savez_compressed(
                path,
                obstacle=np.asarray(
                    [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
                    dtype=np.float32,
                ),
                uncertainty=np.zeros((1, 9), dtype=np.float32),
                map_types=np.asarray((OCCUPANCY_MAP_TYPE,)),
                origin_xy=np.asarray((0.0, 0.0)),
                resolution=np.asarray(0.25),
            )
            stack = MapStack(aerial=load_uav_map(path))

        cost, collision = stack.cost(
            np.asarray([0.625, 1.125, 1.625]), np.asarray([0.125] * 3)
        )

        self.assertGreater(float(cost[0]), 0.0)
        self.assertGreater(float(cost[2]), 0.0)
        self.assertLess(float(cost[0]), 3.5)
        self.assertLess(float(cost[2]), 3.5)
        self.assertEqual(collision.tolist(), [False, True, False])

    def test_meridian_map_flips_north_up_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.npz"
            np.savez_compressed(
                path,
                label=np.asarray([[1, 0], [0, 0]], dtype=np.uint8),
                classes=np.asarray(["free", "obstacle"]),
                scores=np.asarray([1.0, 0.0], dtype=np.float32),
                transform=np.asarray([1.0, 0.0, 10.0, 0.0, -1.0, 22.0]),
                crs=np.asarray("local"),
                meta=np.asarray(json.dumps({"frame": "world", "sequence": 9})),
            )
            layer = load_uav_map(path)
            _, south_obstacle, _, _ = layer.sample(np.asarray([10.5]), np.asarray([20.5]))
            _, north_obstacle, _, _ = layer.sample(np.asarray([10.5]), np.asarray([21.5]))
            self.assertEqual(float(south_obstacle[0]), 0.0)
            self.assertEqual(float(north_obstacle[0]), 1.0)
            self.assertEqual(layer.sequence, 9)

    def test_counterfactual_request_holds_until_new_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_path = root / "map.npz"
            manager = AssistanceManager(
                "counterfactual_uav",
                map_path,
                root / "request.json",
                root / "status.json",
                MapStack(),
                persistence_s=0.0,
                stop_settle_s=0.0,
                fusion_settle_s=0.0,
            )
            manager.update(
                1.0,
                (0.0, 0.0, 5.0, 5.0),
                source="lidar_occupancy",
                map_type="canopy_obstacle",
            )
            self.assertTrue(manager.hold)
            manager.update(
                1.0,
                (0.0, 0.0, 5.0, 5.0),
                source="lidar_occupancy",
                map_type="canopy_obstacle",
                speed_mps=0.0,
            )
            request = json.loads((root / "request.json").read_text())
            history = json.loads((root / "request_history.json").read_text())
            self.assertTrue(request["hold_requested"])
            self.assertEqual(len(history["requests"]), 1)
            self.assertEqual(history["requests"][0]["source"], "lidar_occupancy")
            self.assertFalse(history["requests"][0]["decision_relevant"])
            self.assertEqual(request["map_types"], ["canopy_obstacle"])
            xs = [point[0] for point in request["roi_xy"]]
            ys = [point[1] for point in request["roi_xy"]]
            self.assertAlmostEqual(max(xs) - min(xs), 25.0)
            self.assertAlmostEqual(max(ys) - min(ys), 25.0)
            np.savez_compressed(
                map_path,
                cost=np.zeros((10, 10), dtype=np.float32),
                obstacle=np.zeros((10, 10), dtype=np.float32),
                uncertainty=np.zeros((10, 10), dtype=np.float32),
                origin_xy=np.asarray((0.0, 0.0)),
                resolution=np.asarray(1.0),
                sequence=np.asarray(1),
            )
            manager.update(0.0, None)
            manager.update(0.0, None)
            self.assertFalse(manager.hold)
            self.assertEqual(manager.map_stack.aerial.sequence, 1)

    def test_persistence_is_measured_on_the_supplied_clock(self) -> None:
        """Timing windows are spans of vehicle time, not of wall time.

        The node paces everything else on the simulator clock, so a campaign
        that runs the world slower than real time must not shorten the source
        persistence the request policy was calibrated against.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sim_s = 0.0
            manager = AssistanceManager(
                "counterfactual_uav",
                root / "map.npz",
                root / "request.json",
                root / "status.json",
                MapStack(),
                uncertainty_threshold=0.5,
                persistence_s=2.0,
                stop_settle_s=0.0,
                clock=lambda: sim_s,
            )
            roi = (0.0, 0.0, 5.0, 5.0)
            for sim_s in (0.0, 0.5, 1.0, 1.5):
                manager.update(
                    1.0, roi, source="lidar_occupancy", map_type=OCCUPANCY_MAP_TYPE
                )
                # Wall time has advanced far past 2 s while running this loop;
                # only the supplied clock may retire the persistence window.
                self.assertFalse(manager.hold, f"requested early at {sim_s} s")
            sim_s = 2.0
            manager.update(
                1.0, roi, source="lidar_occupancy", map_type=OCCUPANCY_MAP_TYPE
            )
            self.assertTrue(manager.hold)

    def test_stalled_motion_requests_a_mature_roi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = AssistanceManager(
                "counterfactual_uav",
                root / "map.npz",
                root / "request.json",
                root / "status.json",
                MapStack(),
                uncertainty_threshold=0.75,
                persistence_s=0.0,
                stop_settle_s=0.0,
            )

            manager.update(
                0.10,
                (0.0, 0.0, 5.0, 5.0),
                source="lidar_occupancy",
                map_type=OCCUPANCY_MAP_TYPE,
                speed_mps=0.0,
                mobility_stalled=True,
            )
            self.assertTrue(manager.hold)
            manager.update(
                0.10,
                (0.0, 0.0, 5.0, 5.0),
                source="lidar_occupancy",
                map_type=OCCUPANCY_MAP_TYPE,
                speed_mps=0.0,
                mobility_stalled=True,
            )

            request = json.loads((root / "request.json").read_text())
            trigger_history = json.loads(
                (root / "request_trigger_history.json").read_text()
            )
            self.assertTrue(request["mobility_relevant"])
            self.assertFalse(request["decision_relevant"])
            self.assertLess(request["uncertainty_exposure"], 0.75)
            self.assertEqual(
                trigger_history["triggers"][0]["trigger_kind"], "mobility"
            )

    def test_forward_probe_is_recorded_as_its_own_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = AssistanceManager(
                "counterfactual_uav",
                root / "map.npz",
                root / "request.json",
                root / "status.json",
                MapStack(),
                uncertainty_threshold=0.75,
                persistence_s=100.0,
                stop_settle_s=0.0,
            )
            update = {
                "source": "lidar_occupancy",
                "map_type": COMBINED_MAP_TYPE,
                "probe_relevant": True,
                "speed_mps": 0.0,
            }
            manager.update(0.03, (1.0, 2.0, 5.0, 6.0), **update)
            manager.update(0.03, (1.0, 2.0, 5.0, 6.0), **update)

            trigger_history = json.loads(
                (root / "request_trigger_history.json").read_text()
            )
            request = json.loads((root / "request.json").read_text())
            self.assertEqual(
                trigger_history["triggers"][0]["trigger_kind"], "forward_probe"
            )
            status = json.loads((root / "status.json").read_text())
            self.assertTrue(status["probe_relevant"])
            self.assertTrue(request["probe_relevant"])
            self.assertEqual(
                set(request["map_types"]),
                {OCCUPANCY_MAP_TYPE, SEMANTIC_MAP_TYPE},
            )

    def test_ground_truth_uav_writes_occupancy_and_goose_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "uav.npz"
            producer = GroundTruthUav(
                path,
                [
                    Plant("grass", -2.0, 0.0, 0.0, 1.0, 0.0),
                    Plant("bush", 0.0, 0.0, 0.0, 1.0, 0.0),
                    Plant("tree", 2.0, 0.0, 0.0, 1.0, 0.0),
                ],
                [LabeledEllipse(0.0, 2.0, 0.8, 0.5, 0.0, 40)],
            )
            producer(
                (-12.5, -12.5, 12.5, 12.5),
                3,
                COMBINED_MAP_TYPE,
            )
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive["occupancy"].shape, (100, 100))
                self.assertEqual(archive["semantic_label"].shape, (100, 100))
                self.assertIn(17, archive["semantic_label"])
                self.assertIn(28, archive["semantic_label"])
                self.assertIn(40, archive["semantic_label"])
                self.assertIn(50, archive["semantic_label"])
                self.assertGreater(np.count_nonzero(archive["occupancy"]), 0)
                self.assertEqual(int(archive["sequence"]), 3)
                self.assertEqual(
                    set(archive["map_types"].tolist()),
                    {OCCUPANCY_MAP_TYPE, SEMANTIC_MAP_TYPE},
                )
            layer = load_uav_map(path)
            self.assertEqual(layer.shape, (100, 100))
            _, grass_obstacle, _, grass_valid = layer.sample(
                np.asarray([-2.0]), np.asarray([0.0])
            )
            self.assertTrue(bool(grass_valid[0]))
            self.assertAlmostEqual(float(grass_obstacle[0]), 0.04, places=6)
            grass_cost, grass_collision = MapStack(aerial=layer).cost(
                np.asarray([-2.0]), np.asarray([0.0])
            )
            self.assertAlmostEqual(float(grass_cost[0]), 1.2, places=5)
            self.assertFalse(bool(grass_collision[0]))

    def test_ground_truth_uav_delivers_only_the_requested_product(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "uav.npz"
            producer = GroundTruthUav(
                path, [Plant("tree", 0.0, 0.0, 0.0, 1.0, 0.0)]
            )

            producer((-12.5, -12.5, 12.5, 12.5), 1, SEMANTIC_MAP_TYPE)
            with np.load(path, allow_pickle=False) as archive:
                semantic_names = set(archive.files)
                self.assertIn(28, archive["semantic_label"])
            self.assertIn("cost", semantic_names)
            self.assertNotIn("occupancy", semantic_names)
            semantic_map = load_uav_map(path)
            self.assertEqual(semantic_map.map_types, (SEMANTIC_MAP_TYPE,))
            self.assertTrue(semantic_map.provides(SEMANTIC_MAP_TYPE))
            self.assertFalse(semantic_map.provides(OCCUPANCY_MAP_TYPE))

            producer((-12.5, -12.5, 12.5, 12.5), 2, OCCUPANCY_MAP_TYPE)
            with np.load(path, allow_pickle=False) as archive:
                occupancy_names = set(archive.files)
                self.assertGreater(np.count_nonzero(archive["occupancy"]), 0)
            self.assertIn("obstacle", occupancy_names)
            self.assertNotIn("cost", occupancy_names)
            self.assertNotIn("semantic_label", occupancy_names)
            occupancy_map = load_uav_map(path)
            self.assertEqual(occupancy_map.map_types, (OCCUPANCY_MAP_TYPE,))
            self.assertFalse(occupancy_map.provides(SEMANTIC_MAP_TYPE))

            with self.assertRaises(ValueError):
                producer((-12.5, -12.5, 12.5, 12.5), 3, "terrain_slope")

    def test_semantic_uav_map_does_not_resolve_occupancy_uncertainty(self) -> None:
        uncertain = np.full((4, 4), 0.25, dtype=np.float32)
        stack = MapStack(
            ground_occupancy_uncertainty=LocalGridMap(uncertain, 0.0, 0.0, 1.0),
            ground_semantic_uncertainty=LocalGridMap(uncertain, 0.0, 0.0, 1.0),
            uncertainty_maturity_s=0.0,
        )
        trajectories = np.asarray([[[0.5, 0.5], [1.5, 0.5], [2.5, 0.5], [3.5, 0.5]]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "semantic.npz"
            np.savez_compressed(
                path,
                cost=np.zeros((4, 4), dtype=np.float32),
                obstacle=np.zeros((4, 4), dtype=np.float32),
                uncertainty=np.zeros((4, 4), dtype=np.float32),
                map_types=np.asarray((SEMANTIC_MAP_TYPE,)),
                origin_xy=np.asarray((0.0, 0.0)),
                resolution=np.asarray(1.0),
            )
            stack.aerial = load_uav_map(path)

        # The semantic answer clears its own channel, so occupancy is now the
        # only source still raised and keeps the full swept exposure.
        semantic = stack._evidence_exposure(
            trajectories,
            stack.ground_semantic_uncertainty,
            source="ground_semantic_cost",
            map_type=SEMANTIC_MAP_TYPE,
            uncertainty_min=0.04,
        )
        occupancy = stack.evaluate_assistance(trajectories, None, 0.2)
        self.assertEqual(semantic.uncertainty_exposure, 0.0)
        self.assertEqual(occupancy.source, "lidar_occupancy")
        self.assertEqual(occupancy.uncertainty_exposure, 1.0)

    def test_uav_products_accumulate_across_channels_and_regions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stack = MapStack()
            for sequence, map_type, origin, obstacle in (
                (1, OCCUPANCY_MAP_TYPE, (0.0, 0.0), 1.0),
                (2, SEMANTIC_MAP_TYPE, (10.0, 0.0), 1.0),
            ):
                path = root / f"{sequence}.npz"
                np.savez_compressed(
                    path,
                    cost=np.asarray(
                        [[0.8 if map_type == SEMANTIC_MAP_TYPE else 0.0]],
                        dtype=np.float32,
                    ),
                    obstacle=np.asarray([[obstacle]], dtype=np.float32),
                    uncertainty=np.zeros((1, 1), dtype=np.float32),
                    map_types=np.asarray((map_type,)),
                    origin_xy=np.asarray(origin),
                    resolution=np.asarray(1.0),
                    sequence=np.asarray(sequence),
                )
                stack.add_aerial(load_uav_map(path))

        cost, collision = stack.cost(
            np.asarray([0.5, 10.5]), np.asarray([0.5, 0.5])
        )
        self.assertEqual(len(stack.aerial_history), 2)
        self.assertEqual(stack.aerial.sequence, 2)
        self.assertGreater(float(cost[0]), 0.0)
        self.assertGreater(float(cost[1]), 0.0)
        self.assertEqual(collision.tolist(), [True, False])

    def test_ground_only_never_loads_a_stale_uav_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_path = root / "map.npz"
            np.savez_compressed(
                map_path,
                cost=np.zeros((2, 2), dtype=np.float32),
                obstacle=np.ones((2, 2), dtype=np.float32),
                uncertainty=np.zeros((2, 2), dtype=np.float32),
                origin_xy=np.asarray((0.0, 0.0)),
                resolution=np.asarray(1.0),
            )
            stack = MapStack()
            manager = AssistanceManager(
                "ground_only",
                map_path,
                root / "request.json",
                root / "status.json",
                stack,
            )
            self.assertIsNone(stack.aerial)
            self.assertFalse(manager.reload_map())


class AerialSamplingTest(unittest.TestCase):
    """Retained-product sampling was made cheaper; it must stay identical."""

    @staticmethod
    def _product(origin_x: float, sequence: int, map_type: str) -> UavMap:
        rng = np.random.default_rng(sequence)
        shape = (100, 100)
        return UavMap(
            rng.random(shape),
            rng.random(shape),
            rng.random(shape),
            rng.random(shape),
            origin_x,
            0.0,
            0.25,
            sequence,
            (map_type,),
        )

    def test_extent_prefilter_matches_sampling_every_product(self) -> None:
        stack = MapStack()
        for index in range(6):
            stack.add_aerial(
                self._product(
                    index * 20.0,
                    index,
                    OCCUPANCY_MAP_TYPE if index % 2 else SEMANTIC_MAP_TYPE,
                )
            )
        rng = np.random.default_rng(11)
        x = rng.uniform(-10.0, 140.0, 500)
        y = rng.uniform(-10.0, 35.0, 500)

        def reference(map_type: str) -> tuple[np.ndarray, ...]:
            """The loop as it read before the extent and early-exit shortcuts."""
            cost = np.zeros_like(x)
            obstacle = np.zeros_like(x)
            clearance = np.zeros_like(x)
            uncertainty = np.ones_like(x)
            valid = np.zeros_like(x, dtype=bool)
            for product in reversed(stack.aerial_history):
                if not product.provides(map_type):
                    continue
                p_cost, p_obstacle, p_clearance, p_valid = product.sample_planner(x, y)
                _, _, p_uncertainty, _ = product.sample(x, y)
                selected = p_valid & ~valid
                cost = np.where(selected, p_cost, cost)
                obstacle = np.where(selected, p_obstacle, obstacle)
                clearance = np.where(selected, p_clearance, clearance)
                uncertainty = np.where(selected, p_uncertainty, uncertainty)
                valid |= p_valid
            return cost, obstacle, clearance, uncertainty, valid

        for map_type in (OCCUPANCY_MAP_TYPE, SEMANTIC_MAP_TYPE):
            with self.subTest(map_type=map_type):
                for expected, actual in zip(
                    reference(map_type), stack._sample_aerial(x, y, map_type)
                ):
                    np.testing.assert_array_equal(expected, actual)

    def test_a_product_elsewhere_on_the_route_is_not_consulted(self) -> None:
        stack = MapStack()
        stack.add_aerial(self._product(500.0, 0, OCCUPANCY_MAP_TYPE))
        x = np.linspace(0.0, 10.0, 50)
        y = np.zeros_like(x)
        self.assertFalse(stack.aerial_history[0].covers_any(x, y))
        *_, valid = stack._sample_aerial(x, y, OCCUPANCY_MAP_TYPE)
        self.assertFalse(bool(valid.any()))


class UncertaintyDiagnosticsTest(unittest.TestCase):
    """The stuck-correlation instrumentation is measured, so it must be sane."""

    @staticmethod
    def _stack() -> MapStack:
        cells = 60
        # Rows index y and columns index x, both from a -5 m origin at 0.25 m.
        # The occupied body sits 2.5-4.0 m straight ahead of the origin; the
        # unknown patch sits alongside it at y = 3.25-4.5 m.
        probability = np.full((cells, cells), 0.05)
        probability[18:23, 30:36] = 0.95
        probability[33:38, 20:28] = np.nan
        variance = np.full((cells, cells), 0.005)
        variance[33:38, 20:28] = np.nan

        def grid(values: np.ndarray) -> LocalGridMap:
            return LocalGridMap(values, -5.0, -5.0, 0.25)

        return MapStack(
            ground_obstacle_probability=grid(probability),
            ground_occupancy_uncertainty=grid(variance),
            ground_semantics=grid(np.full((cells, cells), 0.3)),
            ground_semantic_obstacles=grid(np.full((cells, cells), 0.1)),
            ground_semantic_uncertainty=grid(np.full((cells, cells), 0.005)),
        )

    @staticmethod
    def _straight(length_m: float, steps: int = 60) -> np.ndarray:
        x = np.linspace(0.0, length_m, steps)
        zeros = np.zeros_like(x)
        return np.stack([x, zeros, zeros, zeros, zeros], axis=-1)

    def test_diagnostics_are_json_serialisable_scalars(self) -> None:
        stack = self._stack()
        path = self._straight(5.0)
        population = np.stack([path, path + 0.1])
        diagnostics = stack.uncertainty_diagnostics(
            population, path, path, (0.0, 0.0), 1.0
        )
        for key, value in diagnostics.items():
            with self.subTest(key=key):
                self.assertTrue(
                    value is None or isinstance(value, float),
                    f"{key} is {type(value).__name__}, not a float",
                )
        # A numpy bool here silently killed a whole shadow run.
        json.dumps(diagnostics)

    def test_fixed_probe_keeps_reach_a_shrunken_path_loses(self) -> None:
        """The point of the probe: a slowed vehicle still sees what is ahead."""
        stack = self._stack()
        crawling = self._straight(0.4)
        probe = self._straight(8.0)
        diagnostics = stack.uncertainty_diagnostics(
            None, crawling, probe, (0.0, 0.0), 1.0
        )
        self.assertLess(diagnostics["path_extent_m"], 1.0)
        self.assertGreater(diagnostics["probe_extent_m"], 7.0)
        # The blocked patch sits about 2.5 m ahead of the origin.
        self.assertEqual(diagnostics["path_occ_blocked_frac"], 0.0)
        self.assertGreater(diagnostics["probe_occ_blocked_frac"], 0.0)

    def test_probe_evaluator_matches_the_calibrated_trace_metric(self) -> None:
        stack = self._stack()
        probe = self._straight(8.0)
        probe[:, 1] = 3.8
        diagnostics = stack.uncertainty_diagnostics(
            None, None, probe, (0.0, 0.0), 1.0
        )
        evaluation = stack.evaluate_probe_assistance(probe, 0.02)

        self.assertAlmostEqual(
            evaluation.uncertainty_exposure,
            diagnostics["probe_occ_exposure"],
        )
        self.assertTrue(evaluation.probe_relevant)
        self.assertIsNotNone(evaluation.roi)
        self.assertEqual(evaluation.map_type, COMBINED_MAP_TYPE)

        clear = np.zeros((4, 11), dtype=np.float32)
        stack.add_aerial(
            UavMap(
                clear,
                clear,
                clear,
                clear,
                -1.0,
                2.0,
                1.0,
                1,
                (OCCUPANCY_MAP_TYPE,),
            )
        )
        answered = stack.evaluate_probe_assistance(probe, 0.02)
        self.assertEqual(answered.uncertainty_exposure, 0.0)
        self.assertFalse(answered.probe_relevant)
        self.assertIsNone(answered.roi)

    def test_unknown_evidence_separates_from_ambiguous_evidence(self) -> None:
        stack = self._stack()
        # y = +3.75 m crosses the NaN patch, which is unknown, not ambiguous.
        unknown_path = self._straight(2.0)
        unknown_path[:, 1] = 3.8
        diagnostics = stack.uncertainty_diagnostics(
            None, unknown_path, None, (0.0, 0.0), 1.0
        )
        self.assertGreater(diagnostics["path_occ_unknown_frac"], 0.5)
        self.assertEqual(diagnostics["path_occ_ambiguous_frac"], 0.0)
        self.assertGreaterEqual(
            diagnostics["path_occ_exposure"], diagnostics["path_occ_unknown_frac"]
        )

    def test_missing_layers_report_none_rather_than_zero(self) -> None:
        """A blank map must not look like confidently clear ground."""
        diagnostics = MapStack().uncertainty_diagnostics(
            None, self._straight(5.0), None, (0.0, 0.0), 1.0
        )
        self.assertIsNone(diagnostics["path_occ_exposure"])
        self.assertIsNone(diagnostics["probe_occ_exposure"])
        self.assertIsNone(diagnostics["here_occ_probability"])


if __name__ == "__main__":
    unittest.main()
