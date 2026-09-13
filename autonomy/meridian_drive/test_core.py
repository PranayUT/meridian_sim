"""Focused tests for the simulator autonomy boundary."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from autonomy.meridian_drive.assistance import AssistanceManager
from autonomy.meridian_drive.core import MPPI, MppiConfig, Route, VehicleModel, rollout
from autonomy.meridian_drive.ground_mapping import SemanticMapper
from autonomy.meridian_drive.maps import LocalGridMap, MapStack, TerrainMap, load_uav_map
from autonomy.meridian_drive.routes import load_route


class DynamicsTests(unittest.TestCase):
    def test_ackermann_inside_wheel_is_limited_to_60_degrees(self) -> None:
        model = VehicleModel()
        track_width = 0.34
        center_radius = model.wheelbase / math.tan(model.steer_max)
        inside_angle = math.atan(model.wheelbase / (center_radius - track_width / 2.0))
        self.assertAlmostEqual(math.degrees(inside_angle), 60.0, places=6)

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

    def test_route_11_projects_to_terrain_coordinates(self) -> None:
        path = Path(__file__).resolve().parents[2] / "paths" / "Route 11.kmz"
        points = load_route(path)
        self.assertEqual(len(points), 15)
        self.assertAlmostEqual(points[0][0], -8.802, places=2)
        self.assertAlmostEqual(points[0][1], -94.620, places=2)
        self.assertTrue(all(abs(x) <= 256.0 and abs(y) <= 256.0 for x, y in points))


class MapTests(unittest.TestCase):
    def test_meridian_ground_classes_reach_planner_cost(self) -> None:
        grid = np.asarray([[0, 50, 100]], dtype=np.int8)
        stack = MapStack(ground_obstacles=LocalGridMap(grid, 0.0, 0.0, 1.0))
        cost, collision = stack.cost(np.asarray([0.5, 1.5, 2.5]), np.asarray([0.5] * 3))
        self.assertEqual(cost.tolist(), [0.0, 3.0, 0.0])
        self.assertEqual(collision.tolist(), [False, False, True])

    def test_perfect_camera_labels_project_to_semantic_cost(self) -> None:
        mapper = SemanticMapper()
        labels = np.zeros((5, 5), dtype=np.uint8)
        depth = np.full((5, 5), np.inf, dtype=np.float32)
        labels[4, 0] = 17
        depth[4, 0] = 2.0
        mapper.set_labels(labels, 1.0)
        mapper.set_depth(depth, 1.0)
        self.assertTrue(mapper.project_if_ready((0.0, 0.0, 0.0, 0.0), 1.0))
        _, cost, _, observed, obstacle = mapper.render((0.0, 0.0), 1.0)
        self.assertEqual(np.count_nonzero(observed), 1)
        self.assertAlmostEqual(float(cost[np.isfinite(cost)][0]), 0.70, places=5)
        self.assertAlmostEqual(float(obstacle[np.isfinite(obstacle)][0]), 1.0)

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
            )
            manager.update(1.0, (0.0, 0.0, 5.0, 5.0))
            self.assertTrue(manager.hold)
            request = json.loads((root / "request.json").read_text())
            self.assertTrue(request["hold_requested"])
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
            self.assertFalse(manager.hold)
            self.assertEqual(manager.map_stack.aerial.sequence, 1)


if __name__ == "__main__":
    unittest.main()
