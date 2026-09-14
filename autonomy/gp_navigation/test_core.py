"""Focused tests for the ROS-free GP-Navigation adaptation."""

from __future__ import annotations

import unittest

import numpy as np

from autonomy.gp_navigation.core import (
    GPNavigationConfig,
    GPNavigationPlanner,
    RRTStarPlanner,
    SparseGPTerrainMapper,
    TraversabilityGrid,
)
from autonomy.meridian_drive.core import Route


def _manual_grid(values: np.ndarray, resolution: float = 0.25) -> TraversabilityGrid:
    zeros = np.zeros_like(values, dtype=np.float64)
    radius = resolution * (values.shape[0] - 1) / 2.0
    return TraversabilityGrid(
        elevation=zeros,
        uncertainty=zeros,
        slope=zeros,
        step_height=zeros,
        flatness=zeros,
        traversability=np.asarray(values, dtype=np.float64),
        center_x=0.0,
        center_y=0.0,
        yaw=0.0,
        resolution=resolution,
        radius=radius,
    )


class SparseGPTests(unittest.TestCase):
    def test_flat_plane_is_reconstructed_with_lower_central_uncertainty(self) -> None:
        config = GPNavigationConfig(radius=2.0, resolution=0.25, inducing_points=80)
        mapper = SparseGPTerrainMapper(config)
        x, y = np.meshgrid(np.linspace(-1.8, 1.8, 20), np.linspace(-1.8, 1.8, 20))
        points = np.column_stack((x.ravel(), y.ravel(), (0.1 * x).ravel()))
        grid = mapper.build(points, (0.0, 0.0, 0.0))
        self.assertIsNotNone(grid)
        assert grid is not None
        center = grid.shape[0] // 2
        self.assertAlmostEqual(float(grid.elevation[center, center]), 0.0, delta=0.08)
        self.assertAlmostEqual(float(grid.slope[center, center]), np.arctan(0.1), delta=0.08)
        self.assertLess(
            float(grid.uncertainty[center, center]),
            float(grid.uncertainty[0, 0]),
        )

    def test_rotated_grid_samples_in_world_coordinates(self) -> None:
        values = np.zeros((9, 9), dtype=np.float64)
        values[4, 6] = 0.75
        grid = _manual_grid(values, 0.25)
        rotated = TraversabilityGrid(
            **{
                **grid.__dict__,
                "center_x": 10.0,
                "center_y": 20.0,
                "yaw": np.pi / 2.0,
            }
        )
        value, valid = rotated.sample(np.asarray([10.0]), np.asarray([20.5]))
        self.assertTrue(bool(valid[0]))
        self.assertAlmostEqual(float(value[0]), 0.75)


class RRTStarTests(unittest.TestCase):
    def test_rrt_star_routes_around_high_traversability_barrier(self) -> None:
        values = np.zeros((41, 41), dtype=np.float64)
        # A finite wall across the direct route, leaving room above and below.
        values[14:27, 27:30] = 1.0
        grid = _manual_grid(values)
        config = GPNavigationConfig(
            radius=5.0,
            resolution=0.25,
            iter_max=1500,
            traversability_limit=0.3,
            goal_bias=0.15,
        )
        planner = RRTStarPlanner(config, seed=3)
        path = planner.plan(grid, (0.0, 0.0), (4.0, 0.0))
        self.assertIsNotNone(path)
        assert path is not None
        self.assertGreater(len(path), 2)
        self.assertGreater(float(np.max(np.abs(path[:, 1]))), 1.5)

    def test_clear_goal_uses_direct_path(self) -> None:
        grid = _manual_grid(np.zeros((41, 41), dtype=np.float64))
        planner = RRTStarPlanner(GPNavigationConfig(), seed=1)
        path = planner.plan(grid, (0.0, 0.0), (3.0, 0.0))
        np.testing.assert_allclose(path, np.asarray(((0.0, 0.0), (3.0, 0.0))))

    def test_shared_collision_map_is_rasterized_once_per_plan(self) -> None:
        grid = _manual_grid(np.zeros((41, 41), dtype=np.float64))

        class ClearMap:
            calls = 0

            def cost(self, x, _y):
                self.calls += 1
                return np.zeros_like(x), np.zeros_like(x, dtype=bool)

        shared = ClearMap()
        planner = RRTStarPlanner(GPNavigationConfig(), seed=1)
        path = planner.plan(grid, (0.0, 0.0), (3.0, 0.0), shared)
        self.assertIsNotNone(path)
        self.assertEqual(shared.calls, 1)


class PlannerFacadeTests(unittest.TestCase):
    def test_clear_route_produces_ackermann_command_and_full_horizon(self) -> None:
        config = GPNavigationConfig(iter_max=20, horizon=12)
        planner = GPNavigationPlanner(
            Route.from_waypoints([(0.0, 0.0), (10.0, 0.0)]), config=config
        )
        planner._grid = _manual_grid(np.zeros((41, 41), dtype=np.float64))
        command = planner.command(np.zeros(5), None, np.empty((0, 2)))
        self.assertGreater(float(command[0]), 0.0)
        self.assertAlmostEqual(float(command[1]), 0.0)
        self.assertEqual(planner.last_trajectories.shape, (1, 13, 5))

    def test_failed_plan_observes_replanning_period(self) -> None:
        config = GPNavigationConfig(iter_max=20, horizon=12)
        planner = GPNavigationPlanner(
            Route.from_waypoints([(0.0, 0.0), (10.0, 0.0)]), config=config
        )
        planner._grid = _manual_grid(np.zeros((41, 41), dtype=np.float64))

        class NoPath:
            calls = 0

            def plan(self, *_args, **_kwargs):
                self.calls += 1
                return None

        no_path = NoPath()
        planner.rrt = no_path
        for _ in range(5):
            command = planner.command(np.zeros(5), None, np.empty((0, 2)))
            np.testing.assert_allclose(command, np.zeros(2))
        self.assertEqual(no_path.calls, 1)
        self.assertEqual(planner.last_trajectories.shape, (1, 13, 5))


if __name__ == "__main__":
    unittest.main()
