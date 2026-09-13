"""Meridian-compatible lidar and perfect-simulation semantic mapping.

The lidar path uses the same rolling geometric classifier and probabilistic
filter as the vehicle repository.  The semantic path replaces only neural
inference: Gazebo supplies perfect per-pixel labels and depth, which are
projected into Meridian's 0.25 m evidence grid.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np

from .obstacle_grid_logic import LocalGrid, ProbabilisticOccupancyGrid, UNSEEN
from .semantic_grids import SemanticEvidenceGrid

RESOLUTION = 0.25
WINDOW_M = 40.0
GRID_CELLS = 160
CAMERA_X = 0.16
CAMERA_Z = 0.26
CAMERA_PITCH = 0.22
CAMERA_HFOV = 1.3962634

# GOOSE PP-LiteSeg costs used by Meridian Drive. Labels absent from the
# simulated world retain the conservative default of 1.0.
GOOSE_COSTS = np.ones(64, dtype=np.float64)
GOOSE_COSTS[[3, 24, 31]] = 0.20
GOOSE_COSTS[[7, 9, 11, 21, 23]] = 0.05
GOOSE_COSTS[5] = 0.40
GOOSE_COSTS[18] = 0.40
GOOSE_COSTS[50] = 0.40
GOOSE_COSTS[51] = 0.60
GOOSE_COSTS[[17, 30, 52, 59]] = 0.70
GOOSE_COSTS[40] = 0.80
GOOSE_COSTS[[16, 27, 28, 62]] = 0.95
SIM_LABEL_NAMES = {17: "brush", 28: "trees", 31: "soil", 40: "rock", 50: "grass"}
SEMANTIC_OBSTACLE_LABELS = (17, 28, 40)


class GroundMapper:
    """Run Meridian's rolling lidar classifier on Gazebo point sweeps."""

    def __init__(self) -> None:
        self.local = LocalGrid(RESOLUTION, WINDOW_M, history=10, ground_history=100)
        self.occupancy = ProbabilisticOccupancyGrid(RESOLUTION, WINDOW_M)
        self.classes = np.full((GRID_CELLS, GRID_CELLS), UNSEEN, dtype=np.int8)

    def update(
        self,
        xyz: np.ndarray,
        sensor_xyz: tuple[float, float, float],
        ground_z: float,
        stamp_s: float,
    ) -> None:
        sx, sy, _ = sensor_xyz
        self.local.recentre(sx, sy)
        self.occupancy.recentre(sx, sy)
        newest = self.local.add(xyz, stamp_s)
        merged = self.local.classify(
            (sx, sy),
            sensor_ground_z=ground_z,
            body_band_lo_m=0.10,
            body_band_hi_m=0.75,
            body_band_returns=2,
            step_height_m=0.10,
            ground_slope=0.35,
            ground_gap_m=0.75,
            shadow_depth_m=0.75,
            classify_range_m=12.0,
        )
        direct = self.local.direct_occupancy_observation(
            newest,
            merged,
            sensor_xyz,
            ground_z,
            body_band_lo_m=0.10,
            body_band_hi_m=0.75,
            ground_slope=0.35,
            ground_gap_m=0.75,
            solid_min_vertical_extent_m=0.30,
        )
        self.occupancy.update(direct, direct != UNSEEN, stamp_s, (sx, sy))
        self.classes = self.occupancy.classes()

    @property
    def origin(self) -> tuple[float, float]:
        return self.occupancy.origin_x, self.occupancy.origin_y


class SemanticMapper:
    """Back-project paired Gazebo label/depth images into semantic evidence."""

    def __init__(self) -> None:
        self.grid = SemanticEvidenceGrid(
            RESOLUTION,
            GOOSE_COSTS,
            evidence_half_life=20.0,
            evidence_cap=30.0,
            weak_evidence_scale=5.0,
            age_half_life=10.0,
        )
        self.latest_labels: tuple[np.ndarray, float] | None = None
        self.latest_depth: tuple[np.ndarray, float] | None = None
        self.last_pair_s = -math.inf

    @staticmethod
    def decode_labels(message: object) -> np.ndarray:
        width, height = int(message.width), int(message.height)
        raw = np.frombuffer(message.data, dtype=np.uint8)
        step = int(message.step) or width
        if raw.size < step * height:
            raise ValueError("semantic label image is shorter than its declared shape")
        return raw[: step * height].reshape(height, step)[:, :width].copy()

    @staticmethod
    def decode_depth(message: object) -> np.ndarray:
        width, height = int(message.width), int(message.height)
        raw = np.frombuffer(message.data, dtype="<f4")
        step_pixels = (int(message.step) // 4) if int(message.step) else width
        if raw.size < step_pixels * height:
            raise ValueError("depth image is shorter than its declared shape")
        return raw[: step_pixels * height].reshape(height, step_pixels)[:, :width].copy()

    def set_labels(self, labels: np.ndarray, received_s: float | None = None) -> None:
        self.latest_labels = (np.asarray(labels, dtype=np.uint8), received_s or time.monotonic())

    def set_depth(self, depth: np.ndarray, received_s: float | None = None) -> None:
        self.latest_depth = (np.asarray(depth, dtype=np.float32), received_s or time.monotonic())

    def project_if_ready(self, pose: tuple[float, float, float, float], stamp_s: float) -> bool:
        if self.latest_labels is None or self.latest_depth is None:
            return False
        labels, label_s = self.latest_labels
        depth, depth_s = self.latest_depth
        pair_s = min(label_s, depth_s)
        if abs(label_s - depth_s) > 0.15 or pair_s <= self.last_pair_s or labels.shape != depth.shape:
            return False
        self.last_pair_s = pair_s

        stride = 4
        rows, cols = np.mgrid[0 : labels.shape[0] : stride, 0 : labels.shape[1] : stride]
        selected_labels = labels[::stride, ::stride].reshape(-1)
        ranges = depth[::stride, ::stride].reshape(-1).astype(np.float64)
        valid = (
            np.isfinite(ranges)
            & (ranges >= 0.30)
            & (ranges <= 8.0)
            & np.isin(selected_labels, tuple(SIM_LABEL_NAMES))
        )
        if not np.any(valid):
            return False
        u = cols.reshape(-1)[valid].astype(np.float64)
        v = rows.reshape(-1)[valid].astype(np.float64)
        d = ranges[valid]
        label = selected_labels[valid]
        height, width = labels.shape
        focal = width / (2.0 * math.tan(CAMERA_HFOV / 2.0))
        camera = np.column_stack((d, -(u - (width - 1) / 2.0) * d / focal, -(v - (height - 1) / 2.0) * d / focal))

        cp, sp = math.cos(CAMERA_PITCH), math.sin(CAMERA_PITCH)
        forward = cp * camera[:, 0] + sp * camera[:, 2]
        up = -sp * camera[:, 0] + cp * camera[:, 2]
        lateral = camera[:, 1]
        x, y, yaw, base_z = pose
        cy, sy = math.cos(yaw), math.sin(yaw)
        camera_x = x + cy * CAMERA_X
        camera_y = y + sy * CAMERA_X
        world_x = camera_x + cy * forward - sy * lateral
        world_y = camera_y + sy * forward + cy * lateral
        world_z = base_z + CAMERA_Z + up
        low_enough = world_z <= base_z + 0.5
        if not np.any(low_enough):
            return False
        label = label[low_enough]
        xy = np.column_stack((world_x[low_enough], world_y[low_enough]))
        probabilities = np.zeros((len(label), 64), dtype=np.float64)
        probabilities[np.arange(len(label)), label] = 1.0
        self.grid.update(xy, probabilities, np.ones(len(label)), stamp_s)
        self.grid.prune(np.asarray((x, y)), 40.0)
        return True

    def render(self, center_xy: tuple[float, float], now_s: float):
        origin, cost, uncertainty, observed, probabilities = self.grid.render_with_probabilities(
            np.asarray(center_xy), GRID_CELLS, GRID_CELLS, now_s
        )
        obstacle_probability = np.nansum(
            probabilities[..., SEMANTIC_OBSTACLE_LABELS], axis=2
        ).astype(np.float32)
        obstacle_probability[observed <= 0.0] = np.nan
        return origin, cost, uncertainty, observed, obstacle_probability


def write_snapshot(
    path: Path,
    occupancy: GroundMapper,
    semantic: SemanticMapper,
    pose: tuple[float, float, float, float],
    stamp_s: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Atomically publish the two local grids for the planner and viewer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    (
        occupancy_probability,
        occupancy_variance,
        occupancy_support,
        occupancy_age,
    ) = occupancy.occupancy.evidence_grid(stamp_s)
    (
        origin,
        semantic_cost,
        semantic_uncertainty,
        semantic_observed,
        semantic_obstacle_probability,
    ) = semantic.render(pose[:2], stamp_s)
    # A process-specific temporary name keeps a stale or accidentally doubled
    # autonomy process from stealing this writer's file between save and rename.
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            temp,
            occupancy=occupancy.classes,
            occupancy_probability=occupancy_probability,
            occupancy_variance=occupancy_variance,
            occupancy_support=occupancy_support,
            occupancy_age=occupancy_age,
            occupancy_origin=np.asarray(occupancy.origin),
            semantic_cost=semantic_cost,
            semantic_uncertainty=semantic_uncertainty,
            semantic_observed=semantic_observed,
            semantic_obstacle_probability=semantic_obstacle_probability,
            semantic_origin=np.asarray(origin),
            resolution=np.asarray(RESOLUTION),
            pose=np.asarray(pose),
            camera_hfov=np.asarray(CAMERA_HFOV),
            camera_range=np.asarray(8.0),
            timestamp=np.asarray(time.time()),
        )
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return semantic_cost, (float(origin[0]), float(origin[1]))
