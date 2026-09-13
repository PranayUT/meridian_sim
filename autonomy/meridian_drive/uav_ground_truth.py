"""Ground-truth 25 m local maps used to imitate a UAV response in Gazebo."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools.vegetation import Plant, plants_from_masks

from .ground_mapping import GOOSE_COSTS, SEMANTIC_OBSTACLE_LABELS
from .maps import OCCUPANCY_MAP_TYPE, SEMANTIC_MAP_TYPE


@dataclass(frozen=True)
class LabeledEllipse:
    x: float
    y: float
    radius_x: float
    radius_y: float
    yaw: float
    label: int


class GroundTruthUav:
    """Rasterize the known simulated world at the requested counterfactual."""

    def __init__(
        self,
        result_path: Path,
        plants: list[Plant],
        fixed_features: list[LabeledEllipse] | None = None,
        resolution: float = 0.25,
    ) -> None:
        if resolution <= 0.0:
            raise ValueError("UAV resolution must be positive")
        self.result_path = result_path
        self.plants = plants
        self.fixed_features = fixed_features or []
        self.resolution = resolution

    @classmethod
    def from_world(
        cls,
        result_path: Path,
        masks_path: Path,
        world_path: Path,
        vegetation_seed: int | None = None,
        resolution: float = 0.25,
    ) -> "GroundTruthUav":
        with np.load(masks_path, allow_pickle=False) as archive:
            masks = {kind: np.asarray(archive[kind], dtype=bool) for kind in ("grass", "bush", "tree")}
            density = float(np.asarray(archive["density"]).item())
            saved_seed = int(np.asarray(archive["seed"]).item())
        # Elevation does not affect a top-down label or footprint. Supplying a
        # tiny flat DEM reproduces the exact seeded X/Y plant layout cheaply.
        plants = plants_from_masks(
            masks,
            np.zeros((2, 2)),
            density,
            saved_seed if vegetation_seed is None else vegetation_seed,
        )
        return cls(result_path, plants, _labeled_ellipses(world_path), resolution)

    def __call__(
        self,
        roi: tuple[float, float, float, float],
        sequence: int,
        map_type: str = "",
    ) -> None:
        """Write the product the request raised, and only that product.

        A request names one evidence channel because one channel crossed its
        uncertainty rule. Delivering the other channel as well would let an
        occupancy question silently resolve semantic cells that no source ever
        asked about, and vice versa. An unnamed type keeps the original
        both-layer response for external callers.
        """
        if map_type and map_type not in (SEMANTIC_MAP_TYPE, OCCUPANCY_MAP_TYPE):
            raise ValueError(f"unknown UAV map type: {map_type}")
        map_types = (map_type,) if map_type else (SEMANTIC_MAP_TYPE, OCCUPANCY_MAP_TYPE)
        wants_semantic = SEMANTIC_MAP_TYPE in map_types
        wants_occupancy = OCCUPANCY_MAP_TYPE in map_types
        x0, y0, x1, y1 = roi
        width = int(round((x1 - x0) / self.resolution))
        height = int(round((y1 - y0) / self.resolution))
        if width <= 0 or height <= 0:
            raise ValueError("requested UAV ROI is empty")
        labels = np.full((height, width), 31, dtype=np.uint8)  # GOOSE soil
        obstacle = np.zeros((height, width), dtype=np.float32)
        # The two products disagree on extent by design: occupancy paints the
        # physical body a wheel can strike, while semantic traversability
        # paints the labeled canopy, matching what the ground semantic layer
        # already treats as untraversable.
        semantic_obstacle = np.zeros((height, width), dtype=np.float32)

        # Plants are ordered grass, bush, tree. Later, taller vegetation wins
        # the top-down semantic label just as it does in a UAV image.
        semantic_radius = {"grass": 0.20, "bush": 0.75, "tree": 1.35}
        occupancy_radius = {"bush": 0.43, "tree": 0.20}
        goose_label = {"grass": 50, "bush": 17, "tree": 28}
        for plant in self.plants:
            radius = semantic_radius[plant.kind] * plant.scale
            if not _intersects(plant.x, plant.y, radius, x0, y0, x1, y1):
                continue
            _paint_ellipse(labels, goose_label[plant.kind], plant.x, plant.y, radius, radius, 0.0, x0, y0, self.resolution)
            if goose_label[plant.kind] in SEMANTIC_OBSTACLE_LABELS:
                _paint_ellipse(semantic_obstacle, 1.0, plant.x, plant.y, radius, radius, 0.0, x0, y0, self.resolution)
            if plant.kind in occupancy_radius:
                body = occupancy_radius[plant.kind] * plant.scale
                _paint_ellipse(obstacle, 1.0, plant.x, plant.y, body, body, 0.0, x0, y0, self.resolution)

        for feature in self.fixed_features:
            radius = max(feature.radius_x, feature.radius_y)
            if not _intersects(feature.x, feature.y, radius, x0, y0, x1, y1):
                continue
            _paint_ellipse(labels, feature.label, feature.x, feature.y, feature.radius_x, feature.radius_y, feature.yaw, x0, y0, self.resolution)
            if feature.label in SEMANTIC_OBSTACLE_LABELS:
                _paint_ellipse(obstacle, 1.0, feature.x, feature.y, feature.radius_x, feature.radius_y, feature.yaw, x0, y0, self.resolution)
                _paint_ellipse(semantic_obstacle, 1.0, feature.x, feature.y, feature.radius_x, feature.radius_y, feature.yaw, x0, y0, self.resolution)

        payload: dict[str, np.ndarray] = {
            "uncertainty": np.zeros((height, width), dtype=np.float32),
            "map_types": np.asarray(map_types),
            "origin_xy": np.asarray((x0, y0), dtype=np.float64),
            "resolution": np.asarray(self.resolution),
            "sequence": np.asarray(sequence),
        }
        if wants_semantic:
            payload["cost"] = GOOSE_COSTS[labels].astype(np.float32)
            payload["semantic_label"] = labels
        if wants_occupancy:
            payload["occupancy"] = (100.0 * obstacle).astype(np.uint8)
        # Both products carry an obstacle raster, but each carries its own:
        # a combined response keeps occupancy's physical bodies.
        payload["obstacle"] = obstacle if wants_occupancy else semantic_obstacle
        temporary = self.result_path.with_name(
            f".{self.result_path.name}.{os.getpid()}.tmp.npz"
        )
        self.result_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.savez_compressed(temporary, **payload)
            os.replace(temporary, self.result_path)
        finally:
            temporary.unlink(missing_ok=True)


def _intersects(x: float, y: float, radius: float, x0: float, y0: float, x1: float, y1: float) -> bool:
    return x + radius >= x0 and x - radius < x1 and y + radius >= y0 and y - radius < y1


def _paint_ellipse(
    grid: np.ndarray,
    value: float | int,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    yaw: float,
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> None:
    radius = max(radius_x, radius_y)
    col0 = max(0, int(np.floor((center_x - radius - origin_x) / resolution)))
    col1 = min(grid.shape[1], int(np.ceil((center_x + radius - origin_x) / resolution)))
    row0 = max(0, int(np.floor((center_y - radius - origin_y) / resolution)))
    row1 = min(grid.shape[0], int(np.ceil((center_y + radius - origin_y) / resolution)))
    if row0 >= row1 or col0 >= col1:
        return
    rows, cols = np.mgrid[row0:row1, col0:col1]
    x = origin_x + (cols + 0.5) * resolution - center_x
    y = origin_y + (rows + 0.5) * resolution - center_y
    cy, sy = np.cos(yaw), np.sin(yaw)
    local_x = cy * x + sy * y
    local_y = -sy * x + cy * y
    inside = (local_x / radius_x) ** 2 + (local_y / radius_y) ** 2 <= 1.0
    view = grid[row0:row1, col0:col1]
    view[inside] = value


def _labeled_ellipses(world_path: Path) -> list[LabeledEllipse]:
    result: list[LabeledEllipse] = []
    root = ET.parse(world_path).getroot()
    for model in root.findall("./world/model"):
        pose_text = model.findtext("pose")
        radii_text = model.findtext(".//ellipsoid/radii")
        label_text = model.findtext(".//visual/plugin/label")
        if pose_text is None or radii_text is None or label_text is None:
            continue
        pose = [float(value) for value in pose_text.split()]
        radii = [float(value) for value in radii_text.split()]
        if len(pose) >= 6 and len(radii) >= 2:
            result.append(LabeledEllipse(pose[0], pose[1], radii[0], radii[1], pose[5], int(label_text)))
    return result
