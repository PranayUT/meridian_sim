"""Local-grid input for simulated UAV assistance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TerrainMap:
    """Elevation and grade derived from the DEM used by Gazebo."""

    elevation_grid: np.ndarray
    slope_grid: np.ndarray
    origin_x: float
    origin_y: float
    resolution: float
    soft_slope: float = 0.10
    hard_slope: float = 0.42

    @classmethod
    def from_tif(cls, path: Path, size_xy: float = 512.0) -> "TerrainMap":
        from PIL import Image

        elevation = np.asarray(Image.open(path), dtype=np.float64)
        if elevation.ndim != 2 or min(elevation.shape) < 2:
            raise MapFormatError("terrain DEM must be a two-dimensional elevation raster")
        # GeoTIFF row zero is north. Planner grids grow from the southwest.
        elevation = np.flipud(elevation - np.nanmin(elevation))
        resolution_x = size_xy / (elevation.shape[1] - 1)
        resolution_y = size_xy / (elevation.shape[0] - 1)
        if not np.isclose(resolution_x, resolution_y):
            raise MapFormatError("terrain DEM cells must be square")
        dz_dy, dz_dx = np.gradient(elevation, resolution_y, resolution_x)
        slope = np.hypot(dz_dx, dz_dy)
        return cls(elevation, slope, -size_xy / 2.0, -size_xy / 2.0, resolution_x)

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation_grid.shape

    def _indices(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        col = np.rint((x - self.origin_x) / self.resolution).astype(np.int64)
        row = np.rint((y - self.origin_y) / self.resolution).astype(np.int64)
        valid = (row >= 0) & (row < self.shape[0]) & (col >= 0) & (col < self.shape[1])
        return np.clip(row, 0, self.shape[0] - 1), np.clip(col, 0, self.shape[1] - 1), valid

    def elevation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Sample terrain height in Gazebo world coordinates."""
        row, col, valid = self._indices(np.asarray(x), np.asarray(y))
        return np.where(valid, self.elevation_grid[row, col], 0.0)

    def cost(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return a smooth grade cost and an impassable-slope mask."""
        row, col, valid = self._indices(x, y)
        slope = self.slope_grid[row, col]
        span = max(1e-6, self.hard_slope - self.soft_slope)
        cost = np.clip((slope - self.soft_slope) / span, 0.0, 1.0) ** 2
        return np.where(valid, cost, 1.0), (~valid) | (slope >= self.hard_slope)


class MapFormatError(ValueError):
    """The UAV map cannot be used by the planner."""


@dataclass(frozen=True)
class LocalGridMap:
    """One rolling lower-left-origin grid produced by onboard sensors."""

    grid: np.ndarray
    origin_x: float
    origin_y: float
    resolution: float

    def sample(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        col = np.floor((x - self.origin_x) / self.resolution).astype(np.int64)
        row = np.floor((y - self.origin_y) / self.resolution).astype(np.int64)
        valid = (row >= 0) & (row < self.grid.shape[0]) & (col >= 0) & (col < self.grid.shape[1])
        safe_row = np.clip(row, 0, self.grid.shape[0] - 1)
        safe_col = np.clip(col, 0, self.grid.shape[1] - 1)
        return self.grid[safe_row, safe_col], valid


@dataclass(frozen=True)
class UavMap:
    cost_grid: np.ndarray
    obstacle_grid: np.ndarray
    uncertainty_grid: np.ndarray
    origin_x: float
    origin_y: float
    resolution: float
    sequence: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.cost_grid.shape

    def sample(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        col = np.floor((x - self.origin_x) / self.resolution).astype(np.int64)
        row = np.floor((y - self.origin_y) / self.resolution).astype(np.int64)
        covered = (
            (row >= 0)
            & (row < self.shape[0])
            & (col >= 0)
            & (col < self.shape[1])
        )
        safe_row = np.clip(row, 0, self.shape[0] - 1)
        safe_col = np.clip(col, 0, self.shape[1] - 1)
        cost = np.where(covered, self.cost_grid[safe_row, safe_col], 0.0)
        obstacle = np.where(covered, self.obstacle_grid[safe_row, safe_col], 0.0)
        uncertainty = np.where(covered, self.uncertainty_grid[safe_row, safe_col], 1.0)
        valid = covered & np.isfinite(cost) & np.isfinite(obstacle) & np.isfinite(uncertainty)
        return cost, obstacle, uncertainty, valid


def _scalar(archive: object, key: str) -> object:
    return np.asarray(archive[key]).item()


def _load_native(archive: object, names: set[str]) -> UavMap:
    if "cost" not in names and "obstacle" not in names:
        raise MapFormatError("a local map needs a cost or obstacle raster")
    first = np.asarray(archive["cost" if "cost" in names else "obstacle"])
    if first.ndim != 2 or first.size == 0:
        raise MapFormatError("map rasters must be non-empty and two-dimensional")
    cost = np.asarray(archive["cost"], dtype=np.float32) if "cost" in names else np.zeros_like(first, dtype=np.float32)
    obstacle = np.asarray(archive["obstacle"], dtype=np.float32) if "obstacle" in names else np.zeros_like(first, dtype=np.float32)
    uncertainty = np.asarray(archive["uncertainty"], dtype=np.float32) if "uncertainty" in names else np.zeros_like(first, dtype=np.float32)
    origin_key = "origin_xy" if "origin_xy" in names else "origin"
    if origin_key not in names or "resolution" not in names:
        raise MapFormatError("a local map needs origin_xy and resolution")
    origin = np.asarray(archive[origin_key], dtype=np.float64)
    resolution = float(_scalar(archive, "resolution"))
    sequence = int(_scalar(archive, "sequence")) if "sequence" in names else 0
    return _validate(cost, obstacle, uncertainty, origin, resolution, sequence)


def _load_meridian(archive: object, names: set[str]) -> UavMap:
    label = np.asarray(archive["label"])
    classes = [str(item).strip().lower() for item in np.asarray(archive["classes"]).tolist()]
    scores = np.asarray(archive["scores"], dtype=np.float32)
    if label.ndim != 2 or label.dtype != np.uint8 or not classes:
        raise MapFormatError("the Meridian label raster or class table is invalid")
    covered = label != 255
    if np.any(covered) and int(np.max(label[covered])) >= len(classes):
        raise MapFormatError("a Meridian label does not index the class table")
    if scores.shape != (len(classes),):
        raise MapFormatError("the Meridian score table must match the class table")
    safe_label = np.clip(label, 0, len(classes) - 1)
    if {"height_mean", "height_variance"}.issubset(names):
        height = np.asarray(archive["height_mean"], dtype=np.float32)
        traversability = 1.0 - np.clip(height, 0.0, 1.0)
        calculated_variance = np.asarray(archive["height_variance"], dtype=np.float32)
    elif "trav_mean" in names:
        traversability = np.asarray(archive["trav_mean"], dtype=np.float32)
        calculated_variance = (
            np.asarray(archive["trav_var"], dtype=np.float32)
            if "trav_var" in names
            else np.where(covered, 0.25, np.nan).astype(np.float32)
        )
    elif {"label2", "margin"}.issubset(names):
        label2 = np.asarray(archive["label2"])
        margin = np.asarray(archive["margin"], dtype=np.float32)
        if label2.shape != label.shape or margin.shape != label.shape:
            raise MapFormatError("the Meridian uncertainty rasters must match label")
        safe_label2 = np.clip(label2, 0, len(classes) - 1)
        first = scores[safe_label]
        second = scores[safe_label2]
        gap = first - second
        traversability = (first + second) / 2.0 + margin * gap / 2.0
        calculated_variance = (1.0 - margin * margin) * gap * gap / 4.0
    else:
        traversability = scores[safe_label]
        calculated_variance = np.where(covered, 0.25, np.nan).astype(np.float32)
    cost = np.where(covered, 1.0 - traversability, np.nan).astype(np.float32)
    obstacle_indices = [index for index, name in enumerate(classes) if name in {"obstacle", "occupied", "solid"}]
    obstacle = np.isin(label, obstacle_indices).astype(np.float32)
    obstacle[~covered] = np.nan
    uncertainty = np.where(covered, calculated_variance, np.nan).astype(np.float32)
    transform = np.asarray(archive["transform"], dtype=np.float64)
    if transform.shape != (6,) or abs(transform[1]) > 1e-9 or abs(transform[3]) > 1e-9:
        raise MapFormatError("the simulator accepts only axis-aligned Meridian rasters")
    x_scale, _, x_origin, _, y_scale, y_origin = transform
    if x_scale <= 0.0 or y_scale >= 0.0 or not np.isclose(x_scale, -y_scale):
        raise MapFormatError("the Meridian raster must be north-up with square cells")
    meta = json.loads(str(_scalar(archive, "meta")))
    frame = str(meta.get("frame", ""))
    crs = str(_scalar(archive, "crs")).lower()
    if frame not in {"world", "map", "odom"} and crs not in {"local", "enu", "world"}:
        raise MapFormatError("the map must use simulator world or local ENU coordinates")
    # Meridian rasters store the north edge in row zero. The planner uses a
    # lower-left origin and rows that increase toward positive Y.
    cost = np.flipud(cost)
    obstacle = np.flipud(obstacle)
    uncertainty = np.flipud(uncertainty)
    origin = np.asarray((x_origin, y_origin + y_scale * label.shape[0]))
    sequence = int(meta.get("sequence", 0))
    return _validate(cost, obstacle, uncertainty, origin, x_scale, sequence)


def _validate(
    cost: np.ndarray,
    obstacle: np.ndarray,
    uncertainty: np.ndarray,
    origin: np.ndarray,
    resolution: float,
    sequence: int,
) -> UavMap:
    if cost.shape != obstacle.shape or cost.shape != uncertainty.shape:
        raise MapFormatError("cost, obstacle, and uncertainty must have the same shape")
    if cost.ndim != 2 or cost.size > 4_000_000:
        raise MapFormatError("the map must be a two-dimensional raster of at most 4,000,000 cells")
    if origin.shape != (2,) or not np.all(np.isfinite(origin)):
        raise MapFormatError("origin_xy must contain two finite values")
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise MapFormatError("resolution must be positive")
    return UavMap(
        cost_grid=np.clip(cost, 0.0, 1.0),
        obstacle_grid=np.clip(obstacle, 0.0, 1.0),
        uncertainty_grid=np.clip(uncertainty, 0.0, 1.0),
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        resolution=resolution,
        sequence=sequence,
    )


def load_uav_map(path: Path) -> UavMap:
    """Load either the simple simulator contract or Meridian's UAV NPZ."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
            if {"label", "classes", "scores", "transform", "crs", "meta"}.issubset(names):
                return _load_meridian(archive, names)
            return _load_native(archive, names)
    except MapFormatError:
        raise
    except Exception as error:
        raise MapFormatError(f"cannot read UAV map: {error}") from error


@dataclass
class MapStack:
    """The planner-facing fused map view."""

    aerial: UavMap | None = None
    terrain: TerrainMap | None = None
    ground_obstacles: LocalGridMap | None = None
    ground_obstacle_probability: LocalGridMap | None = None
    ground_semantics: LocalGridMap | None = None
    ground_semantic_obstacles: LocalGridMap | None = None
    collision_probability: float = 0.65
    semantic_collision_probability: float = 0.45

    def cost(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        total_cost = np.zeros_like(x)
        collision = np.zeros_like(x, dtype=bool)
        if self.terrain is not None:
            terrain_cost, terrain_collision = self.terrain.cost(x, y)
            total_cost = np.maximum(total_cost, terrain_cost)
            collision |= terrain_collision
        # Sample a small circular approximation of the quarter-scale rover's
        # footprint. Semantic labels retain a traversability cost, while the
        # posterior mass of brush, tree, and rock labels is a physical
        # collision signal in this simulator.
        for dx, dy in (
            (0.0, 0.0),
            (-0.225, 0.0),
            (0.225, 0.0),
            (0.0, -0.225),
            (0.0, 0.225),
            (-0.16, -0.16),
            (-0.16, 0.16),
            (0.16, -0.16),
            (0.16, 0.16),
        ):
            sample_x, sample_y = x + dx, y + dy
            if self.ground_semantics is not None:
                semantic, valid = self.ground_semantics.sample(sample_x, sample_y)
                total_cost = np.maximum(
                    total_cost,
                    np.where(valid & np.isfinite(semantic), semantic, 0.0),
                )
            if self.ground_semantic_obstacles is not None:
                obstacle, valid = self.ground_semantic_obstacles.sample(
                    sample_x, sample_y
                )
                known = valid & np.isfinite(obstacle)
                total_cost = np.maximum(
                    total_cost, np.where(known, 4.0 * obstacle, 0.0)
                )
                collision |= known & (
                    obstacle >= self.semantic_collision_probability
                )
            if self.ground_obstacles is not None:
                cells, valid = self.ground_obstacles.sample(sample_x, sample_y)
                # Meridian's TALL class is soft-but-expensive; SOLID is a hard
                # collision. Unknown remains available to uncertainty logic.
                total_cost = np.maximum(
                    total_cost, np.where(valid & (cells == 50), 3.0, 0.0)
                )
                collision |= valid & (cells == 100)
            if self.ground_obstacle_probability is not None:
                probability, valid = self.ground_obstacle_probability.sample(
                    sample_x, sample_y
                )
                known = valid & np.isfinite(probability)
                total_cost = np.maximum(
                    total_cost, np.where(known, 4.0 * probability, 0.0)
                )
            if self.aerial is not None:
                cost, obstacle, _, valid = self.aerial.sample(sample_x, sample_y)
                total_cost = np.maximum(
                    total_cost, np.where(valid, cost + 20.0 * obstacle, 0.0)
                )
                collision |= valid & (obstacle >= self.collision_probability)
        return total_cost, collision

    def uncertainty_exposure(
        self, trajectories: np.ndarray | None
    ) -> tuple[float, tuple[float, float, float, float] | None]:
        """Measure how much of the sampled rollout population needs evidence."""
        if trajectories is None or len(trajectories) == 0:
            return 0.0, None
        x, y = trajectories[..., 0], trajectories[..., 1]
        if self.aerial is None:
            uncertain = np.ones(x.shape, dtype=bool)
        else:
            _, _, variance, valid = self.aerial.sample(x, y)
            uncertain = (~valid) | (variance >= 0.04)
        if uncertain.ndim == 1:
            exposure = float(np.mean(uncertain))
        else:
            exposure = float(np.mean(np.any(uncertain, axis=1)))
        if not np.any(uncertain):
            return exposure, None
        selected_x, selected_y = x[uncertain], y[uncertain]
        padding = 4.0
        return exposure, (
            float(np.min(selected_x) - padding),
            float(np.min(selected_y) - padding),
            float(np.max(selected_x) + padding),
            float(np.max(selected_y) + padding),
        )
