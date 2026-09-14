"""Local-grid input for simulated UAV assistance."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# A UAV result answers one raised source. Meridian names the product it
# delivers, so the planner can tell which evidence channel the map is
# entitled to speak for.
SEMANTIC_MAP_TYPE = "semantic_traversability"
OCCUPANCY_MAP_TYPE = "canopy_obstacle"
COMBINED_MAP_TYPE = "occupancy_and_semantic"

# Maturity state is held as a sorted key array plus its timestamps so lookups
# are a searchsorted rather than a per-cell dict probe. Cell indices are grid
# coordinates, far inside the 32-bit half each field gets.
_EMPTY_MATURITY: tuple[np.ndarray, np.ndarray] = (
    np.empty(0, dtype=np.int64),
    np.empty(0, dtype=np.float64),
)


def _pack_cells(cell_x: np.ndarray, cell_y: np.ndarray) -> np.ndarray:
    return (cell_x.astype(np.int64) << 32) | (cell_y.astype(np.int64) & 0xFFFFFFFF)


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
    obstacle_clearance_grid: np.ndarray
    uncertainty_grid: np.ndarray
    origin_x: float
    origin_y: float
    resolution: float
    sequence: int
    map_types: tuple[str, ...] = ()

    @property
    def shape(self) -> tuple[int, int]:
        return self.cost_grid.shape

    def provides(self, map_type: str) -> bool:
        """True when this result answers the named evidence channel.

        A map that declares no types is a legacy or external product that
        carries every layer, so it still supersedes both channels.
        """
        return not self.map_types or map_type in self.map_types

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

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """World extent covered by this product, as (x0, y0, x1, y1)."""
        return (
            self.origin_x,
            self.origin_y,
            self.origin_x + self.shape[1] * self.resolution,
            self.origin_y + self.shape[0] * self.resolution,
        )

    def covers_any(self, x: np.ndarray, y: np.ndarray) -> bool:
        """Cheap rejection test before rasterising a 25 m window.

        Retained products accumulate along the route, and most of them are
        nowhere near the current rollout population. Comparing two bounding
        boxes costs four floats; sampling the raster costs a full index pass
        per query point.
        """
        x0, y0, x1, y1 = self.bounds
        return bool(
            np.any((x >= x0) & (x < x1) & (y >= y0) & (y < y1))
        )

    def sample_all(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample every layer, planner-facing and uncertainty, in one pass."""
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
        clearance = np.where(
            covered, self.obstacle_clearance_grid[safe_row, safe_col], 0.0
        )
        uncertainty = np.where(
            covered, self.uncertainty_grid[safe_row, safe_col], 1.0
        )
        # Coverage is decided by the planner layers alone, as it was when the
        # uncertainty layer came from a second sample() call. A non-finite
        # uncertainty inside a covered cell stays non-finite here rather than
        # silently changing which cells the aerial product speaks for.
        valid = (
            covered
            & np.isfinite(cost)
            & np.isfinite(obstacle)
            & np.isfinite(clearance)
        )
        return cost, obstacle, clearance, uncertainty, valid

    def sample_planner(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample all planner-facing layers with one coordinate lookup."""
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
        obstacle = np.where(
            covered, self.obstacle_grid[safe_row, safe_col], 0.0
        )
        clearance = np.where(
            covered, self.obstacle_clearance_grid[safe_row, safe_col], 0.0
        )
        valid = (
            covered
            & np.isfinite(cost)
            & np.isfinite(obstacle)
            & np.isfinite(clearance)
        )
        return cost, obstacle, clearance, valid


@dataclass(frozen=True)
class AssistanceEvaluation:
    """One Meridian-style source selected from a frozen rollout population."""

    uncertainty_exposure: float
    roi: tuple[float, float, float, float] | None
    source: str = ""
    map_type: str = ""
    decision_relevant: bool = False
    action_relevant: bool = False
    probe_relevant: bool = False


@dataclass(frozen=True)
class CounterfactualViability:
    """Meridian's free/occupied hypothesis test, reported rather than reduced."""

    baseline: float
    free: float
    occupied: float
    useful_fraction: float
    decision_relevant: bool


def _arc_length(states: np.ndarray) -> float:
    """Mean path length of a set of rollouts, in metres."""
    if states.shape[-2] < 2:
        return 0.0
    steps = np.hypot(
        np.diff(states[..., 0], axis=-1), np.diff(states[..., 1], axis=-1)
    )
    return float(np.mean(np.sum(steps, axis=-1)))


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
    map_types = (
        tuple(str(item) for item in np.asarray(archive["map_types"]).reshape(-1).tolist())
        if "map_types" in names
        else ()
    )
    return _validate(
        cost, obstacle, uncertainty, origin, resolution, sequence, map_types
    )


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
    map_types: tuple[str, ...] = (),
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
        obstacle_clearance_grid=_obstacle_clearance(
            np.clip(obstacle, 0.0, 1.0), resolution
        ),
        uncertainty_grid=np.clip(uncertainty, 0.0, 1.0),
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        resolution=resolution,
        sequence=sequence,
        map_types=map_types,
    )


def _obstacle_clearance(
    obstacle: np.ndarray, resolution: float, margin_m: float = 0.75
) -> np.ndarray:
    """Return a compact linear falloff around an aerial obstacle raster.

    The hard body raster remains unchanged. This derived layer gives MPPI and
    its local guide a lateral gradient, so they do not choose a path exactly
    one discretized footprint cell from a body and then clip it in Gazebo.
    """
    occupied = np.nan_to_num(obstacle, nan=0.0).astype(np.float32, copy=False)
    clearance = np.zeros_like(occupied, dtype=np.float32)
    cells = int(np.ceil(margin_m / resolution))
    rows, cols = occupied.shape
    for drow in range(-cells, cells + 1):
        for dcol in range(-cells, cells + 1):
            distance = resolution * np.hypot(drow, dcol)
            if distance >= margin_m:
                continue
            weight = 1.0 - distance / margin_m
            source_row = slice(max(0, -drow), min(rows, rows - drow))
            source_col = slice(max(0, -dcol), min(cols, cols - dcol))
            target_row = slice(max(0, drow), min(rows, rows + drow))
            target_col = slice(max(0, dcol), min(cols, cols + dcol))
            clearance[target_row, target_col] = np.maximum(
                clearance[target_row, target_col],
                weight * occupied[source_row, source_col],
            )
    return clearance


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
    aerial_history: list[UavMap] = field(default_factory=list, repr=False)
    terrain: TerrainMap | None = None
    ground_obstacles: LocalGridMap | None = None
    ground_obstacle_probability: LocalGridMap | None = None
    ground_occupancy_uncertainty: LocalGridMap | None = None
    ground_semantics: LocalGridMap | None = None
    ground_semantic_obstacles: LocalGridMap | None = None
    ground_semantic_uncertainty: LocalGridMap | None = None
    collision_probability: float = 0.65
    semantic_collision_probability: float = 0.45
    uncertainty_maturity_s: float = 1.0
    _uncertain_since: dict[str, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, init=False, repr=False
    )
    _maturity_update_s: dict[str, float] = field(
        default_factory=dict, init=False, repr=False
    )

    # Rollouts kept when subsampling the population for diagnostics.
    _DIAGNOSTIC_ROLLOUTS = 24

    _FOOTPRINT_OFFSETS = (
        (0.0, 0.0),
        (-0.225, 0.0),
        (0.225, 0.0),
        (0.0, -0.225),
        (0.0, 0.225),
        (-0.16, -0.16),
        (-0.16, 0.16),
        (0.16, -0.16),
        (0.16, 0.16),
    )

    def clear_aerial(self) -> None:
        self.aerial = None
        self.aerial_history.clear()

    def add_aerial(self, aerial: UavMap) -> None:
        """Retain local products instead of replacing prior UAV evidence."""
        self.aerial = aerial
        self.aerial_history.append(aerial)

    @property
    def has_aerial(self) -> bool:
        return bool(self._aerial_products())

    def _aerial_products(self) -> list[UavMap]:
        products = list(self.aerial_history)
        if self.aerial is not None and not any(
            item is self.aerial for item in products
        ):
            products.append(self.aerial)
        return products

    def _sample_aerial(
        self, x: np.ndarray, y: np.ndarray, map_type: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample the newest covering UAV product of one evidence type."""
        cost = np.zeros_like(x, dtype=np.float64)
        obstacle = np.zeros_like(x, dtype=np.float64)
        clearance = np.zeros_like(x, dtype=np.float64)
        uncertainty = np.ones_like(x, dtype=np.float64)
        valid = np.zeros_like(x, dtype=bool)
        # Newest product wins where fixed 25 m requests overlap. Retained
        # products are cheap to hold but were not cheap to consult: every one
        # of them rasterised the full query on every planner call, so the
        # per-tick cost grew linearly with how much of the route had been
        # answered. Reject by extent first, and stop once every query point
        # has an answer.
        for product in reversed(self._aerial_products()):
            if not product.provides(map_type):
                continue
            if not product.covers_any(x, y):
                continue
            (
                p_cost,
                p_obstacle,
                p_clearance,
                p_uncertainty,
                p_valid,
            ) = product.sample_all(x, y)
            selected = p_valid & ~valid
            cost = np.where(selected, p_cost, cost)
            obstacle = np.where(selected, p_obstacle, obstacle)
            clearance = np.where(selected, p_clearance, clearance)
            uncertainty = np.where(selected, p_uncertainty, uncertainty)
            valid |= p_valid
            if valid.all():
                break
        return cost, obstacle, clearance, uncertainty, valid

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
        for dx, dy in self._FOOTPRINT_OFFSETS:
            sample_x, sample_y = x + dx, y + dy
            aerial_cost = np.zeros_like(sample_x, dtype=np.float64)
            aerial_obstacle = np.zeros_like(sample_x, dtype=np.float64)
            aerial_valid = np.zeros_like(sample_x, dtype=bool)
            semantic_override = np.zeros_like(sample_x, dtype=bool)
            occupancy_override = np.zeros_like(sample_x, dtype=bool)
            if self._aerial_products():
                (
                    occupancy_cost,
                    occupancy_obstacle,
                    occupancy_clearance,
                    _,
                    occupancy_valid,
                ) = self._sample_aerial(
                    sample_x, sample_y, OCCUPANCY_MAP_TYPE
                )
                (
                    semantic_cost,
                    semantic_obstacle,
                    semantic_clearance,
                    _,
                    semantic_valid,
                ) = self._sample_aerial(
                    sample_x, sample_y, SEMANTIC_MAP_TYPE
                )
                semantic_override = semantic_valid
                occupancy_override = occupancy_valid
            if self.ground_semantics is not None:
                semantic, valid = self.ground_semantics.sample(sample_x, sample_y)
                known = valid & np.isfinite(semantic) & ~semantic_override
                total_cost = np.maximum(
                    total_cost,
                    np.where(known, semantic, 0.0),
                )
            if self.ground_semantic_obstacles is not None:
                obstacle, valid = self.ground_semantic_obstacles.sample(
                    sample_x, sample_y
                )
                known = valid & np.isfinite(obstacle) & ~semantic_override
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
                known = valid & ~occupancy_override
                total_cost = np.maximum(
                    total_cost, np.where(known & (cells == 50), 3.0, 0.0)
                )
                collision |= known & (cells == 100)
            if self.ground_obstacle_probability is not None:
                probability, valid = self.ground_obstacle_probability.sample(
                    sample_x, sample_y
                )
                known = (
                    valid & np.isfinite(probability) & ~occupancy_override
                )
                total_cost = np.maximum(
                    total_cost, np.where(known, 4.0 * probability, 0.0)
                )
            if self._aerial_products():
                # Semantic traversability describes what the canopy looks
                # like from above; it is not a second physical-body raster.
                # Treating its full bush/tree extent as hard collision made a
                # semantic answer much more restrictive than Gazebo geometry.
                # Only occupancy supplies physical collision and clearance.
                aerial_cost = np.maximum(occupancy_cost, semantic_cost)
                aerial_obstacle = occupancy_obstacle
                aerial_clearance = occupancy_clearance
                aerial_valid = occupancy_valid | semantic_valid
                total_cost = np.maximum(
                    total_cost,
                    np.where(
                        aerial_valid,
                        aerial_cost + 20.0 * aerial_obstacle,
                        0.0,
                    ),
                )
                # A hard occupied cell says where the body is; this soft ring
                # says not to skim that boundary. It is intentionally below
                # the local guide's 3.5 hard-block cutoff.
                total_cost = np.maximum(
                    total_cost,
                    np.where(aerial_valid, 3.0 * aerial_clearance, 0.0),
                )
                collision |= aerial_valid & (
                    aerial_obstacle >= self.collision_probability
                )
        return total_cost, collision

    def uncertainty_exposure(
        self, trajectories: np.ndarray | None
    ) -> tuple[float, tuple[float, float, float, float] | None]:
        """Compatibility view of :meth:`evaluate_assistance`."""
        evaluation = self.evaluate_assistance(trajectories, None, 0.0)
        return evaluation.uncertainty_exposure, evaluation.roi

    def evaluate_assistance(
        self,
        trajectories: np.ndarray | None,
        initial_xy: tuple[float, float] | None,
        exposure_threshold: float,
        now_s: float | None = None,
    ) -> AssistanceEvaluation:
        """Evaluate occupancy and semantic evidence as Meridian Drive does.

        Each channel reports the fraction of swept-footprint cells above its
        uncertainty rule. Occupancy additionally receives the free/occupied
        counterfactual test. Aerial evidence supersedes ground evidence only
        inside the returned map footprint.
        """
        if trajectories is None or len(trajectories) == 0:
            return AssistanceEvaluation(0.0, None)
        candidates: list[AssistanceEvaluation] = []
        occupancy = self._occupancy_evaluation(trajectories, initial_xy, now_s)
        if occupancy is not None:
            candidates.append(occupancy)
        semantic = self._evidence_exposure(
            trajectories,
            self.ground_semantic_uncertainty,
            source="ground_semantic_cost",
            map_type=SEMANTIC_MAP_TYPE,
            uncertainty_min=0.04,
            now_s=now_s,
        )
        if semantic is not None:
            candidates.append(semantic)
        if not candidates:
            return AssistanceEvaluation(0.0, None)
        relevant = next((item for item in candidates if item.decision_relevant), None)
        if relevant is not None:
            return relevant
        # Meridian evaluates the control occupancy source first, then the
        # remaining evidence topics in configured order. Preserve that stable
        # priority instead of letting small score noise switch sources.
        selected = next(
            (
                item
                for item in candidates
                if item.uncertainty_exposure >= exposure_threshold
            ),
            None,
        )
        if selected is not None:
            return selected
        highest = max(candidates, key=lambda item: item.uncertainty_exposure)
        # Keep the best mature ROI available even below the exposure gate.
        # AssistanceManager still enforces that gate for ordinary requests,
        # but it also needs the ROI when commanded motion has demonstrably
        # failed. Discarding it here made the UAV blind for the full 30 s until
        # the experiment harness dragged a physically wedged rover.
        return highest

    def evaluate_action_assistance(
        self,
        trajectory: np.ndarray | None,
        exposure_threshold: float = 0.20,
        now_s: float | None = None,
    ) -> AssistanceEvaluation:
        """Evaluate uncertainty on the trajectory the controller selected.

        The population policy answers whether uncertainty changes the choice
        among sampled controls. This complementary gate answers whether the
        control that will actually be executed crosses unresolved evidence.
        Source-level persistence is applied by AssistanceManager. Per-cell
        maturity is intentionally not applied here: a moving three-second
        action horizon continually acquires new cells, and requiring those
        future cells to remain in the horizon would suppress the warning until
        after the rover stopped.
        """
        if trajectory is None or len(trajectory) == 0:
            return AssistanceEvaluation(0.0, None)
        trajectories = np.asarray(trajectory)
        if trajectories.ndim == 2:
            trajectories = trajectories[None, ...]
        candidates: list[AssistanceEvaluation] = []
        occupancy = self._evidence_exposure(
            trajectories,
            self.ground_occupancy_uncertainty,
            source="lidar_occupancy",
            map_type=OCCUPANCY_MAP_TYPE,
            uncertainty_min=0.04,
            probability_layer=self.ground_obstacle_probability,
            now_s=None,
        )
        if occupancy is not None:
            candidates.append(occupancy)
        semantic = self._evidence_exposure(
            trajectories,
            self.ground_semantic_uncertainty,
            source="ground_semantic_cost",
            map_type=SEMANTIC_MAP_TYPE,
            uncertainty_min=0.04,
            now_s=None,
        )
        if semantic is not None:
            candidates.append(semantic)
        if not candidates:
            return AssistanceEvaluation(0.0, None)
        selected = next(
            (
                item
                for item in candidates
                if item.roi is not None
                and item.uncertainty_exposure >= exposure_threshold
            ),
            None,
        )
        if selected is not None:
            return AssistanceEvaluation(
                selected.uncertainty_exposure,
                selected.roi,
                selected.source,
                selected.map_type,
                action_relevant=True,
            )
        return max(candidates, key=lambda item: item.uncertainty_exposure)

    def evaluate_probe_assistance(
        self,
        trajectory: np.ndarray | None,
        exposure_threshold: float = 0.02,
    ) -> AssistanceEvaluation:
        """Evaluate the fixed-distance corridor used to warn before contact.

        Unlike the population evaluator, this intentionally counts sampled
        swept-footprint points rather than distinct cells. That exactly
        matches the shadow-trace metric calibrated against Route-11 drags and
        preserves a constant warning distance when the selected time-horizon
        trajectory collapses as the rover slows.
        """
        if trajectory is None or len(trajectory) == 0:
            return AssistanceEvaluation(0.0, None)
        if (
            self.ground_occupancy_uncertainty is None
            and self.ground_obstacle_probability is None
        ):
            return AssistanceEvaluation(0.0, None)
        trajectories = np.asarray(trajectory)
        if trajectories.ndim == 2:
            trajectories = trajectories[None, ...]
        x, y = self._swept_points(trajectories)
        x = x.reshape(-1)
        y = y.reshape(-1)
        unknown = np.ones(x.shape, dtype=bool)
        ambiguous = np.zeros(x.shape, dtype=bool)
        high_variance = np.zeros(x.shape, dtype=bool)
        if self.ground_occupancy_uncertainty is not None:
            variance, valid = self.ground_occupancy_uncertainty.sample(x, y)
            known = valid & np.isfinite(variance)
            high_variance = known & (variance >= 0.04)
            unknown &= ~known
        if self.ground_obstacle_probability is not None:
            probability, valid = self.ground_obstacle_probability.sample(x, y)
            known = valid & np.isfinite(probability)
            ambiguous = known & (probability >= 0.20) & (probability <= 0.80)
            unknown &= ~known
        uncertain = unknown | ambiguous | high_variance
        if self._aerial_products():
            _, _, _, aerial_uncertainty, aerial_valid = self._sample_aerial(
                x, y, OCCUPANCY_MAP_TYPE
            )
            uncertain = np.where(aerial_valid, aerial_uncertainty >= 0.04, uncertain)
        exposure = float(np.mean(uncertain))
        if not np.any(uncertain):
            return AssistanceEvaluation(
                exposure, None, "lidar_occupancy", COMBINED_MAP_TYPE
            )
        selected_x = x[uncertain]
        selected_y = y[uncertain]
        roi = (
            float(np.min(selected_x)),
            float(np.min(selected_y)),
            float(np.max(selected_x)),
            float(np.max(selected_y)),
        )
        return AssistanceEvaluation(
            exposure,
            roi,
            "lidar_occupancy",
            COMBINED_MAP_TYPE,
            probe_relevant=exposure > exposure_threshold,
        )

    def uncertainty_diagnostics(
        self,
        population: np.ndarray | None,
        selected: np.ndarray | None,
        probe: np.ndarray | None,
        initial_xy: tuple[float, float] | None,
        now_s: float | None = None,
    ) -> dict[str, float | None]:
        """Measure every candidate stuck-predictor over one planner cycle.

        This is instrumentation, not policy. Nothing here feeds a request; the
        purpose is to record many uncertainty, hazard, and geometry variables
        at a fixed cadence so that drag events can be correlated against all of
        them offline instead of one hypothesis at a time.

        Three geometries are measured wherever a variable supports it:

        - ``population``: every sampled rollout, which is what the Meridian
          exposure policy consumes;
        - ``selected``: the trajectory the controller actually chose, whose
          extent shrinks with commanded speed;
        - ``probe``: a fixed-distance forward corridor, which keeps a constant
          warning distance even when the vehicle has slowed to a crawl.
        """
        # The rollout population is 192 x 60 states before the nine footprint
        # offsets. Measuring all of it twice a second cost more than the whole
        # 50 ms control tick once retained aerial products were in play, and a
        # starved controller changes the very behaviour being measured. A
        # strided subsample answers the same distributional questions.
        sampled = None
        if population is not None and len(population):
            population = np.asarray(population)
            stride = max(1, len(population) // self._DIAGNOSTIC_ROLLOUTS)
            sampled = population[::stride]
        out: dict[str, float | None] = {}
        for label, states, fused in (
            ("pop", sampled, False),
            ("path", selected, True),
            ("probe", probe, True),
        ):
            out.update(self._geometry_diagnostics(label, states, fused))
        if sampled is not None and initial_xy is not None:
            viability = self._counterfactual_viability(
                sampled, initial_xy, "lidar_occupancy", now_s
            )
            if viability is not None:
                out["cf_baseline_viability"] = viability.baseline
                out["cf_free_viability"] = viability.free
                out["cf_occupied_viability"] = viability.occupied
                out["cf_useful_fraction"] = viability.useful_fraction
                out["cf_decision_relevant"] = float(viability.decision_relevant)
        if initial_xy is not None:
            out.update(self._footprint_diagnostics(initial_xy))
        return out

    def _geometry_diagnostics(
        self, label: str, states: np.ndarray | None, fused: bool = True
    ) -> dict[str, float | None]:
        """Uncertainty, hazard, and extent measured over one set of states."""
        keys = (
            "occ_exposure",
            "occ_unknown_frac",
            "occ_ambiguous_frac",
            "occ_variance_frac",
            "occ_probability_max",
            "occ_probability_mean",
            "occ_blocked_frac",
            "sem_exposure",
            "sem_unknown_frac",
            "sem_cost_max",
            "sem_cost_mean",
            "sem_blocked_frac",
            "fused_cost_max",
            "fused_cost_mean",
            "fused_blocked_frac",
            "extent_m",
        )
        out: dict[str, float | None] = {f"{label}_{key}": None for key in keys}
        if states is None or len(states) == 0:
            return out
        states = np.asarray(states)
        if states.ndim == 2:
            states = states[None, ...]
        x, y = self._swept_points(states)
        x = x.reshape(-1)
        y = y.reshape(-1)
        out[f"{label}_extent_m"] = _arc_length(states)

        variance_layer = self.ground_occupancy_uncertainty
        probability_layer = self.ground_obstacle_probability
        if variance_layer is not None or probability_layer is not None:
            unknown = np.ones(x.shape, dtype=bool)
            ambiguous = np.zeros(x.shape, dtype=bool)
            high_variance = np.zeros(x.shape, dtype=bool)
            probability = np.full(x.shape, np.nan)
            if variance_layer is not None:
                variance, valid = variance_layer.sample(x, y)
                known = valid & np.isfinite(variance)
                high_variance = known & (variance >= 0.04)
                unknown &= ~known
            if probability_layer is not None:
                sampled, valid = probability_layer.sample(x, y)
                known = valid & np.isfinite(sampled)
                probability = np.where(known, sampled, np.nan)
                ambiguous = known & (sampled >= 0.20) & (sampled <= 0.80)
                unknown &= ~known
            out[f"{label}_occ_unknown_frac"] = float(np.mean(unknown))
            out[f"{label}_occ_ambiguous_frac"] = float(np.mean(ambiguous))
            out[f"{label}_occ_variance_frac"] = float(np.mean(high_variance))
            out[f"{label}_occ_exposure"] = float(
                np.mean(unknown | ambiguous | high_variance)
            )
            if np.any(np.isfinite(probability)):
                out[f"{label}_occ_probability_max"] = float(
                    np.nanmax(probability)
                )
                out[f"{label}_occ_probability_mean"] = float(
                    np.nanmean(probability)
                )
                out[f"{label}_occ_blocked_frac"] = float(
                    np.mean(np.nan_to_num(probability, nan=0.0) >= 0.50)
                )

        semantic_layer = self.ground_semantics
        if semantic_layer is not None:
            cost, valid = semantic_layer.sample(x, y)
            known = valid & np.isfinite(cost)
            out[f"{label}_sem_unknown_frac"] = float(np.mean(~known))
            if np.any(known):
                out[f"{label}_sem_cost_max"] = float(np.max(cost[known]))
                out[f"{label}_sem_cost_mean"] = float(np.mean(cost[known]))
        semantic_variance = self.ground_semantic_uncertainty
        if semantic_variance is not None:
            variance, valid = semantic_variance.sample(x, y)
            known = valid & np.isfinite(variance)
            out[f"{label}_sem_exposure"] = float(
                np.mean((~known) | (known & (variance >= 0.04)))
            )
        semantic_obstacles = self.ground_semantic_obstacles
        if semantic_obstacles is not None:
            obstacle, valid = semantic_obstacles.sample(x, y)
            known = valid & np.isfinite(obstacle)
            out[f"{label}_sem_blocked_frac"] = float(
                np.mean(known & (obstacle >= self.semantic_collision_probability))
            )

        if fused:
            cost, blocked = self.cost(states[..., 0], states[..., 1])
            if cost.size:
                out[f"{label}_fused_cost_max"] = float(np.max(cost))
                out[f"{label}_fused_cost_mean"] = float(np.mean(cost))
                out[f"{label}_fused_blocked_frac"] = float(np.mean(blocked))
        return out

    def _footprint_diagnostics(
        self, initial_xy: tuple[float, float]
    ) -> dict[str, float | None]:
        """What the maps say about the cells the vehicle is standing on."""
        out: dict[str, float | None] = {
            "here_occ_probability": None,
            "here_occ_variance": None,
            "here_sem_cost": None,
            "here_fused_cost": None,
            "here_blocked": None,
        }
        x = np.array(
            [initial_xy[0] + dx for dx, _ in self._FOOTPRINT_OFFSETS],
            dtype=np.float64,
        )
        y = np.array(
            [initial_xy[1] + dy for _, dy in self._FOOTPRINT_OFFSETS],
            dtype=np.float64,
        )
        if self.ground_obstacle_probability is not None:
            probability, valid = self.ground_obstacle_probability.sample(x, y)
            known = valid & np.isfinite(probability)
            if np.any(known):
                out["here_occ_probability"] = float(np.max(probability[known]))
        if self.ground_occupancy_uncertainty is not None:
            variance, valid = self.ground_occupancy_uncertainty.sample(x, y)
            known = valid & np.isfinite(variance)
            if np.any(known):
                out["here_occ_variance"] = float(np.max(variance[known]))
        if self.ground_semantics is not None:
            cost, valid = self.ground_semantics.sample(x, y)
            known = valid & np.isfinite(cost)
            if np.any(known):
                out["here_sem_cost"] = float(np.max(cost[known]))
        # MapStack.cost sweeps the footprint itself, so it takes the centre.
        centre_x = np.array([initial_xy[0]], dtype=np.float64)
        centre_y = np.array([initial_xy[1]], dtype=np.float64)
        cost, blocked = self.cost(centre_x, centre_y)
        if cost.size:
            out["here_fused_cost"] = float(np.max(cost))
            out["here_blocked"] = float(np.mean(blocked))
        return out

    def _evidence_exposure(
        self,
        trajectories: np.ndarray,
        layer: LocalGridMap | None,
        source: str,
        map_type: str,
        uncertainty_min: float,
        probability_layer: LocalGridMap | None = None,
        now_s: float | None = None,
    ) -> AssistanceEvaluation | None:
        if layer is None:
            return None
        x, y = self._swept_points(trajectories)
        x = x.reshape(-1)
        y = y.reshape(-1)
        uncertainty, valid = layer.sample(x, y)
        uncertain = valid & ((~np.isfinite(uncertainty)) | (uncertainty >= uncertainty_min))
        if probability_layer is not None:
            probability, probability_valid = probability_layer.sample(x, y)
            known_probability = probability_valid & np.isfinite(probability)
            uncertain |= valid & (
                (~known_probability) | ((probability >= 0.20) & (probability <= 0.80))
            )
        if self._aerial_products():
            _, _, _, aerial_uncertainty, aerial_valid = self._sample_aerial(
                x, y, map_type
            )
            uncertain = np.where(
                aerial_valid,
                aerial_uncertainty >= uncertainty_min,
                uncertain,
            )
            valid |= aerial_valid
        if not np.any(valid):
            return None
        # Meridian counts distinct swept cells, not repeated trajectory visits.
        cell_x = np.floor(x[valid] / layer.resolution).astype(np.int64)
        cell_y = np.floor(y[valid] / layer.resolution).astype(np.int64)
        # np.unique(axis=0) sorts a two-column void view, which falls back to
        # generic element comparison: ~25 ms for a full swept population, twice
        # per control tick. Packing the pair into one int64 and taking the 1-D
        # path is ~10x faster for an identical result. Cell indices are grid
        # coordinates, far inside the 32-bit half each field gets.
        keys = (cell_x << 32) | (cell_y & 0xFFFFFFFF)
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        coordinates = np.column_stack((
            unique_keys >> 32,
            ((unique_keys & 0xFFFFFFFF) ^ 0x80000000) - 0x80000000,
        ))
        raw_flags = np.zeros(len(coordinates), dtype=bool)
        np.logical_or.at(raw_flags, inverse, uncertain[valid])
        flags = self._mature_uncertainty(source, coordinates, raw_flags, now_s)
        exposure = float(np.mean(flags))
        if not np.any(flags):
            return AssistanceEvaluation(exposure, None, source, map_type)
        selected = coordinates[flags].astype(np.float64) * layer.resolution
        return AssistanceEvaluation(
            exposure,
            (
                float(np.min(selected[:, 0])),
                float(np.min(selected[:, 1])),
                float(np.max(selected[:, 0]) + layer.resolution),
                float(np.max(selected[:, 1]) + layer.resolution),
            ),
            source,
            map_type,
        )

    def _occupancy_evaluation(
        self,
        trajectories: np.ndarray,
        initial_xy: tuple[float, float] | None,
        now_s: float | None,
    ) -> AssistanceEvaluation | None:
        exposure = self._evidence_exposure(
            trajectories,
            self.ground_occupancy_uncertainty,
            source="lidar_occupancy",
            map_type=OCCUPANCY_MAP_TYPE,
            uncertainty_min=0.04,
            probability_layer=self.ground_obstacle_probability,
            now_s=now_s,
        )
        if exposure is None or initial_xy is None or self.ground_obstacle_probability is None:
            return exposure
        viability = self._counterfactual_viability(
            trajectories, initial_xy, exposure.source, now_s
        )
        if viability is None:
            return exposure
        return AssistanceEvaluation(
            exposure.uncertainty_exposure,
            exposure.roi,
            exposure.source,
            exposure.map_type,
            viability.decision_relevant,
        )

    def _counterfactual_viability(
        self,
        trajectories: np.ndarray,
        initial_xy: tuple[float, float],
        source: str,
        now_s: float | None,
    ) -> "CounterfactualViability | None":
        """Meridian's free/occupied hypothesis test over a rollout population.

        Shared by the live occupancy evaluation and by the offline uncertainty
        diagnostics so the two cannot report different viabilities for the same
        population.
        """
        if self.ground_obstacle_probability is None:
            return None
        states_x = trajectories[..., 0]
        states_y = trajectories[..., 1]
        probabilities: list[np.ndarray] = []
        uncertain_samples: list[np.ndarray] = []
        for dx, dy in self._FOOTPRINT_OFFSETS:
            x, y = states_x + dx, states_y + dy
            probability, probability_valid = self.ground_obstacle_probability.sample(x, y)
            variance, variance_valid = self.ground_occupancy_uncertainty.sample(x, y)
            known = probability_valid & variance_valid & np.isfinite(probability) & np.isfinite(variance)
            probability = np.where(known, probability, 0.5)
            uncertain = (~known) | (variance >= 0.04) | (
                (probability >= 0.20) & (probability <= 0.80)
            )
            if self._aerial_products():
                (
                    _,
                    aerial_probability,
                    _,
                    aerial_uncertainty,
                    aerial_valid,
                ) = self._sample_aerial(x, y, OCCUPANCY_MAP_TYPE)
                probability = np.where(aerial_valid, aerial_probability, probability)
                uncertain = np.where(aerial_valid, aerial_uncertainty >= 0.04, uncertain)
            uncertain = self._mature_sample_uncertainty(
                source,
                x,
                y,
                uncertain,
                self.ground_occupancy_uncertainty.resolution,
                now_s,
            )
            probabilities.append(probability)
            uncertain_samples.append(uncertain)
        probability = np.maximum.reduce(probabilities)
        uncertain = np.logical_or.reduce(uncertain_samples)
        baseline_max = np.max(probability, axis=-1)
        free_max = np.max(np.where(uncertain, 0.05, probability), axis=-1)
        occupied_max = np.max(np.where(uncertain, 0.95, probability), axis=-1)
        progress = np.hypot(
            states_x[..., -1] - initial_xy[0],
            states_y[..., -1] - initial_xy[1],
        )
        useful = np.isfinite(progress) & (progress >= 0.25)
        baseline_viability = float(np.mean(useful & (baseline_max < 0.50)))
        free_viability = float(np.mean(useful & (free_max < 0.50)))
        occupied_viability = float(np.mean(useful & (occupied_max < 0.50)))
        # bool(): `np.any(...) and ...` returns np.bool_ when it short-circuits
        # on a False first operand, and np.bool_ is not JSON serializable.
        decision_relevant = bool(
            np.any(uncertain)
            and baseline_viability < 0.20
            and max(free_viability, occupied_viability) - baseline_viability >= 0.15
            and abs(free_viability - occupied_viability) >= 0.15
        )
        return CounterfactualViability(
            baseline_viability,
            free_viability,
            occupied_viability,
            float(np.mean(useful)),
            decision_relevant,
        )

    def _mature_uncertainty(
        self,
        source: str,
        coordinates: np.ndarray,
        uncertain: np.ndarray,
        now_s: float | None,
    ) -> np.ndarray:
        """Require the same unresolved cells to persist in the swept set.

        Meridian's policy already persists a source-level exposure.  This
        source-side gate prevents different newly encountered frontier cells
        from satisfying that timer on one another's behalf.  Calls without a
        clock retain the immediate behavior used by compatibility callers.
        """
        if now_s is None or self.uncertainty_maturity_s <= 0.0:
            return uncertain
        now_s = float(now_s)
        previous_update = self._maturity_update_s.get(source)
        if previous_update is not None and now_s < previous_update:
            self._uncertain_since.pop(source, None)
        self._maturity_update_s[source] = now_s

        mature = np.zeros_like(uncertain)
        selected = np.flatnonzero(uncertain)
        # Dropping a cell from the current rollout population resets its
        # opportunity window. This is what distinguishes a moving frontier
        # from one location that remains unresolved under repeated planning,
        # so only the cells uncertain right now are carried forward.
        if selected.size == 0:
            self._uncertain_since[source] = _EMPTY_MATURITY
            return mature
        keys = _pack_cells(coordinates[selected, 0], coordinates[selected, 1])
        order = np.argsort(keys, kind="stable")
        keys, selected = keys[order], selected[order]
        since = np.full(keys.shape, now_s, dtype=np.float64)
        known_keys, known_since = self._uncertain_since.get(source, _EMPTY_MATURITY)
        if known_keys.size:
            index = np.minimum(
                np.searchsorted(known_keys, keys), known_keys.size - 1
            )
            carried = known_keys[index] == keys
            since[carried] = known_since[index[carried]]
        self._uncertain_since[source] = (keys, since)
        mature[selected] = (now_s - since) >= self.uncertainty_maturity_s
        return mature

    def _mature_sample_uncertainty(
        self,
        source: str,
        x: np.ndarray,
        y: np.ndarray,
        uncertain: np.ndarray,
        resolution: float,
        now_s: float | None,
    ) -> np.ndarray:
        """Apply the exposure maturity state to counterfactual samples too."""
        if now_s is None or self.uncertainty_maturity_s <= 0.0:
            return uncertain
        raw = uncertain.reshape(-1)
        mature = np.zeros_like(raw)
        known_keys, known_since = self._uncertain_since.get(source, _EMPTY_MATURITY)
        selected = np.flatnonzero(raw)
        if known_keys.size == 0 or selected.size == 0:
            return mature.reshape(uncertain.shape)
        keys = _pack_cells(
            np.floor(x.reshape(-1)[selected] / resolution),
            np.floor(y.reshape(-1)[selected] / resolution),
        )
        index = np.minimum(np.searchsorted(known_keys, keys), known_keys.size - 1)
        # A cell absent from the exposure state was never counted as a mature
        # uncertain cell, so it cannot be substituted by the counterfactual.
        mature[selected] = (known_keys[index] == keys) & (
            (float(now_s) - known_since[index]) >= self.uncertainty_maturity_s
        )
        return mature.reshape(uncertain.shape)

    def _swept_points(self, trajectories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = trajectories[..., 0]
        y = trajectories[..., 1]
        return (
            np.concatenate([x + dx for dx, _ in self._FOOTPRINT_OFFSETS]),
            np.concatenate([y + dy for _, dy in self._FOOTPRINT_OFFSETS]),
        )
