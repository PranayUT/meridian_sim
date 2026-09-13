#!/usr/bin/env python3
"""Turn timestamped lidar scans into obstacle evidence, without ROS.

The ROS boundary deskews raw scans into the continuous ``odom`` frame before
they reach this module. Each scan keeps the sensor origin that it was taken
from. This module is the arithmetic that turns those points into
cells: a per-cell floor with a few seconds of memory, a ground model
that knows which floors are really ground, a count of returns in the
*body band* above that ground, a ledge test, and — the part that needs
the sensor origin — a *shadow* test that splits the tall cells in two.

Four cell classes, in nav_msgs/OccupancyGrid's own vocabulary:

* ``UNSEEN`` (-1) — no returns, or no return anywhere near the ground
  in the cell's neighbourhood: a patch whose only returns are canopy
  metres overhead has not had its ground seen, and says so. (A single
  canopy cell ringed by seen ground takes its neighbours' ground — the
  3x3 median interpolates across one cell, as it always has.)
* ``GROUND`` (0) — ground seen, and nothing standing in the body band.
* ``TALL`` (50) — ``body_band_returns`` or more returns between
  ``body_band_lo_m`` and ``body_band_hi_m`` above the local ground (or a
  ``step_height_m`` ledge) whose column is *porous*: the rays got past
  it and painted ground BEHIND it in the same window. Grass, ferns,
  light brush. Also the default for every tall cell we cannot judge —
  beyond ``classify_range_m``, or with no sensor origin.
* ``SOLID`` (100) — the same body-band test, but nothing at ground level
  was seen behind it along the ray (within ``shadow_depth_m``) although
  the cell itself is within ``classify_range_m`` of the sensor. Rock,
  log, trunk, wall.

**SOLID is the claim that needs evidence.** A tall cell is TALL until a
shadow proves otherwise; nothing here promotes a cell to SOLID on
range, height or density alone. That asymmetry is deliberate: TALL means
"tall, and we do not know more", and the operator (and later the cost
function) should read it that way.

**Why a body band and not the tallest return** (2026-08-18/19, bag
``field_20260818_200921``): scored against the cells the truck had
physically driven through, the max-height test called 29-31 % of them
TALL or SOLID, and raising the height threshold to 1.3 m barely moved
it, because a third of the driven cells had their tallest return 4-13 m
up — tree canopy, charged to the ground cell beneath it. The truck
drove UNDER the branches. Counting only returns in the band a trunk,
rock or wall occupies and canopy does not is the fix, and it is the same
fix lang2auto's fusion work reached for the same reason.

**Why the ground has memory and a slope model**: the band is measured
above the local ground, and the ground under a canopy is hit sparsely —
a 1 s window at 8 m often holds canopy returns and no ground return at
all, and the Mid-360 (-7 deg lower FOV, ~0.25 m up) never sees the
ground within ~2 m of a parked truck. A canopy floor is as wrong a
reference for the band as it was for max-height, so the floor is the
minimum over ``ground_history`` scans (~10 s), and a floor is only
believed to be ground when it sits within ``ground_gap_m`` of a lower
envelope built from every seen floor in the window plus the ground the
truck itself is standing on, sloping up at ``ground_slope``. A cell
whose floor fails that test has not had its ground seen: it is UNSEEN
unless something stands in the band above the envelope.

It is split out of the node for the same reason ``route_logic.py`` is
split out of ``route_sequencer``: the whole decision is then testable
anywhere numpy is installed, including CI's venv, which has no ROS. Do
not import rclpy or any message type here.

What this is NOT — read before trusting a green cell:

* **No negative obstacles.** A hole, ditch or drop-off returns no
  points, so it reads as UNSEEN at best and as flat ground at worst (a
  pit rimmed with grass hits the same cells as the grass). This is gap 2
  in ``docs/autonomy/AUTONOMY_GAPS.md``, and it is the classic way an
  off-road robot destroys itself. Nothing here addresses it, and the
  2026-08-18 bag has a surveyed retaining wall with a drop beyond it
  that reads GROUND on the cells around it.
* **No vegetation semantics beyond porosity.** TALL does not mean
  "grass" and it does not mean "drivable". It means the rays got
  through, which a chain-link fence, a bush and a hedge all do.
* **A body deeper than ``shadow_depth_m``** — a berm, a thick wall —
  casts its shadow beyond the sampled window from its near face. The
  walk skips over a leading run of tall cells (up to ``_MAX_BODY_CELLS``)
  for exactly this reason, but a body deeper than that reads TALL until
  something behind it is either seen or missing.
* **Overhead clearance is not checked.** Anything above
  ``body_band_hi_m`` is ignored on purpose; a branch at 1.2 m over the
  path is TALL, a branch at 1.6 m is not there. The truck is ~0.4 m
  tall, so the band is generous, but it is a band.
* **The planner reads the local grid.** ``mppi_node``'s obstacle term
  charges rollouts by the class of the cell under them, so a false TALL
  on the path is a wall to the planner, not just a colour on the Route
  tab.

``res`` = 0.25 m because that is DLIO's voxel leaf, so a finer grid
would only interpolate a quantity the input does not carry.
The 2026-08-18 bag used ``body_band_lo_m`` = 0.30 m. It took driven-ground
false positives from 29 % to under 1 % per look inside 12 m. The 2026-09-09
field rule lowers ``body_band_lo_m`` and ``step_height_m`` to 0.10 m so the
map can detect rocks and ledges above about 4 in. The configured upper band is
0.75 m. This covers the 0.5 m truck and rejects higher canopy.
``shadow_depth_m`` = 2.0 m is eight cells. Three cells (0.75 m) did not
clear the footprint of a body about a metre across: the window landed under
the near overhang, where the ground is genuinely visible, and never reached
the shadow past the far edge. The figure is a footprint, not a shadow depth,
and it wants a sweep against ground truth like every other one here.
``classify_range_m`` = 12 m is where a Mid-360 scan
is still dense enough that an unseen cell means occlusion rather than a
sampling gap. Nobody has driven the truck at a measured obstacle to find
the real thresholds, and nobody has swept the shadow parameters against
ground truth.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

# The four cell values, in nav_msgs/OccupancyGrid's own vocabulary
# (-1 unknown, 0..100 probability of occupancy). The console server pins
# these same numbers so the browser can colour a grid it did not
# compute; a drift test reads them out of this file with `ast`, so keep
# them literal module-level assignments.
UNSEEN = -1
GROUND = 0
TALL = 50
SOLID = 100

# The three-class names this grid shipped with, kept for the readers
# that predate the split (and for anything that only ever asked "is this
# cell driveable"). Literal, not `UNKNOWN = UNSEEN`, because the drift
# test reads them with `ast.literal_eval`.
UNKNOWN = -1
OBSTACLE = 100

# How far the shadow walk may skip over a leading run of tall cells
# before it starts looking for evidence: the near face of a thick body
# has more of the same body behind it, which says nothing about
# porosity. 8 cells = 2 m at the default resolution.
_MAX_BODY_CELLS = 8

# Sample states along the ray behind a tall cell.
_UNJUDGED = 0  # outside the window — no information either way
_BODY = 1  # another tall cell: more obstacle, not evidence
_LIT = 2  # seen and not tall: the ray got through to the ground
_SHADOW = 3  # nothing seen here at all


@dataclass
class Grid:
    """A raster in the cloud's own frame, in OccupancyGrid's convention.

    ``data`` is row-major with ``index = row * w + col``: ``col`` runs
    along +x from ``origin_x``, ``row`` runs along +y from ``origin_y``.
    That is exactly how OccupancyGrid is defined, so the node can hand
    ``data`` straight over without a flip — and a flip is the classic
    silent bug here, because a symmetric test scene cannot see it.
    """

    origin_x: float
    origin_y: float
    res: float
    w: int
    h: int
    data: np.ndarray  # int8, length w * h


@dataclass
class WindowRaster:
    """What one scan left in a fixed window: the raw per-cell numbers.

    Kept unclassified on purpose. The ring buffer merges these (min of
    mins, sum of counts, concatenation of points) and classifies the
    merge — a single Mid-360 scan paints the ground too sparsely for "no
    returns behind it" to mean occlusion rather than a sampling gap.

    Shape is ``(h, w)`` throughout; empty cells carry +inf / 0 so that
    merging is a plain ``minimum``/``add``.

    ``xyz`` is the scan's finite points, ALL of them and in the cloud's
    own frame, not just the ones inside the window: the band count needs
    each return's height above a ground estimate that only exists at
    classification time, and a window that re-anchors would otherwise
    have to shift a point set. Ten scans of ~20 k float32 points is a
    couple of megabytes; re-binning them per classification is a few
    milliseconds.

    Float32, not 64: z is metres of terrain, the ring holds ten of these
    at 160x160 cells, and halving the bytes the merge and the 3x3 median
    walk over is a measurable slice of the per-scan budget.
    """

    min_z: np.ndarray  # float32 (h, w), +inf where empty
    count: np.ndarray  # int32 (h, w)
    xyz: np.ndarray  # float32 (N, 3), the scan's finite points


def _empty(res: float) -> Grid:
    return Grid(
        origin_x=0.0,
        origin_y=0.0,
        res=res,
        w=0,
        h=0,
        data=np.zeros(0, dtype=np.int8),
    )


def _empty_window(w: int, h: int) -> WindowRaster:
    return WindowRaster(
        min_z=np.full((h, w), np.inf, dtype=np.float32),
        count=np.zeros((h, w), dtype=np.int32),
        xyz=np.zeros((0, 3), dtype=np.float32),
    )


def _neighbourhood_median(field: np.ndarray) -> np.ndarray:
    """3x3 median of a 2-D field where NaN means "no data".

    Deliberately numpy-only: the system ROS python is not guaranteed to
    have scipy, and a node that only runs where someone happened to
    ``pip install scipy`` is a node that fails on the truck.

    Hand-rolled rather than ``np.nanmedian(stack, axis=0)``, which is
    the same answer to the bit but measured 18 ms on one 160x160 window
    of real scan data — over half the 10 Hz budget on its own. Sorting
    nine elements with NaN pushed to +inf and picking the middle by the
    valid count is a fifth of that. A cell with no valid neighbour at
    all comes back NaN, which is what we want: it stays UNSEEN.
    """
    h, w = field.shape
    padded = np.full((h + 2, w + 2), np.nan, dtype=field.dtype)
    padded[1 : h + 1, 1 : w + 1] = field
    stack = np.stack(
        [padded[dy : dy + h, dx : dx + w] for dy in range(3) for dx in range(3)]
    )
    valid = ~np.isnan(stack)
    n = valid.sum(axis=0)
    ordered = np.sort(np.where(valid, stack, np.inf), axis=0)
    # Lower and upper middle, averaged — nanmedian's own rule for an
    # even number of values. Clipped so an all-NaN column indexes
    # something harmless before the np.where below throws it away.
    lo = np.clip((n - 1) // 2, 0, stack.shape[0] - 1)[None]
    hi = np.clip(n // 2, 0, stack.shape[0] - 1)[None]
    middle = (
        np.take_along_axis(ordered, lo, axis=0)[0]
        + np.take_along_axis(ordered, hi, axis=0)[0]
    ) / 2
    return np.where(n > 0, middle, np.nan)


def _clean(xyz: np.ndarray) -> np.ndarray:
    """(N, 3) float64, non-finite rows dropped."""
    pts = np.asarray(xyz, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 3)
    pts = pts.reshape(-1, 3)
    return pts[np.isfinite(pts).all(axis=1)]


def exclude_oriented_box(
    xyz: np.ndarray,
    position_xyz: tuple[float, float, float],
    quaternion_xyzw: tuple[float, float, float, float],
    bounds_xyz: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    """Remove points inside a body-frame box from a world-frame cloud.

    DLIO publishes its deskewed scan in ``odom``. The filter box is fixed
    to ``base_link``. Therefore, each point must be rotated through the
    inverse odom-to-base pose before the axis-aligned bounds are applied.
    The quaternion is normalized here because odometry messages can carry
    small numerical drift. A zero or non-finite quaternion is not a usable
    pose and raises ``ValueError`` instead of silently disabling the filter.

    Bounds are ``(x_min, x_max, y_min, y_max, z_min, z_max)``. The limits
    are inclusive. This removes a point on the vehicle skin as well as one
    just inside it.
    """
    pts = _clean(xyz)
    if pts.shape[0] == 0:
        return pts

    position = np.asarray(position_xyz, dtype=np.float64)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    bounds = np.asarray(bounds_xyz, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("position must contain three finite values")
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("quaternion must contain four finite values")
    if bounds.shape != (6,) or not np.isfinite(bounds).all():
        raise ValueError("bounds must contain six finite values")
    if not (bounds[0] < bounds[1] and bounds[2] < bounds[3] and bounds[4] < bounds[5]):
        raise ValueError("box minimums must be smaller than maximums")

    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-9:
        raise ValueError("quaternion norm is zero")
    x, y, z, w = quaternion / norm
    # base_link -> odom. Points are row vectors below, so multiplying by
    # this matrix (not its transpose) applies the inverse rotation.
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    body = (pts - position) @ rotation
    inside = (
        (body[:, 0] >= bounds[0])
        & (body[:, 0] <= bounds[1])
        & (body[:, 1] >= bounds[2])
        & (body[:, 1] <= bounds[3])
        & (body[:, 2] >= bounds[4])
        & (body[:, 2] <= bounds[5])
    )
    return pts[~inside]


def rasterize_window(
    xyz: np.ndarray,
    origin_x: float,
    origin_y: float,
    w: int,
    h: int,
    res: float = 0.25,
) -> WindowRaster:
    """Rasterize points into a window with FIXED bounds.

    Unlike ``rasterize``, nothing here is derived from the data: the
    caller owns the extent, so two scans rasterized into the same window
    can be merged cell-for-cell. Points outside the window are **dropped,
    not clipped** — clipping smears everything beyond 20 m onto the
    border row, which then reads as a wall around the truck.
    """
    if res <= 0.0:
        raise ValueError("res must be positive")
    if w < 1 or h < 1:
        raise ValueError("window must be at least one cell")

    out = _empty_window(w, h)
    pts = _clean(xyz)
    if pts.shape[0] == 0:
        return out
    out.xyz = pts.astype(np.float32)

    col = np.floor((pts[:, 0] - origin_x) / res).astype(np.int64)
    row = np.floor((pts[:, 1] - origin_y) / res).astype(np.int64)
    inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    if not inside.any():
        return out
    idx = row[inside] * w + col[inside]
    # float32 to match the raster: np.minimum.at is unbuffered and pays
    # for a per-element cast otherwise.
    z = pts[inside, 2].astype(np.float32)

    n = w * h
    min_z = out.min_z.reshape(-1)
    np.minimum.at(min_z, idx, z)
    out.count[...] = np.bincount(idx, minlength=n).reshape(h, w).astype(np.int32)
    return out


# The ground envelope is built at this many metres per coarse cell: a
# lower bound on the terrain does not need 0.25 m detail, and the L1
# cone below is a pass per coarse cell per axis, so 1 m keeps a 40 m
# window at ~160 shifts of a 40x40 array instead of ~640 of 160x160.
_ENVELOPE_COARSE_M = 1.0


def _lower_envelope(
    floor: np.ndarray,
    res: float,
    slope: float,
    seed: tuple[int, int, float] | None,
) -> np.ndarray:
    """A lower bound on the terrain under every cell, from the seen floors.

    ``floor`` is ``(h, w)`` with NaN where nothing was seen. Every seen
    floor is a point the ground is at or below (a canopy return is above
    the ground; a ground return IS the ground), and the ground cannot
    rise faster than ``slope`` (m per m) away from any of them, so the
    envelope at a cell is the minimum over the window of
    ``floor + slope * distance``. Distance is L1 (|dx| + |dy|), which
    makes the min separable into a row pass and a column pass and costs
    an over-estimate of at most sqrt(2) on the diagonal — the cone is
    steeper there, which only makes the bound looser, never wrong.

    ``seed`` is ``(row, col, z)`` for the one ground observation that
    is always available: the truck is standing on it. Without it a truck
    parked under a tree, seeing canopy and no ground anywhere in the
    window, has no bound at all and the canopy floor passes as ground —
    exactly the 2026-08-18 start, 229 s stationary under a canopy.

    Computed on a coarse lattice (``_ENVELOPE_COARSE_M``) and repeated
    back up: the coarse cell takes the min of its members, so the bound
    is at or below every fine floor in it, and the slope penalty is
    charged in coarse cells, which under-charges by at most one coarse
    cell of slope. Both errors point the same way, looser, and
    ``ground_gap_m`` absorbs them.
    """
    h, w = floor.shape
    b = max(1, int(round(_ENVELOPE_COARSE_M / res)))
    hc = -(-h // b)
    wc = -(-w // b)
    padded = np.full((hc * b, wc * b), np.inf, dtype=np.float32)
    padded[:h, :w] = np.where(np.isnan(floor), np.inf, floor)
    coarse = padded.reshape(hc, b, wc, b).min(axis=(1, 3))
    if seed is not None:
        r, c, z = seed
        if 0 <= r < h and 0 <= c < w:
            rc, cc = r // b, c // b
            coarse[rc, cc] = min(float(coarse[rc, cc]), float(z))
    pen = np.float32(slope * b * res)
    # Row pass then column pass. Each pass is a full sweep of shifts;
    # the window is small enough at coarse resolution that the O(n)
    # shifts beat a two-pointer scan written in Python.
    rows = coarse.copy()
    for k in range(1, hc):
        rows[k:, :] = np.minimum(rows[k:, :], coarse[:-k, :] + pen * k)
        rows[:-k, :] = np.minimum(rows[:-k, :], coarse[k:, :] + pen * k)
    env = rows.copy()
    for k in range(1, wc):
        env[:, k:] = np.minimum(env[:, k:], rows[:, :-k] + pen * k)
        env[:, :-k] = np.minimum(env[:, :-k], rows[:, k:] + pen * k)
    return np.repeat(np.repeat(env, b, axis=0), b, axis=1)[:h, :w]


def ground_model(
    floor: np.ndarray,
    res: float,
    seed: tuple[int, int, float] | None,
    ground_slope: float,
    ground_gap_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(ground, ground_seen) for a window from its per-cell floors.

    ``floor`` is ``(h, w)`` minimum z per cell over the ground-memory
    window, +inf where nothing was ever seen. The ground estimate is the
    3x3 median of the seen floors — a single low outlier, or a cell
    whose only points are the top of a rock, does not define the local
    ground plane — and it is *believed* only where it lies within
    ``ground_gap_m`` of the lower envelope: a floor metres above every
    plausible terrain is canopy, and ``ground_seen`` is False there.
    Where the floor is not believed the envelope stands in as the ground
    reference, so the band test above it can still catch a trunk whose
    foot the rays never reached; the cell itself does not become GROUND
    on the strength of a return that never touched the ground.

    Returns ``ground`` (float, NaN only where the envelope is infinite,
    i.e. nothing seen and no seed) and ``ground_seen`` (bool).
    """
    seen = np.isfinite(floor)
    cell_floor = np.where(seen, floor, np.nan)
    median = _neighbourhood_median(cell_floor)
    env = _lower_envelope(median, res, ground_slope, seed)
    with np.errstate(invalid="ignore"):
        ground_seen = np.isfinite(median) & (median <= env + ground_gap_m)
    ground = np.where(ground_seen, median, np.where(np.isfinite(env), env, np.nan))
    return ground, ground_seen


def _band_counts(
    xyz: np.ndarray,
    ground: np.ndarray,
    origin: tuple[float, float],
    res: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Per-cell count of returns between ``lo`` and ``hi`` above ground."""
    h, w = ground.shape
    if xyz.shape[0] == 0:
        return np.zeros((h, w), dtype=np.int32)
    col = np.floor((xyz[:, 0] - origin[0]) / res).astype(np.int64)
    row = np.floor((xyz[:, 1] - origin[1]) / res).astype(np.int64)
    inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    idx = row[inside] * w + col[inside]
    dz = xyz[inside, 2] - ground.reshape(-1)[idx]
    # NaN ground compares False on both sides: a return over a cell with
    # no ground reference at all counts for nothing.
    with np.errstate(invalid="ignore"):
        in_band = (dz >= lo) & (dz <= hi)
    return np.bincount(idx[in_band], minlength=h * w).reshape(h, w).astype(np.int32)


def _band_height_extent(
    xyz: np.ndarray,
    ground: np.ndarray,
    origin: tuple[float, float],
    res: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Return the vertical extent of body-band returns in each cell.

    A flat raised surface can produce many returns above old ground. It does
    not have the vertical column that a trunk, wall, or large bush has. Keep
    that distinction separate from return count and shadow evidence.
    """
    h, w = ground.shape
    extent = np.zeros((h, w), dtype=np.float32)
    if xyz.shape[0] == 0:
        return extent
    col = np.floor((xyz[:, 0] - origin[0]) / res).astype(np.int64)
    row = np.floor((xyz[:, 1] - origin[1]) / res).astype(np.int64)
    inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    idx = row[inside] * w + col[inside]
    dz = xyz[inside, 2] - ground.reshape(-1)[idx]
    with np.errstate(invalid="ignore"):
        in_band = (dz >= lo) & (dz <= hi)
    if not np.any(in_band):
        return extent
    idx = idx[in_band]
    dz = dz[in_band]
    order = np.argsort(idx, kind="stable")
    idx = idx[order]
    dz = dz[order]
    unique, starts = np.unique(idx, return_index=True)
    lows = np.minimum.reduceat(dz, starts)
    highs = np.maximum.reduceat(dz, starts)
    extent.reshape(-1)[unique] = (highs - lows).astype(np.float32)
    return extent


def _connected_physical_components(
    mask: np.ndarray,
    *,
    viewpoint_diverse: np.ndarray,
    res: float,
    min_cells: int,
    min_extent_m: float,
    immediate_cells: int,
    min_diverse_fraction: float,
) -> np.ndarray:
    """Keep large objects or smaller objects seen from separate positions."""
    keep = np.zeros_like(mask, dtype=bool)
    pending = np.asarray(mask, dtype=bool).copy()
    height, width = pending.shape
    for start_row, start_col in zip(*np.nonzero(pending), strict=True):
        if not pending[start_row, start_col]:
            continue
        stack = [(int(start_row), int(start_col))]
        pending[start_row, start_col] = False
        cells: list[tuple[int, int]] = []
        while stack:
            row, col = stack.pop()
            cells.append((row, col))
            for drow in (-1, 0, 1):
                for dcol in (-1, 0, 1):
                    if drow == 0 and dcol == 0:
                        continue
                    next_row = row + drow
                    next_col = col + dcol
                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and pending[next_row, next_col]
                    ):
                        pending[next_row, next_col] = False
                        stack.append((next_row, next_col))
        rows = [cell[0] for cell in cells]
        cols = [cell[1] for cell in cells]
        extent_m = max(
            (max(rows) - min(rows) + 1) * res,
            (max(cols) - min(cols) + 1) * res,
        )
        diverse_fraction = float(
            np.mean([viewpoint_diverse[row, col] for row, col in cells])
        )
        confirmed = len(cells) >= immediate_cells or (
            diverse_fraction >= min_diverse_fraction
        )
        if len(cells) >= min_cells and extent_m >= min_extent_m and confirmed:
            for row, col in cells:
                keep[row, col] = True
    return keep


def _ledges(ground: np.ndarray, step_height_m: float) -> np.ndarray:
    """Cells on either side of a 4-neighbour ground step over the limit.

    NaN comparisons are False, which is what we want: an unseen
    neighbour is not evidence of a step.
    """
    h, w = ground.shape
    ledge = np.zeros((h, w), dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if w > 1:
            step_x = np.abs(ground[:, 1:] - ground[:, :-1]) > step_height_m
            ledge[:, 1:] |= step_x
            ledge[:, :-1] |= step_x
        if h > 1:
            step_y = np.abs(ground[1:, :] - ground[:-1, :]) > step_height_m
            ledge[1:, :] |= step_y
            ledge[:-1, :] |= step_y
    return ledge


def classify(
    min_z: np.ndarray,
    count: np.ndarray,
    xyz: np.ndarray,
    res: float,
    origin: tuple[float, float],
    sensor_xy: tuple[float, float] | None,
    floor: np.ndarray | None = None,
    sensor_ground_z: float | None = None,
    body_band_lo_m: float = 0.10,
    body_band_hi_m: float = 1.50,
    body_band_returns: int = 1,
    step_height_m: float = 0.10,
    ground_slope: float = 0.35,
    ground_gap_m: float = 0.75,
    shadow_depth_m: float = 0.75,
    classify_range_m: float = 12.0,
) -> np.ndarray:
    """Classify a window into UNSEEN / GROUND / TALL / SOLID.

    ``min_z``/``count`` are ``(h, w)`` as produced by ``rasterize_window``
    (or merged from several of them) and ``xyz`` the points behind them;
    ``origin`` is the window's lower-left corner in the same frame as
    ``sensor_xy``. ``floor`` is the longer-memory per-cell minimum z
    (``LocalGrid`` keeps ``ground_history`` scans of it); it defaults to
    ``min_z``, which is right for a one-shot raster and wrong for a
    rolling one — see the module docstring for why the ground needs
    memory. ``sensor_ground_z`` is the z of the ground under the sensor,
    the one ground observation a parked truck always has.

    A cell with no returns is UNSEEN, and so is a cell whose
    neighbourhood's floor sits far above the ground model — canopy
    overhead, ground never seen — unless something stands in the band.
    A cell is *tall* when ``body_band_returns`` or more of the window's
    returns lie between ``body_band_lo_m`` and ``body_band_hi_m`` above
    its ground reference, or when it sits across a ledge — a 4-neighbour
    ground difference over ``step_height_m`` between two believed
    grounds. Every other seen cell is GROUND.

    Tall cells are TALL unless the shadow test promotes them: walk out
    along the ray from ``sensor_xy`` through the cell, skip the leading
    run of tall cells (the rest of the same body), and look at the next
    ``shadow_depth_m`` of window. Cells in there with their ground seen
    and nothing standing on them are the ray having got through; cells
    with no ground-level return are shadow. The cell is SOLID when there
    is shadow and it is at least as much of the window as the lit cells
    are — porosity is a reading of the window, not one cell, because a
    body that meets the ground on a curve shows its own ground under the
    overhang. A cell whose only returns are canopy counts as unseen for
    this purpose: the ray reaching a branch behind the trunk says
    nothing about the ground behind the trunk.

    ``sensor_xy = None`` (no odometry yet, or too stale to trust) leaves
    every tall cell TALL, as does any cell further than
    ``classify_range_m`` from the sensor. SOLID is the claim that needs
    evidence; TALL is what we say when we do not have it.
    """
    h, w = count.shape
    seen = count > 0
    if floor is None:
        floor = min_z

    seed = None
    if sensor_xy is not None and sensor_ground_z is not None:
        seed = (
            int(np.floor((sensor_xy[1] - origin[1]) / res)),
            int(np.floor((sensor_xy[0] - origin[0]) / res)),
            float(sensor_ground_z),
        )
    ground, ground_seen = ground_model(
        np.where(np.isfinite(floor), floor, np.inf),
        res,
        seed,
        ground_slope,
        ground_gap_m,
    )

    band = _band_counts(xyz, ground, origin, res, body_band_lo_m, body_band_hi_m)
    occupied = band >= max(1, int(body_band_returns))
    # A ledge between two believed grounds only: a canopy floor next to a
    # real one is a 4 m "step" that no wheel will ever meet.
    ledge = _ledges(np.where(ground_seen, ground, np.nan), step_height_m)

    grounded = seen & ground_seen
    cls = np.full((h, w), UNSEEN, dtype=np.int8)
    cls[grounded] = GROUND
    candidate = seen & (occupied | (ledge & ground_seen))
    cls[candidate] = TALL

    if sensor_xy is None:
        return cls
    rows, cols = np.nonzero(candidate)
    if rows.size == 0:
        return cls

    solid = _shadow_test(
        candidate=candidate,
        seen=grounded,
        rows=rows,
        cols=cols,
        res=res,
        origin=origin,
        sensor_xy=sensor_xy,
        shadow_depth_m=shadow_depth_m,
        classify_range_m=classify_range_m,
    )
    cls[rows[solid], cols[solid]] = SOLID
    return cls


def _shadow_test(
    candidate: np.ndarray,
    seen: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    res: float,
    origin: tuple[float, float],
    sensor_xy: tuple[float, float],
    shadow_depth_m: float,
    classify_range_m: float,
) -> np.ndarray:
    """Which of the tall cells at ``rows``/``cols`` cast a shadow.

    Vectorised over cells, looped over the (handful of) steps along the
    ray — the opposite of the obvious per-cell loop, because there are
    thousands of tall cells in a 40 m window and never more than a dozen
    steps.
    """
    h, w = candidate.shape
    sx, sy = float(sensor_xy[0]), float(sensor_xy[1])
    cx = origin[0] + (cols + 0.5) * res
    cy = origin[1] + (rows + 0.5) * res
    dist = np.hypot(cx - sx, cy - sy)
    # A cell under the sensor has no ray direction, and a cell past
    # classify_range_m is in scan territory too sparse for "unseen" to
    # mean "occluded".
    near = (dist > 1e-6) & (dist <= classify_range_m)
    out = np.zeros(rows.size, dtype=bool)
    # Walk ONLY the cells that could come out SOLID. A 40 m window holds
    # tall cells out to 28 m and the range gate keeps 12 m of them, so
    # subsetting here cuts the loop below by about three quarters — it
    # was the largest single cost in the per-scan budget.
    keep = np.nonzero(near)[0]
    if keep.size == 0:
        return out
    rows = rows[keep]
    cols = cols[keep]
    cx = cx[keep]
    cy = cy[keep]
    inv = 1.0 / dist[keep]
    ux = (cx - sx) * inv
    uy = (cy - sy) * inv

    k_shadow = max(1, int(round(shadow_depth_m / res)))
    k_span = k_shadow + _MAX_BODY_CELLS
    m = rows.size
    state = np.full((k_span, m), _UNJUDGED, dtype=np.int8)
    seen_flat = seen.reshape(-1)
    cand_flat = candidate.reshape(-1)
    for k in range(k_span):
        step = (k + 1) * res
        c2 = np.floor((cx + ux * step - origin[0]) / res).astype(np.int64)
        r2 = np.floor((cy + uy * step - origin[1]) / res).astype(np.int64)
        inside = (c2 >= 0) & (c2 < w) & (r2 >= 0) & (r2 < h)
        idx = np.where(inside, r2 * w + c2, 0)
        # The cell itself can come back as a sample on a diagonal ray
        # (half a cell of offset does not always leave the cell); it is
        # a tall cell, so _BODY is the right answer and the leading-run
        # skip below steps over it.
        s = np.where(cand_flat[idx], _BODY, np.where(seen_flat[idx], _LIT, _SHADOW))
        state[k] = np.where(inside, s, _UNJUDGED)

    # Skip the leading run of body cells: the near face of a rock has
    # more rock behind it, which is not evidence about porosity.
    body = state == _BODY
    first_free = np.argmax(~body, axis=0)
    all_body = body.all(axis=0)
    ks = np.arange(k_span)[:, None]
    window = (
        (ks >= first_free[None, :])
        & (ks < (first_free + k_shadow)[None, :])
        & ~all_body[None, :]
    )
    # Porosity has to be the dominant reading of the window, not a single
    # cell. A body whose surface curves down to meet the ground — a bush, a
    # boulder — is widest well above the band and tapers to nothing at z=0,
    # so the lidar sees its own ground under the overhang. Those cells are
    # lit, they sit inside the body's footprint, and under an `any()` veto
    # one of them denied SOLID to the whole body: the 0.25 m bush window
    # measured three lit cells under the overhang against a full shadow that
    # only starts past the far edge.
    lit = ((state == _LIT) & window).sum(axis=0)
    shadowed = ((state == _SHADOW) & window).sum(axis=0)
    out[keep] = (shadowed > 0) & (shadowed >= lit)
    return out


def rasterize(
    xyz: np.ndarray,
    res: float = 0.25,
    body_band_lo_m: float = 0.10,
    body_band_hi_m: float = 1.50,
    body_band_returns: int = 1,
    step_height_m: float = 0.10,
    max_side_cells: int = 1200,
    sensor_xy: tuple[float, float] | None = None,
    sensor_ground_z: float | None = None,
    ground_slope: float = 0.35,
    ground_gap_m: float = 0.75,
    shadow_depth_m: float = 0.75,
    classify_range_m: float = 12.0,
) -> Grid:
    """Rasterize an (N, 3) array of points into a Grid, extent from data.

    Non-finite rows are dropped. The extent is the point bounds plus one
    cell of margin on every side, so every cell that has points has a
    full 3x3 neighbourhood inside the grid.

    If either side would exceed ``max_side_cells``, ``res`` doubles until
    it fits. Doubling (rather than fitting the extent exactly) keeps the
    result a deterministic function of the input: the same cloud always
    produces the same cell boundaries, so a grid does not shimmer between
    keyframes. The resolution actually used is on the returned Grid.

    Without a ``sensor_xy`` there is no shadow evidence, so every tall
    cell comes back TALL — never SOLID. This is the one-shot form, kept
    for offline work and for the tests; the node drives ``LocalGrid`` and
    ``GlobalGrid`` instead. The ground floor is this one cloud's, with
    no memory — fine for a cloud that already spans seconds, wrong for a
    single scan under a canopy (see ``LocalGrid``).
    """
    if res <= 0.0:
        raise ValueError("res must be positive")
    if max_side_cells < 1:
        raise ValueError("max_side_cells must be at least 1")

    pts = _clean(xyz)
    if pts.shape[0] == 0:
        return _empty(res)

    x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
    y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())

    # Grow the cell until the raster fits. The bound is on cells, not on
    # metres, because it is a memory/serialisation bound: OccupancyGrid
    # carries one byte per cell over DDS.
    while True:
        col0 = int(np.floor(x_min / res)) - 1
        row0 = int(np.floor(y_min / res)) - 1
        w = int(np.floor(x_max / res)) - col0 + 2
        h = int(np.floor(y_max / res)) - row0 + 2
        if w <= max_side_cells and h <= max_side_cells:
            break
        res *= 2.0

    origin_x = col0 * res
    origin_y = row0 * res
    window = rasterize_window(pts, origin_x, origin_y, w, h, res)
    cls = classify(
        window.min_z,
        window.count,
        window.xyz,
        res=res,
        origin=(origin_x, origin_y),
        sensor_xy=sensor_xy,
        sensor_ground_z=sensor_ground_z,
        body_band_lo_m=body_band_lo_m,
        body_band_hi_m=body_band_hi_m,
        body_band_returns=body_band_returns,
        step_height_m=step_height_m,
        ground_slope=ground_slope,
        ground_gap_m=ground_gap_m,
        shadow_depth_m=shadow_depth_m,
        classify_range_m=classify_range_m,
    )
    return Grid(
        origin_x=origin_x,
        origin_y=origin_y,
        res=res,
        w=w,
        h=h,
        data=cls.reshape(-1),
    )


def _shift(arr: np.ndarray, drow: int, dcol: int, fill: float | int) -> np.ndarray:
    """Move a raster's CONTENTS by whole cells when its window moves.

    ``drow``/``dcol`` are how far the window's origin moved, in cells, so
    the data moves the other way. Whole cells only: a fractional shift
    would resample the history and blur it a little more on every
    re-anchor.
    """
    out = np.full_like(arr, fill)
    h, w = arr.shape
    if abs(drow) >= h or abs(dcol) >= w:
        return out
    src_r = slice(max(drow, 0), h + min(drow, 0))
    dst_r = slice(max(-drow, 0), h + min(-drow, 0))
    src_c = slice(max(dcol, 0), w + min(dcol, 0))
    dst_c = slice(max(-dcol, 0), w + min(-dcol, 0))
    out[dst_r, dst_c] = arr[src_r, src_c]
    return out


class LocalGrid:
    """The last ``history`` scans, in a window that follows the truck.

    Holds raw ``WindowRaster``s rather than classified cells: merging
    min/count/points and classifying the merge is what makes "nothing
    was seen behind it" mean occlusion. One Mid-360 scan leaves most
    ground cells at 20 m empty simply because the pattern did not sweep
    them, and classifying per scan turns that into a field of false
    SOLID.

    Alongside the ring it keeps ``ground_history`` scans of per-cell
    minimum z — the floor memory the ground model reads. Merging is a
    reduce over arrays of ``n*n`` cells, which is cheap; re-rasterizing
    ``history`` scans of points would not be.
    """

    def __init__(
        self,
        res: float = 0.25,
        window_m: float = 40.0,
        history: int = 10,
        ground_history: int = 100,
    ) -> None:
        if res <= 0.0:
            raise ValueError("res must be positive")
        self.res = float(res)
        self.n = max(1, int(round(window_m / self.res)))
        self.history = max(1, int(history))
        # The floor remembers longer than the classification window: at
        # 8 m under a canopy one second of scans often holds no ground
        # return at all, and a band measured above a canopy floor is
        # the max-height bug back again. Ten seconds is 20 m of travel
        # at 2 m/s — long enough that the ground ahead was densely
        # sampled before the truck reaches it, short enough that a
        # below-ground artefact (a puddle mirroring the sky) heals.
        self.ground_history = max(self.history, int(ground_history))
        self.origin_x = 0.0
        self.origin_y = 0.0
        self._ring: list[WindowRaster] = []
        self._ring_stamps: list[float | None] = []
        # Floor memory in two tiers: the per-scan minima of the current
        # ring, and one merged raster per full ring's worth of scans
        # before it. min over 10 + 10 arrays per classification instead
        # of 100 — the flat list cost 1.2 ms a scan offline, ~3 ms on the
        # loaded Jetson, for the same answer.
        self._floor_blocks: list[np.ndarray] = []
        self._floor_pending: list[np.ndarray] = []
        self._anchored = False

    @property
    def w(self) -> int:
        return self.n

    @property
    def h(self) -> int:
        return self.n

    @property
    def scans(self) -> int:
        return len(self._ring)

    @property
    def centre(self) -> tuple[float, float]:
        half = self.n * self.res / 2.0
        return (self.origin_x + half, self.origin_y + half)

    def recentre(self, x: float, y: float) -> bool:
        """Re-anchor the window on the truck if it has drifted too far.

        Returns True when the window moved. The origin is snapped to the
        cell lattice so the shift is a whole number of cells and the
        stored history survives it — a window that re-anchored to an
        arbitrary float would resample every scan it holds.
        """
        half = self.n * self.res / 2.0
        cx, cy = self.centre
        if self._anchored and float(np.hypot(x - cx, y - cy)) <= self.n * self.res / 4:
            return False

        new_ox = float(np.floor((x - half) / self.res) * self.res)
        new_oy = float(np.floor((y - half) / self.res) * self.res)
        dcol = int(round((new_ox - self.origin_x) / self.res))
        drow = int(round((new_oy - self.origin_y) / self.res))
        self._anchored = True
        if dcol == 0 and drow == 0:
            return False
        for i, raster in enumerate(self._ring):
            self._ring[i] = WindowRaster(
                min_z=_shift(raster.min_z, drow, dcol, np.inf),
                count=_shift(raster.count, drow, dcol, 0),
                xyz=raster.xyz,
            )
        for tier in (self._floor_blocks, self._floor_pending):
            for i, floor in enumerate(tier):
                tier[i] = _shift(floor, drow, dcol, np.inf)
        self.origin_x = new_ox
        self.origin_y = new_oy
        return True

    def add(self, xyz: np.ndarray, stamp_s: float | None = None) -> WindowRaster:
        """Rasterize one scan into the window and push it on the ring."""
        if stamp_s is not None and not np.isfinite(stamp_s):
            raise ValueError("scan stamp must be finite")
        raster = rasterize_window(
            xyz, self.origin_x, self.origin_y, self.n, self.n, self.res
        )
        self._ring.append(raster)
        self._ring_stamps.append(None if stamp_s is None else float(stamp_s))
        del self._ring[: max(0, len(self._ring) - self.history)]
        del self._ring_stamps[: max(0, len(self._ring_stamps) - self.history)]
        self._floor_pending.append(raster.min_z)
        if len(self._floor_pending) >= self.history:
            self._floor_blocks.append(np.minimum.reduce(self._floor_pending))
            self._floor_pending = []
            blocks = -(-self.ground_history // self.history)
            del self._floor_blocks[: max(0, len(self._floor_blocks) - blocks)]
        return raster

    def merged(self) -> WindowRaster:
        if not self._ring:
            return _empty_window(self.n, self.n)
        if len(self._ring) == 1:
            return self._ring[0]
        return WindowRaster(
            min_z=np.minimum.reduce([r.min_z for r in self._ring]),
            count=np.add.reduce([r.count for r in self._ring]),
            xyz=np.concatenate([r.xyz for r in self._ring]),
        )

    def scan_coverage(self) -> np.ndarray:
        """Number of recent scans with at least one return in each cell."""
        if not self._ring:
            return np.zeros((self.n, self.n), dtype=np.int16)
        return np.add.reduce(
            [raster.count > 0 for raster in self._ring], dtype=np.int16
        )

    def observation_age(self, now_s: float) -> np.ndarray:
        """Age of the newest direct return in each cell, in seconds."""
        if not np.isfinite(now_s):
            raise ValueError("now_s must be finite")
        age = np.full((self.n, self.n), np.nan, dtype=np.float32)
        for raster, stamp in zip(self._ring, self._ring_stamps, strict=True):
            if stamp is None:
                continue
            observed = raster.count > 0
            value = max(0.0, float(now_s) - stamp)
            age[observed] = np.where(
                np.isnan(age[observed]), value, np.minimum(age[observed], value)
            )
        return age

    def oldest_scan_age(self, now_s: float) -> float:
        """Age of the oldest timestamped scan still used by the local grid."""
        if not np.isfinite(now_s):
            raise ValueError("now_s must be finite")
        stamps = [stamp for stamp in self._ring_stamps if stamp is not None]
        if not stamps:
            return 0.0
        return max(0.0, float(now_s) - min(stamps))

    def elevation_evidence(
        self,
        *,
        sensor_std_m: float = 0.05,
        single_scan_std_m: float = 0.15,
        max_effective_scans: float = 3.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a robust ground-height estimate from independent sweeps.

        Each sweep contributes at most one lower-surface observation per
        cell. The median rejects an isolated low or high return. The variance
        includes the lidar floor, finite support, and measured disagreement.
        Repeated sweeps are correlated, so the effective support is capped.
        """
        values = (sensor_std_m, single_scan_std_m, max_effective_scans)
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("elevation uncertainty parameters must be positive")
        shape = (self.n, self.n)
        mean = np.full(shape, np.nan, dtype=np.float32)
        variance = np.full(shape, np.nan, dtype=np.float32)
        evidence = np.zeros(shape, dtype=np.float32)
        if not self._ring:
            return mean, variance, evidence

        samples = np.stack(
            [
                np.where(np.isfinite(raster.min_z), raster.min_z, np.nan)
                for raster in self._ring
            ]
        ).astype(np.float64, copy=False)
        support = np.count_nonzero(np.isfinite(samples), axis=0)
        active = support > 0
        if not np.any(active):
            return mean, variance, evidence

        median = np.full(shape, np.nan, dtype=np.float64)
        mad = np.full(shape, np.nan, dtype=np.float64)
        active_samples = samples[:, active]
        active_median = np.nanmedian(active_samples, axis=0)
        median[active] = active_median
        mad[active] = np.nanmedian(
            np.abs(active_samples - active_median[np.newaxis, :]), axis=0
        )
        effective = np.minimum(support.astype(np.float64), max_effective_scans)
        robust_sigma = 1.4826 * np.nan_to_num(mad, nan=0.0)
        estimate_variance = sensor_std_m**2 + (
            single_scan_std_m**2 + robust_sigma**2
        ) / np.maximum(effective, 1.0)
        mean[active] = median[active].astype(np.float32)
        variance[active] = estimate_variance[active].astype(np.float32)
        evidence[active] = effective[active].astype(np.float32)
        return mean, variance, evidence

    def floor(self) -> np.ndarray:
        """Per-cell minimum z over roughly the last ``ground_history`` scans.

        Roughly: the memory is whole blocks of ``history`` scans plus the
        partial block in progress, so it holds between ``ground_history``
        and ``ground_history + history - 1`` scans.
        """
        tiers = self._floor_blocks + self._floor_pending
        if not tiers:
            return np.full((self.n, self.n), np.inf, dtype=np.float32)
        if len(tiers) == 1:
            return tiers[0]
        return np.minimum.reduce(tiers)

    def classify(
        self,
        sensor_xy: tuple[float, float] | None,
        sensor_ground_z: float | None = None,
        body_band_lo_m: float = 0.10,
        body_band_hi_m: float = 1.50,
        body_band_returns: int = 1,
        step_height_m: float = 0.10,
        ground_slope: float = 0.35,
        ground_gap_m: float = 0.75,
        shadow_depth_m: float = 0.75,
        classify_range_m: float = 12.0,
    ) -> np.ndarray:
        merged = self.merged()
        return classify(
            merged.min_z,
            merged.count,
            merged.xyz,
            res=self.res,
            origin=(self.origin_x, self.origin_y),
            sensor_xy=sensor_xy,
            floor=self.floor(),
            sensor_ground_z=sensor_ground_z,
            body_band_lo_m=body_band_lo_m,
            body_band_hi_m=body_band_hi_m,
            body_band_returns=body_band_returns,
            step_height_m=step_height_m,
            ground_slope=ground_slope,
            ground_gap_m=ground_gap_m,
            shadow_depth_m=shadow_depth_m,
            classify_range_m=classify_range_m,
        )

    def direct_occupancy_observation(
        self,
        newest: WindowRaster,
        merged_classes: np.ndarray,
        sensor_xyz: tuple[float, float, float] | None,
        sensor_ground_z: float | None,
        *,
        body_band_lo_m: float,
        body_band_hi_m: float,
        ground_slope: float,
        ground_gap_m: float,
        solid_min_vertical_extent_m: float = 0.30,
    ) -> np.ndarray:
        """Classify only evidence directly sampled by the newest sweep.

        The longer merged window decides whether a current body return casts
        a shadow. It does not decide whether the newest sweep observed that
        body return. This distinction prevents one old return from being
        counted again whenever a later ground point lands in the same cell.
        """
        seed = None
        if sensor_xyz is not None and sensor_ground_z is not None:
            seed = (
                int(np.floor((sensor_xyz[1] - self.origin_y) / self.res)),
                int(np.floor((sensor_xyz[0] - self.origin_x) / self.res)),
                float(sensor_ground_z),
            )
        floor = self.floor()
        ground, ground_seen = ground_model(
            np.where(np.isfinite(floor), floor, np.inf),
            self.res,
            seed,
            ground_slope,
            ground_gap_m,
        )
        band = _band_counts(
            newest.xyz,
            ground,
            (self.origin_x, self.origin_y),
            self.res,
            body_band_lo_m,
            body_band_hi_m,
        )
        body = band > 0
        vertical_extent = _band_height_extent(
            self.merged().xyz,
            ground,
            (self.origin_x, self.origin_y),
            self.res,
            body_band_lo_m,
            body_band_hi_m,
        )
        structure = vertical_extent >= solid_min_vertical_extent_m
        clear = self._ray_clear_mask(
            newest.xyz,
            ground,
            sensor_xyz,
            body_band_hi_m=body_band_hi_m,
        )
        direct_ground = ((newest.count > 0) & ground_seen) | clear
        direct_ground &= ~body
        result = np.full((self.n, self.n), UNSEEN, dtype=np.int8)
        result[direct_ground] = GROUND
        result[body] = TALL
        result[body & structure & (merged_classes == SOLID)] = SOLID
        return result

    def _ray_clear_mask(
        self,
        xyz: np.ndarray,
        ground: np.ndarray,
        sensor_xyz: tuple[float, float, float] | None,
        *,
        body_band_hi_m: float,
    ) -> np.ndarray:
        """Mark cells crossed by a measured ray inside vehicle height.

        One far return per one-degree azimuth bin bounds the work. At the
        12 m classification limit, one 0.25 m cell spans about 1.2 degrees.
        Finer angular bins repeat work in the same cells without adding map
        resolution. At each
        crossed cell, the interpolated 3-D ray must lie between local ground
        and the top of the collision band. Canopy rays therefore do not clear
        obstacles below them. The endpoint is excluded because it is the hit.
        """
        clear = np.zeros((self.n, self.n), dtype=bool)
        if sensor_xyz is None or xyz.size == 0:
            return clear
        points = np.asarray(xyz, dtype=np.float64)
        delta = points[:, :2] - np.asarray(sensor_xyz[:2], dtype=np.float64)
        distance = np.hypot(delta[:, 0], delta[:, 1])
        usable = np.isfinite(points).all(axis=1) & (distance >= self.res)
        if not np.any(usable):
            return clear
        points = points[usable]
        delta = delta[usable]
        distance = distance[usable]
        angle = np.arctan2(delta[:, 1], delta[:, 0])
        bins = np.floor((angle + np.pi) * (180.0 / np.pi)).astype(np.int32)
        order = np.lexsort((distance, bins))
        sorted_bins = bins[order]
        last = np.r_[sorted_bins[1:] != sorted_bins[:-1], True]
        endpoints = points[order[last]]

        sx, sy, sz = (float(value) for value in sensor_xyz)
        for endpoint in endpoints:
            dx = float(endpoint[0]) - sx
            dy = float(endpoint[1]) - sy
            horizontal = float(np.hypot(dx, dy))
            steps = max(1, int(np.ceil(horizontal / self.res)))
            for step in range(1, steps):
                fraction = step / steps
                x = sx + fraction * dx
                y = sy + fraction * dy
                col = int(np.floor((x - self.origin_x) / self.res))
                row = int(np.floor((y - self.origin_y) / self.res))
                if not (0 <= row < self.n and 0 <= col < self.n):
                    continue
                floor = float(ground[row, col])
                if not np.isfinite(floor):
                    continue
                ray_z = sz + fraction * (float(endpoint[2]) - sz)
                height = ray_z - floor
                if 0.0 <= height <= body_band_hi_m:
                    clear[row, col] = True
        return clear

    def to_grid(self, cls: np.ndarray) -> Grid:
        return Grid(
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            res=self.res,
            w=self.n,
            h=self.n,
            data=cls.reshape(-1),
        )


class ProbabilisticOccupancyGrid:
    """A bounded, decaying Bernoulli obstacle estimate in ``odom``.

    The geometric classifier supplies one measurement class per cell. Only
    cells sampled by the newest lidar sweep update this filter. This prevents
    the overlapping ten-scan classification window from counting one return
    ten times. Repeated body returns increase occupancy. Direct ground
    returns clear it. Stale evidence decays toward the unknown prior.

    ``SOLID`` is an output decision, not stored state. It requires both a high
    posterior probability and persistent occupied support. The filter keeps
    probability and support separate so MPPI can later consume them directly.
    """

    def __init__(
        self,
        res: float = 0.25,
        window_m: float = 40.0,
        *,
        occupied_log_odds: float = 0.85,
        tall_log_odds: float = 0.35,
        free_log_odds: float = -0.85,
        decay_half_life_s: float = 5.0,
        evidence_cap: float = 10.0,
        observed_evidence_min: float = 0.5,
        solid_evidence_min: float = 4.5,
        solid_probability: float = 0.90,
        free_probability: float = 0.35,
        solid_min_component_cells: int = 3,
        solid_min_component_extent_m: float = 0.50,
        solid_immediate_component_cells: int = 4,
        solid_viewpoint_baseline_m: float = 0.75,
        solid_viewpoint_min_fraction: float = 0.50,
    ) -> None:
        if res <= 0.0 or window_m <= 0.0:
            raise ValueError("resolution and window must be positive")
        positive = (
            occupied_log_odds,
            tall_log_odds,
            decay_half_life_s,
            evidence_cap,
            observed_evidence_min,
            solid_evidence_min,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("occupancy filter positive parameters are invalid")
        if not np.isfinite(free_log_odds) or free_log_odds >= 0.0:
            raise ValueError("free_log_odds must be finite and negative")
        if not 0.5 < solid_probability < 1.0:
            raise ValueError("solid_probability must be between 0.5 and 1")
        if not 0.0 < free_probability < 0.5:
            raise ValueError("free_probability must be between 0 and 0.5")
        if solid_min_component_cells < 1:
            raise ValueError("solid_min_component_cells must be positive")
        if solid_immediate_component_cells < solid_min_component_cells:
            raise ValueError("immediate component size must not be smaller")
        if (
            not np.isfinite(solid_min_component_extent_m)
            or solid_min_component_extent_m <= 0.0
        ):
            raise ValueError("solid_min_component_extent_m must be positive")
        if (
            not np.isfinite(solid_viewpoint_baseline_m)
            or solid_viewpoint_baseline_m <= 0
        ):
            raise ValueError("solid_viewpoint_baseline_m must be positive")
        if not 0.0 < solid_viewpoint_min_fraction <= 1.0:
            raise ValueError("solid_viewpoint_min_fraction must be in (0, 1]")
        self.res = float(res)
        self.n = max(1, int(round(window_m / self.res)))
        self.occupied_log_odds = float(occupied_log_odds)
        self.tall_log_odds = float(tall_log_odds)
        self.free_log_odds = float(free_log_odds)
        self.decay_half_life_s = float(decay_half_life_s)
        self.evidence_cap = float(evidence_cap)
        self.observed_evidence_min = float(observed_evidence_min)
        self.solid_evidence_min = float(solid_evidence_min)
        self.solid_probability = float(solid_probability)
        self.free_probability = float(free_probability)
        self.solid_min_component_cells = int(solid_min_component_cells)
        self.solid_min_component_extent_m = float(solid_min_component_extent_m)
        self.solid_immediate_component_cells = int(solid_immediate_component_cells)
        self.solid_viewpoint_baseline_m = float(solid_viewpoint_baseline_m)
        self.solid_viewpoint_min_fraction = float(solid_viewpoint_min_fraction)
        shape = (self.n, self.n)
        self.log_odds = np.zeros(shape, dtype=np.float32)
        self.evidence = np.zeros(shape, dtype=np.float32)
        self.occupied_evidence = np.zeros(shape, dtype=np.float32)
        self.solid_evidence = np.zeros(shape, dtype=np.float32)
        self.solid_anchor_x = np.full(shape, np.nan, dtype=np.float32)
        self.solid_anchor_y = np.full(shape, np.nan, dtype=np.float32)
        self.solid_viewpoint_diverse = np.zeros(shape, dtype=bool)
        self.last_observed_s = np.full(shape, np.nan, dtype=np.float64)
        self.origin_x = 0.0
        self.origin_y = 0.0
        self._anchored = False
        self._last_update_s: float | None = None

    def recentre(self, x: float, y: float) -> bool:
        """Move the rolling state by whole cells without resampling it."""
        half = self.n * self.res / 2.0
        old_x = self.origin_x + half
        old_y = self.origin_y + half
        if self._anchored and float(np.hypot(x - old_x, y - old_y)) <= half / 2.0:
            return False
        new_x = float(np.floor((x - half) / self.res) * self.res)
        new_y = float(np.floor((y - half) / self.res) * self.res)
        dcol = int(round((new_x - self.origin_x) / self.res))
        drow = int(round((new_y - self.origin_y) / self.res))
        self._anchored = True
        if dcol == 0 and drow == 0:
            return False
        self.log_odds = _shift(self.log_odds, drow, dcol, 0.0)
        self.evidence = _shift(self.evidence, drow, dcol, 0.0)
        self.occupied_evidence = _shift(self.occupied_evidence, drow, dcol, 0.0)
        self.solid_evidence = _shift(self.solid_evidence, drow, dcol, 0.0)
        self.solid_anchor_x = _shift(self.solid_anchor_x, drow, dcol, np.nan)
        self.solid_anchor_y = _shift(self.solid_anchor_y, drow, dcol, np.nan)
        self.solid_viewpoint_diverse = _shift(
            self.solid_viewpoint_diverse, drow, dcol, False
        )
        self.last_observed_s = _shift(self.last_observed_s, drow, dcol, np.nan)
        self.origin_x = new_x
        self.origin_y = new_y
        return True

    def update(
        self,
        classes: np.ndarray,
        observed: np.ndarray,
        stamp_s: float,
        sensor_xy: tuple[float, float] | None = None,
    ) -> None:
        """Fuse one independent sweep of occupied and direct-ground evidence."""
        classes = np.asarray(classes)
        observed = np.asarray(observed, dtype=bool)
        if classes.shape != self.log_odds.shape or observed.shape != classes.shape:
            raise ValueError("occupancy update arrays must match the rolling grid")
        if not np.isfinite(stamp_s):
            raise ValueError("occupancy update stamp must be finite")
        if self._last_update_s is not None:
            dt = max(0.0, float(stamp_s) - self._last_update_s)
            retain = float(2.0 ** (-dt / self.decay_half_life_s))
            self.log_odds *= retain
            self.evidence *= retain
            self.occupied_evidence *= retain
            self.solid_evidence *= retain
            expired = self.solid_evidence < self.observed_evidence_min
            self.solid_anchor_x[expired] = np.nan
            self.solid_anchor_y[expired] = np.nan
            self.solid_viewpoint_diverse[expired] = False
        self._last_update_s = float(stamp_s)

        ground = observed & (classes == GROUND)
        tall = observed & (classes == TALL)
        solid = observed & (classes == SOLID)
        occupied = tall | solid
        self.log_odds[ground] += self.free_log_odds
        self.log_odds[tall] += self.tall_log_odds
        self.log_odds[solid] += self.occupied_log_odds
        np.clip(self.log_odds, -8.0, 8.0, out=self.log_odds)
        sampled = ground | occupied
        self.evidence[sampled] += 1.0
        self.occupied_evidence[occupied] += 1.0
        self.occupied_evidence[ground] = np.maximum(
            0.0, self.occupied_evidence[ground] - 1.0
        )
        self.solid_evidence[solid] += 1.0
        if sensor_xy is not None:
            sensor_x, sensor_y = (float(value) for value in sensor_xy)
            if not np.isfinite(sensor_x) or not np.isfinite(sensor_y):
                raise ValueError("sensor_xy must be finite")
            first = solid & ~np.isfinite(self.solid_anchor_x)
            self.solid_anchor_x[first] = sensor_x
            self.solid_anchor_y[first] = sensor_y
            baseline = np.hypot(
                sensor_x - self.solid_anchor_x,
                sensor_y - self.solid_anchor_y,
            )
            self.solid_viewpoint_diverse[
                solid & (baseline >= self.solid_viewpoint_baseline_m)
            ] = True
        not_structure = ground | tall
        self.solid_evidence[not_structure] = np.maximum(
            0.0, self.solid_evidence[not_structure] - 1.0
        )
        np.minimum(self.evidence, self.evidence_cap, out=self.evidence)
        np.minimum(
            self.occupied_evidence,
            self.evidence_cap,
            out=self.occupied_evidence,
        )
        np.minimum(self.solid_evidence, self.evidence_cap, out=self.solid_evidence)
        self.last_observed_s[sampled] = float(stamp_s)

    def probability(self) -> np.ndarray:
        return (1.0 / (1.0 + np.exp(-self.log_odds))).astype(np.float32)

    def classes(self) -> np.ndarray:
        """Derive the compatibility grid from probability and support."""
        probability = self.probability()
        known = self.evidence >= self.observed_evidence_min
        result = np.full(probability.shape, UNSEEN, dtype=np.int8)
        result[known & (probability <= self.free_probability)] = GROUND
        possible = known & (probability > self.free_probability)
        result[possible] = TALL
        hard = (
            possible
            & (probability >= self.solid_probability)
            & (self.occupied_evidence >= self.solid_evidence_min)
            & (self.solid_evidence >= self.solid_evidence_min)
        )
        hard = _connected_physical_components(
            hard,
            viewpoint_diverse=self.solid_viewpoint_diverse,
            res=self.res,
            min_cells=self.solid_min_component_cells,
            min_extent_m=self.solid_min_component_extent_m,
            immediate_cells=self.solid_immediate_component_cells,
            min_diverse_fraction=self.solid_viewpoint_min_fraction,
        )
        result[hard] = SOLID
        return result

    def evidence_grid(
        self, now_s: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return Bernoulli mean, posterior variance, support, and age."""
        probability = self.probability()
        known = self.evidence >= self.observed_evidence_min
        mean = np.full(probability.shape, np.nan, dtype=np.float32)
        variance = np.full(probability.shape, np.nan, dtype=np.float32)
        mean[known] = probability[known]
        variance[known] = (
            probability[known]
            * (1.0 - probability[known])
            / (self.evidence[known] + 1.0)
        )
        support = np.zeros(probability.shape, dtype=np.float32)
        support[known] = self.evidence[known]
        age = np.full(probability.shape, np.nan, dtype=np.float32)
        observed = known & np.isfinite(self.last_observed_s)
        age[observed] = np.maximum(
            0.0, float(now_s) - self.last_observed_s[observed]
        ).astype(np.float32)
        return mean, variance, support, age

    def to_grid(self) -> Grid:
        return Grid(
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            res=self.res,
            w=self.n,
            h=self.n,
            data=self.classes().reshape(-1),
        )


def occupancy_evidence(
    cls: np.ndarray,
    scan_coverage: np.ndarray,
    *,
    history_scans: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert categorical cells into a soft Bernoulli occupancy claim.

    The probabilities are deliberately broad because no field obstacle has
    been surveyed as occupied/free ground truth. Evidence is the fraction of
    the rolling scan history that sampled a cell. It never exceeds one
    effective observation because adjacent Mid-360 scans and windows are
    correlated.
    """
    classes = np.asarray(cls)
    coverage = np.asarray(scan_coverage)
    if classes.shape != coverage.shape or classes.ndim != 2:
        raise ValueError("classes and scan coverage must have the same 2-D shape")
    if history_scans < 1:
        raise ValueError("history_scans must be positive")
    active = classes != UNSEEN
    known = active & np.isin(classes, (GROUND, TALL, SOLID))
    mean = np.full(classes.shape, np.nan, dtype=np.float32)
    variance = np.full(classes.shape, np.nan, dtype=np.float32)
    evidence = np.zeros(classes.shape, dtype=np.float32)
    mean[classes == GROUND] = 0.05
    mean[classes == TALL] = 0.50
    mean[classes == SOLID] = 0.95
    # A neighbourhood can establish ground for a cell with no direct return.
    # Give that claim one history slot, not zero and not a full observation.
    evidence[known] = np.maximum(1, coverage[known]) / float(history_scans)
    np.minimum(evidence, 1.0, out=evidence)
    variance[known] = mean[known] * (1.0 - mean[known]) / (evidence[known] + 1.0)
    return mean, variance, evidence


# How much slack to add on a side that has to grow, in cells. Expanding
# by exactly what the newest window needs would reallocate and copy on
# nearly every scan while the truck is driving in a straight line.
_EXPAND_MARGIN_CELLS = 40


class GlobalGrid:
    """Every classified window ever seen, as per-cell vote counts.

    Votes rather than last-writer-wins: a cell the truck drove past
    once, from one angle, in one scan window, should not out-shout
    twenty later looks at it. A cell is TALL when at least
    ``tall_share`` of its votes called it tall — one look in five by
    default — and SOLID needs ``solid_votes`` of shadow evidence AND at
    least as many solid votes as tall ones, so a rock seen through grass
    stays TALL until the shadow wins.

    ``tall_share`` replaces the rule this grid shipped with, "any tall
    vote ever, TALL forever". On the 2026-08-18 bag that rule left 7 %
    of the ground the truck had driven over TALL in the accumulated map
    even after every per-look false positive inside 12 m was under 0.5 %:
    a cell within range of the truck for a minute collects hundreds of
    votes, and one stray look stuck. The driven cells that had any tall
    vote had a tall share under 0.20 (median 10 of 378); the surveyed
    trunks were never under 0.88. A share is also what lets a person who
    stood somewhere for a while stop being a wall once they have left.
    """

    def __init__(
        self,
        res: float = 0.25,
        max_side_cells: int = 1200,
        solid_votes: int = 2,
        tall_share: float = 0.2,
    ) -> None:
        if res <= 0.0:
            raise ValueError("res must be positive")
        if max_side_cells < 1:
            raise ValueError("max_side_cells must be at least 1")
        self.res = float(res)
        self.max_side_cells = int(max_side_cells)
        self.solid_votes = max(1, int(solid_votes))
        self.tall_share = min(1.0, max(0.0, float(tall_share)))
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.w = 0
        self.h = 0
        self.coarsened = False
        self.ground = np.zeros((0, 0), dtype=np.int32)
        self.tall = np.zeros((0, 0), dtype=np.int32)
        self.solid = np.zeros((0, 0), dtype=np.int32)

    def add(
        self,
        cls: np.ndarray,
        origin_x: float,
        origin_y: float,
        res: float,
        sensor_xy: tuple[float, float] | None = None,
        vote_range_m: float | None = None,
    ) -> None:
        """Fold one classified window in, expanding (or coarsening) to fit.

        With ``sensor_xy`` and ``vote_range_m`` given, only cells within
        that range of the sensor vote; the rest of the window is folded
        in as UNSEEN (no vote). Beyond ~12 m one second of Mid-360 scans
        is a sprinkle of lone returns, and on the 2026-08-18 bag the far
        half of the window called 60-78 % of the ground the truck later
        drove over TALL under every classifier tried — a vote from there
        is noise, and TALL is sticky. Gating the votes rather than the
        classification keeps the live local grid honest about what it
        thinks it sees while the accumulated map only remembers what was
        seen well.
        """
        h_in, w_in = cls.shape
        if h_in == 0 or w_in == 0:
            return
        if sensor_xy is not None and vote_range_m is not None:
            cx = origin_x + (np.arange(w_in) + 0.5) * res - sensor_xy[0]
            cy = origin_y + (np.arange(h_in) + 0.5) * res - sensor_xy[1]
            far = np.hypot(cx[None, :], cy[:, None]) > vote_range_m
            cls = np.where(far, UNSEEN, cls).astype(np.int8)
        self._ensure(origin_x, origin_y, origin_x + w_in * res, origin_y + h_in * res)

        # Cell CENTRES, so an incoming cell always lands wholly in one
        # global cell even after the global res has doubled away from
        # the incoming one.
        cols = np.floor(
            (origin_x + (np.arange(w_in) + 0.5) * res - self.origin_x) / self.res
        ).astype(np.int64)
        rows = np.floor(
            (origin_y + (np.arange(h_in) + 0.5) * res - self.origin_y) / self.res
        ).astype(np.int64)
        np.clip(cols, 0, self.w - 1, out=cols)
        np.clip(rows, 0, self.h - 1, out=rows)

        # Accumulate into the BLOCK the window covers, not into the whole
        # map: a bincount with minlength = w*h costs the size of the map
        # on every scan, which is 1.4 M cells once the drive is long
        # enough — the one thing here that would quietly stop keeping up
        # with 10 Hz halfway through a field day.
        r_lo, r_hi = int(rows[0]), int(rows[-1]) + 1
        c_lo, c_hi = int(cols[0]), int(cols[-1]) + 1
        bh, bw = r_hi - r_lo, c_hi - c_lo
        idx = ((rows - r_lo)[:, None] * bw + (cols - c_lo)[None, :]).reshape(-1)
        flat = cls.reshape(-1)
        n = bh * bw
        for value, votes in (
            (GROUND, self.ground),
            (TALL, self.tall),
            (SOLID, self.solid),
        ):
            hit = idx[flat == value]
            if hit.size:
                # astype, not a bare +=: bincount is int64 and numpy
                # refuses to cast it down into the int32 vote array.
                votes[r_lo:r_hi, c_lo:c_hi] += (
                    np.bincount(hit, minlength=n).reshape(bh, bw).astype(np.int32)
                )

    def _ensure(self, x0: float, y0: float, x1: float, y1: float) -> None:
        if self.w == 0:
            self.origin_x = float(np.floor(x0 / self.res) * self.res)
            self.origin_y = float(np.floor(y0 / self.res) * self.res)
        while True:
            col0 = int(np.floor((x0 - self.origin_x) / self.res))
            row0 = int(np.floor((y0 - self.origin_y) / self.res))
            col1 = int(np.ceil((x1 - self.origin_x) / self.res))
            row1 = int(np.ceil((y1 - self.origin_y) / self.res))
            # Only a side that actually has to grow gets the margin, so
            # a window well inside the map is a no-op instead of a copy.
            # The margin is capped against max_side_cells as well, or a
            # small enough bound could never be met and the coarsening
            # loop below would never terminate.
            margin = min(_EXPAND_MARGIN_CELLS, max(1, self.max_side_cells // 8))
            lo_c = col0 - margin if col0 < 0 else 0
            lo_r = row0 - margin if row0 < 0 else 0
            hi_c = col1 + margin if col1 > self.w else self.w
            hi_r = row1 + margin if row1 > self.h else self.h
            if (
                hi_c - lo_c <= self.max_side_cells
                and hi_r - lo_r <= self.max_side_cells
            ):
                break
            self._coarsen()
        if (lo_c, lo_r, hi_c, hi_r) == (0, 0, self.w, self.h):
            return

        new_w = hi_c - lo_c
        new_h = hi_r - lo_r
        for name in ("ground", "tall", "solid"):
            old = getattr(self, name)
            new = np.zeros((new_h, new_w), dtype=np.int32)
            if self.w and self.h:
                new[-lo_r : -lo_r + self.h, -lo_c : -lo_c + self.w] = old
            setattr(self, name, new)
        self.origin_x += lo_c * self.res
        self.origin_y += lo_r * self.res
        self.w = new_w
        self.h = new_h

    def _coarsen(self) -> None:
        """Halve the cell count per side by summing 2x2 blocks of votes.

        The origin stays where it is (so the lattice shifts by at most
        half a coarse cell relative to the incoming windows, which
        ``add`` handles by binning cell centres). Same rule as
        ``rasterize``: double, never fit exactly, so the map does not
        shimmer.
        """
        pad_r = self.h % 2
        pad_c = self.w % 2
        for name in ("ground", "tall", "solid"):
            old = getattr(self, name)
            if pad_r or pad_c:
                old = np.pad(old, ((0, pad_r), (0, pad_c)))
            hh, ww = old.shape
            setattr(self, name, old.reshape(hh // 2, 2, ww // 2, 2).sum(axis=(1, 3)))
        self.res *= 2.0
        self.h = (self.h + pad_r) // 2
        self.w = (self.w + pad_c) // 2
        self.coarsened = True

    def to_int8(self) -> np.ndarray:
        """The (h, w) grid of cell classes. Values are UNSEEN/GROUND/TALL/SOLID."""
        out = np.full((self.h, self.w), UNSEEN, dtype=np.int8)
        if self.w == 0 or self.h == 0:
            return out
        total = self.ground + self.tall + self.solid
        seen = total > 0
        out[seen] = GROUND
        standing = self.tall + self.solid
        # Strict > 0 as well as the share: with tall_share 0 the rule
        # degrades to the old "any tall vote", never to "every seen
        # cell is tall".
        tall = (standing > 0) & (standing >= self.tall_share * total)
        out[tall] = TALL
        out[tall & (self.solid >= self.solid_votes) & (self.solid >= self.tall)] = SOLID
        return out

    def to_grid(self) -> Grid:
        return Grid(
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            res=self.res,
            w=self.w,
            h=self.h,
            data=self.to_int8().reshape(-1),
        )


def census(cls: np.ndarray) -> dict[str, int]:
    """How many cells of each class — the line worth logging."""
    return {
        "unseen": int((cls == UNSEEN).sum()),
        "ground": int((cls == GROUND).sum()),
        "tall": int((cls == TALL).sum()),
        "solid": int((cls == SOLID).sum()),
    }
