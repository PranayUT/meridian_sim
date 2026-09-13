from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

CellKey = tuple[int, int]


def aggregate_class_probabilities(
    probabilities: np.ndarray, groups: Iterable[Iterable[int]]
) -> np.ndarray:
    """Sum disjoint native class probabilities into an external ontology."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim < 1:
        raise ValueError("probabilities must have a class axis")
    group_ids = [tuple(int(value) for value in group) for group in groups]
    if not group_ids:
        return np.empty((*probabilities.shape[:-1], 0), dtype=np.float64)
    flat_ids = [value for group in group_ids for value in group]
    if len(set(flat_ids)) != len(flat_ids):
        raise ValueError("ontology source classes must be disjoint")
    if any(value < 0 or value >= probabilities.shape[-1] for value in flat_ids):
        raise ValueError("ontology source class is outside the probability tensor")
    return np.stack(
        [probabilities[..., group].sum(axis=-1) for group in group_ids], axis=-1
    )


def linear_ramp(value: np.ndarray, safe: float, bad: float) -> np.ndarray:
    if bad <= safe:
        raise ValueError("bad threshold must be greater than safe threshold")
    return np.clip((value - safe) / (bad - safe), 0.0, 1.0)


@dataclass
class SemanticCell:
    evidence: np.ndarray
    last_update: float


@dataclass
class ElevationCell:
    mean: float
    variance: float
    evidence: float
    last_update: float


class SemanticEvidenceGrid:
    def __init__(
        self,
        resolution: float,
        class_costs: Iterable[float],
        evidence_half_life: float,
        evidence_cap: float,
        weak_evidence_scale: float,
        age_half_life: float,
    ) -> None:
        self.resolution = float(resolution)
        self.class_costs = np.asarray(list(class_costs), dtype=np.float64)
        self.num_classes = int(self.class_costs.size)
        self.evidence_half_life = float(evidence_half_life)
        self.evidence_cap = float(evidence_cap)
        self.weak_evidence_scale = float(weak_evidence_scale)
        self.age_half_life = float(age_half_life)
        self.cells: dict[CellKey, SemanticCell] = {}

    def _key_array(self, xy: np.ndarray) -> np.ndarray:
        return np.floor(np.asarray(xy) / self.resolution).astype(np.int64)

    def update(
        self,
        xy: np.ndarray,
        probabilities: np.ndarray,
        weights: np.ndarray,
        stamp: float,
    ) -> None:
        if xy.size == 0:
            return
        probabilities = np.asarray(probabilities, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if probabilities.shape != (xy.shape[0], self.num_classes):
            raise ValueError(
                f"Expected probabilities {(xy.shape[0], self.num_classes)}, got {probabilities.shape}"
            )

        keys = self._key_array(xy)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        for group_index, key_values in enumerate(unique):
            mask = inverse == group_index
            addition = (probabilities[mask] * weights[mask, None]).sum(axis=0)
            key = (int(key_values[0]), int(key_values[1]))
            cell = self.cells.get(key)
            if cell is None:
                evidence = addition
            else:
                dt = max(0.0, stamp - cell.last_update)
                decay = 2.0 ** (-dt / max(self.evidence_half_life, 1e-6))
                evidence = cell.evidence * decay + addition
            total = float(evidence.sum())
            if total > self.evidence_cap:
                evidence *= self.evidence_cap / total
            self.cells[key] = SemanticCell(evidence=evidence, last_update=stamp)

    def render(self, center_xy: np.ndarray, width: int, height: int, now: float):
        return self.render_with_probabilities(center_xy, width, height, now)[:4]

    def render_with_probabilities(
        self, center_xy: np.ndarray, width: int, height: int, now: float
    ):
        """Render cost and uncertainty while retaining the class posterior."""
        origin, probabilities, _, support, age = self.render_evidence(
            center_xy, width, height, now
        )
        cost = np.full((height, width), np.nan, dtype=np.float32)
        uncertainty = np.full_like(cost, np.nan)
        observed = (support > 0.0).astype(np.float32)

        known = support > 0.0
        if np.any(known):
            p = probabilities[known]
            entropy_denominator = np.log(max(self.num_classes, 2))
            entropy = (
                -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1) / entropy_denominator
            )
            weak = np.exp(-support[known] / max(self.weak_evidence_scale, 1e-6))
            age_u = 1.0 - 2.0 ** (-age[known] / max(self.age_half_life, 1e-6))
            cost[known] = p @ self.class_costs
            uncertainty[known] = np.maximum.reduce((entropy, weak, age_u))
        return origin, cost, uncertainty, observed, probabilities

    def render_evidence(
        self, center_xy: np.ndarray, width: int, height: int, now: float
    ):
        """Render class marginals for ``MapEvidence`` publishers.

        The accumulated class weights act as Dirichlet evidence. Each class
        marginal is therefore a Bernoulli probability with the corresponding
        Beta variance. Zero support means that the cell makes no claim.
        """
        origin = (
            np.asarray(center_xy, dtype=np.float64)
            - np.array([width * self.resolution, height * self.resolution]) / 2.0
        )
        origin_key = np.floor(origin / self.resolution).astype(np.int64)
        origin = origin_key.astype(np.float64) * self.resolution

        probabilities = np.full(
            (height, width, self.num_classes), np.nan, dtype=np.float32
        )
        variance = np.full_like(probabilities, np.nan)
        support_grid = np.zeros((height, width), dtype=np.float32)
        age_grid = np.full((height, width), np.nan, dtype=np.float32)

        for (gx, gy), cell in self.cells.items():
            x, y = gx - origin_key[0], gy - origin_key[1]
            if not (0 <= x < width and 0 <= y < height):
                continue
            age = max(0.0, now - cell.last_update)
            decay = 2.0 ** (-age / max(self.evidence_half_life, 1e-6))
            evidence = cell.evidence * decay
            support = float(evidence.sum())
            if support <= 1e-9:
                continue
            p = evidence / support
            probabilities[y, x] = p
            variance[y, x] = p * (1.0 - p) / (support + 1.0)
            support_grid[y, x] = support
            age_grid[y, x] = age
        return origin, probabilities, variance, support_grid, age_grid

    def prune(self, center_xy: np.ndarray, retention_radius: float) -> None:
        center_key = np.asarray(center_xy, dtype=np.float64) / self.resolution
        radius_cells_sq = (retention_radius / self.resolution) ** 2
        self.cells = {
            key: value
            for key, value in self.cells.items()
            if (key[0] - center_key[0]) ** 2 + (key[1] - center_key[1]) ** 2
            <= radius_cells_sq
        }


class ElevationEvidenceGrid:
    def __init__(
        self,
        resolution: float,
        min_samples: int,
        variance_bad: float,
        slope_safe_deg: float,
        slope_bad_deg: float,
        roughness_safe: float,
        roughness_bad: float,
        age_half_life: float,
        evidence_half_life: float = 10.0,
        evidence_cap: float = 10.0,
    ) -> None:
        self.resolution = float(resolution)
        self.min_samples = int(min_samples)
        self.variance_bad = float(variance_bad)
        self.slope_safe_deg = float(slope_safe_deg)
        self.slope_bad_deg = float(slope_bad_deg)
        self.roughness_safe = float(roughness_safe)
        self.roughness_bad = float(roughness_bad)
        self.age_half_life = float(age_half_life)
        self.evidence_half_life = float(evidence_half_life)
        self.evidence_cap = float(evidence_cap)
        self.cells: dict[CellKey, ElevationCell] = {}

    def update(self, xyz: np.ndarray, stamp: float) -> None:
        if xyz.size == 0:
            return
        xyz = np.asarray(xyz, dtype=np.float64)
        keys = np.floor(xyz[:, :2] / self.resolution).astype(np.int64)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        for group_index, key_values in enumerate(unique):
            values = xyz[inverse == group_index, 2]
            mean_b = float(np.median(values))
            mad_b = float(np.median(np.abs(values - mean_b)))
            variance_b = (1.4826 * mad_b) ** 2
            key = (int(key_values[0]), int(key_values[1]))
            old = self.cells.get(key)
            if old is None:
                merged = ElevationCell(mean_b, variance_b, 1.0, stamp)
            else:
                dt = max(0.0, stamp - old.last_update)
                old_evidence = old.evidence * 2.0 ** (
                    -dt / max(self.evidence_half_life, 1e-6)
                )
                n = old_evidence + 1.0
                delta = mean_b - old.mean
                mean = old.mean + delta / n
                variance = (
                    old_evidence * old.variance
                    + variance_b
                    + delta * delta * old_evidence / n
                ) / n
                merged = ElevationCell(mean, variance, min(n, self.evidence_cap), stamp)
            self.cells[key] = merged

    def _cost_from_elevation(
        self, elevation: np.ndarray, within_cell_variance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return geometric cost and the mask with full local geometry."""
        dzdx = np.full_like(elevation, np.nan)
        dzdy = np.full_like(elevation, np.nan)
        dzdx[:, 1:-1] = (elevation[:, 2:] - elevation[:, :-2]) / (2.0 * self.resolution)
        dzdy[1:-1, :] = (elevation[2:, :] - elevation[:-2, :]) / (2.0 * self.resolution)
        slope = np.degrees(np.arctan(np.sqrt(dzdx * dzdx + dzdy * dzdy)))

        finite = np.isfinite(elevation)
        values = np.where(finite, elevation, 0.0)
        count = np.zeros_like(elevation)
        total = np.zeros_like(elevation)
        total_sq = np.zeros_like(elevation)
        # Sum each 3x3 neighborhood without adding a SciPy dependency. This
        # path also runs for the uncertainty ensemble below, so the old nested
        # Python loop was too expensive for field use.
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                source_y = slice(
                    max(0, -dy), min(elevation.shape[0], elevation.shape[0] - dy)
                )
                source_x = slice(
                    max(0, -dx), min(elevation.shape[1], elevation.shape[1] - dx)
                )
                target_y = slice(
                    max(0, dy), min(elevation.shape[0], elevation.shape[0] + dy)
                )
                target_x = slice(
                    max(0, dx), min(elevation.shape[1], elevation.shape[1] + dx)
                )
                valid = finite[source_y, source_x]
                sample = values[source_y, source_x]
                count[target_y, target_x] += valid
                total[target_y, target_x] += sample
                total_sq[target_y, target_x] += sample * sample
        local_std = np.full_like(elevation, np.nan)
        enough = count >= self.min_samples
        local_variance = np.maximum(
            total_sq / np.maximum(count, 1.0)
            - np.square(total / np.maximum(count, 1.0)),
            0.0,
        )
        local_std[enough] = np.sqrt(local_variance[enough])

        within_cell_std = np.sqrt(np.maximum(within_cell_variance, 0.0))
        roughness = np.fmax(local_std, within_cell_std)
        slope_cost = linear_ramp(slope, self.slope_safe_deg, self.slope_bad_deg)
        roughness_cost = linear_ramp(roughness, self.roughness_safe, self.roughness_bad)
        cost = np.fmax(slope_cost, roughness_cost)
        full_geometry = finite & np.isfinite(slope) & np.isfinite(local_std)
        cost[~finite] = np.nan
        return cost, full_geometry

    def render_cost_evidence(
        self,
        center_xy: np.ndarray,
        width: int,
        height: int,
        now: float,
        sensor_variance: float = 0.0,
    ):
        """Render geometric cost with a cost-space posterior variance.

        Eight fixed antithetic elevation samples propagate uncertainty in the
        cell-height means through the nonlinear slope and roughness transform.
        A bounded-cost sampling term also prevents small populations from
        claiming false precision. Cells without the neighbors needed to
        compute both slope and roughness make no fusion claim.
        """
        origin, elevation, height_variance, support, age = self.render_evidence(
            center_xy, width, height, now
        )
        cost, full_geometry = self._cost_from_elevation(elevation, height_variance)
        posterior_height_variance = np.full_like(height_variance, np.nan)
        active = support > 0.0
        posterior_height_variance[active] = (
            np.maximum(height_variance[active], sensor_variance) / support[active]
        )
        sigma = np.sqrt(np.maximum(posterior_height_variance, 0.0))
        yy, xx = np.indices(elevation.shape, dtype=np.uint32)
        samples = []
        hashed = xx * np.uint32(73856093) ^ yy * np.uint32(19349663)
        for bit in (0, 5, 11, 17):
            sign = np.where(((hashed >> bit) & np.uint32(1)) == 0, -1.0, 1.0)
            for direction in (-1.0, 1.0):
                sample_cost, _ = self._cost_from_elevation(
                    elevation + direction * sign * sigma, height_variance
                )
                samples.append(sample_cost)
        sample_stack = np.stack(samples)
        sample_valid = np.isfinite(sample_stack)
        sample_count = sample_valid.sum(axis=0)
        sample_mean = np.nansum(sample_stack, axis=0) / np.maximum(sample_count, 1)
        sample_error = np.where(
            sample_valid, sample_stack - sample_mean[None, :, :], 0.0
        )
        cost_variance = np.sum(np.square(sample_error), axis=0) / np.maximum(
            sample_count, 1
        )
        effective_support = np.where(full_geometry, support, 0.0)
        # Cost is bounded to [0, 1]. This is the maximum-variance prior for a
        # bounded observation, reduced by the effective independent support.
        cost_variance = np.maximum(cost_variance, 0.25 / (effective_support + 1.0))
        cost_variance[~full_geometry] = np.nan
        return origin, cost, cost_variance, effective_support, age

    def render(self, center_xy: np.ndarray, width: int, height: int, now: float):
        origin, elevation, variance, support, age = self.render_evidence(
            center_xy, width, height, now
        )
        cost, full_geometry = self._cost_from_elevation(elevation, variance)

        variance_u = np.clip(variance / max(self.variance_bad, 1e-12), 0.0, 1.0)
        support_u = np.exp(-support / max(float(self.min_samples), 1.0))
        age_u = 1.0 - 2.0 ** (-age / max(self.age_half_life, 1e-6))
        neighborhood_u = np.where(full_geometry, 0.0, 1.0)
        uncertainty = np.maximum.reduce((variance_u, support_u, age_u, neighborhood_u))

        known = np.isfinite(elevation)
        cost[~known] = np.nan
        uncertainty[~known] = np.nan
        return (
            origin,
            cost.astype(np.float32),
            uncertainty.astype(np.float32),
            known.astype(np.float32),
        )

    def render_evidence(
        self, center_xy: np.ndarray, width: int, height: int, now: float
    ):
        """Render the accumulated elevation estimate and its native evidence."""
        origin = (
            np.asarray(center_xy, dtype=np.float64)
            - np.array([width * self.resolution, height * self.resolution]) / 2.0
        )
        origin_key = np.floor(origin / self.resolution).astype(np.int64)
        origin = origin_key.astype(np.float64) * self.resolution

        elevation = np.full((height, width), np.nan, dtype=np.float64)
        variance = np.full_like(elevation, np.nan)
        support = np.zeros_like(elevation)
        age = np.full_like(elevation, np.inf)

        for (gx, gy), cell in self.cells.items():
            x, y = gx - origin_key[0], gy - origin_key[1]
            if 0 <= x < width and 0 <= y < height:
                elevation[y, x] = cell.mean
                variance[y, x] = cell.variance
                support[y, x] = cell.evidence * 2.0 ** (
                    -max(0.0, now - cell.last_update)
                    / max(self.evidence_half_life, 1e-6)
                )
                age[y, x] = max(0.0, now - cell.last_update)
        return origin, elevation, variance, support, age

    def prune(self, center_xy: np.ndarray, retention_radius: float) -> None:
        center_key = np.asarray(center_xy, dtype=np.float64) / self.resolution
        radius_cells_sq = (retention_radius / self.resolution) ** 2
        self.cells = {
            key: value
            for key, value in self.cells.items()
            if (key[0] - center_key[0]) ** 2 + (key[1] - center_key[1]) ** 2
            <= radius_cells_sq
        }
