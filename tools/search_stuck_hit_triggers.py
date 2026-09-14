#!/usr/bin/env python3
"""Search rolling-hit UAV trigger rules on ground-only shadow traces.

Lidar uncertainty near vegetation flickers as successive rays classify a cell.
Requiring consecutive samples can therefore reject a real approach while a
rolling hit count retains repeated evidence and ignores isolated frontier
spikes.  This script selects a rule on earlier seeds and reports later seeds
without tuning on them.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from search_stuck_triggers import (
    Run,
    discover,
    finite_values,
    load_run,
    spatially_unique,
    thresholds,
)


SECONDARY = (
    "probe_occ_probability_mean",
    "probe_occ_probability_max",
    "probe_occ_blocked_frac",
    "probe_fused_cost_mean",
    "probe_fused_cost_max",
)


@dataclass(frozen=True)
class HitRule:
    threshold: float
    window_s: float
    hits: int
    policy_persistence_s: float = 0.0
    min_world_speed: float = 0.02
    secondary: str | None = None
    secondary_direction: str = ">"
    secondary_threshold: float = 0.0

    def describe(self) -> str:
        text = (
            f"{self.hits} hits of probe_occ_exposure > {self.threshold:.6g} "
            f"within {self.window_s:g} s"
        )
        text += f" while world_speed_mps > {self.min_world_speed:.6g}"
        if self.secondary is not None:
            text += (
                f" with {self.secondary} {self.secondary_direction} "
                f"{self.secondary_threshold:.6g}"
            )
        return text + f", then {self.policy_persistence_s:g} s policy persistence"


@dataclass
class Metrics:
    regions: int = 0
    recalled: int = 0
    triggers: int = 0
    false_triggers: int = 0
    runs: int = 0
    leads: list[float] | None = None

    @property
    def recall(self) -> float:
        return self.recalled / self.regions if self.regions else 0.0

    @property
    def requests_per_run(self) -> float:
        return self.triggers / self.runs if self.runs else math.inf

    @property
    def false_per_run(self) -> float:
        return self.false_triggers / self.runs if self.runs else math.inf

    @property
    def median_lead(self) -> float:
        return float(np.median(self.leads)) if self.leads else math.nan


def rolling_condition(rule: HitRule, run: Run) -> np.ndarray:
    exposure = run.values("probe_occ_exposure")
    hit = np.isfinite(exposure) & (exposure > rule.threshold)
    speed = run.values("world_speed_mps")
    hit &= np.isfinite(speed) & (speed > rule.min_world_speed)
    if rule.secondary is not None:
        values = run.values(rule.secondary)
        if rule.secondary_direction == ">":
            hit &= np.isfinite(values) & (values > rule.secondary_threshold)
        else:
            hit &= np.isfinite(values) & (values < rule.secondary_threshold)

    active = np.zeros(len(hit), dtype=bool)
    left = 0
    count = 0
    for right, value in enumerate(hit):
        count += int(value)
        while run.times[right] - run.times[left] > rule.window_s:
            count -= int(hit[left])
            left += 1
        active[right] = count >= rule.hits
    return active


def persistent_episode_indices(rule: HitRule, run: Run) -> list[int]:
    active = rolling_condition(rule, run)
    result: list[int] = []
    began: float | None = None
    fired = False
    for index, value in enumerate(active):
        if not value:
            began = None
            fired = False
            continue
        if began is None:
            began = float(run.times[index])
        if not fired and run.times[index] - began >= rule.policy_persistence_s - 1e-9:
            result.append(index)
            fired = True
    return result


def score(
    rule: HitRule,
    runs: list[Run],
    map_radius_m: float,
    approach_s: float,
    stuck_s: float,
) -> Metrics:
    result = Metrics(runs=len(runs), leads=[])
    for run in runs:
        indices = spatially_unique(
            persistent_episode_indices(rule, run), run, map_radius_m
        )
        result.triggers += len(indices)
        matched: set[int] = set()
        # A spawn collision is not an approaching hazard and cannot be avoided
        # by a pre-contact request.  It is a placement/world-generation issue.
        regions = [
            region
            for region in run.regions
            if float(region.get("route_progress_m", 0.0)) >= 20.0
        ]
        for region in regions:
            result.regions += 1
            stall = float(region["sim_time_s"]) - stuck_s
            matches = [
                index
                for index in indices
                if stall - approach_s <= run.times[index] <= stall
                and math.dist(run.xy[index], region["stuck_xy"]) <= map_radius_m
            ]
            if matches:
                chosen = min(matches, key=lambda index: run.times[index])
                matched.add(chosen)
                result.recalled += 1
                assert result.leads is not None
                result.leads.append(stall - float(run.times[chosen]))
        result.false_triggers += len(indices) - len(matched)
    return result


def score_retained_products(
    rule: HitRule,
    runs: list[Run],
    dedupe_radius_m: float,
    product_reach_m: float,
    stuck_s: float,
) -> Metrics:
    """Score whether any retained UAV product covers each later drag region."""
    result = Metrics(runs=len(runs), leads=[])
    for run in runs:
        indices = spatially_unique(
            persistent_episode_indices(rule, run), run, dedupe_radius_m
        )
        result.triggers += len(indices)
        used: set[int] = set()
        regions = [
            region
            for region in run.regions
            if float(region.get("route_progress_m", 0.0)) >= 20.0
        ]
        for region in regions:
            result.regions += 1
            stall = float(region["sim_time_s"]) - stuck_s
            matches = [
                index
                for index in indices
                if run.times[index] <= stall
                and math.dist(run.xy[index], region["stuck_xy"]) <= product_reach_m
            ]
            if matches:
                chosen = max(matches, key=lambda index: run.times[index])
                used.add(chosen)
                result.recalled += 1
                assert result.leads is not None
                result.leads.append(stall - float(run.times[chosen]))
        result.false_triggers += len(indices) - len(used)
    return result


def rules(
    train: list[Run],
    *,
    include_secondary: bool = True,
    min_world_speed: float = 0.02,
) -> list[HitRule]:
    exposure = finite_values(train, "probe_occ_exposure")
    primary = thresholds(
        exposure, (0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.97, 0.98, 0.99)
    )
    primary = sorted(set(primary + [0.02, 0.03, 0.04, 0.04895, 0.06, 0.08]))
    windows = (2.0, 3.0, 4.0, 5.0, 6.0)
    output: list[HitRule] = []
    bases: list[tuple[float, float, int]] = []
    for threshold in primary:
        for window in windows:
            for hits in range(1, min(6, int(2 * window) + 1)):
                bases.append((threshold, window, hits))
                output.append(
                    HitRule(
                        threshold,
                        window,
                        hits,
                        min_world_speed=min_world_speed,
                    )
                )
    if not include_secondary:
        return output
    for name in SECONDARY:
        values = finite_values(train, name)
        gates = thresholds(values, (0.25, 0.50, 0.75, 0.90))
        for threshold, window, hits in bases:
            for gate in gates:
                for direction in (">", "<"):
                    output.append(
                        HitRule(
                            threshold,
                            window,
                            hits,
                            min_world_speed=min_world_speed,
                            secondary=name,
                            secondary_direction=direction,
                            secondary_threshold=gate,
                        )
                    )
    return output


def rank(metrics: Metrics) -> tuple[float, float, float, float]:
    return (
        metrics.recall,
        -metrics.false_per_run,
        -metrics.requests_per_run,
        metrics.median_lead if math.isfinite(metrics.median_lead) else -math.inf,
    )


def report(label: str, rule: HitRule, metrics: Metrics) -> None:
    lead = "-" if not math.isfinite(metrics.median_lead) else f"{metrics.median_lead:.1f}s"
    print(
        f"{label:<10} recall {metrics.recalled}/{metrics.regions} "
        f"({100 * metrics.recall:5.1f}%), requests/run {metrics.requests_per_run:5.2f}, "
        f"false/run {metrics.false_per_run:5.2f}, median pre-stall lead {lead}; "
        f"{rule.describe()}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--validation-seed", type=int, required=True)
    parser.add_argument("--region-radius-m", type=float, default=12.5)
    parser.add_argument("--map-radius-m", type=float, default=12.5)
    parser.add_argument(
        "--retained-products",
        action="store_true",
        help="score whether any earlier retained UAV product covers a drag region",
    )
    parser.add_argument(
        "--product-dedupe-radius-m",
        type=float,
        default=16.5,
        help="distance used to merge overlapping persistent products",
    )
    parser.add_argument(
        "--product-reach-m",
        type=float,
        default=20.5,
        help="map half-width plus the forward probe's reach",
    )
    parser.add_argument("--min-world-speed", type=float, default=0.02)
    parser.add_argument("--approach-s", type=float, default=15.0)
    parser.add_argument("--stuck-s", type=float, default=30.0)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument(
        "--candidate",
        nargs=3,
        type=float,
        metavar=("THRESHOLD", "WINDOW_S", "HITS"),
        help="report one fixed primary rule before any search; use --top 0 to stop there",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="skip secondary hazard gates for a fast rolling-hit sweep",
    )
    args = parser.parse_args()

    loaded = [
        run
        for path in discover(args.paths)
        if (run := load_run(path, args.region_radius_m)) is not None
    ]
    train = [run for run in loaded if run.seed < args.validation_seed]
    validation = [run for run in loaded if run.seed >= args.validation_seed]
    train_regions = sum(
        float(region.get("route_progress_m", 0.0)) >= 20.0
        for run in train for region in run.regions
    )
    validation_regions = sum(
        float(region.get("route_progress_m", 0.0)) >= 20.0
        for run in validation for region in run.regions
    )
    if not train or not validation:
        raise SystemExit("validation seed must split the runs")
    if not train_regions:
        raise SystemExit("selection runs have no post-startup drag regions yet")
    print(
        f"loaded {len(loaded)} runs: selection {len(train)} runs/{train_regions} regions, "
        f"validation {len(validation)} runs/{validation_regions} regions"
    )

    def evaluate(rule: HitRule, subset: list[Run]) -> Metrics:
        if args.retained_products:
            return score_retained_products(
                rule,
                subset,
                args.product_dedupe_radius_m,
                args.product_reach_m,
                args.stuck_s,
            )
        return score(rule, subset, args.map_radius_m, args.approach_s, args.stuck_s)

    if args.candidate is not None:
        threshold, window_s, hits_float = args.candidate
        hits = int(hits_float)
        if not 0.0 <= threshold <= 1.0 or window_s < 0.5:
            parser.error("candidate threshold/window is outside the executable range")
        if hits < 1 or hits != hits_float:
            parser.error("candidate hits must be a positive integer")
        candidate = HitRule(
            threshold,
            window_s,
            hits,
            min_world_speed=args.min_world_speed,
        )
        print("fixed candidate")
        report("selection", candidate, evaluate(candidate, train))
        report("validation", candidate, evaluate(candidate, validation))
        print()
        if args.top <= 0:
            return 0

    candidates = [
        (rule, evaluate(rule, train))
        for rule in rules(
            train,
            include_secondary=not args.primary_only,
            min_world_speed=args.min_world_speed,
        )
    ]
    candidates.sort(key=lambda item: rank(item[1]), reverse=True)
    print(f"searched {len(candidates)} rolling-hit rules\n")
    for rule, selected in candidates[: args.top]:
        report("selection", rule, selected)
        report("validation", rule, evaluate(rule, validation))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
