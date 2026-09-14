#!/usr/bin/env python3
"""Search simple, executable rules that warn before Route drag regions.

The correlation report ranks individual samples.  A UAV policy acts on
episodes instead: a condition must persist, one request covers a 25 m region,
and a useful warning has to occur before the harness starts its stuck timer.
This tool searches small one- and two-variable rules under those constraints,
selects them on one set of seeds, and reports their untouched validation-seed
performance.

Example::

    python tools/search_stuck_triggers.py \
      runtime/experiments/r11_trigger_shadow_48 --validation-seed 32
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


SEED_RE = re.compile(r"(?:^|/)s(?P<seed>[0-9]+)_")
SECONDARY_FEATURES = (
    "probe_occ_probability_mean",
    "probe_occ_probability_max",
    "probe_occ_blocked_frac",
    "probe_fused_cost_mean",
    "probe_fused_cost_max",
    "path_occ_probability_mean",
    "path_occ_probability_max",
    "path_occ_blocked_frac",
    "path_fused_cost_mean",
    "path_fused_cost_max",
    "pop_occ_probability_mean",
    "pop_occ_probability_max",
    "cf_baseline_viability",
    "cf_free_viability",
    "planner_best_cost",
    "planner_cost_spread",
    "command_speed_mps",
    "world_speed_mps",
)


@dataclass
class Run:
    path: Path
    seed: int
    samples: list[dict]
    times: np.ndarray
    xy: np.ndarray
    regions: list[dict]

    def values(self, name: str) -> np.ndarray:
        out = np.full(len(self.samples), np.nan, dtype=np.float64)
        for index, sample in enumerate(self.samples):
            value = sample.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[index] = float(value)
        return out


@dataclass(frozen=True)
class Rule:
    primary_threshold: float
    persistence_s: float
    secondary: str | None = None
    secondary_direction: str = ">"
    secondary_threshold: float = 0.0

    def describe(self) -> str:
        text = f"probe_occ_exposure > {self.primary_threshold:.6g}"
        if self.secondary is not None:
            text += (
                f" AND {self.secondary} {self.secondary_direction} "
                f"{self.secondary_threshold:.6g}"
            )
        return f"{text} for {self.persistence_s:g} s"


@dataclass
class Metrics:
    regions: int = 0
    recalled: int = 0
    triggers: int = 0
    false_triggers: int = 0
    run_count: int = 0
    leads: list[float] | None = None

    @property
    def recall(self) -> float:
        return self.recalled / self.regions if self.regions else 0.0

    @property
    def requests_per_run(self) -> float:
        return self.triggers / self.run_count if self.run_count else math.inf

    @property
    def false_per_run(self) -> float:
        return self.false_triggers / self.run_count if self.run_count else math.inf

    @property
    def median_prestall_s(self) -> float:
        return float(np.median(self.leads)) if self.leads else math.nan


def discover(inputs: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for path in inputs:
        if (path / "assistance_trace.jsonl").is_file():
            found.add(path)
        elif path.is_dir():
            found.update(item.parent for item in path.rglob("assistance_trace.jsonl"))
    return sorted(found)


def cluster_regions(items: list[dict], radius_m: float) -> list[dict]:
    regions: list[dict] = []
    for item in sorted(items, key=lambda value: float(value["sim_time_s"])):
        xy = item.get("stuck_xy")
        if not isinstance(xy, list) or len(xy) != 2:
            continue
        if any(math.dist(xy, prior["stuck_xy"]) <= radius_m for prior in regions):
            continue
        regions.append(item)
    return regions


def load_run(path: Path, region_radius_m: float) -> Run | None:
    match = SEED_RE.search(str(path))
    if match is None:
        return None
    samples = [
        json.loads(line)
        for line in (path / "assistance_trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not samples:
        return None
    interventions: list[dict] = []
    history = path / "intervention_history.json"
    if history.is_file():
        interventions = json.loads(history.read_text(encoding="utf-8")).get(
            "interventions", []
        )
    return Run(
        path=path,
        seed=int(match.group("seed")),
        samples=samples,
        times=np.asarray([float(item["sim_time_s"]) for item in samples]),
        xy=np.asarray([item["vehicle_xy"] for item in samples], dtype=np.float64),
        regions=cluster_regions(interventions, region_radius_m),
    )


def episode_indices(condition: np.ndarray, times: np.ndarray, persistence_s: float) -> list[int]:
    """First qualifying sample of every continuous true episode."""
    starts: list[int] = []
    began: float | None = None
    fired = False
    for index, active in enumerate(condition):
        if not active:
            began = None
            fired = False
            continue
        if began is None:
            began = float(times[index])
        if not fired and float(times[index]) - began + 1e-9 >= persistence_s:
            starts.append(index)
            fired = True
    return starts


def spatially_unique(indices: list[int], run: Run, radius_m: float) -> list[int]:
    """Approximate persistent 25 m products: one trigger per covered region."""
    accepted: list[int] = []
    for index in indices:
        if any(math.dist(run.xy[index], run.xy[prior]) <= radius_m for prior in accepted):
            continue
        accepted.append(index)
    return accepted


def condition_for(rule: Rule, run: Run) -> np.ndarray:
    primary = run.values("probe_occ_exposure")
    condition = np.isfinite(primary) & (primary > rule.primary_threshold)
    if rule.secondary is not None:
        values = run.values(rule.secondary)
        if rule.secondary_direction == ">":
            condition &= np.isfinite(values) & (values > rule.secondary_threshold)
        else:
            condition &= np.isfinite(values) & (values < rule.secondary_threshold)
    return condition


def score_rule(
    rule: Rule,
    runs: list[Run],
    *,
    map_radius_m: float,
    approach_s: float,
    stuck_s: float,
) -> Metrics:
    metrics = Metrics(run_count=len(runs), leads=[])
    for run in runs:
        indices = episode_indices(condition_for(rule, run), run.times, rule.persistence_s)
        indices = spatially_unique(indices, run, map_radius_m)
        metrics.triggers += len(indices)
        matched_indices: set[int] = set()
        for region in run.regions:
            metrics.regions += 1
            drag_s = float(region["sim_time_s"])
            stall_s = drag_s - stuck_s
            stuck_xy = region["stuck_xy"]
            matches = [
                index
                for index in indices
                if stall_s - approach_s <= run.times[index] <= stall_s
                and math.dist(run.xy[index], stuck_xy) <= map_radius_m
            ]
            if matches:
                chosen = min(matches, key=lambda index: run.times[index])
                metrics.recalled += 1
                matched_indices.add(chosen)
                assert metrics.leads is not None
                metrics.leads.append(stall_s - float(run.times[chosen]))
        metrics.false_triggers += len(indices) - len(matched_indices)
    return metrics


def finite_values(runs: list[Run], name: str) -> np.ndarray:
    parts = [run.values(name) for run in runs]
    if not parts:
        return np.empty(0)
    values = np.concatenate(parts)
    return values[np.isfinite(values)]


def thresholds(values: np.ndarray, quantiles: tuple[float, ...]) -> list[float]:
    if values.size == 0:
        return []
    return sorted(set(float(value) for value in np.quantile(values, quantiles)))


def candidate_rules(train: list[Run]) -> list[Rule]:
    primary_values = finite_values(train, "probe_occ_exposure")
    primary_thresholds = thresholds(
        primary_values, (0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.97, 0.98, 0.99)
    )
    primary_thresholds = sorted(set(primary_thresholds + [0.04895]))
    persistence = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
    rules = [Rule(value, duration) for value in primary_thresholds for duration in persistence]
    for name in SECONDARY_FEATURES:
        values = finite_values(train, name)
        secondary_thresholds = thresholds(values, (0.25, 0.50, 0.75, 0.90, 0.95))
        for primary in primary_thresholds:
            for secondary in secondary_thresholds:
                for direction in (">", "<"):
                    for duration in persistence:
                        rules.append(Rule(primary, duration, name, direction, secondary))
    return rules


def selection_key(item: tuple[Rule, Metrics]) -> tuple[float, float, float, float]:
    _, metrics = item
    # Missing a drag region is much more expensive than buying an extra map,
    # but equal-recall rules should be sparse and early.
    lead = metrics.median_prestall_s
    return (
        metrics.recall,
        -metrics.false_per_run,
        -metrics.requests_per_run,
        lead if math.isfinite(lead) else -math.inf,
    )


def show(label: str, rule: Rule, metrics: Metrics) -> str:
    lead = metrics.median_prestall_s
    lead_text = "-" if not math.isfinite(lead) else f"{lead:.1f}s"
    return (
        f"{label:<10} recall {metrics.recalled}/{metrics.regions} "
        f"({100.0 * metrics.recall:5.1f}%), requests/run {metrics.requests_per_run:5.2f}, "
        f"false/run {metrics.false_per_run:5.2f}, median pre-stall lead {lead_text}; "
        f"{rule.describe()}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument(
        "--validation-seed", type=int, required=True,
        help="seeds below this value select the rule; later seeds only validate it",
    )
    parser.add_argument("--region-radius-m", type=float, default=12.5)
    parser.add_argument("--map-radius-m", type=float, default=12.5)
    parser.add_argument("--approach-s", type=float, default=15.0)
    parser.add_argument("--stuck-s", type=float, default=30.0)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    runs = [
        run
        for path in discover(args.paths)
        if (run := load_run(path, args.region_radius_m)) is not None
    ]
    train = [run for run in runs if run.seed < args.validation_seed]
    validation = [run for run in runs if run.seed >= args.validation_seed]
    if not train or not validation:
        raise SystemExit("the validation seed must split the discovered runs into two sets")
    train_regions = sum(len(run.regions) for run in train)
    validation_regions = sum(len(run.regions) for run in validation)
    if train_regions == 0:
        raise SystemExit("the selection runs do not yet contain any drag interventions")

    scorer: Callable[[Rule, list[Run]], Metrics] = lambda rule, subset: score_rule(
        rule,
        subset,
        map_radius_m=args.map_radius_m,
        approach_s=args.approach_s,
        stuck_s=args.stuck_s,
    )
    candidates = [(rule, scorer(rule, train)) for rule in candidate_rules(train)]
    candidates.sort(key=selection_key, reverse=True)
    print(
        f"loaded {len(runs)} runs: selection {len(train)} runs/{train_regions} regions, "
        f"validation {len(validation)} runs/{validation_regions} regions"
    )
    print(f"searched {len(candidates)} executable rules\n")
    seen: set[str] = set()
    shown = 0
    for rule, train_metrics in candidates:
        description = rule.describe()
        if description in seen:
            continue
        seen.add(description)
        print(show("selection", rule, train_metrics))
        print(show("validation", rule, scorer(rule, validation)))
        print()
        shown += 1
        if shown >= args.top:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
