#!/usr/bin/env python3
"""Rank uncertainty variables by how well they predict a drag intervention.

`assistance_trace.jsonl` records many candidate stuck-predictors at a fixed
cadence; `intervention_history.json` records when the harness had to drag the
rover. This joins the two and asks, for every variable, whether it separates
the run-up to a drag from ordinary driving.

The separation statistic is AUC: the probability that a randomly chosen sample
from the window scores above a randomly chosen baseline sample. 0.50 is no
signal, 1.00 is perfect separation, and below 0.50 means the variable falls
rather than spikes.

Three windows are scored separately because the distinction matters:

  approach  the seconds before the vehicle stopped making progress. Only a
            variable that separates *here* can warn early enough to act.
  stalled   the stuck timer running, before the harness intervenes. A variable
            that only separates here is a detector, not a predictor.
  drag      immediately around the intervention.

Usage:
  python tools/analyze_stuck_correlation.py runtime/experiments/<run-id> [...]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

CONTEXT_KEYS = {
    "sim_time_s",
    "vehicle_xy",
    "yaw_rad",
    "action_roi",
    "action_source",
    "population_source",
    "assistance_state",
}


def load_run(run_dir: Path) -> tuple[list[dict], list[dict]]:
    trace_path = run_dir / "assistance_trace.jsonl"
    if not trace_path.exists():
        raise SystemExit(f"{trace_path} is missing; the run predates the trace")
    samples = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            samples.append(json.loads(line))
    interventions: list[dict] = []
    history_path = run_dir / "intervention_history.json"
    if history_path.exists():
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        interventions = list(payload["interventions"])
    return samples, interventions


def cluster_interventions(
    interventions: list[dict], radius_m: float
) -> list[dict]:
    """Keep the first intervention in each local UAV-map-sized region.

    Several harness drags can be repeated attempts through one wedge. Counting
    each as an independent prediction target exaggerates both sample size and
    lead time. The retained UAV products are local and persistent, so the unit
    that matters is the first failure in each spatial region.
    """
    if radius_m <= 0.0:
        return interventions.copy()
    regions: list[dict] = []
    for item in interventions:
        xy = item.get("stuck_xy")
        if not isinstance(xy, list) or len(xy) != 2:
            regions.append(item)
            continue
        if any(
            isinstance(region.get("stuck_xy"), list)
            and len(region["stuck_xy"]) == 2
            and math.dist(xy, region["stuck_xy"]) <= radius_m
            for region in regions
        ):
            continue
        regions.append(item)
    return regions


def add_derived_variables(samples: list[dict]) -> None:
    """Add interpretable estimates of when a map could change the decision.

    Uncertainty by itself mostly measures a moving sensor frontier. Multiplying
    it by obstacle probability or fused cost asks the more useful question:
    is unresolved evidence colocated with something that is constraining the
    rover? These remain diagnostics; they do not affect the live policy.
    """
    for sample in samples:
        for geometry in ("pop", "path", "probe"):
            def product(left: str, right: str) -> float | None:
                a = sample.get(f"{geometry}_{left}")
                b = sample.get(f"{geometry}_{right}")
                if not isinstance(a, (int, float)) or isinstance(a, bool):
                    return None
                if not isinstance(b, (int, float)) or isinstance(b, bool):
                    return None
                return float(a) * float(b)

            sample[f"{geometry}_occ_uncertain_probability"] = product(
                "occ_exposure", "occ_probability_mean"
            )
            sample[f"{geometry}_occ_unknown_probability"] = product(
                "occ_unknown_frac", "occ_probability_mean"
            )
            sample[f"{geometry}_occ_uncertain_cost"] = product(
                "occ_exposure", "fused_cost_mean"
            )
            sample[f"{geometry}_sem_unknown_cost"] = product(
                "sem_unknown_frac", "sem_cost_mean"
            )


def variable_names(samples: list[dict]) -> list[str]:
    names: dict[str, None] = {}
    for sample in samples:
        for key, value in sample.items():
            if key in CONTEXT_KEYS:
                continue
            if isinstance(value, bool) or isinstance(value, (int, float)):
                names[key] = None
    return list(names)


def column(samples: list[dict], name: str) -> np.ndarray:
    values = np.full(len(samples), np.nan)
    for index, sample in enumerate(samples):
        value = sample.get(name)
        if isinstance(value, bool):
            values[index] = float(value)
        elif isinstance(value, (int, float)) and value is not None:
            values[index] = float(value)
    return values


def auc(window: np.ndarray, baseline: np.ndarray) -> float | None:
    """Mann-Whitney U as a probability, tie-corrected."""
    window = window[np.isfinite(window)]
    baseline = baseline[np.isfinite(baseline)]
    if window.size == 0 or baseline.size == 0:
        return None
    joined = np.concatenate([window, baseline])
    order = joined.argsort(kind="mergesort")
    ranks = np.empty(joined.size, dtype=np.float64)
    ranks[order] = np.arange(1, joined.size + 1, dtype=np.float64)
    sorted_values = joined[order]
    start = 0
    for index in range(1, joined.size + 1):
        if index == joined.size or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    rank_sum = ranks[: window.size].sum()
    return float(
        (rank_sum - window.size * (window.size + 1) / 2.0)
        / (window.size * baseline.size)
    )


def sustained(values: np.ndarray, threshold: float, run: int) -> np.ndarray:
    """Mask of samples that begin `run` consecutive samples above threshold.

    The comparison is deliberately strict. With zero-inflated uncertainty
    fields, the 95th percentile is often exactly zero; treating equality as a
    crossing would turn every ordinary zero into an alarm and violate the
    requested false-positive rate.
    """
    above = np.isfinite(values) & (values > threshold)
    if run <= 1:
        return above
    starts = above.copy()
    for offset in range(1, run):
        shifted = np.zeros_like(above)
        if offset < above.size:
            shifted[: above.size - offset] = above[offset:]
        starts &= shifted
    return starts


def episode_starts(active: np.ndarray) -> np.ndarray:
    """Return only the first sample of each contiguous alarm episode."""
    starts = active.copy()
    if starts.size > 1:
        starts[1:] &= ~active[:-1]
    return starts


def lead_times(
    times: np.ndarray,
    values: np.ndarray,
    drags: list[float],
    threshold: float,
    horizon_s: float,
    run: int,
) -> list[float]:
    """How early each drag was first warned about.

    The earliest sustained crossing inside the horizon, not the crossing that
    happens to still be true at the drag. A variable that spikes on approach
    and then falls once the vehicle is pinned is the one worth having, and
    measuring from the drag backwards would score it zero.
    """
    starts = sustained(values, threshold, run)
    leads: list[float] = []
    previous_drag = -math.inf
    for drag in drags:
        # A warning for the next failure cannot predate the previous drag. Give
        # the relocation two seconds to settle, matching the drag window used
        # elsewhere in this report.
        earliest = max(drag - horizon_s, previous_drag + 2.0)
        window = starts & (times <= drag) & (times >= earliest)
        if not np.any(window):
            previous_drag = drag
            continue
        leads.append(float(drag - np.min(times[window])))
        previous_drag = drag
    return leads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument(
        "--stuck-s",
        type=float,
        default=30.0,
        help="harness stuck timer; sets where the approach window ends",
    )
    parser.add_argument(
        "--approach-s",
        type=float,
        default=15.0,
        help="length of the approach window before the stall began",
    )
    parser.add_argument(
        "--baseline-guard-s",
        type=float,
        default=20.0,
        help="extra seconds around a drag excluded from the baseline",
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--timeline",
        type=int,
        default=0,
        help="also print a per-drag time series for the top N variables",
    )
    parser.add_argument(
        "--min-run",
        type=int,
        default=2,
        help="consecutive samples above threshold that count as a warning",
    )
    parser.add_argument(
        "--false-positive-rate",
        type=float,
        default=0.05,
        help="baseline exceedance used to set the lead-time threshold",
    )
    parser.add_argument(
        "--drag-region-radius-m",
        type=float,
        default=12.5,
        help=(
            "nearby interventions counted as one map opportunity; defaults "
            "to half the 25 m UAV product width (0 disables clustering)"
        ),
    )
    args = parser.parse_args()
    if args.drag_region_radius_m < 0.0:
        parser.error("drag-region-radius-m may not be negative")

    times_all: list[np.ndarray] = []
    samples_all: list[dict] = []
    drag_marks: list[float] = []
    all_drag_marks: list[float] = []
    intervention_count = 0
    offset = 0.0
    for run_dir in args.run_dirs:
        samples, interventions = load_run(run_dir)
        add_derived_variables(samples)
        times = column(samples, "sim_time_s")
        if times.size == 0:
            continue
        # Concatenated runs are shifted onto one axis so windows never overlap.
        shift = offset - float(np.nanmin(times))
        times_all.append(times + shift)
        samples_all.extend(samples)
        regions = cluster_interventions(
            interventions, args.drag_region_radius_m
        )
        drag_marks.extend(float(item["sim_time_s"]) + shift for item in regions)
        all_drag_marks.extend(
            float(item["sim_time_s"]) + shift for item in interventions
        )
        intervention_count += len(interventions)
        offset = float(np.nanmax(times + shift)) + 1e6
        print(
            f"{run_dir.name}: {len(samples)} trace samples, "
            f"{len(interventions)} drags in {len(regions)} regions",
        )
    if not samples_all:
        raise SystemExit("no trace samples found")
    times = np.concatenate(times_all)
    if not drag_marks:
        raise SystemExit("no drag interventions recorded; nothing to correlate")

    approach = np.zeros(times.size, dtype=bool)
    stalled = np.zeros(times.size, dtype=bool)
    at_drag = np.zeros(times.size, dtype=bool)
    excluded = np.zeros(times.size, dtype=bool)
    for drag in drag_marks:
        stall_start = drag - args.stuck_s
        approach |= (times >= stall_start - args.approach_s) & (times < stall_start)
        stalled |= (times >= stall_start) & (times < drag - 2.0)
        at_drag |= (times >= drag - 2.0) & (times <= drag + 2.0)
    # Every intervention, including a repeat in an already-counted region, is
    # non-baseline behavior and must be kept out of ordinary driving.
    for drag in all_drag_marks:
        stall_start = drag - args.stuck_s
        excluded |= (
            times >= stall_start - args.approach_s - args.baseline_guard_s
        ) & (times <= drag + args.baseline_guard_s)
    baseline = ~excluded
    intervals = np.diff(np.sort(times))
    intervals = intervals[intervals > 0.0]
    cadence_s = float(np.median(intervals)) if intervals.size else 0.5

    print(
        f"\nwindows: approach {approach.sum()}, stalled {stalled.sum()}, "
        f"drag {at_drag.sum()}, baseline {baseline.sum()} samples "
        f"over {len(drag_marks)} drag regions ({intervention_count} interventions)"
    )
    print(
        "\nAUC 0.50 = no separation, below 0.50 means the variable falls. "
        "'dir' gives the predictive direction. 'lead' is the median seconds "
        "before each drag that a variable first crossed the corresponding "
        f"{100 * (1 - args.false_positive_rate):.0f}th baseline-tail threshold "
        f"for {args.min_run} samples running; 'prestall' subtracts the "
        f"{args.stuck_s:g} s harness timer, so positive values are true early "
        "warnings. 'alarm/min' counts episodes during ordinary driving."
    )

    rows = []
    for name in variable_names(samples_all):
        values = column(samples_all, name)
        base_values = values[baseline]
        base_values = base_values[np.isfinite(base_values)]
        if base_values.size < 10:
            continue
        if np.nanstd(values[np.isfinite(values)]) == 0.0:
            continue
        approach_auc = auc(values[approach], values[baseline])
        stalled_auc = auc(values[stalled], values[baseline])
        drag_auc = auc(values[at_drag], values[baseline])
        # Some strong precursors fall on approach (viability and speed are the
        # common cases). Orient every candidate so a warning is always a high
        # crossing, then convert the reported threshold back to native units.
        rising = approach_auc is None or approach_auc >= 0.5
        oriented_values = values if rising else -values
        oriented_base = oriented_values[baseline]
        oriented_base = oriented_base[np.isfinite(oriented_base)]
        threshold = float(
            np.quantile(oriented_base, 1.0 - args.false_positive_rate)
        )
        leads = lead_times(
            times,
            oriented_values,
            drag_marks,
            threshold,
            args.stuck_s + args.approach_s,
            args.min_run,
        )
        # A variable that alarms constantly explains nothing, however well it
        # separates. Count alarm *episodes*, not every overlapping run-sized
        # slice of one long plateau.
        active = sustained(oriented_values, threshold, args.min_run)
        baseline_starts = episode_starts(active) & baseline
        baseline_minutes = max(1e-9, float(np.sum(baseline)) * cadence_s / 60.0)
        alarms_per_min = float(np.sum(baseline_starts)) / baseline_minutes
        baseline_active = 100.0 * float(np.mean(active[baseline]))
        early = [lead for lead in leads if lead >= args.stuck_s]
        rows.append(
            {
                "name": name,
                "direction": ">" if rising else "<",
                "threshold": threshold if rising else -threshold,
                "approach": approach_auc,
                "stalled": stalled_auc,
                "drag": drag_auc,
                "baseline_median": float(np.median(base_values)),
                "approach_median": float(
                    np.nanmedian(values[approach])
                    if np.any(np.isfinite(values[approach]))
                    else math.nan
                ),
                "lead": float(np.median(leads)) if leads else math.nan,
                "prestall": (
                    float(np.median(leads) - args.stuck_s)
                    if leads
                    else math.nan
                ),
                "fired": len(leads),
                "early": len(early),
                "alarms": alarms_per_min,
                "baseline_active": baseline_active,
            }
        )

    def strength(row: dict) -> float:
        value = row["approach"]
        return 0.0 if value is None else abs(value - 0.5)

    rows.sort(key=strength, reverse=True)
    header = (
        f"{'variable':<34}{'dir':>4}{'threshold':>10}{'approach':>9}"
        f"{'stalled':>9}{'drag':>7}{'base':>10}{'window':>10}"
        f"{'lead s':>8}{'prestall':>9}{'fired':>7}{'early':>7}"
        f"{'alarm/min':>10}{'active%':>9}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for row in rows[: args.top]:
        def show(value: float | None) -> str:
            if value is None or not math.isfinite(value):
                return "    -"
            if abs(value) >= 1000.0:
                return f"{value:.1e}"
            return f"{value:.3f}"

        print(
            f"{row['name']:<34}{row['direction']:>4}"
            f"{show(row['threshold']):>10}{show(row['approach']):>9}"
            f"{show(row['stalled']):>9}"
            f"{show(row['drag']):>7}{show(row['baseline_median']):>10}"
            f"{show(row['approach_median']):>10}"
            f"{show(row['lead']):>8}{show(row['prestall']):>9}"
            f"{row['fired']:>5}/{len(drag_marks)}"
            f"{row['early']:>5}/{len(drag_marks)}"
            f"{row['alarms']:>10.2f}{row['baseline_active']:>9.1f}"
        )
    if args.timeline:
        show_timeline(
            times, samples_all, rows[: args.timeline], drag_marks, args.stuck_s
        )
    return 0


def show_timeline(
    times: np.ndarray,
    samples: list[dict],
    rows: list[dict],
    drags: list[float],
    stuck_s: float,
) -> None:
    """Print each drag's run-up so a spike can be seen, not just scored.

    An AUC says a variable separates; it does not say whether it separated on
    every drag or on one very hard one. Columns are 5 s bins relative to the
    intervention, so the stall boundary at -stuck_s is visible in place.
    """
    edges = np.arange(-stuck_s - 20.0, 5.1, 5.0)
    print("\n\nper-drag timeline, 5 s bins, seconds relative to the drag")
    print(f"the stall timer starts at {-stuck_s:+.0f} s\n")
    label_row = "".join(f"{edge:+6.0f}" for edge in edges)
    for row in rows:
        name = row["name"]
        values = column(samples, name)
        print(f"{name}")
        print(f"{'drag':<6}{label_row}")
        for index, drag in enumerate(drags, start=1):
            cells = []
            for edge in edges:
                window = (times >= drag + edge) & (times < drag + edge + 5.0)
                selected = values[window]
                selected = selected[np.isfinite(selected)]
                cells.append(
                    "     ." if selected.size == 0 else f"{np.max(selected):6.2f}"
                )
            print(f"{index:<6}" + "".join(cells))
        print()


if __name__ == "__main__":
    raise SystemExit(main())
