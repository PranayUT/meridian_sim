#!/usr/bin/env python3
"""Compare drag interventions over matched simulator-time windows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autonomy.meridian_drive.core import Route
from autonomy.meridian_drive.routes import load_route


TRIAL_RE = re.compile(
    r"^s(?P<seed>[0-9]+)(?:_[A-Za-z0-9]+)*_Route-11_"
    r"(?P<direction>forward|reverse)$"
)


@dataclass(frozen=True)
class Trial:
    path: Path
    seed: int
    direction: str
    final_time_s: float
    startup_drags: int

    @property
    def key(self) -> tuple[int, str]:
        return self.seed, self.direction


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def endpoint_row(trial: Trial) -> dict[str, str] | None:
    path = trial.path / "campaign.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def discover(root: Path, min_progress_m: float) -> dict[tuple[int, str], Trial]:
    found: dict[tuple[int, str], Trial] = {}
    for trace in root.glob("*/assistance_trace.jsonl"):
        match = TRIAL_RE.match(trace.parent.name)
        if match is None:
            continue
        lines = [
            line
            for line in trace.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if not lines:
            continue
        history = trace.parent / "intervention_history.json"
        startup_drags = 0
        if history.is_file():
            startup_drags = sum(
                float(item.get("route_progress_m", 0.0)) < min_progress_m
                for item in read_json(history).get("interventions", [])
            )
        trial = Trial(
            path=trace.parent,
            seed=int(match.group("seed")),
            direction=match.group("direction"),
            final_time_s=float(json.loads(lines[-1])["sim_time_s"]),
            startup_drags=startup_drags,
        )
        previous = found.get(trial.key)
        if previous is None or (
            trial.startup_drags == 0,
            trial.final_time_s,
        ) > (
            previous.startup_drags == 0,
            previous.final_time_s,
        ):
            found[trial.key] = trial
    return found


def interventions(
    trial: Trial, horizon_s: float, min_progress_m: float
) -> tuple[int, int]:
    path = trial.path / "intervention_history.json"
    if not path.is_file():
        return 0, 0
    retained = startup = 0
    for item in read_json(path).get("interventions", []):
        if float(item.get("sim_time_s", float("inf"))) > horizon_s:
            continue
        if float(item.get("route_progress_m", 0.0)) < min_progress_m:
            startup += 1
        else:
            retained += 1
    return retained, startup


def requests(trial: Trial, horizon_s: float) -> int:
    path = trial.path / "uav_request_history.json"
    if not path.is_file():
        return 0
    count = 0
    for item in read_json(path).get("requests", []):
        time_s = item.get("sim_time_s")
        if time_s is None or float(time_s) <= horizon_s:
            count += 1
    return count


def request_kinds(trial: Trial, horizon_s: float) -> Counter[str]:
    path = trial.path / "uav_request_history.json"
    result: Counter[str] = Counter()
    if not path.is_file():
        return result
    for item in read_json(path).get("requests", []):
        time_s = item.get("sim_time_s")
        if time_s is not None and float(time_s) > horizon_s:
            continue
        if item.get("probe_relevant"):
            kind = "forward_probe"
        elif item.get("mobility_relevant"):
            kind = "mobility"
        elif item.get("action_relevant"):
            kind = "selected_trajectory"
        elif item.get("decision_relevant"):
            kind = "counterfactual"
        else:
            kind = "exposure"
        result[kind] += 1
    return result


def covered_drag_leads(
    trial: Trial,
    horizon_s: float,
    min_progress_m: float,
    product_reach_m: float,
    stuck_s: float,
) -> list[float]:
    intervention_path = trial.path / "intervention_history.json"
    request_path = trial.path / "uav_request_history.json"
    if not intervention_path.is_file() or not request_path.is_file():
        return []
    request_items = read_json(request_path).get("requests", [])
    leads: list[float] = []
    for drag in read_json(intervention_path).get("interventions", []):
        drag_time = float(drag.get("sim_time_s", float("inf")))
        if (
            drag_time > horizon_s
            or float(drag.get("route_progress_m", 0.0)) < min_progress_m
        ):
            continue
        stall_time = drag_time - stuck_s
        stuck_xy = drag.get("stuck_xy")
        if not isinstance(stuck_xy, list) or len(stuck_xy) != 2:
            continue
        matches: list[float] = []
        for request in request_items:
            request_time = request.get("sim_time_s")
            request_xy = request.get("vehicle_xy")
            if (
                request_time is None
                or not isinstance(request_xy, list)
                or len(request_xy) != 2
            ):
                continue
            request_time = float(request_time)
            if request_time <= stall_time and math.dist(request_xy, stuck_xy) <= product_reach_m:
                matches.append(request_time)
        if matches:
            leads.append(stall_time - max(matches))
    return leads


def recovery_outcomes(
    trial: Trial,
    horizon_s: float,
    evaluation_s: float,
    success_distance_m: float,
) -> tuple[int, int, int]:
    samples: list[dict] = []
    history = trial.path / "intervention_history.json"
    drag_times = (
        [
            float(item["sim_time_s"])
            for item in read_json(history).get("interventions", [])
        ]
        if history.is_file()
        else []
    )
    trace = trial.path / "assistance_trace.jsonl"
    for line in trace.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        item = json.loads(line)
        if float(item["sim_time_s"]) > horizon_s:
            break
        samples.append(item)
    attempts = evaluated = successes = 0
    previous_count = 0
    for index, item in enumerate(samples):
        count = int(item.get("mapped_recovery_count", 0))
        if count <= previous_count:
            continue
        attempts += count - previous_count
        previous_count = count
        start_time = float(item["sim_time_s"])
        start_xy = item.get("vehicle_xy")
        if not isinstance(start_xy, list) or len(start_xy) != 2:
            continue
        next_drag = min(
            (drag_time for drag_time in drag_times if drag_time > start_time),
            default=float("inf"),
        )
        window_end = min(start_time + evaluation_s, next_drag)
        if horizon_s < start_time + evaluation_s and next_drag > horizon_s:
            continue
        evaluated += 1
        maximum_distance = 0.0
        for later in samples[index + 1 :]:
            if float(later["sim_time_s"]) >= window_end:
                break
            later_xy = later.get("vehicle_xy")
            if isinstance(later_xy, list) and len(later_xy) == 2:
                maximum_distance = max(
                    maximum_distance, math.dist(start_xy, later_xy)
                )
        successes += maximum_distance >= success_distance_m
    return attempts, evaluated, successes


def route_progress(trial: Trial, horizon_s: float, routes: dict[str, Route]) -> float:
    route = routes[trial.direction]
    progress_m = 0.0
    history = trial.path / "intervention_history.json"
    drags = (
        sorted(
            read_json(history).get("interventions", []),
            key=lambda item: float(item["sim_time_s"]),
        )
        if history.is_file()
        else []
    )
    drag_index = 0
    trace = trial.path / "assistance_trace.jsonl"
    for line in trace.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        item = json.loads(line)
        sample_time = float(item["sim_time_s"])
        if sample_time > horizon_s:
            break
        while (
            drag_index < len(drags)
            and float(drags[drag_index]["sim_time_s"]) <= sample_time
        ):
            progress_m = max(
                progress_m,
                float(drags[drag_index].get("route_progress_m", 0.0)),
            )
            drag_index += 1
        xy = item.get("vehicle_xy")
        if not isinstance(xy, list) or len(xy) != 2:
            continue
        _, measured, _ = route.nearest(
            np.asarray([float(xy[0])]),
            np.asarray([float(xy[1])]),
            max(0.0, progress_m - 0.5),
            min(float(route.distance[-1]), progress_m + 8.0),
        )
        progress_m = max(progress_m, float(measured[0]))
    while (
        drag_index < len(drags)
        and float(drags[drag_index]["sim_time_s"]) <= horizon_s
    ):
        progress_m = max(
            progress_m,
            float(drags[drag_index].get("route_progress_m", 0.0)),
        )
        drag_index += 1
    return progress_m


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("assisted", type=Path)
    parser.add_argument("--min-progress-m", type=float, default=20.0)
    parser.add_argument("--product-reach-m", type=float, default=20.5)
    parser.add_argument("--stuck-s", type=float, default=30.0)
    parser.add_argument("--recovery-evaluation-s", type=float, default=5.0)
    parser.add_argument("--recovery-success-m", type=float, default=0.5)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()

    control = discover(args.control, args.min_progress_m)
    assisted = discover(args.assisted, args.min_progress_m)
    keys = sorted(control.keys() & assisted.keys())
    if not keys:
        raise SystemExit("no seed/direction pairs with usable traces")

    waypoints = load_route(PROJECT_ROOT / "paths" / "from_truck" / "Route-11.json")
    routes = {
        "forward": Route.from_waypoints(waypoints),
        "reverse": Route.from_waypoints(list(reversed(waypoints))),
    }

    control_drags = assisted_drags = startup_drags = request_count = 0
    recovery_count = recovery_evaluated = recovery_succeeded = 0
    request_kind_counts: Counter[str] = Counter()
    drag_leads: list[float] = []
    horizon_total_s = 0.0
    control_progress_m = assisted_progress_m = 0.0
    if args.details:
        print(
            "seed direction horizon_s control assisted requests recoveries "
            "startup covered progress_control/assisted"
        )
    for key in keys:
        left, right = control[key], assisted[key]
        horizon_s = min(left.final_time_s, right.final_time_s)
        left_drags, _ = interventions(left, horizon_s, args.min_progress_m)
        right_drags, right_startup = interventions(
            right, horizon_s, args.min_progress_m
        )
        right_requests = requests(right, horizon_s)
        right_recoveries, right_recovery_evaluated, right_recovery_succeeded = (
            recovery_outcomes(
                right,
                horizon_s,
                args.recovery_evaluation_s,
                args.recovery_success_m,
            )
        )
        right_leads = covered_drag_leads(
            right,
            horizon_s,
            args.min_progress_m,
            args.product_reach_m,
            args.stuck_s,
        )
        control_drags += left_drags
        assisted_drags += right_drags
        startup_drags += right_startup
        request_count += right_requests
        request_kind_counts.update(request_kinds(right, horizon_s))
        recovery_count += right_recoveries
        recovery_evaluated += right_recovery_evaluated
        recovery_succeeded += right_recovery_succeeded
        drag_leads.extend(right_leads)
        horizon_total_s += horizon_s
        left_progress = route_progress(left, horizon_s, routes)
        right_progress = route_progress(right, horizon_s, routes)
        control_progress_m += left_progress
        assisted_progress_m += right_progress
        if args.details:
            print(
                left.seed,
                left.direction,
                f"{horizon_s:.1f}",
                left_drags,
                right_drags,
                right_requests,
                right_recoveries,
                right_startup,
                len(right_leads),
                f"progress={left_progress:.1f}/{right_progress:.1f}",
            )

    reduction = (
        100.0 * (control_drags - assisted_drags) / control_drags
        if control_drags
        else float("nan")
    )
    coverage = (
        100.0 * len(drag_leads) / assisted_drags
        if assisted_drags
        else float("nan")
    )
    median_lead = statistics.median(drag_leads) if drag_leads else float("nan")
    recovery_success = (
        100.0 * recovery_succeeded / recovery_evaluated
        if recovery_evaluated
        else float("nan")
    )
    coverage_text = (
        f"{len(drag_leads)}/{assisted_drags} ({coverage:.1f}%)"
        if assisted_drags
        else "-"
    )
    lead_text = f"{median_lead:.1f}s" if drag_leads else "-"
    recovery_text = (
        f"{recovery_succeeded}/{recovery_evaluated} ({recovery_success:.1f}%)"
        if recovery_evaluated
        else "-"
    )
    request_kind_text = ",".join(
        f"{kind}:{count}" for kind, count in sorted(request_kind_counts.items())
    ) or "-"
    print(
        f"pairs={len(keys)} matched_time={horizon_total_s / 3600.0:.2f} run-hours "
        f"control_drags={control_drags} assisted_drags={assisted_drags} "
        f"reduction={reduction:.1f}% requests={request_count} "
        f"request_kinds={request_kind_text} "
        f"recoveries={recovery_count} excluded_startup_drags={startup_drags} "
        f"predrag_map_coverage={coverage_text} median_map_lead={lead_text} "
        f"progress_control/assisted={control_progress_m:.0f}/{assisted_progress_m:.0f}m "
        f"recovery_success={recovery_text}"
    )
    endpoint_pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    for key in keys:
        left_row = endpoint_row(control[key])
        right_row = endpoint_row(assisted[key])
        if left_row is not None and right_row is not None:
            endpoint_pairs.append((left_row, right_row))
    if endpoint_pairs:
        left_successes = sum(int(left["success"]) for left, _ in endpoint_pairs)
        right_successes = sum(int(right["success"]) for _, right in endpoint_pairs)
        left_time = sum(float(left["sim_time_s"]) for left, _ in endpoint_pairs)
        right_time = sum(float(right["sim_time_s"]) for _, right in endpoint_pairs)
        left_path = sum(float(left["path_length_m"]) for left, _ in endpoint_pairs)
        right_path = sum(float(right["path_length_m"]) for _, right in endpoint_pairs)
        left_drags = sum(int(left["interventions"]) for left, _ in endpoint_pairs)
        right_drags = sum(int(right["interventions"]) for _, right in endpoint_pairs)
        print(
            f"completed_pairs={len(endpoint_pairs)} success_control/assisted="
            f"{left_successes}/{right_successes} "
            f"time_control/assisted={left_time:.0f}/{right_time:.0f}s "
            f"path_control/assisted={left_path:.0f}/{right_path:.0f}m "
            f"endpoint_drags_control/assisted={left_drags}/{right_drags}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
