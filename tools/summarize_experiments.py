#!/usr/bin/env python3
"""Aggregate campaign CSVs from one or more experiment runs."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# A standalone run writes runtime/experiments/<run-id>/, while a campaign nests
# its trials one level deeper under runtime/experiments/<campaign>/. Pick up
# both so the default still pools everything on disk.
DEFAULT_GLOBS = ("runtime/experiments/*/campaign.csv",
                 "runtime/experiments/*/*/campaign.csv")

NUMERIC = ("sim_time_s", "wall_time_s", "path_length_m", "route_length_m",
           "interventions", "interventions_resolved", "success", "cycle", "seed")


def run_label(path: Path) -> str:
    """Name a trial by its path under runtime/experiments.

    A campaign trial reads as "<campaign>/<trial>" and a standalone run as
    just "<run-id>", so pooled output says which campaign a row came from.
    """
    experiments = ROOT / "runtime" / "experiments"
    try:
        # Resolve first: paths given on the command line are usually relative,
        # and relative_to would miss the campaign directory without this.
        return path.resolve().parent.relative_to(experiments).as_posix()
    except ValueError:
        return path.parent.name


def load(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for key in NUMERIC:
                    if row.get(key):
                        row[key] = float(row[key])
                row["run"] = run_label(path)
                rows.append(row)
    return rows


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path,
                        help="campaign CSVs; defaults to "
                             + " and ".join(DEFAULT_GLOBS))
    args = parser.parse_args()

    if args.paths:
        paths = args.paths
    else:
        found: set[Path] = set()
        for pattern in DEFAULT_GLOBS:
            found.update(ROOT.glob(pattern))
        paths = sorted(found)
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise SystemExit("no campaign CSVs found under "
                         + " or ".join(str(ROOT / g) for g in DEFAULT_GLOBS))
    rows = load(paths)
    if not rows:
        raise SystemExit("campaign CSVs contain no trials")

    runs = sorted({r["run"] for r in rows})
    successes = [r for r in rows if r["success"]]
    print(f"{len(rows)} trials from {len(runs)} run(s): {', '.join(runs)}")
    print(f"success rate {100.0 * len(successes) / len(rows):.1f}%  "
          f"({len(successes)}/{len(rows)})")

    # Both halves of the ratio must come from the same trials, or CSVs written
    # before wall_time_s existed inflate it.
    timed = [r for r in rows if r.get("wall_time_s") and r.get("sim_time_s")]
    wall = sum(r["wall_time_s"] for r in timed)
    if wall > 0:
        sim = sum(r["sim_time_s"] for r in timed)
        span = f"{wall / 3600.0:.1f} h" if wall >= 3600.0 else f"{wall / 60.0:.0f} min"
        note = "" if len(timed) == len(rows) else f", {len(timed)}/{len(rows)} trials timed"
        print(f"achieved {sim / wall:.2f}x real time over {span} of wall clock{note}")

    drags = sum(r["interventions"] for r in rows)
    fixed = sum(r["interventions_resolved"] for r in rows)
    if drags:
        print(f"interventions {drags:.0f}, resolved {fixed:.0f} "
              f"({100.0 * fixed / drags:.0f}%), "
              f"{drags / len(rows):.1f} per trial")

    print(f"\n{'route':<12} {'dir':<8} {'n':>4} {'succ':>6} {'sim s':>8} "
          f"{'driven m':>9} {'ratio':>6} {'drags':>6}")
    keys = sorted({(r["route"], r["direction"]) for r in rows})
    for route, direction in keys:
        subset = [r for r in rows if r["route"] == route and r["direction"] == direction]
        ok = [r for r in subset if r["success"]]
        ratio = mean([r["path_length_m"] / r["route_length_m"] for r in ok
                      if r.get("route_length_m")])
        print(f"{route:<12} {direction:<8} {len(subset):>4} "
              f"{100.0 * len(ok) / len(subset):>5.0f}% "
              f"{mean([r['sim_time_s'] for r in ok]):>8.0f} "
              f"{mean([r['path_length_m'] for r in ok]):>9.0f} "
              f"{ratio:>6.2f} "
              f"{mean([r['interventions'] for r in subset]):>6.1f}")

    failures = [r for r in rows if not r["success"]]
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for row in failures[:20]:
            print(f"  run {row['run']} cycle {row['cycle']:.0f} seed {row['seed']:.0f} "
                  f"{row['route']} {row['direction']}: {row['outcome']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
