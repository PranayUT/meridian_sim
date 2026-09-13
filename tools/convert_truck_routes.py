#!/usr/bin/env python3
"""Split the truck's multi-route export into per-route simulator route files."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autonomy.meridian_drive.routes import RouteFileError, load_route, wgs84_to_world

DEFAULT_SOURCE = ROOT / "paths" / "from_truck.txt"
# A subdirectory on purpose: find_default_route only scans files at the top of
# paths/, so dropping several routes here keeps the single-route auto-selection
# working. Pick one with --route-file.
DEFAULT_OUTPUT = ROOT / "paths" / "from_truck"
TERRAIN_HALF_M = 256.0


def read_export(source: Path) -> list[dict]:
    """Parse the truck export, which indents with U+00A0 instead of spaces."""
    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as error:
        raise RouteFileError(f"cannot read {source}: {error}") from error
    try:
        routes = json.loads(raw.replace(" ", " "))
    except json.JSONDecodeError as error:
        raise RouteFileError(f"{source} is not valid JSON: {error}") from error
    if not isinstance(routes, list):
        raise RouteFileError("expected a JSON array of routes")
    return routes


def safe_stem(route: dict, index: int) -> str:
    name = str(route.get("name") or route.get("id") or f"route_{index}")
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return cleaned or f"route_{index}"


def summarize(waypoints: list) -> tuple[int, float, float, float, float, float]:
    points = [wgs84_to_world(float(lon), float(lat)) for lat, lon in waypoints]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    length = sum(math.dist(points[i], points[i + 1]) for i in range(len(points) - 1))
    return len(points), min(xs), max(xs), min(ys), max(ys), length


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    routes = read_export(args.source)
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"{'route':<12} {'wps':>4} {'length':>9} {'x range':>17} {'y range':>19}  status")
    written = 0
    usable = 0
    for index, route in enumerate(routes):
        waypoints = route.get("wps")
        if not isinstance(waypoints, list) or len(waypoints) < 2:
            print(f"{safe_stem(route, index):<12} {'-':>4} {'-':>9} {'-':>17} {'-':>19}  skipped: no waypoint list")
            continue
        stem = safe_stem(route, index)
        target = args.output / f"{stem}.json"
        # Keep id/name/site for provenance; the loader reads "wps" and ignores
        # the rest.
        target.write_text(
            json.dumps(
                {
                    "id": route.get("id"),
                    "name": route.get("name"),
                    "frame": route.get("frame", "enu"),
                    "site": route.get("site"),
                    "wps": [[float(lat), float(lon)] for lat, lon in waypoints],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        written += 1
        count, x0, x1, y0, y1, length = summarize(waypoints)
        try:
            load_route(target)
            status = "ok"
            usable += 1
        except RouteFileError as error:
            status = f"WILL NOT LOAD: {error}"
        print(
            f"{stem:<12} {count:>4} {length:>8.1f}m "
            f"[{x0:6.1f},{x1:6.1f}] [{y0:7.1f},{y1:7.1f}]  {status}"
        )

    print(f"\nwrote {written} route file(s) to {args.output.relative_to(ROOT)}; {usable} load cleanly")
    print(f"terrain is {2 * TERRAIN_HALF_M:.0f} m square, so the loader rejects any point past "
          f"+/-{TERRAIN_HALF_M:.0f} m")
    print("drive one with: ./scripts/run_autonomy.sh --route-file "
          f"{(args.output / 'Route-11.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
