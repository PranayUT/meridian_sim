#!/usr/bin/env python3
"""Run route campaigns with ground-only or simulated-UAV assistance.

Each trial drives one route in one direction with one seed. When the rover
stops making progress the harness drags it along the route and counts that as
an intervention, so a single bad spot cannot silently end a run. Intervening
from out here rather than inside gazebo_node keeps the planner itself the same
code that runs against Meridian Drive.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node

from autonomy.meridian_drive.core import Route
from autonomy.meridian_drive.maps import TerrainMap
from autonomy.meridian_drive.routes import load_route

WORLD = "hill_country"
MODEL = "hill_rover"
DEFAULT_ROUTES = ("Route-11", "Route-12", "Route-13")
ROUTE_DIR = ROOT / "paths" / "from_truck"
RUNTIME = ROOT / "runtime"

# PX4 MPC_XY_CRUISE: default horizontal velocity in autonomous modes,
# including missions when a waypoint does not specify another speed.
PX4_MISSION_CRUISE_SPEED_MPS = 5.0


@dataclass
class Trial:
    cycle: int
    seed: int
    route: str
    direction: str
    route_file: Path
    route_length_m: float
    veg_seed: str


def route_coverage_distance_m(route: Route, swath_m: float) -> float:
    """Approximate an exhaustive route-area survey with a lawnmower path.

    The rectangle is the same abstraction used to build the route-wide UAV
    evidence: the route bounds padded by half a local-map width. Straight
    survey legs run along the longer axis and adjacent legs are joined by a
    cross-track transition. Takeoff, landing, and travel from an unspecified
    depot are deliberately excluded.
    """
    if swath_m <= 0.0:
        raise ValueError("survey swath must be positive")
    padding = swath_m / 2.0
    width = float(np.ptp(route.xy[:, 0])) + 2.0 * padding
    height = float(np.ptp(route.xy[:, 1])) + 2.0 * padding
    along_track = max(width, height)
    across_track = min(width, height)
    passes = max(1, math.ceil(across_track / swath_m))
    transitions = min(across_track, (passes - 1) * swath_m)
    return passes * along_track + transitions


def estimate_uav_usage(
    mode: str,
    route: Route,
    ugv_navigation_time_s: float,
    request_count: int,
    map_size_m: float,
) -> dict[str, float | int]:
    """Return the deliberately simple analytical UAV accounting model.

    A reactive assist is one map-width observation transect at PX4 cruise
    speed. Explore-then-drive covers the padded route rectangle before the UGV
    begins. Greedy and always-on fly concurrently for the UGV run duration.
    """
    if ugv_navigation_time_s < 0.0 or request_count < 0 or map_size_m <= 0.0:
        raise ValueError("UAV usage inputs must be non-negative and map size positive")
    unit_assist_time_s = map_size_m / PX4_MISSION_CRUISE_SPEED_MPS
    survey_distance_m = 0.0
    sorties = 0
    if mode == "ground_only":
        flight_time_s = 0.0
        total_navigation_time_s = ugv_navigation_time_s
    elif mode == "counterfactual_uav":
        sorties = request_count
        flight_time_s = request_count * unit_assist_time_s
        # The real request policy holds the UGV, while its ground-truth map
        # producer currently answers instantly. Charge the omitted service
        # time sequentially instead of slowing Gazebo down.
        total_navigation_time_s = ugv_navigation_time_s + flight_time_s
    elif mode == "explore_then_drive":
        sorties = 1
        survey_distance_m = route_coverage_distance_m(route, map_size_m)
        flight_time_s = survey_distance_m / PX4_MISSION_CRUISE_SPEED_MPS
        total_navigation_time_s = flight_time_s + ugv_navigation_time_s
    elif mode in ("greedy_uav", "always_on_uav"):
        sorties = 1
        flight_time_s = ugv_navigation_time_s
        total_navigation_time_s = ugv_navigation_time_s
    else:
        raise ValueError(f"unknown assistance mode: {mode}")
    return {
        "uav_flight_time_s": round(flight_time_s, 2),
        "total_navigation_time_s": round(total_navigation_time_s, 2),
        "uav_sorties": sorties,
        "uav_survey_distance_m": round(survey_distance_m, 2),
        "uav_assist_unit_time_s": round(unit_assist_time_s, 2),
    }


class RoverWatcher:
    """Track sim time, position, and distance driven from the world pose feed."""

    def __init__(self, node: Node, topic: str) -> None:
        self.lock = threading.Lock()
        self.sim_s = 0.0
        self.xy: tuple[float, float] | None = None
        self.z = 0.0
        self.yaw = 0.0
        self.path_m = 0.0
        self._previous: tuple[float, float] | None = None
        if not node.subscribe(Pose_V, topic, self._on_pose):
            raise RuntimeError(f"cannot subscribe to {topic}")

    def _on_pose(self, message: Pose_V) -> None:
        entry = next((p for p in message.pose if p.name == MODEL), None)
        if entry is None:
            return
        stamp = message.header.stamp
        x, y = float(entry.position.x), float(entry.position.y)
        orientation = entry.orientation
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
        )
        with self.lock:
            self.sim_s = stamp.sec + stamp.nsec * 1e-9
            if self._previous is not None:
                step = math.dist((x, y), self._previous)
                # Teleports are applied by this harness, so ignore the jump.
                if step < 1.0:
                    self.path_m += step
            self._previous = (x, y)
            self.xy = (x, y)
            self.z = float(entry.position.z)
            self.yaw = yaw

    def reset(self) -> None:
        with self.lock:
            self.path_m = 0.0
            self._previous = None

    def snapshot(self) -> tuple[float, tuple[float, float] | None, float, float, float]:
        with self.lock:
            return self.sim_s, self.xy, self.z, self.yaw, self.path_m


def gz_service(service: str, request: str, reqtype: str, timeout_ms: int = 10000,
               attempts: int = 4) -> bool:
    """Call a Gazebo service, retrying while the bus is busy.

    When a campaign starts, every trial boots a simulator and reaches for this
    bus at the same moment, and discovery alone can outlast a short timeout. A
    reply that is merely late is not a failed trial, so back off and ask again.
    """
    for attempt in range(attempts):
        result = subprocess.run(
            ["gz", "service", "-s", service, "--reqtype", reqtype,
             "--reptype", "gz.msgs.Boolean", "--timeout", str(timeout_ms),
             "--req", request],
            capture_output=True, text=True,
        )
        if "data: true" in result.stdout:
            return True
        if attempt + 1 < attempts:
            time.sleep(2.0 * (attempt + 1))
    return False


def reset_world() -> bool:
    return gz_service(f"/world/{WORLD}/control", "reset: {all: true}", "gz.msgs.WorldControl")


def pause_world() -> bool:
    return gz_service(
        f"/world/{WORLD}/control",
        "pause: true",
        "gz.msgs.WorldControl",
    )


def place(x: float, y: float, z: float, yaw: float) -> bool:
    request = (
        f'name: "{MODEL}", position: {{x: {x}, y: {y}, z: {z}}}, '
        f"orientation: {{x: 0, y: 0, z: {math.sin(yaw / 2.0)}, w: {math.cos(yaw / 2.0)}}}"
    )
    return gz_service(f"/world/{WORLD}/set_pose", request, "gz.msgs.Pose")


def terrain_height(terrain: TerrainMap | None, x: float, y: float) -> float | None:
    if terrain is None:
        return None
    col = int((x - terrain.origin_x) / terrain.resolution)
    row = int((y - terrain.origin_y) / terrain.resolution)
    rows, cols = terrain.elevation_grid.shape
    if not (0 <= row < rows and 0 <= col < cols):
        return None
    return float(terrain.elevation_grid[row, col])


def build_route_files(names: list[str]) -> list[tuple[str, str, Path]]:
    """Return (route, direction, file) including a reversed copy of each route."""
    output = RUNTIME / "experiment_routes"
    output.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[str, str, Path]] = []
    for name in names:
        source = ROUTE_DIR / f"{name}.json"
        if not source.is_file():
            raise SystemExit(f"missing {source}; run tools/convert_truck_routes.py first")
        payload = json.loads(source.read_text(encoding="utf-8"))
        entries.append((name, "forward", source))
        reversed_payload = dict(payload)
        reversed_payload["name"] = f"{payload.get('name', name)}-reverse"
        reversed_payload["wps"] = list(reversed(payload["wps"]))
        target = output / f"{name}-reverse.json"
        target.write_text(json.dumps(reversed_payload, indent=2) + "\n", encoding="utf-8")
        entries.append((name, "reverse", target))
    return entries


DEFAULT_OBSTACLE_MESH = ROOT / "models" / "painted_vegetation" / "meshes" / "collision.obj"
OBSTACLE_CELL_M = 0.25


def obstacle_cells(mesh: Path = DEFAULT_OBSTACLE_MESH, cache: Path | None = None) -> set[tuple[int, int]]:
    """Grid cells covered by the baked vegetation collision mesh.

    The mesh has to be the one this run's simulator loaded. A randomised
    campaign gives each round its own variant, and grading drag targets against
    a different round's bushes would aim the rover straight into them.
    """
    cache = cache or RUNTIME / "obstacle_cells.npy"
    if cache.is_file() and mesh.is_file() and cache.stat().st_mtime >= mesh.stat().st_mtime:
        return {tuple(cell) for cell in np.load(cache)}
    if not mesh.is_file():
        return set()
    points = []
    with mesh.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                points.append((float(parts[1]), float(parts[2])))
    grid = np.unique(np.floor(np.asarray(points) / OBSTACLE_CELL_M).astype(np.int32), axis=0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, grid)
    return {tuple(cell) for cell in grid}


def is_clear(cells: set[tuple[int, int]], terrain: TerrainMap | None,
             x: float, y: float, radius_m: float = 0.6) -> bool:
    """True when a body-sized patch has no known obstacle and drivable grade."""
    if terrain is not None:
        col = int((x - terrain.origin_x) / terrain.resolution)
        row = int((y - terrain.origin_y) / terrain.resolution)
        rows, cols = terrain.slope_grid.shape
        if 0 <= row < rows and 0 <= col < cols:
            if float(terrain.slope_grid[row, col]) >= terrain.hard_slope:
                return False
    if not cells:
        return True
    span = int(math.ceil(radius_m / OBSTACLE_CELL_M))
    base_x = int(math.floor(x / OBSTACLE_CELL_M))
    base_y = int(math.floor(y / OBSTACLE_CELL_M))
    for dx in range(-span, span + 1):
        for dy in range(-span, span + 1):
            if (base_x + dx, base_y + dy) in cells:
                return False
    return True


def drag_target(route: Route, xy: tuple[float, float], drag_m: float,
                cells: set[tuple[int, int]] | None = None,
                terrain: TerrainMap | None = None,
                search_m: float = 20.0,
                lateral_search_m: float = 4.0,
                progress_m: float | None = None) -> tuple[float, float, float, float]:
    """A pose at least drag_m further along the route, facing the route tangent.

    Two reasons not to drag straight at the next waypoint: that bearing usually
    points through whatever is wedging the rover, and these routes thread dense
    vegetation, so a fixed hop lands in the next bush about half the time.
    Following the route arc and skipping past occupied ground fixes both.
    """
    if progress_m is None:
        distances = np.hypot(route.xy[:, 0] - xy[0], route.xy[:, 1] - xy[1])
        progress = float(route.distance[int(np.argmin(distances))])
    else:
        progress = progress_m

    def pose_at(offset: float) -> tuple[float, float, float, float]:
        index = int(np.searchsorted(route.distance, progress + offset))
        index = min(index, len(route.distance) - 1)
        return (
            float(route.xy[index, 0]),
            float(route.xy[index, 1]),
            float(route.yaw[index]),
            float(route.distance[index]),
        )

    fallback = pose_at(drag_m)
    if cells is None:
        return fallback
    lateral_offsets = [0.0]
    lateral = OBSTACLE_CELL_M * 2.0
    while lateral <= lateral_search_m:
        lateral_offsets.extend((lateral, -lateral))
        lateral += OBSTACLE_CELL_M * 2.0
    offset = drag_m
    while offset <= drag_m + search_m:
        x, y, yaw, target_progress = pose_at(offset)
        normal_x, normal_y = -math.sin(yaw), math.cos(yaw)
        for lateral_offset in lateral_offsets:
            candidate_x = x + lateral_offset * normal_x
            candidate_y = y + lateral_offset * normal_y
            if is_clear(cells, terrain, candidate_x, candidate_y):
                return candidate_x, candidate_y, yaw, target_progress
        offset += OBSTACLE_CELL_M * 2.0
    raise RuntimeError(
        f"no clear drag target within {search_m:g} m along and "
        f"{lateral_search_m:g} m beside the route"
    )


def run_trial(
    trial: Trial, watcher: RoverWatcher, terrain: TerrainMap | None,
    cells: set[tuple[int, int]], args: argparse.Namespace
) -> dict:
    waypoints = load_route(trial.route_file)
    anchors = np.asarray(waypoints, dtype=np.float64)
    route = Route.from_waypoints(waypoints)
    start = anchors[0]
    heading = math.atan2(anchors[1][1] - start[1], anchors[1][0] - start[0])

    if not reset_world():
        # Worth saying out loud: a reset that never landed leaves the previous
        # trial's world in place, which quietly changes what this trial measures.
        print("    warning: world reset did not confirm; continuing", flush=True)
    time.sleep(1.0)
    if args.lockstep and not pause_world():
        # A late request commonly lands even when discovery loses its reply.
        # The autonomy node verifies and repeats the pause before it steps.
        print("    warning: initial world pause did not confirm; continuing", flush=True)
    height = terrain_height(terrain, start[0], start[1])
    start_z = (height + args.ride_height) if height is not None else args.fallback_z
    if not place(start[0], start[1], start_z, heading):
        raise SystemExit(
            "set_pose failed after retries; the simulator is up but its service "
            "bus never answered. Lower --jobs so fewer simulators start at once."
        )
    time.sleep(1.5)
    watcher.reset()

    command = [
        sys.executable, "-m", "autonomy.meridian_drive.gazebo_node",
        "--route-file", str(trial.route_file),
        "--planner", args.planner,
        "--assistance", args.assistance,
        "--uav-source", args.uav_source,
        "--uav-uncertainty-threshold", str(args.uav_uncertainty_threshold),
        "--uav-path-uncertainty-threshold", str(args.uav_path_uncertainty_threshold),
        "--mapping-uncertainty-maturity", str(args.mapping_uncertainty_maturity),
        "--assistance-period", str(args.assistance_period),
        "--uav-map-size", str(args.uav_map_size),
        "--uav-map", str(args.run_dir / "uav_map.npz"),
        "--seed", str(trial.seed),
        "--arrival-radius", str(args.arrival_radius),
        "--arrival-progress-fraction", str(args.min_route_progress),
        "--ground-map", str(args.run_dir / "ground_maps.npz"),
        "--assistance-trace", str(args.run_dir / "assistance_trace.jsonl"),
        "--assistance-probe-m", str(args.assistance_probe_m),
        "--uav-probe-uncertainty-threshold",
        str(args.uav_probe_uncertainty_threshold),
        "--uav-probe-hit-window", str(args.uav_probe_hit_window),
        "--uav-probe-hits", str(args.uav_probe_hits),
        "--uav-probe-min-world-speed", str(args.uav_probe_min_world_speed),
        "--uav-grass-occupancy-probability",
        str(args.uav_grass_occupancy_probability),
        "--uav-recovery-duration", str(args.uav_recovery_duration),
        "--uav-recovery-speed", str(args.uav_recovery_speed),
        "--uav-recovery-steering", str(args.uav_recovery_steering),
        "--uav-recovery-cooldown", str(args.uav_recovery_cooldown),
        "--uav-recovery-max-attempts", str(args.uav_recovery_max_attempts),
        "--uav-recovery-rearm-progress", str(args.uav_recovery_rearm_progress),
        "--status", str(args.run_dir / "autonomy_status.json"),
        "--uav-request", str(args.run_dir / "uav_request.json"),
        "--no-visualization",
    ]
    if args.lockstep:
        command.extend(("--lockstep", "--physics-step", str(args.physics_step)))
    if trial.veg_seed:
        command.extend(("--uav-vegetation-seed", str(int(trial.veg_seed))))
    log_path = args.run_dir / f"node_c{trial.cycle}_{trial.route}_{trial.direction}.log"
    goal = anchors[-1]
    wall_start = time.monotonic()
    interventions = 0
    intervention_history: list[dict[str, object]] = []
    intervention_history_path = args.run_dir / "intervention_history.json"
    resolved = 0
    pending_drag = False
    outcome = "timeout"
    route_progress_m = 0.0
    with log_path.open("w", encoding="utf-8") as log:
        node = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            sim_start, _, _, _, _ = watcher.snapshot()
            deadline_s = sim_start + trial.route_length_m / args.min_speed + args.grace_s
            progress_required_m = args.min_route_progress * float(route.distance[-1])
            anchor_s, anchor_xy = sim_start, None
            while True:
                time.sleep(0.25)
                if node.poll() is not None:
                    outcome = "node exited"
                    break
                sim_s, xy, z, yaw, path_m = watcher.snapshot()
                if xy is None:
                    continue
                _, measured_progress, _ = route.nearest(
                    np.asarray([xy[0]]),
                    np.asarray([xy[1]]),
                    max(0.0, route_progress_m - 0.5),
                    min(route.distance[-1], route_progress_m + 8.0),
                )
                route_progress_m = max(
                    route_progress_m, float(measured_progress[0])
                )
                # Routes are loops: they finish within ~12 m of where they
                # start, so proximity to the goal alone is not evidence the
                # route was driven. Require the trial to have covered the route
                # before arrival counts, or a rover that merely passes near the
                # end point early is scored a success.
                if (math.dist(xy, (goal[0], goal[1])) <= args.arrival_radius
                        and route_progress_m >= progress_required_m):
                    outcome = "success"
                    break
                if sim_s > deadline_s:
                    outcome = "timeout"
                    break
                if anchor_xy is None or math.dist(xy, anchor_xy) > args.stuck_epsilon:
                    if pending_drag:
                        # It drove clear of where the drag put it, so that
                        # intervention did its job.
                        resolved += 1
                        pending_drag = False
                    anchor_s, anchor_xy = sim_s, xy
                elif sim_s - anchor_s >= args.stuck_s:
                    pending_drag = False
                    if interventions >= args.max_interventions:
                        outcome = "stuck"
                        break
                    drag_x, drag_y, bearing, drag_progress = drag_target(
                        route, xy, args.drag_m, cells, terrain,
                        progress_m=route_progress_m,
                    )
                    ground = terrain_height(terrain, drag_x, drag_y)
                    drag_z = (ground + args.ride_height) if ground is not None else z
                    place(drag_x, drag_y, drag_z, bearing)
                    interventions += 1
                    route_progress_m = max(route_progress_m, drag_progress)
                    pending_drag = True
                    intervention_history.append(
                        {
                            "intervention_number": interventions,
                            "trial_time_s": sim_s - sim_start,
                            "sim_time_s": sim_s,
                            "stuck_xy": [xy[0], xy[1]],
                            "target_xy": [drag_x, drag_y],
                            "route_progress_m": route_progress_m,
                            "path_length_m": path_m,
                        }
                    )
                    temporary_history = intervention_history_path.with_name(
                        f".{intervention_history_path.name}.{os.getpid()}.tmp"
                    )
                    temporary_history.write_text(
                        json.dumps(
                            {"version": 1, "interventions": intervention_history},
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    temporary_history.replace(intervention_history_path)
                    print(
                        f"    drag {interventions}: to ({drag_x:.1f}, {drag_y:.1f}) "
                        f"at sim {sim_s:.0f}s",
                        flush=True,
                    )
                    time.sleep(1.0)
                    # Re-anchor on where the drag actually left the rover, so
                    # the intervention only counts as resolved once it drives
                    # clear of that spot under its own power.
                    settled_s, settled_xy = watcher.snapshot()[:2]
                    anchor_s = settled_s
                    anchor_xy = settled_xy if settled_xy is not None else (drag_x, drag_y)
        finally:
            node.send_signal(signal.SIGTERM)
            try:
                node.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                node.kill()
                node.wait(timeout=5.0)

    sim_s, _, _, _, path_m = watcher.snapshot()
    uav_requests = 0
    status_path = args.run_dir / "autonomy_status.json"
    try:
        uav_requests = int(json.loads(status_path.read_text(encoding="utf-8"))["request_count"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass
    ugv_navigation_time_s = sim_s - sim_start
    uav_usage = estimate_uav_usage(
        args.assistance,
        route,
        ugv_navigation_time_s,
        uav_requests,
        args.uav_map_size,
    )
    return {
        "cycle": trial.cycle,
        "seed": trial.seed,
        "route": trial.route,
        "direction": trial.direction,
        "outcome": outcome,
        "success": int(outcome == "success"),
        "sim_time_s": round(ugv_navigation_time_s, 2),
        "wall_time_s": round(time.monotonic() - wall_start, 2),
        "path_length_m": round(path_m, 2),
        "route_length_m": round(trial.route_length_m, 2),
        "interventions": interventions,
        "interventions_resolved": resolved,
        "route_progress_m": round(route_progress_m, 2),
        "uav_requests": uav_requests,
        **uav_usage,
        "uav_speed_mps": PX4_MISSION_CRUISE_SPEED_MPS,
        "uav_map_size_m": args.uav_map_size,
        "planner": args.planner,
        "assistance": args.assistance,
        "uav_uncertainty_threshold": args.uav_uncertainty_threshold,
        "uav_path_uncertainty_threshold": args.uav_path_uncertainty_threshold,
        "uav_probe_uncertainty_threshold": args.uav_probe_uncertainty_threshold,
        "uav_probe_hit_window": args.uav_probe_hit_window,
        "uav_probe_hits": args.uav_probe_hits,
        "uav_probe_min_world_speed": args.uav_probe_min_world_speed,
        "uav_grass_occupancy_probability": args.uav_grass_occupancy_probability,
        "uav_recovery_duration": args.uav_recovery_duration,
        "uav_recovery_speed": args.uav_recovery_speed,
        "uav_recovery_steering": args.uav_recovery_steering,
        "uav_recovery_cooldown": args.uav_recovery_cooldown,
        "uav_recovery_max_attempts": args.uav_recovery_max_attempts,
        "uav_recovery_rearm_progress": args.uav_recovery_rearm_progress,
        "mapping_uncertainty_maturity": args.mapping_uncertainty_maturity,
        "veg_seed": trial.veg_seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=3, help="0 runs until interrupted")
    parser.add_argument("--routes", nargs="+", default=list(DEFAULT_ROUTES))
    parser.add_argument("--directions", nargs="+", choices=["forward", "reverse"],
                        default=["forward", "reverse"],
                        help="run a subset so long routes can be split across sittings")
    parser.add_argument("--seed", type=int, default=7, help="seed of the first cycle")
    parser.add_argument(
        "--planner",
        choices=("meridian_mppi", "gp_navigation"),
        default="meridian_mppi",
        help="local navigation planner under test",
    )
    parser.add_argument(
        "--assistance",
        choices=(
            "ground_only",
            "greedy_uav",
            "counterfactual_uav",
            "explore_then_drive",
            "always_on_uav",
        ),
        default="ground_only",
    )
    parser.add_argument("--uav-source", choices=("ground_truth", "file"), default="ground_truth")
    parser.add_argument("--uav-uncertainty-threshold", type=float, default=0.75)
    parser.add_argument("--uav-path-uncertainty-threshold", type=float, default=0.20)
    parser.add_argument("--assistance-probe-m", type=float, default=8.0,
                        help="forward distance measured by the trace corridor probe")
    parser.add_argument("--uav-probe-uncertainty-threshold", type=float, default=0.04895)
    parser.add_argument("--uav-probe-hit-window", type=float, default=2.0)
    parser.add_argument("--uav-probe-hits", type=int, default=3)
    parser.add_argument("--uav-probe-min-world-speed", type=float, default=0.02)
    parser.add_argument("--uav-grass-occupancy-probability", type=float, default=0.04)
    parser.add_argument("--uav-recovery-duration", type=float, default=2.0)
    parser.add_argument("--uav-recovery-speed", type=float, default=0.6)
    parser.add_argument("--uav-recovery-steering", type=float, default=0.65)
    parser.add_argument("--uav-recovery-cooldown", type=float, default=3.0)
    parser.add_argument("--uav-recovery-max-attempts", type=int, default=1)
    parser.add_argument("--uav-recovery-rearm-progress", type=float, default=2.0)
    parser.add_argument("--mapping-uncertainty-maturity", type=float, default=1.0)
    parser.add_argument("--assistance-period", type=float, default=0.0,
                        help="seconds between assistance evaluations; 0 evaluates every tick")
    parser.add_argument("--uav-map-size", type=float, default=25.0)
    parser.add_argument("--stuck-s", type=float, default=30.0, help="sim seconds without progress")
    parser.add_argument("--stuck-epsilon", type=float, default=0.5, help="m that counts as progress")
    parser.add_argument("--drag-m", type=float, default=3.0)
    parser.add_argument("--max-interventions", type=int, default=20)
    parser.add_argument("--min-route-progress", type=float, default=0.9,
                        help="fraction of the route that must be covered before "
                             "reaching the goal counts as success")
    parser.add_argument("--min-speed", type=float, default=0.5, help="sets the per-trial deadline")
    parser.add_argument(
        "--lockstep", action="store_true",
        help="step the world from the planner so its rate in simulator time "
             "is the same whether one trial or twenty share the machine",
    )
    parser.add_argument("--physics-step", type=float, default=0.001)
    parser.add_argument("--grace-s", type=float, default=120.0)
    # 0.25 m was tighter than the rover's own footprint: a trial could drive its
    # route correctly and still never be credited, circling the goal until the
    # clock ran out. Verified on Route-12 forward seed 7, which timed out twice
    # at 0.25 m and finished in 62% of its budget at 1.0 m.
    parser.add_argument("--arrival-radius", type=float, default=1.0)
    parser.add_argument("--ride-height", type=float, default=0.25)
    parser.add_argument("--fallback-z", type=float, default=22.0)
    parser.add_argument("--veg-root", type=Path, default=None,
                        help="resource root holding this run's painted_vegetation variant; "
                             "must be the same one on GZ_SIM_RESOURCE_PATH")
    parser.add_argument("--veg-seed", default="",
                        help="recorded per trial so a CSV says which variant it drove")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--run-id", default=None,
                        help="names runtime/experiments/<id>/; defaults to a timestamp")
    args = parser.parse_args()
    if not 0.0 <= args.min_route_progress <= 1.0:
        parser.error("min-route-progress must be between 0 and 1")
    if not 0.0 <= args.uav_uncertainty_threshold <= 1.0:
        parser.error("uav-uncertainty-threshold must be between 0 and 1")
    if not 0.0 <= args.uav_path_uncertainty_threshold <= 1.0:
        parser.error("uav-path-uncertainty-threshold must be between 0 and 1")
    if not 0.0 <= args.uav_probe_uncertainty_threshold <= 1.0:
        parser.error("uav-probe-uncertainty-threshold must be between 0 and 1")
    if args.uav_probe_hit_window < 0.5:
        parser.error("uav-probe-hit-window must be at least 0.5 s")
    if args.uav_probe_hits < 1:
        parser.error("uav-probe-hits must be positive")
    if args.uav_probe_min_world_speed < 0.0:
        parser.error("uav-probe-min-world-speed must be non-negative")
    if not 0.0 <= args.uav_grass_occupancy_probability <= 1.0:
        parser.error("uav-grass-occupancy-probability must be between 0 and 1")
    if min(args.uav_recovery_duration, args.uav_recovery_speed, args.uav_recovery_cooldown) < 0.0:
        parser.error("uav recovery timing and speed must be non-negative")
    if not 0.0 <= args.uav_recovery_steering <= 1.0:
        parser.error("uav-recovery-steering must be between 0 and 1")
    if args.uav_recovery_max_attempts < 1:
        parser.error("uav-recovery-max-attempts must be positive")
    if args.uav_recovery_rearm_progress < 0.0:
        parser.error("uav-recovery-rearm-progress must be non-negative")
    if args.mapping_uncertainty_maturity < 0.0:
        parser.error("mapping-uncertainty-maturity must be non-negative")

    # Everything a run writes lives under one directory so several campaigns
    # can share a machine. Pair that with a distinct GZ_PARTITION per run.
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    args.run_dir = RUNTIME / "experiments" / run_id
    args.run_dir.mkdir(parents=True, exist_ok=True)

    entries = [e for e in build_route_files(args.routes) if e[1] in args.directions]
    if not entries:
        raise SystemExit("no route/direction combinations selected")
    terrain_path = ROOT / "models" / "hill_terrain" / "meshes" / "terrain.tif"
    terrain = TerrainMap.from_tif(terrain_path) if terrain_path.is_file() else None
    if args.veg_root is not None:
        mesh = args.veg_root / "painted_vegetation" / "meshes" / "collision.obj"
        if not mesh.is_file():
            raise SystemExit(f"no vegetation variant at {mesh}; build it with tools/make_vegetation.py")
        cells = obstacle_cells(mesh, args.veg_root / "obstacle_cells.npy")
        print(f"vegetation variant {args.veg_root}", flush=True)
    else:
        cells = obstacle_cells()
    print(f"{len(cells)} obstacle cells loaded for drag placement", flush=True)

    node = Node()
    watcher = RoverWatcher(node, f"/world/{WORLD}/dynamic_pose/info")
    print(f"Waiting for {WORLD} pose data...", flush=True)
    for _ in range(60):
        if watcher.snapshot()[1] is not None:
            break
        time.sleep(0.5)
    else:
        raise SystemExit(f"no pose on /world/{WORLD}/dynamic_pose/info; start the simulator first")

    results_path = args.results or (args.run_dir / "campaign.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "cycle", "seed", "route", "direction", "outcome", "success",
        "sim_time_s", "wall_time_s", "path_length_m", "route_length_m",
        "interventions", "interventions_resolved", "route_progress_m",
        "uav_requests", "uav_flight_time_s", "total_navigation_time_s",
        "uav_sorties", "uav_survey_distance_m", "uav_assist_unit_time_s",
        "uav_speed_mps", "uav_map_size_m", "planner", "assistance",
        "uav_uncertainty_threshold",
        "uav_path_uncertainty_threshold", "uav_probe_uncertainty_threshold",
        "uav_probe_hit_window", "uav_probe_hits", "uav_probe_min_world_speed",
        "uav_grass_occupancy_probability", "uav_recovery_duration",
        "uav_recovery_speed", "uav_recovery_steering",
        "uav_recovery_cooldown", "uav_recovery_max_attempts",
        "uav_recovery_rearm_progress", "mapping_uncertainty_maturity", "veg_seed",
    ]
    rows: list[dict] = []
    stop = False

    def handle(_signum, _frame):
        nonlocal stop
        stop = True
        print("\nFinishing the current trial, then stopping.", flush=True)

    signal.signal(signal.SIGINT, handle)

    with results_path.open("w", newline="", encoding="utf-8") as handle_file:
        writer = csv.DictWriter(handle_file, fieldnames=columns)
        writer.writeheader()
        cycle = 0
        while not stop and (args.cycles == 0 or cycle < args.cycles):
            seed = args.seed + cycle
            for route_name, direction, route_file in entries:
                if stop:
                    break
                waypoints = load_route(route_file)
                length = float(Route.from_waypoints(waypoints).distance[-1])
                trial = Trial(cycle, seed, route_name, direction, route_file, length, args.veg_seed)
                print(f"[cycle {cycle} seed {seed}] {route_name} {direction} ({length:.0f} m)", flush=True)
                row = run_trial(trial, watcher, terrain, cells, args)
                rows.append(row)
                writer.writerow(row)
                handle_file.flush()
                print(
                    f"    {row['outcome']}: {row['sim_time_s']:.0f}s sim, "
                    f"{row['total_navigation_time_s']:.0f}s total navigation, "
                    f"{row['uav_flight_time_s']:.0f}s UAV flight, "
                    f"{row['path_length_m']:.0f} m driven, {row['interventions']} drags, "
                    f"{row['uav_requests']} UAV requests",
                    flush=True,
                )
            cycle += 1

    if rows:
        successes = sum(r["success"] for r in rows)
        print(f"\n{len(rows)} trials, {successes} succeeded ({100.0 * successes / len(rows):.0f}%)")
        print(f"{'route':<12} {'dir':<8} {'trials':>6} {'success':>8} {'sim s':>8} {'driven m':>9} {'drags':>6}")
        for route_name, direction, _ in entries:
            subset = [r for r in rows if r["route"] == route_name and r["direction"] == direction]
            if not subset:
                continue
            ok = [r for r in subset if r["success"]]
            print(
                f"{route_name:<12} {direction:<8} {len(subset):>6} "
                f"{100.0 * len(ok) / len(subset):>7.0f}% "
                f"{(sum(r['sim_time_s'] for r in ok) / len(ok)) if ok else float('nan'):>8.0f} "
                f"{(sum(r['path_length_m'] for r in ok) / len(ok)) if ok else float('nan'):>9.0f} "
                f"{sum(r['interventions'] for r in subset) / len(subset):>6.1f}"
            )
        wall = sum(r["wall_time_s"] for r in rows)
        sim = sum(r["sim_time_s"] for r in rows)
        if wall > 0:
            print(f"achieved {sim / wall:.2f}x real time over {wall / 60.0:.0f} min of wall clock")
        drags = sum(r["interventions"] for r in rows)
        fixed = sum(r["interventions_resolved"] for r in rows)
        if drags:
            print(f"\ndrags: {drags}, resolved {fixed} ({100.0 * fixed / drags:.0f}%)")
        print(f"\nresults: {results_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    code = main()
    # gz-transport calls std::terminate if its Node is destroyed while the
    # callback thread is still delivering a message. Results are already
    # written and flushed by this point, so skip interpreter teardown rather
    # than end a finished campaign on a core dump.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
