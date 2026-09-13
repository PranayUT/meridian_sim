#!/usr/bin/env python3
"""Run the Meridian Drive autonomy core on the Gazebo rover."""

from __future__ import annotations

import argparse
import math
import signal
import threading
import time
from pathlib import Path

import numpy as np
from gz.msgs10.laserscan_pb2 import LaserScan
from gz.msgs10.image_pb2 import Image
from gz.msgs10.odometry_pb2 import Odometry
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.twist_pb2 import Twist
from gz.transport13 import Node

from .assistance import MODES, AssistanceManager
from .core import MPPI, MppiConfig, Route
from .ground_mapping import GroundMapper, SemanticMapper, write_snapshot
from .maps import LocalGridMap, MapStack, TerrainMap
from .routes import find_default_route, load_route
from .visualization import GazeboMarkers

DEFAULT_ROUTE = [(0.0, 0.0), (12.0, 0.0), (18.0, 12.0), (5.0, 20.0), (-8.0, 8.0)]


class GazeboAutonomy:
    """Connect simulator sensor topics to the ROS-free planner."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.node = Node()
        self.publisher = self.node.advertise(args.command_topic, Twist)
        waypoints = load_route(args.route_file) if args.route_file else (args.waypoint or DEFAULT_ROUTE)
        self.route_anchors = np.asarray(waypoints, dtype=np.float64)
        self.route = Route.from_waypoints(waypoints)
        self.route_name = args.route_file.name if args.route_file else "built-in route"
        config = MppiConfig(
            samples=args.samples,
            horizon=args.horizon,
            target_speed=args.target_speed,
            speed_max=args.speed_max,
        )
        self.planner = MPPI(self.route, config=config, seed=args.seed)
        terrain = TerrainMap.from_tif(args.terrain_dem) if args.terrain_dem else None
        self.map_stack = MapStack(terrain=terrain)
        self.ground_mapper = GroundMapper()
        self.semantic_mapper = SemanticMapper()
        self.ground_map_path = args.ground_map
        self.last_map_write_s = 0.0
        self.latest_ground_obstacles: LocalGridMap | None = None
        self.latest_ground_obstacle_probability: LocalGridMap | None = None
        self.latest_ground_semantics: LocalGridMap | None = None
        self.latest_ground_semantic_obstacles: LocalGridMap | None = None
        self.markers = None if args.no_visualization else GazeboMarkers(
            self.node, terrain, args.marker_service
        )
        self.assistance = AssistanceManager(
            mode=args.assistance,
            map_path=args.uav_map,
            request_path=args.uav_request,
            status_path=args.status,
            map_stack=self.map_stack,
        )
        self.arrival_radius = args.arrival_radius
        self.lock = threading.Lock()
        self.mapping_lock = threading.Lock()
        self.mapping_event = threading.Event()
        self.pending_lidar: tuple[np.ndarray, tuple[float, float, float], float, float] | None = None
        self.pending_labels: tuple[np.ndarray, float] | None = None
        self.pending_depth: tuple[np.ndarray, float] | None = None
        self.pose: tuple[float, float, float, float] | None = None
        self.speed = 0.0
        self.base_z = 0.0
        self.last_pose_s = 0.0
        # This node is paced on simulator time, not wall time, so the stack
        # still runs at its nominal rate in-sim when the world runs faster than
        # real time. World-pose messages carry the sim stamp and already arrive
        # at ~60 Hz, so they double as the clock; /clock publishes near 1 kHz
        # and would cost far more callbacks for no extra resolution.
        self.sim_s = 0.0
        self.clock = threading.Condition()
        self.clock_started = threading.Event()
        self.running = True
        self.arrived = False
        self.steer_state = 0.0
        self.velocity_command = 0.0
        self.wheel_angle_command = 0.0
        if not self.node.subscribe(Odometry, args.odometry_topic, self._on_odometry):
            raise RuntimeError(f"cannot subscribe to {args.odometry_topic}")
        self.model_name = args.model_name
        if not self.node.subscribe(Pose_V, args.world_pose_topic, self._on_world_pose):
            raise RuntimeError(f"cannot subscribe to {args.world_pose_topic}")
        if not self.node.subscribe(LaserScan, args.lidar_topic, self._on_lidar):
            raise RuntimeError(f"cannot subscribe to {args.lidar_topic}")
        if not self.node.subscribe(Image, args.semantic_topic, self._on_semantic):
            raise RuntimeError(f"cannot subscribe to {args.semantic_topic}")
        if not self.node.subscribe(Image, args.depth_topic, self._on_depth):
            raise RuntimeError(f"cannot subscribe to {args.depth_topic}")
        self.mapping_thread = threading.Thread(
            target=self._mapping_worker, name="ground-mapping", daemon=True
        )
        self.mapping_thread.start()

    def now(self) -> float:
        """Simulator clock in seconds."""
        with self.clock:
            return self.sim_s

    def _stamp_s(self, message) -> float:
        """Sim time a message was produced, not the time it was delivered."""
        stamp = message.header.stamp
        seconds = stamp.sec + stamp.nsec * 1e-9
        return seconds if seconds > 0.0 else self.now()

    def _on_odometry(self, message: Odometry) -> None:
        with self.lock:
            self.speed = max(0.0, float(message.twist.linear.x))
            if self.pose is not None:
                self.pose = (*self.pose[:3], self.speed)

    def _on_world_pose(self, message: Pose_V) -> None:
        stamp_s = self._stamp_s(message)
        model_pose = next((pose for pose in message.pose if pose.name == self.model_name), None)
        if model_pose is None:
            return
        orientation = model_pose.orientation
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
        )
        with self.lock:
            self.pose = (
                float(model_pose.position.x),
                float(model_pose.position.y),
                yaw,
                self.speed,
            )
            self.last_pose_s = stamp_s
            self.base_z = float(model_pose.position.z)
        with self.clock:
            self.sim_s = stamp_s
            self.clock.notify_all()
        self.clock_started.set()

    def _on_semantic(self, message: Image) -> None:
        try:
            labels = self.semantic_mapper.decode_labels(message)
        except ValueError as error:
            print(f"Ignoring semantic frame: {error}", flush=True)
            return
        with self.mapping_lock:
            self.pending_labels = (labels, self._stamp_s(message))
        self.mapping_event.set()

    def _on_depth(self, message: Image) -> None:
        try:
            depth = self.semantic_mapper.decode_depth(message)
        except ValueError as error:
            print(f"Ignoring depth frame: {error}", flush=True)
            return
        with self.mapping_lock:
            self.pending_depth = (depth, self._stamp_s(message))
        self.mapping_event.set()

    def _on_lidar(self, message: LaserScan) -> None:
        count = int(message.count)
        vertical_count = max(1, int(message.vertical_count))
        ranges = np.asarray(message.ranges, dtype=np.float64)
        if count <= 0 or ranges.size < count:
            return
        usable = ranges[: count * vertical_count].reshape((-1, count))
        valid = np.isfinite(usable) & (usable > max(0.22, float(message.range_min))) & (usable < float(message.range_max))
        if not np.any(valid):
            return
        with self.lock:
            if self.pose is None:
                return
            x, y, yaw, _ = self.pose
            base_z = self.base_z
        vertical_indices, horizontal_indices = np.nonzero(valid)
        distances = usable[valid]
        bearings = float(message.angle_min) + horizontal_indices * float(message.angle_step)
        vertical = float(message.vertical_angle_min) + vertical_indices * float(message.vertical_angle_step)
        horizontal_range = distances * np.cos(vertical)
        local_x = 0.0625 + horizontal_range * np.cos(bearings)
        local_y = horizontal_range * np.sin(bearings)
        cy, sy = math.cos(yaw), math.sin(yaw)
        world_x = x + cy * local_x - sy * local_y
        world_y = y + sy * local_x + cy * local_y
        world_z = base_z + 0.3725 + distances * np.sin(vertical)
        xyz = np.column_stack((world_x, world_y, world_z))
        sensor_xyz = (x + cy * 0.0625, y + sy * 0.0625, base_z + 0.3725)
        stamp_s = self._stamp_s(message)
        # A one-element mailbox prevents old scans queuing behind mapping.
        # Meridian processes this layer at 10 Hz even when a sensor is faster.
        with self.mapping_lock:
            self.pending_lidar = (xyz, sensor_xyz, base_z, stamp_s)
        self.mapping_event.set()

    def _mapping_worker(self) -> None:
        """Build and publish maps without delaying the 20 Hz controller."""
        last_lidar_s = -math.inf
        last_semantic_s = -math.inf
        while self.running:
            self.mapping_event.wait(0.1)
            self.mapping_event.clear()
            with self.mapping_lock:
                lidar = self.pending_lidar
                labels = self.pending_labels
                depth = self.pending_depth
                # Consume lidar once. Camera frames stay until a synchronized
                # pair is available, then SemanticMapper rejects duplicates.
                self.pending_lidar = None
            with self.lock:
                pose = self.pose
                base_z = self.base_z
            now_s = self.now()

            if lidar is not None and now_s - last_lidar_s >= 0.095:
                xyz, sensor_xyz, ground_z, scan_s = lidar
                self.ground_mapper.update(xyz, sensor_xyz, ground_z, scan_s)
                origin = self.ground_mapper.origin
                self.latest_ground_obstacles = LocalGridMap(
                    self.ground_mapper.classes.copy(), origin[0], origin[1], 0.25
                )
                occupancy_probability = self.ground_mapper.occupancy.evidence_grid(
                    scan_s
                )[0]
                self.latest_ground_obstacle_probability = LocalGridMap(
                    occupancy_probability, origin[0], origin[1], 0.25
                )
                last_lidar_s = now_s

            semantic_changed = False
            if pose is not None and labels is not None and depth is not None and now_s - last_semantic_s >= 0.48:
                label_image, label_s = labels
                depth_image, depth_s = depth
                self.semantic_mapper.set_labels(label_image, label_s)
                self.semantic_mapper.set_depth(depth_image, depth_s)
                semantic_changed = self.semantic_mapper.project_if_ready(
                    (pose[0], pose[1], pose[2], base_z), now_s
                )
                if semantic_changed:
                    origin, cost, _, _, obstacle = self.semantic_mapper.render(
                        pose[:2], now_s
                    )
                    self.latest_ground_semantics = LocalGridMap(
                        cost, float(origin[0]), float(origin[1]), 0.25
                    )
                    self.latest_ground_semantic_obstacles = LocalGridMap(
                        obstacle, float(origin[0]), float(origin[1]), 0.25
                    )
                    last_semantic_s = now_s

            if pose is not None and now_s - self.last_map_write_s >= 0.2:
                write_snapshot(
                    self.ground_map_path,
                    self.ground_mapper,
                    self.semantic_mapper,
                    (pose[0], pose[1], pose[2], base_z),
                    now_s,
                )
                self.last_map_write_s = now_s

    def _zero(self) -> None:
        self.velocity_command = 0.0
        self.wheel_angle_command = 0.0
        self.publisher.publish(Twist())

    def step(self) -> None:
        with self.lock:
            pose = self.pose
            pose_age = self.now() - self.last_pose_s
        if pose is None or pose_age > 0.5:
            self._zero()
            return
        x, y, yaw, speed = pose
        self.map_stack.ground_obstacles = self.latest_ground_obstacles
        self.map_stack.ground_obstacle_probability = (
            self.latest_ground_obstacle_probability
        )
        self.map_stack.ground_semantics = self.latest_ground_semantics
        self.map_stack.ground_semantic_obstacles = (
            self.latest_ground_semantic_obstacles
        )
        if math.hypot(x - self.route.xy[-1, 0], y - self.route.xy[-1, 1]) <= self.arrival_radius:
            if not self.arrived:
                print("Final route point reached.", flush=True)
            self.arrived = True
            self._zero()
            return

        self.assistance.reload_map()
        state = np.asarray((x, y, yaw, speed, self.steer_state), dtype=np.float64)
        # Meridian MPPI consumes the classified local occupancy grid. Feeding
        # raw endpoints here as well double-counts obstacles and mistakes
        # uphill ground for a body-height collision.
        command = self.planner.command(state, self.map_stack, np.empty((0, 2)))
        if self.markers is not None:
            self.markers.update(self.route.xy, self.route_anchors, self.planner.best_trajectory())
        trajectories = self.planner.last_trajectories
        planned = trajectories[:, 1:] if trajectories is not None else trajectories
        exposure, roi = self.map_stack.uncertainty_exposure(planned)
        self.assistance.update(exposure, roi)
        if self.assistance.hold:
            command[:] = 0.0

        model = self.planner.model
        dt = self.planner.config.dt
        # Gazebo's Ackermann plugin shares a single acceleration limiter between
        # its linear and angular channels, and that limiter is sized for yaw
        # response. Enforce the rollout's linear envelope here instead, where it
        # constrains one axis only.
        velocity = float(
            np.clip(
                float(command[0]),
                self.velocity_command - model.brake_max * dt,
                self.velocity_command + model.acceleration_max * dt,
            )
        )
        self.velocity_command = velocity
        # Final backstop on the published wheel angle, matching Meridian's
        # steer_cmd_slew_rad_s: a one-cycle jump slams the wheels and the
        # rover oscillates behind it.
        slew = self.planner.config.steer_cmd_slew_rad_s * dt
        wheel_angle = float(
            np.clip(
                command[1] * model.steer_max,
                self.wheel_angle_command - slew,
                self.wheel_angle_command + slew,
            )
        )
        self.wheel_angle_command = wheel_angle
        alpha = 1.0 - math.exp(-dt / model.steer_tau)
        self.steer_state += alpha * (wheel_angle - self.steer_state)
        output = Twist()
        output.linear.x = velocity
        # Gazebo's Ackermann system physically steers the front wheels, but its
        # Twist interface accepts yaw rate. Convert Meridian's wheel angle at
        # this transport boundary using the same bicycle geometry.
        yaw_rate = velocity * math.tan(wheel_angle) / model.wheelbase
        # Apply the same lateral-acceleration constraint used by the MPPI
        # rollout. Without this clamp, the real vehicle receives motion that
        # the optimizer never predicted and can exceed valid Ackermann motion.
        yaw_limit = model.lateral_acceleration_max / max(abs(velocity), 0.1)
        output.angular.z = float(np.clip(yaw_rate, -yaw_limit, yaw_limit))
        self.publisher.publish(output)

    def _wait_until(self, sim_deadline: float) -> bool:
        """Block until the world reaches sim_deadline; False if already past it.

        Bounded in wall time so a paused or stalled clock still lets step() run
        its staleness check and publish a zero command.
        """
        with self.clock:
            if self.sim_s >= sim_deadline:
                return False
            limit = time.monotonic() + 0.5
            while self.running and self.sim_s < sim_deadline:
                remaining = limit - time.monotonic()
                if remaining <= 0.0:
                    break
                self.clock.wait(timeout=min(0.05, remaining))
        return True

    def run(self) -> None:
        print(
            f"Meridian Drive MPPI is following {self.route_name} with {self.assistance.mode}. "
            f"UAV requests: {self.assistance.request_path}",
            flush=True,
        )
        period = self.planner.config.dt
        if not self.clock_started.wait(timeout=10.0):
            print("No world-pose messages yet; is the simulator running?", flush=True)
        deadline = self.now()
        late_cycles = 0
        last_report_s = self.now()
        try:
            while self.running:
                self.step()
                now = self.now()
                deadline += period
                if now < deadline - period:
                    # The world was reset; re-anchor instead of racing to replay
                    # the skipped span.
                    deadline = now + period
                if not self._wait_until(deadline):
                    late_cycles += 1
                now = self.now()
                if now - last_report_s >= 5.0:
                    if late_cycles:
                        print(
                            f"Controller missed {late_cycles} of the last "
                            f"{round(5.0 / period)} cycles: the world is stepping "
                            "faster than the planner can keep up with.",
                            flush=True,
                        )
                    last_report_s, late_cycles = now, 0
        finally:
            self.running = False
            self.mapping_event.set()
            self.mapping_thread.join(timeout=2.0)
            if self.markers is not None:
                self.markers.close()
            self._zero()


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    runtime = project_root / "runtime"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waypoint", nargs=2, type=float, action="append", metavar=("X", "Y"))
    parser.add_argument("--route-file", type=Path, help="KMZ, KML, or Meridian GPS JSON route")
    parser.add_argument("--assistance", choices=MODES, default="ground_only")
    parser.add_argument("--uav-map", type=Path, default=runtime / "uav_map.npz")
    parser.add_argument("--uav-request", type=Path, default=runtime / "uav_request.json")
    parser.add_argument("--status", type=Path, default=runtime / "autonomy_status.json")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--target-speed", type=float, default=1.7)
    parser.add_argument("--speed-max", type=float, default=2.2)
    parser.add_argument("--arrival-radius", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--odometry-topic", default="/model/hill_rover/odometry")
    parser.add_argument("--world-pose-topic", default="/world/hill_country/dynamic_pose/info")
    parser.add_argument("--model-name", default="hill_rover")
    parser.add_argument("--lidar-topic", default="/model/hill_rover/lidar")
    parser.add_argument("--semantic-topic", default="/model/hill_rover/semantic/labels_map")
    parser.add_argument("--depth-topic", default="/model/hill_rover/depth")
    parser.add_argument("--ground-map", type=Path, default=runtime / "ground_maps.npz")
    parser.add_argument("--command-topic", default="/model/hill_rover/cmd_vel")
    parser.add_argument(
        "--terrain-dem",
        type=Path,
        default=project_root / "models" / "hill_terrain" / "meshes" / "terrain.tif",
    )
    parser.add_argument("--marker-service", default="/marker_array")
    parser.add_argument("--no-visualization", action="store_true")
    args = parser.parse_args()
    if args.route_file is not None and args.waypoint:
        parser.error("use route-file or waypoint, not both")
    if args.route_file is None and not args.waypoint:
        args.route_file = find_default_route(project_root)
    if args.samples < 8 or args.horizon < 2:
        parser.error("samples must be at least 8 and horizon must be at least 2")
    if args.target_speed <= 0.0 or args.speed_max < args.target_speed:
        parser.error("speed limits must be positive and speed-max must include target-speed")
    return args


def main() -> None:
    autonomy = GazeboAutonomy(parse_args())

    def stop(_signal: int, _frame: object) -> None:
        autonomy.running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    autonomy.run()


if __name__ == "__main__":
    main()
