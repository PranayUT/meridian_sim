#!/usr/bin/env python3
"""Run the Meridian Drive autonomy core on the Gazebo rover."""

from __future__ import annotations

import argparse
from collections import deque
import json
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
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean
from gz.transport13 import Node

from autonomy.gp_navigation import GPNavigationConfig, GPNavigationPlanner

from .assistance import MODES, AssistanceManager, MappedRecovery
from .core import MPPI, MppiConfig, Route, rollout
from .ground_mapping import GroundMapper, SemanticMapper, write_snapshot
from .maps import (
    COMBINED_MAP_TYPE,
    AssistanceEvaluation,
    LocalGridMap,
    MapStack,
    TerrainMap,
)
from .routes import find_default_route, load_route
from .uav_ground_truth import GroundTruthUav
from .visualization import GazeboMarkers

DEFAULT_ROUTE = [(0.0, 0.0), (12.0, 0.0), (18.0, 12.0), (5.0, 20.0), (-8.0, 8.0)]


class GazeboAutonomy:
    """Connect simulator sensor topics to the ROS-free planner."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.node = Node()
        world_name = args.world_pose_topic.strip('/').split('/')[1]
        self.world_control_service = f"/world/{world_name}/control"
        self.lockstep = bool(args.lockstep)
        self.physics_step_s = float(args.physics_step)
        self.publisher = self.node.advertise(args.command_topic, Twist)
        waypoints = load_route(args.route_file) if args.route_file else (args.waypoint or DEFAULT_ROUTE)
        self.route_anchors = np.asarray(waypoints, dtype=np.float64)
        self.route = Route.from_waypoints(waypoints)
        self.route_name = args.route_file.name if args.route_file else "built-in route"
        self.planner_kind = args.planner
        if args.planner == "gp_navigation":
            gp_config = GPNavigationConfig(
                resolution=args.gp_resolution,
                radius=args.gp_radius,
                inducing_points=args.gp_inducing_points,
                step_len=args.gp_step_len,
                iter_max=args.gp_iterations,
                traversability_limit=args.gp_traversability_limit,
                replan_period_s=args.gp_replan_period,
                target_speed=args.target_speed,
                speed_max=args.speed_max,
                horizon=args.horizon,
            )
            self.planner = GPNavigationPlanner(
                self.route, config=gp_config, seed=args.seed
            )
        else:
            config = MppiConfig(
                samples=args.samples,
                horizon=args.horizon,
                target_speed=args.target_speed,
                speed_max=args.speed_max,
            )
            self.planner = MPPI(self.route, config=config, seed=args.seed)
        terrain = TerrainMap.from_tif(args.terrain_dem) if args.terrain_dem else None
        self.map_stack = MapStack(
            terrain=terrain,
            uncertainty_maturity_s=args.mapping_uncertainty_maturity,
        )
        self.ground_mapper = GroundMapper()
        self.semantic_mapper = SemanticMapper()
        self.ground_map_path = args.ground_map
        self.assistance_trace_path = args.assistance_trace
        self.assistance_trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.assistance_trace_path.unlink(missing_ok=True)
        self.last_assistance_trace_s = -math.inf
        self.assistance_probe_m = float(args.assistance_probe_m)
        self.last_trace_pose: tuple[float, float, float] | None = None
        self.probe_uncertainty_threshold = float(
            args.uav_probe_uncertainty_threshold
        )
        self.probe_hit_window_s = float(args.uav_probe_hit_window)
        self.probe_required_hits = int(args.uav_probe_hits)
        self.probe_min_world_speed_mps = float(args.uav_probe_min_world_speed)
        self.probe_hits: deque[tuple[float, AssistanceEvaluation]] = deque()
        self.mapped_recovery = MappedRecovery(
            duration_s=args.uav_recovery_duration,
            speed_mps=args.uav_recovery_speed,
            steering_fraction=args.uav_recovery_steering,
            cooldown_s=args.uav_recovery_cooldown,
            max_attempts_per_location=args.uav_recovery_max_attempts,
            rearm_progress_m=args.uav_recovery_rearm_progress,
        )
        self.last_probe_evaluation = AssistanceEvaluation(0.0, None)
        self.last_raw_probe_evaluation = AssistanceEvaluation(0.0, None)
        self.last_probe_world_speed_mps: float | None = None
        self.last_probe_hit = False
        self.last_map_write_s = 0.0
        self.latest_ground_obstacles: LocalGridMap | None = None
        self.latest_ground_obstacle_probability: LocalGridMap | None = None
        self.latest_ground_occupancy_uncertainty: LocalGridMap | None = None
        self.latest_ground_semantics: LocalGridMap | None = None
        self.latest_ground_semantic_obstacles: LocalGridMap | None = None
        self.latest_ground_semantic_uncertainty: LocalGridMap | None = None
        self.markers = None if args.no_visualization else GazeboMarkers(
            self.node, terrain, args.marker_service
        )
        ground_truth_uav = None
        if args.assistance != "ground_only" and args.uav_source == "ground_truth":
            ground_truth_uav = GroundTruthUav.from_world(
                args.uav_map,
                args.uav_semantic_masks,
                args.world_file,
                args.uav_vegetation_seed,
                args.uav_resolution,
                args.uav_grass_occupancy_probability,
            )
        self.assistance = AssistanceManager(
            mode=args.assistance,
            map_path=args.uav_map,
            request_path=args.uav_request,
            status_path=args.status,
            map_stack=self.map_stack,
            uncertainty_threshold=args.uav_uncertainty_threshold,
            map_size_m=args.uav_map_size,
            request_handler=ground_truth_uav,
            clock=self.now,
        )
        if args.assistance in ("explore_then_drive", "always_on_uav"):
            # Both route-wide arms expose the same information to the UGV.
            # explore_then_drive is charged for a sequential lawnmower survey
            # by the campaign harness; always_on_uav remains an oracle upper
            # bound. Keep the equivalent of a local UAV view around every
            # route point available from the first planner tick onward. Align
            # outward to raster cells so rounding cannot lose the far edge.
            assert ground_truth_uav is not None
            padding = args.uav_map_size / 2.0
            resolution = args.uav_resolution
            route_x = self.route.xy[:, 0]
            route_y = self.route.xy[:, 1]
            roi = (
                math.floor((float(np.min(route_x)) - padding) / resolution)
                * resolution,
                math.floor((float(np.min(route_y)) - padding) / resolution)
                * resolution,
                math.ceil((float(np.max(route_x)) + padding) / resolution)
                * resolution,
                math.ceil((float(np.max(route_y)) + padding) / resolution)
                * resolution,
            )
            # The combined product contains both evidence channels. Unlike
            # reactive assistance, these proactive baselines have no request
            # count, stop, wait, or fusion-settling state in the controller.
            ground_truth_uav(roi, 1, COMBINED_MAP_TYPE)
            if not self.assistance.reload_map(force=True):
                raise RuntimeError("could not load the route-wide UAV baseline map")
            self.assistance.detail = (
                "route-wide occupancy and semantic aerial survey loaded"
                if args.assistance == "explore_then_drive"
                else "route-wide occupancy and semantic aerial evidence loaded"
            )
        # Assistance evaluation is the most expensive thing in the control
        # tick. It runs every tick by default so the counterfactual sees the
        # same rollout population the planner just acted on, and so per-cell
        # uncertainty maturity is sampled at the full 20 Hz it was calibrated
        # against. Raising this trades that fidelity for tick budget: the
        # state machine still advances every tick on the cached evaluation,
        # so stops, requests, and fusion stay responsive, but a longer period
        # weakens the maturity gate, which biases toward more requests.
        self.assistance_period_s = float(args.assistance_period)
        self.last_assistance_s = -math.inf
        self.last_evaluation = AssistanceEvaluation(0.0, None)
        self.last_action_evaluation = AssistanceEvaluation(0.0, None)
        self.path_uncertainty_threshold = float(
            args.uav_path_uncertainty_threshold
        )
        self.arrival_radius = args.arrival_radius
        self.arrival_progress_fraction = float(args.arrival_progress_fraction)
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
        self.mobility_anchor_s = 0.0
        self.mobility_anchor_xy: tuple[float, float] | None = None
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
            self.speed = float(message.twist.linear.x)
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
                if isinstance(self.planner, GPNavigationPlanner) and pose is not None:
                    self.planner.update_point_cloud(xyz, pose[:3], scan_s)
                origin = self.ground_mapper.origin
                self.latest_ground_obstacles = LocalGridMap(
                    self.ground_mapper.classes.copy(), origin[0], origin[1], 0.25
                )
                occupancy_probability, occupancy_variance = (
                    self.ground_mapper.occupancy.evidence_grid(scan_s)[:2]
                )
                self.latest_ground_obstacle_probability = LocalGridMap(
                    occupancy_probability, origin[0], origin[1], 0.25
                )
                self.latest_ground_occupancy_uncertainty = LocalGridMap(
                    occupancy_variance, origin[0], origin[1], 0.25
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
                    (
                        origin,
                        cost,
                        _,
                        _,
                        obstacle,
                        cost_variance,
                    ) = self.semantic_mapper.render_layers(pose[:2], now_s)
                    self.latest_ground_semantics = LocalGridMap(
                        cost, float(origin[0]), float(origin[1]), 0.25
                    )
                    self.latest_ground_semantic_obstacles = LocalGridMap(
                        obstacle, float(origin[0]), float(origin[1]), 0.25
                    )
                    self.latest_ground_semantic_uncertainty = LocalGridMap(
                        cost_variance, float(origin[0]), float(origin[1]), 0.25
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
        self.map_stack.ground_occupancy_uncertainty = (
            self.latest_ground_occupancy_uncertainty
        )
        self.map_stack.ground_semantics = self.latest_ground_semantics
        self.map_stack.ground_semantic_obstacles = (
            self.latest_ground_semantic_obstacles
        )
        self.map_stack.ground_semantic_uncertainty = (
            self.latest_ground_semantic_uncertainty
        )
        if (
            math.hypot(x - self.route.xy[-1, 0], y - self.route.xy[-1, 1])
            <= self.arrival_radius
            and self.planner.progress_m
            >= self.arrival_progress_fraction * float(self.route.distance[-1])
        ):
            if not self.arrived:
                print("Final route point reached.", flush=True)
            self.arrived = True
            self._zero()
            return

        if self.assistance.reload_map():
            # The retained product answers the rolling episode that requested
            # it. Do not let the pre-map hit survive fusion long enough to buy
            # a duplicate product for the same corridor.
            self.probe_hits.clear()
            self.last_probe_evaluation = AssistanceEvaluation(0.0, None)
        state = np.asarray((x, y, yaw, speed, self.steer_state), dtype=np.float64)
        # Meridian MPPI consumes the classified local occupancy grid. Feeding
        # raw endpoints here as well double-counts obstacles and mistakes
        # uphill ground for a body-height collision.
        command = self.planner.command(state, self.map_stack, np.empty((0, 2)))
        if self.markers is not None:
            self.markers.update(self.route.xy, self.route_anchors, self.planner.best_trajectory())
        trajectories = self.planner.last_trajectories
        planned = trajectories[:, 1:] if trajectories is not None else trajectories
        now_s = self.now()
        # A world reset moves the clock backwards; re-anchor instead of
        # blocking evaluation until the old timestamp comes around again.
        if now_s < self.last_assistance_s:
            self.last_assistance_s = -math.inf
        if now_s - self.last_assistance_s >= self.assistance_period_s:
            self.last_evaluation = self.map_stack.evaluate_assistance(
                planned,
                (x, y),
                self.assistance.uncertainty_threshold,
                now_s,
            )
            self.last_action_evaluation = (
                self.map_stack.evaluate_action_assistance(
                    self.planner.best_trajectory(),
                    self.path_uncertainty_threshold,
                    now_s,
                )
            )
            self.last_assistance_s = now_s
        action_evaluation = self.last_action_evaluation
        trace_due = now_s - self.last_assistance_trace_s >= 0.5
        probe: np.ndarray | None = None
        if trace_due:
            probe_state = np.asarray(
                (x, y, yaw, speed, self.steer_state), dtype=np.float64
            )
            probe = self._forward_probe(probe_state)
            self._sample_probe_assistance(now_s, (x, y), probe)
        probe_evaluation = self.last_probe_evaluation
        evaluation = (
            probe_evaluation
            if probe_evaluation.probe_relevant
            else self.last_evaluation
        )
        # Odometry is driven by wheel rotation and remains high when the rover
        # spins its wheels against vegetation. Detect loss of mobility from
        # world displacement, which is also what the intervention harness
        # measures. Reset while assistance deliberately holds the rover so a
        # completed request cannot trigger another request on its own stop.
        if (
            now_s < self.mobility_anchor_s
            or self.mobility_anchor_xy is None
            or math.dist((x, y), self.mobility_anchor_xy) >= 0.25
        ):
            self.mobility_anchor_s = now_s
            self.mobility_anchor_xy = (x, y)
        mobility_stalled = now_s - self.mobility_anchor_s >= 5.0
        # A stall is uncertainty-relevant only when the selected path itself
        # still contains unresolved evidence. Do not attach a stall to an ROI
        # drawn from unrelated rejected rollouts.
        mobility_uncertain = mobility_stalled and action_evaluation.roi is not None
        if mobility_uncertain and not evaluation.action_relevant:
            evaluation = action_evaluation
        self.assistance.update(
            evaluation.uncertainty_exposure,
            evaluation.roi,
            source=evaluation.source,
            map_type=evaluation.map_type,
            decision_relevant=evaluation.decision_relevant,
            action_relevant=evaluation.action_relevant,
            probe_relevant=evaluation.probe_relevant,
            speed_mps=speed,
            mobility_stalled=mobility_uncertain,
            sim_time_s=now_s,
            position_xy=(x, y),
        )
        recovery = self.mapped_recovery.update(
            now_s,
            stalled=mobility_stalled,
            progress_m=self.planner.progress_m,
            enabled=(
                self.assistance.mode != "ground_only"
                and self.map_stack.has_aerial
                and not self.assistance.hold
            ),
        )
        if recovery is not None:
            command[0], command[1], started = recovery
            if started:
                self.planner.nominal.fill(0.0)
                self.planner.previous.fill(0.0)
                print(
                    f"Mapped recovery {self.mapped_recovery.count}: backing out "
                    f"after {now_s - self.mobility_anchor_s:.1f} s without progress",
                    flush=True,
                )
        if self.assistance.hold:
            command[:] = 0.0
        if trace_due:
            self._write_assistance_trace(
                now_s,
                (x, y, yaw),
                speed,
                float(command[0]),
                planned,
                mobility_stalled,
                action_evaluation,
                probe,
                recovery is not None,
            )

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

    def _world_control(self, **fields: object) -> bool:
        """Send one WorldControl request, returning whether it was confirmed."""
        request = WorldControl()
        for name, value in fields.items():
            setattr(request, name, value)
        try:
            confirmed, response = self.node.request(
                self.world_control_service, request, WorldControl, Boolean, 2000
            )
        except Exception:  # transport errors here are not worth ending a trial
            return False
        return bool(confirmed and response.data)

    def _advance_world(self, steps: int, period: float) -> bool:
        """Step the world by exactly one control period and wait for its clock.

        Free-running, the world advances on its own schedule and the planner
        sheds whatever ticks it cannot afford, so the autonomy silently falls
        out of step with the physics whenever the machine is busy. Driving the
        steps from here inverts that: the world only moves when the controller
        is ready for it, so a trial sees the same 20 Hz in simulator time no
        matter how many trials share the machine.
        """
        with self.clock:
            target = self.sim_s + period
        # The acknowledgement can arrive later than the physics it asked for,
        # so the simulator clock decides whether the step happened, not the
        # reply. Giving up on an unconfirmed request instead would re-send on
        # the next tick and step the world twice for one control period.
        self._world_control(pause=True, multi_step=steps)
        # The pose feed carries the simulator stamp, so it reports when the
        # requested block of physics has actually been integrated.
        limit = time.monotonic() + 10.0
        with self.clock:
            while self.running and self.sim_s < target - 1e-9:
                remaining = limit - time.monotonic()
                if remaining <= 0.0:
                    return False
                self.clock.wait(timeout=min(0.05, remaining))
        return True

    def run(self) -> None:
        print(
            f"{self.planner_kind} is following {self.route_name} with {self.assistance.mode}. "
            f"UAV requests: {self.assistance.request_path}",
            flush=True,
        )
        period = self.planner.config.dt
        if not self.clock_started.wait(timeout=10.0):
            print("No world-pose messages yet; is the simulator running?", flush=True)
        deadline = self.now()
        late_cycles = 0
        last_report_s = self.now()
        steps_per_tick = int(round(period / self.physics_step_s))
        if self.lockstep:
            if abs(steps_per_tick * self.physics_step_s - period) > 1e-9:
                raise RuntimeError(
                    f"control period {period}s is not a whole number of "
                    f"{self.physics_step_s}s physics steps"
                )
            # Take the world under control before the first tick, so no physics
            # runs that this controller has not asked for.
            if not any(self._world_control(pause=True) for _ in range(3)):
                print(
                    "World pause was not acknowledged; continuing lockstep "
                    "with simulator-clock verification.",
                    flush=True,
                )
            print(
                f"Lockstep: driving {steps_per_tick} x {self.physics_step_s}s "
                f"physics steps per {period}s control tick.",
                flush=True,
            )
        try:
            while self.running:
                self.step()
                if self.lockstep:
                    if not self._advance_world(steps_per_tick, period):
                        late_cycles += 1
                    now = self.now()
                    if now - last_report_s >= 5.0:
                        if late_cycles:
                            print(
                                f"Lockstep stalled on {late_cycles} of the last "
                                f"{round(5.0 / period)} steps: the simulator did "
                                "not confirm the requested physics.",
                                flush=True,
                            )
                        last_report_s, late_cycles = now, 0
                    continue
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
            # Hand the world back, so a simulator that outlives this process is
            # not left frozen with its clock stopped.
            if self.lockstep:
                self._world_control(pause=False)
            self.mapping_event.set()
            self.mapping_thread.join(timeout=2.0)
            if self.markers is not None:
                self.markers.close()
            self._zero()


    def _forward_probe(self, state: np.ndarray) -> np.ndarray | None:
        """Roll the intended steering out over a fixed forward distance.

        The selected MPPI trajectory is a fixed *time* horizon, so its reach
        collapses from about 5 m to well under 1 m exactly when the controller
        slows in front of something. A diagnostic that is meant to warn before
        contact has to hold its warning distance constant, so this replays the
        nominal steering at target speed for as many steps as the requested
        distance needs.
        """
        if self.assistance_probe_m <= 0.0:
            return None
        config = self.planner.config
        nominal = self.planner.nominal
        step_m = max(1e-6, config.target_speed * config.dt)
        steps = int(math.ceil(self.assistance_probe_m / step_m))
        steer = np.asarray(nominal[:, 1], dtype=np.float64)
        if steer.size == 0:
            return None
        if steps > steer.size:
            steer = np.concatenate(
                [steer, np.full(steps - steer.size, steer[-1])]
            )
        controls = np.empty((1, steps, 2), dtype=np.float64)
        controls[0, :, 0] = config.target_speed
        controls[0, :, 1] = steer[:steps]
        initial = state.copy()
        initial[3] = config.target_speed
        return rollout(initial[None, :], controls, self.planner.model, config.dt)[
            :, 1:
        ]

    def _sample_probe_assistance(
        self,
        now_s: float,
        position_xy: tuple[float, float],
        probe: np.ndarray | None,
    ) -> None:
        """Update the calibrated rolling-hit gate at the trace's 2 Hz cadence."""
        world_speed: float | None = None
        if self.last_trace_pose is not None:
            previous_x, previous_y, previous_s = self.last_trace_pose
            elapsed = now_s - previous_s
            if elapsed > 0.0:
                world_speed = (
                    math.dist(position_xy, (previous_x, previous_y)) / elapsed
                )
            else:
                self.probe_hits.clear()
        self.last_trace_pose = (*position_xy, now_s)
        raw = self.map_stack.evaluate_probe_assistance(
            probe, self.probe_uncertainty_threshold
        )
        while (
            self.probe_hits
            and now_s - self.probe_hits[0][0] > self.probe_hit_window_s
        ):
            self.probe_hits.popleft()
        hit = bool(
            raw.probe_relevant
            and world_speed is not None
            and world_speed > self.probe_min_world_speed_mps
        )
        if hit:
            self.probe_hits.append((now_s, raw))
        if len(self.probe_hits) >= self.probe_required_hits:
            active = self.probe_hits[-1][1]
        else:
            active = AssistanceEvaluation(
                raw.uncertainty_exposure,
                raw.roi,
                raw.source,
                raw.map_type,
                raw.decision_relevant,
                raw.action_relevant,
                probe_relevant=False,
            )
        self.last_raw_probe_evaluation = raw
        self.last_probe_evaluation = active
        self.last_probe_world_speed_mps = world_speed
        self.last_probe_hit = hit

    def _write_assistance_trace(
        self,
        now_s: float,
        pose: tuple[float, float, float],
        wheel_speed: float,
        command_speed: float,
        population: np.ndarray | None,
        mobility_stalled: bool,
        action_evaluation: AssistanceEvaluation,
        probe: np.ndarray | None,
        recovery_active: bool,
    ) -> None:
        """Record every candidate stuck-predictor at a fixed cadence.

        Written in every assistance mode, including ground_only, so a control
        run measures the same variables without a UAV map perturbing the route.
        """
        x, y, yaw = pose
        # World speed, unlike wheel odometry, goes to zero when the rover is
        # spinning its wheels against vegetation. The gap between them is the
        # slip that precedes a wedge.
        world_speed = self.last_probe_world_speed_mps
        selected = self.planner.best_trajectory()
        raw_probe = self.last_raw_probe_evaluation
        active_probe = self.last_probe_evaluation
        trace: dict[str, object] = {
            "sim_time_s": now_s,
            "vehicle_xy": [x, y],
            "yaw_rad": yaw,
            "wheel_speed_mps": wheel_speed,
            "world_speed_mps": world_speed,
            "command_speed_mps": command_speed,
            "slip_mps": None
            if world_speed is None
            else max(0.0, abs(wheel_speed) - world_speed),
            "mobility_window_s": now_s - self.mobility_anchor_s,
            "mobility_stalled": bool(mobility_stalled),
            "mapped_recovery_active": bool(recovery_active),
            "mapped_recovery_count": self.mapped_recovery.count,
            "mapped_recovery_location_attempts": (
                self.mapped_recovery.location_attempts
            ),
            "population_source": self.last_evaluation.source,
            "population_exposure": self.last_evaluation.uncertainty_exposure,
            "population_decision_relevant": bool(
                self.last_evaluation.decision_relevant
            ),
            "action_source": action_evaluation.source,
            "action_exposure": action_evaluation.uncertainty_exposure,
            "action_relevant": bool(action_evaluation.action_relevant),
            "action_roi": list(action_evaluation.roi)
            if action_evaluation.roi is not None
            else None,
            "probe_source": raw_probe.source,
            "probe_trigger_exposure": raw_probe.uncertainty_exposure,
            "probe_hit": self.last_probe_hit,
            "probe_hit_count": len(self.probe_hits),
            "probe_relevant": bool(active_probe.probe_relevant),
            "probe_roi": list(active_probe.roi)
            if active_probe.roi is not None
            else None,
            "assistance_state": self.assistance.state,
            "retained_uav_products": len(self.map_stack.aerial_history),
            "planner_best_cost": None
            if self.planner.last_costs is None
            else float(np.min(self.planner.last_costs)),
            "planner_cost_spread": None
            if self.planner.last_costs is None
            else float(
                np.mean(self.planner.last_costs) - np.min(self.planner.last_costs)
            ),
        }
        trace.update(
            self.map_stack.uncertainty_diagnostics(
                population, selected, probe, (x, y), now_s
            )
        )
        with self.assistance_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace, separators=(",", ":")) + "\n")
        self.last_assistance_trace_s = now_s


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    runtime = project_root / "runtime"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waypoint", nargs=2, type=float, action="append", metavar=("X", "Y"))
    parser.add_argument("--route-file", type=Path, help="KMZ, KML, or Meridian GPS JSON route")
    parser.add_argument(
        "--planner",
        choices=("meridian_mppi", "gp_navigation"),
        default="meridian_mppi",
        help="local navigation baseline to run",
    )
    parser.add_argument("--assistance", choices=MODES, default="ground_only")
    parser.add_argument("--uav-map", type=Path, default=runtime / "uav_map.npz")
    parser.add_argument("--uav-request", type=Path, default=runtime / "uav_request.json")
    parser.add_argument(
        "--uav-source", choices=("ground_truth", "file"), default="ground_truth",
        help="generate exact simulator maps or wait for an external NPZ producer",
    )
    parser.add_argument("--uav-uncertainty-threshold", type=float, default=0.75)
    parser.add_argument(
        "--uav-path-uncertainty-threshold",
        type=float,
        default=0.20,
        help="selected-trajectory uncertainty fraction that requests UAV evidence",
    )
    parser.add_argument(
        "--mapping-uncertainty-maturity", type=float, default=1.0,
        help="seconds one unresolved swept cell must persist before exposure",
    )
    parser.add_argument(
        "--assistance-period", type=float, default=0.0,
        help="seconds between swept-map assistance evaluations; 0 evaluates every tick",
    )
    parser.add_argument("--uav-map-size", type=float, default=25.0)
    parser.add_argument("--uav-resolution", type=float, default=0.25)
    parser.add_argument(
        "--uav-vegetation-seed", type=int,
        help="seed used to bake Gazebo vegetation; defaults to the paint file seed",
    )
    parser.add_argument(
        "--uav-semantic-masks", type=Path,
        default=project_root / "maps" / "vegetation_paint.npz",
    )
    parser.add_argument(
        "--world-file", type=Path,
        default=project_root / "worlds" / "hill_country.sdf",
    )
    parser.add_argument("--status", type=Path, default=runtime / "autonomy_status.json")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--target-speed", type=float, default=1.7)
    parser.add_argument("--speed-max", type=float, default=2.2)
    parser.add_argument("--gp-resolution", type=float, default=0.25)
    parser.add_argument("--gp-radius", type=float, default=5.0)
    parser.add_argument("--gp-inducing-points", type=int, default=160)
    parser.add_argument("--gp-step-len", type=float, default=0.5)
    parser.add_argument("--gp-iterations", type=int, default=1000)
    parser.add_argument("--gp-traversability-limit", type=float, default=0.6)
    parser.add_argument("--gp-replan-period", type=float, default=0.5)
    # Matches the harness default; see tools/run_experiment.py.
    parser.add_argument("--arrival-radius", type=float, default=1.0)
    parser.add_argument("--arrival-progress-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--odometry-topic", default="/model/hill_rover/odometry")
    parser.add_argument("--world-pose-topic", default="/world/hill_country/dynamic_pose/info")
    parser.add_argument("--model-name", default="hill_rover")
    parser.add_argument("--lidar-topic", default="/model/hill_rover/lidar")
    parser.add_argument("--semantic-topic", default="/model/hill_rover/semantic/labels_map")
    parser.add_argument("--depth-topic", default="/model/hill_rover/depth")
    parser.add_argument("--ground-map", type=Path, default=runtime / "ground_maps.npz")
    parser.add_argument(
        "--assistance-trace",
        type=Path,
        default=runtime / "assistance_trace.jsonl",
    )
    parser.add_argument(
        "--assistance-probe-m",
        type=float,
        default=8.0,
        help="fixed forward distance measured by the trace corridor probe",
    )
    parser.add_argument(
        "--uav-probe-uncertainty-threshold",
        type=float,
        default=0.04895,
        help="occupancy uncertainty fraction counted as one forward-probe hit",
    )
    parser.add_argument(
        "--uav-probe-hit-window",
        type=float,
        default=2.0,
        help="simulator seconds over which forward-probe hits accumulate",
    )
    parser.add_argument(
        "--uav-probe-hits",
        type=int,
        default=3,
        help="forward-probe hits required before the request activates",
    )
    parser.add_argument(
        "--uav-probe-min-world-speed",
        type=float,
        default=0.02,
        help="minimum measured displacement speed for a probe sample to count",
    )
    parser.add_argument(
        "--uav-grass-occupancy-probability",
        type=float,
        default=0.04,
        help="soft occupancy probability assigned to physical grass bodies",
    )
    parser.add_argument("--uav-recovery-duration", type=float, default=2.0)
    parser.add_argument("--uav-recovery-speed", type=float, default=0.6)
    parser.add_argument("--uav-recovery-steering", type=float, default=0.65)
    parser.add_argument("--uav-recovery-cooldown", type=float, default=3.0)
    parser.add_argument("--uav-recovery-max-attempts", type=int, default=1)
    parser.add_argument("--uav-recovery-rearm-progress", type=float, default=2.0)
    parser.add_argument("--command-topic", default="/model/hill_rover/cmd_vel")
    parser.add_argument(
        "--terrain-dem",
        type=Path,
        default=project_root / "models" / "hill_terrain" / "meshes" / "terrain.tif",
    )
    parser.add_argument("--marker-service", default="/marker_array")
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument(
        "--lockstep", action="store_true",
        help="drive the world one control period at a time, so the planner's "
             "rate in simulator time does not depend on machine load",
    )
    parser.add_argument(
        "--physics-step", type=float, default=0.001,
        help="the world's max_step_size; the control period must be a whole "
             "multiple of it under --lockstep",
    )
    args = parser.parse_args()
    if args.route_file is not None and args.waypoint:
        parser.error("use route-file or waypoint, not both")
    if args.route_file is None and not args.waypoint:
        args.route_file = find_default_route(project_root)
    if args.samples < 8 or args.horizon < 2:
        parser.error("samples must be at least 8 and horizon must be at least 2")
    if args.target_speed <= 0.0 or args.speed_max < args.target_speed:
        parser.error("speed limits must be positive and speed-max must include target-speed")
    if args.arrival_radius <= 0.0:
        parser.error("arrival-radius must be positive")
    if args.gp_resolution <= 0.0 or args.gp_radius <= args.gp_resolution:
        parser.error("gp-radius must be larger than the positive gp-resolution")
    if args.gp_inducing_points < 8 or args.gp_iterations < 1:
        parser.error("GP inducing points must be at least 8 and iterations positive")
    if args.gp_step_len <= 0.0 or args.gp_replan_period <= 0.0:
        parser.error("GP step length and replan period must be positive")
    if not 0.0 <= args.gp_traversability_limit <= 1.0:
        parser.error("gp-traversability-limit must be between 0 and 1")
    if not 0.0 <= args.arrival_progress_fraction <= 1.0:
        parser.error("arrival-progress-fraction must be between 0 and 1")
    if not 0.0 <= args.uav_uncertainty_threshold <= 1.0:
        parser.error("uav-uncertainty-threshold must be between 0 and 1")
    if not 0.0 <= args.uav_path_uncertainty_threshold <= 1.0:
        parser.error("uav-path-uncertainty-threshold must be between 0 and 1")
    if not 0.0 <= args.uav_probe_uncertainty_threshold <= 1.0:
        parser.error("uav-probe-uncertainty-threshold must be between 0 and 1")
    if args.uav_probe_hit_window < 0.5:
        parser.error("uav-probe-hit-window must be at least the 0.5 s sample period")
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
    if args.assistance_period < 0.0:
        parser.error("assistance-period may not be negative")
    if args.mapping_uncertainty_maturity < 0.0:
        parser.error("mapping-uncertainty-maturity must be non-negative")
    if args.uav_map_size <= 0.0 or args.uav_resolution <= 0.0:
        parser.error("uav-map-size and uav-resolution must be positive")
    cells = args.uav_map_size / args.uav_resolution
    if not math.isclose(cells, round(cells), abs_tol=1e-9):
        parser.error("uav-map-size must be an integer multiple of uav-resolution")
    if args.assistance != "ground_only" and args.uav_source == "ground_truth":
        if not args.uav_semantic_masks.is_file():
            parser.error(f"simulated UAV semantic masks do not exist: {args.uav_semantic_masks}")
        if not args.world_file.is_file():
            parser.error(f"simulator world does not exist: {args.world_file}")
    if args.assistance in ("explore_then_drive", "always_on_uav") \
            and args.uav_source != "ground_truth":
        parser.error(
            "explore_then_drive and always_on_uav require --uav-source ground_truth"
        )
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
