#!/usr/bin/env python3
"""Small Gazebo-native waypoint follower with lidar collision avoidance.

This is intentionally middleware-light: it proves the simulated vehicle, sensors,
and control loop before a full ROS 2 / Nav2 stack is introduced.
"""

from __future__ import annotations

import argparse
import math
import signal
import threading
import time

from gz.msgs10.laserscan_pb2 import LaserScan
from gz.msgs10.odometry_pb2 import Odometry
from gz.msgs10.twist_pb2 import Twist
from gz.transport13 import Node


DEFAULT_WAYPOINTS = [(12.0, 0.0), (18.0, 12.0), (5.0, 20.0), (-8.0, 8.0), (0.0, 0.0)]


class WaypointController:
    def __init__(self, waypoints: list[tuple[float, float]]) -> None:
        self.node = Node()
        self.publisher = self.node.advertise("/model/hill_rover/cmd_vel", Twist)
        self.waypoints = waypoints
        self.index = 0
        self.pose: tuple[float, float, float] | None = None
        self.front_clearance = math.inf
        self.left_clearance = math.inf
        self.right_clearance = math.inf
        self.lock = threading.Lock()
        self.running = True

        if not self.node.subscribe(Odometry, "/model/hill_rover/odometry", self.on_odom):
            raise RuntimeError("Could not subscribe to rover odometry")
        if not self.node.subscribe(LaserScan, "/model/hill_rover/lidar", self.on_lidar):
            raise RuntimeError("Could not subscribe to rover lidar")

    def on_odom(self, msg: Odometry) -> None:
        q = msg.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        with self.lock:
            self.pose = (msg.pose.position.x, msg.pose.position.y, yaw)

    def on_lidar(self, msg: LaserScan) -> None:
        ranges = list(msg.ranges)
        if not ranges:
            return

        def sector(start: float, end: float) -> float:
            values = []
            for i, value in enumerate(ranges):
                angle = msg.angle_min + i * msg.angle_step
                if start <= angle <= end and math.isfinite(value) and value > msg.range_min:
                    values.append(value)
            return min(values, default=math.inf)

        with self.lock:
            self.front_clearance = sector(-0.42, 0.42)
            self.left_clearance = sector(0.42, 1.35)
            self.right_clearance = sector(-1.35, -0.42)

    @staticmethod
    def wrap(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def command(self) -> Twist:
        cmd = Twist()
        with self.lock:
            pose = self.pose
            front, left, right = self.front_clearance, self.left_clearance, self.right_clearance
        if pose is None:
            return cmd

        x, y, yaw = pose
        goal_x, goal_y = self.waypoints[self.index]
        distance = math.hypot(goal_x - x, goal_y - y)
        if distance < 1.5:
            self.index = (self.index + 1) % len(self.waypoints)
            goal_x, goal_y = self.waypoints[self.index]
            print(f"Next waypoint {self.index + 1}: ({goal_x:.1f}, {goal_y:.1f})", flush=True)

        heading_error = self.wrap(math.atan2(goal_y - y, goal_x - x) - yaw)
        if front < 2.2:
            cmd.linear.x = 0.15
            cmd.angular.z = 0.85 if left >= right else -0.85
        else:
            cmd.linear.x = min(2.2, 0.55 + 0.18 * distance) * max(0.15, 1.0 - abs(heading_error) / 1.4)
            cmd.angular.z = max(-0.9, min(0.9, 1.35 * heading_error))
        return cmd

    def run(self) -> None:
        print("Waiting for odometry and lidar; ensure the simulation is running...", flush=True)
        while self.running:
            self.publisher.publish(self.command())
            time.sleep(0.05)
        self.publisher.publish(Twist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--waypoint",
        nargs=2,
        type=float,
        action="append",
        metavar=("X", "Y"),
        help="Waypoint in world metres; repeat for a route.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller = WaypointController(args.waypoint or DEFAULT_WAYPOINTS)

    def stop(_signum: int, _frame: object) -> None:
        controller.running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    controller.run()


if __name__ == "__main__":
    main()

