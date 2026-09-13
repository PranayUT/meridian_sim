#!/usr/bin/env python3
"""Observe a running demo and fail unless core sensors publish and the rover moves."""

import argparse
import math
import threading
import time

from gz.msgs10.imu_pb2 import IMU
from gz.msgs10.laserscan_pb2 import LaserScan
from gz.msgs10.navsat_pb2 import NavSat
from gz.msgs10.odometry_pb2 import Odometry
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node


lock = threading.Lock()
poses: list[tuple[float, float, float, float]] = []
world_poses: list[tuple[float, float, float, float]] = []
counts = {"odom": 0, "world_pose": 0, "imu": 0, "lidar": 0, "navsat": 0}


def odom_callback(msg: Odometry) -> None:
    q = msg.pose.orientation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    with lock:
        poses.append((msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, yaw))
        counts["odom"] += 1


def count(name: str):
    def callback(_msg: object) -> None:
        with lock:
            counts[name] += 1

    return callback


def world_pose_callback(msg: Pose_V) -> None:
    pose = next((item for item in msg.pose if item.name == "hill_rover"), None)
    if pose is None:
        return
    q = pose.orientation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    with lock:
        world_poses.append((pose.position.x, pose.position.y, pose.position.z, yaw))
        counts["world_pose"] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-lidar", action="store_true", help="Do not require a rendering sensor")
    parser.add_argument("--world-pose-topic", default="/world/hill_country/dynamic_pose/info")
    args = parser.parse_args()
    node = Node()
    subscriptions = [
        node.subscribe(Odometry, "/model/hill_rover/odometry", odom_callback),
        node.subscribe(Pose_V, args.world_pose_topic, world_pose_callback),
        node.subscribe(IMU, "/model/hill_rover/imu", count("imu")),
        node.subscribe(NavSat, "/model/hill_rover/navsat", count("navsat")),
    ]
    if args.no_lidar:
        counts.pop("lidar")
    else:
        subscriptions.append(node.subscribe(LaserScan, "/model/hill_rover/lidar", count("lidar")))
    if not all(subscriptions):
        raise SystemExit("One or more Gazebo topic subscriptions failed")
    deadline = time.monotonic() + 12.0
    while time.monotonic() < deadline:
        time.sleep(0.1)
    with lock:
        received = counts.copy()
        samples = poses.copy()
        ground_truth = world_poses.copy()
    if not samples or not ground_truth or any(received[name] == 0 for name in received):
        raise SystemExit(f"Missing runtime data: {received}")
    distance = math.hypot(
        ground_truth[-1][0] - ground_truth[0][0], ground_truth[-1][1] - ground_truth[0][1]
    )
    if distance < 0.25:
        raise SystemExit(f"Rover did not move enough: {distance:.3f} m; messages={received}")
    dx = ground_truth[-1][0] - ground_truth[0][0]
    dy = ground_truth[-1][1] - ground_truth[0][1]
    forward = dx * math.cos(ground_truth[0][3]) + dy * math.sin(ground_truth[0][3])
    if forward < 0.25:
        raise SystemExit(
            f"Rover moved in the wrong direction: forward={forward:.3f} m; messages={received}"
        )
    print(f"Runtime OK: moved {distance:.2f} m; forward={forward:.2f} m; messages={received}")


if __name__ == "__main__":
    main()
