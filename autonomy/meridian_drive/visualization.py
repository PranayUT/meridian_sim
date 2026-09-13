"""Gazebo GUI markers for the route and MPPI solution."""

from __future__ import annotations

import time
import threading

import numpy as np
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.marker_pb2 import Marker
from gz.msgs10.marker_v_pb2 import Marker_V
from gz.transport13 import Node

from .maps import TerrainMap


class GazeboMarkers:
    """Send persistent GUI-only markers through Gazebo's marker service."""

    def __init__(self, node: Node, terrain: TerrainMap | None, service: str) -> None:
        self.node = node
        self.terrain = terrain
        self.service = service
        self.route_sent = False
        self.reported_failure = False
        self.next_retry_s = 0.0
        self.next_rollout_s = 0.0
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.running = True
        self.latest: tuple[np.ndarray, np.ndarray, np.ndarray | None] | None = None
        self.thread = threading.Thread(target=self._worker, name="gazebo-markers", daemon=True)
        self.thread.start()

    def _marker(
        self,
        marker_id: int,
        marker_type: int,
        points_xy: np.ndarray,
        color: tuple[float, float, float, float],
        scale: float,
        z_offset: float,
    ) -> Marker:
        marker = Marker()
        marker.action = Marker.ADD_MODIFY
        marker.ns = "meridian_autonomy"
        marker.id = marker_id
        marker.type = marker_type
        marker.visibility = Marker.GUI
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.material.diffuse.r = color[0]
        marker.material.diffuse.g = color[1]
        marker.material.diffuse.b = color[2]
        marker.material.diffuse.a = color[3]
        marker.material.emissive.r = color[0]
        marker.material.emissive.g = color[1]
        marker.material.emissive.b = color[2]
        marker.material.emissive.a = color[3]
        xy = np.asarray(points_xy, dtype=np.float64)
        if self.terrain is None:
            heights = np.zeros(len(xy), dtype=np.float64)
        else:
            heights = self.terrain.elevation(xy[:, 0], xy[:, 1])
        for (x, y), height in zip(xy, heights):
            point = marker.point.add()
            point.x = float(x)
            point.y = float(y)
            point.z = float(height + z_offset)
        return marker

    def _send(self, markers: tuple[Marker, ...], timeout_ms: int = 100) -> bool:
        request = Marker_V()
        for marker in markers:
            request.marker.add().CopyFrom(marker)
        try:
            executed, response = self.node.request(
                self.service, request, Marker_V, Boolean, timeout_ms
            )
            return bool(executed and response.data)
        except Exception:
            return False

    def _publish(
        self,
        route_xy: np.ndarray,
        route_anchors: np.ndarray,
        best_trajectory: np.ndarray | None,
    ) -> None:
        now = time.monotonic()
        if not self.route_sent:
            if now < self.next_retry_s:
                return
            try:
                service_available = self.service in self.node.service_list()
            except Exception:
                service_available = False
            if not service_available:
                if not self.reported_failure:
                    print(f"Gazebo visualization is waiting for marker service {self.service}.", flush=True)
                    self.reported_failure = True
                self.next_retry_s = now + 2.0
                return
            # Gazebo Harmonic applies LINE_STRIP scale to point coordinates;
            # keep it at one so world positions are unchanged.
            # The imported GPS route is piecewise linear, so its source fixes
            # produce the same curve without sending thousands of dense MPPI
            # association samples through the GUI service.
            route_line = self._marker(1, Marker.LINE_STRIP, route_anchors, (0.1, 1.0, 0.25, 1.0), 1.0, 0.30)
            anchors = self._marker(2, Marker.POINTS, route_anchors, (1.0, 0.75, 0.05, 1.0), 0.45, 0.35)
            # Probe with the small anchor message first. A headless physics
            # server has no marker service, so it should never wait on the
            # larger route request.
            self.route_sent = self._send((anchors, route_line), 500)
            if not self.route_sent:
                if not self.reported_failure:
                    print(f"Gazebo visualization is waiting for marker service {self.service}.", flush=True)
                    self.reported_failure = True
                self.next_retry_s = now + 2.0
                return
            print("Gazebo route and MPPI visualization active.", flush=True)
        if best_trajectory is None or now < self.next_rollout_s:
            return
        xy = np.asarray(best_trajectory[:, :2], dtype=np.float64)
        rollout_line = self._marker(3, Marker.LINE_STRIP, xy, (1.0, 0.05, 0.8, 1.0), 1.0, 0.40)
        rollout_points = self._marker(4, Marker.POINTS, xy, (1.0, 0.45, 0.95, 1.0), 0.25, 0.42)
        self._send((rollout_line, rollout_points), 100)
        self.next_rollout_s = now + 0.10

    def update(
        self,
        route_xy: np.ndarray,
        route_anchors: np.ndarray,
        best_trajectory: np.ndarray | None,
    ) -> None:
        """Replace the pending visualization frame without blocking control."""
        with self.lock:
            self.latest = (
                np.asarray(route_xy).copy(),
                np.asarray(route_anchors).copy(),
                None if best_trajectory is None else np.asarray(best_trajectory).copy(),
            )
        self.event.set()

    def _worker(self) -> None:
        while self.running:
            self.event.wait(0.1)
            self.event.clear()
            with self.lock:
                latest = self.latest
            if latest is not None:
                self._publish(*latest)

    def close(self) -> None:
        self.running = False
        self.event.set()
        self.thread.join(timeout=1.0)
