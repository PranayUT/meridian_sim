"""File-based air-ground policy for the Gazebo research deployment."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .maps import MapFormatError, MapStack, load_uav_map

MODES = ("ground_only", "greedy_uav", "counterfactual_uav")


@dataclass
class AssistanceManager:
    mode: str
    map_path: Path
    request_path: Path
    status_path: Path
    map_stack: MapStack

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"assistance mode must be one of {', '.join(MODES)}")
        self.state = "driving"
        self.detail = f"{self.mode} policy is active"
        self._map_mtime_ns = -1
        self._request_id = ""
        self._request_count = 0
        self._region_counts: dict[tuple[int, int], int] = {}
        self._region_key: tuple[int, int] | None = None
        self._last_request_s = 0.0
        self._last_status_s = 0.0
        self.reload_map()

    @property
    def hold(self) -> bool:
        return self.mode == "counterfactual_uav" and self.state in {"waiting", "exhausted"}

    def reload_map(self) -> bool:
        # Ground-only trials must remain ground-only even if a UAV result from
        # an earlier campaign is still present at the configured path.
        if self.mode == "ground_only":
            self.map_stack.aerial = None
            return False
        try:
            stat = self.map_path.stat()
        except FileNotFoundError:
            return False
        if stat.st_mtime_ns == self._map_mtime_ns:
            return False
        try:
            candidate = load_uav_map(self.map_path)
        except MapFormatError as error:
            self.detail = str(error)
            return False
        self.map_stack.aerial = candidate
        self._map_mtime_ns = stat.st_mtime_ns
        if self.state == "waiting":
            self.state = "driving"
            self.detail = f"UAV map sequence {candidate.sequence} entered the planner"
        else:
            self.detail = f"UAV map sequence {candidate.sequence} loaded"
        return True

    def update(self, exposure: float, roi: tuple[float, float, float, float] | None) -> None:
        if self.reload_map():
            # The exposure came from the prior map. Let the next planner cycle
            # evaluate the new evidence before it can request another result.
            return
        now = time.monotonic()
        should_request = exposure >= 0.10 and roi is not None
        if self.mode == "ground_only":
            if should_request:
                self.detail = f"uncertainty exposure {exposure:.2f} recorded"
        elif self.mode == "greedy_uav":
            if should_request and now - self._last_request_s >= 10.0:
                self._write_request(roi, exposure, hold=False)
        elif should_request:
            assert roi is not None
            key = self._key(roi)
            if self.state == "exhausted" and key != self._region_key:
                self.state = "driving"
            if self.state == "driving" and now - self._last_request_s >= 2.0:
                count = self._region_counts.get(key, 0)
                if count >= 2:
                    self.state = "exhausted"
                    self.detail = "two UAV maps did not clear this region"
                else:
                    self.state = "waiting"
                    self._region_key = key
                    self._region_counts[key] = count + 1
                    if count:
                        x0, y0, x1, y1 = roi
                        roi = (x0 - 3.0, y0 - 3.0, x1 + 3.0, y1 + 3.0)
                    self._write_request(roi, exposure, hold=True)
        if now - self._last_status_s >= 0.5:
            self._write_status(exposure)
            self._last_status_s = now

    def _write_request(self, roi: tuple[float, float, float, float], exposure: float, hold: bool) -> None:
        self._request_id = uuid.uuid4().hex
        self._request_count += 1
        self._last_request_s = time.monotonic()
        x0, y0, x1, y1 = roi
        payload = {
            "version": 1,
            "request_id": self._request_id,
            "request_number": self._request_count,
            "frame": "world",
            "map_types": ["semantic_traversability", "canopy_obstacle", "canopy_height"],
            "roi_xy": [[x0, y1], [x1, y1], [x1, y0], [x0, y0]],
            "uncertainty_exposure": exposure,
            "hold_requested": hold,
            "result_path": str(self.map_path),
            "created_unix_s": time.time(),
        }
        self._atomic_json(self.request_path, payload)
        action = "Holding for" if hold else "Requested"
        self.detail = f"{action} UAV map {self._request_id}"

    def _write_status(self, exposure: float) -> None:
        aerial = self.map_stack.aerial
        payload = {
            "version": 1,
            "mode": self.mode,
            "state": self.state,
            "hold_requested": self.hold,
            "request_id": self._request_id,
            "request_count": self._request_count,
            "uncertainty_exposure": exposure,
            "map_loaded": aerial is not None,
            "map_sequence": aerial.sequence if aerial is not None else None,
            "detail": self.detail,
            "updated_unix_s": time.time(),
        }
        self._atomic_json(self.status_path, payload)

    @staticmethod
    def _key(roi: tuple[float, float, float, float]) -> tuple[int, int]:
        x0, y0, x1, y1 = roi
        return (round((x0 + x1) / 10.0), round((y0 + y1) / 10.0))

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
