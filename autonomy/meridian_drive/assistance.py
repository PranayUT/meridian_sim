"""File-based air-ground policy for the Gazebo research deployment."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .maps import MapFormatError, MapStack, load_uav_map

MODES = ("ground_only", "greedy_uav", "counterfactual_uav")


@dataclass
class AssistanceManager:
    mode: str
    map_path: Path
    request_path: Path
    status_path: Path
    map_stack: MapStack
    uncertainty_threshold: float = 0.20
    map_size_m: float = 25.0
    request_handler: Callable[[tuple[float, float, float, float], int, str], None] | None = None
    persistence_s: float = 2.0
    stop_speed_mps: float = 0.08
    stop_settle_s: float = 0.5
    fusion_settle_s: float = 1.0
    # Every interval below is a span of vehicle time, so it has to be measured
    # on the same clock the planner and the maturity gates use. Reading the
    # wall clock instead made these windows depend on the real-time factor and
    # on how loaded the machine was, which silently changed request behaviour
    # between a solo run and a campaign. Defaults to the wall clock for
    # callers that have no simulator.
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"assistance mode must be one of {', '.join(MODES)}")
        if not 0.0 <= self.uncertainty_threshold <= 1.0:
            raise ValueError("uncertainty threshold must be between 0 and 1")
        if self.map_size_m <= 0.0:
            raise ValueError("UAV map size must be positive")
        if min(self.persistence_s, self.stop_speed_mps, self.stop_settle_s, self.fusion_settle_s) < 0.0:
            raise ValueError("assistance timing and stop speed must be non-negative")
        self.state = "driving"
        self.detail = f"{self.mode} policy is active"
        self._map_mtime_ns = -1
        self._request_id = ""
        self._request_count = 0
        self._request_history: list[dict[str, object]] = []
        self._request_history_path = self.request_path.with_name(
            f"{self.request_path.stem}_history.json"
        )
        self._trigger_history: list[dict[str, object]] = []
        self._trigger_history_path = self.request_path.with_name(
            f"{self.request_path.stem}_trigger_history.json"
        )
        self._trigger_key: tuple[object, ...] | None = None
        self._region_counts: dict[tuple[str, str, int, int], int] = {}
        self._region_key: tuple[str, str, int, int] | None = None
        self._last_request_s = 0.0
        self._last_status_s = 0.0
        self._high_since: dict[str, float] = {}
        self._stopped_since: float | None = None
        self._fusion_started_s = 0.0
        self._pending_roi: tuple[float, float, float, float] | None = None
        self._pending_source = ""
        self._pending_map_type = ""
        self._pending_exposure = 0.0
        self._pending_decision_relevant = False
        self._pending_action_relevant = False
        self._pending_mobility_relevant = False
        self._pending_sim_time_s: float | None = None
        self._pending_position_xy: tuple[float, float] | None = None
        self._mobility_relevant = False
        self._action_relevant = False
        self._mobility_episode_served = False
        # Meridian Drive excludes aerial evidence older than the active trial.
        # Remember any pre-existing file so only a later atomic replacement is
        # admitted, whether this run uses the simulator or an external source.
        self.map_stack.clear_aerial()
        if self.mode != "ground_only":
            try:
                self._map_mtime_ns = self.map_path.stat().st_mtime_ns
            except FileNotFoundError:
                pass

    @property
    def hold(self) -> bool:
        return self.mode == "counterfactual_uav" and self.state != "driving"

    def reload_map(self) -> bool:
        # Ground-only trials must remain ground-only even if a UAV result from
        # an earlier campaign is still present at the configured path.
        if self.mode == "ground_only":
            self.map_stack.clear_aerial()
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
        self.map_stack.add_aerial(candidate)
        self._map_mtime_ns = stat.st_mtime_ns
        if self.state == "waiting":
            self.state = "fusing"
            self._fusion_started_s = self.clock()
            self.detail = f"UAV map sequence {candidate.sequence} is entering the planner"
        else:
            self.detail = f"UAV map sequence {candidate.sequence} loaded"
        return True

    def update(
        self,
        exposure: float,
        roi: tuple[float, float, float, float] | None,
        *,
        source: str = "",
        map_type: str = "",
        decision_relevant: bool = False,
        action_relevant: bool = False,
        speed_mps: float = 0.0,
        mobility_stalled: bool = False,
        sim_time_s: float | None = None,
        position_xy: tuple[float, float] | None = None,
    ) -> None:
        if self.reload_map():
            # The exposure came from the prior map. Let the next planner cycle
            # evaluate the new evidence before it can request another result.
            return
        now = self.clock()
        if self.state == "fusing":
            if now - self._fusion_started_s >= self.fusion_settle_s:
                self.state = "driving"
                self.detail = "aerial evidence entered the planner; ground autonomy resumed"
            else:
                self._write_status(exposure)
                return

        if self.state == "stopping":
            if abs(speed_mps) <= self.stop_speed_mps:
                if self._stopped_since is None:
                    self._stopped_since = now
                if now - self._stopped_since >= self.stop_settle_s:
                    assert self._pending_roi is not None
                    self.state = "requesting"
                    self._write_request(
                        self._pending_roi,
                        self._pending_exposure,
                        hold=True,
                        source=self._pending_source,
                        map_type=self._pending_map_type,
                        decision_relevant=self._pending_decision_relevant,
                        action_relevant=self._pending_action_relevant,
                        mobility_relevant=self._pending_mobility_relevant,
                        sim_time_s=self._pending_sim_time_s,
                        position_xy=self._pending_position_xy,
                    )
                    self.state = "waiting"
            else:
                self._stopped_since = None
            self._write_status(exposure)
            return

        # Sustained loss of mobility in a mature uncertain ROI is itself
        # decision-relevant. This includes both a physical wedge (the model
        # commands motion that does not happen) and planner deadlock (the
        # uncertain evidence makes the model command zero). Waiting for a high
        # population-wide exposure in either case can leave the UAV dormant
        # until the drag harness intervenes. The normal source persistence
        # below rejects transient acceleration, braking, and steering pauses.
        mobility_relevant = (
            roi is not None
            and bool(source)
            and mobility_stalled
            and not self._mobility_episode_served
        )
        if not mobility_stalled:
            self._mobility_episode_served = False
        self._mobility_relevant = mobility_relevant
        self._action_relevant = action_relevant
        should_request = (
            (
                decision_relevant
                or action_relevant
                or mobility_relevant
                or exposure >= self.uncertainty_threshold
            )
            and roi is not None
            and bool(source)
        )
        # The full Meridian evaluator clears persistence independently for
        # every source on every snapshot. This compact interface emits the
        # selected source, so a source switch must retire the prior timer.
        for prior_source in tuple(self._high_since):
            if prior_source != source:
                self._high_since.pop(prior_source, None)
        if should_request:
            self._high_since.setdefault(source, now)
        else:
            self._high_since.pop(source, None)
        persistent = should_request and now - self._high_since[source] >= self.persistence_s
        trigger_kind = (
            "mobility"
            if mobility_relevant
            else (
                "selected_trajectory"
                if action_relevant
                else ("counterfactual" if decision_relevant else "exposure")
            )
        )
        if persistent:
            assert roi is not None
            # One continuous condition is one trigger episode. ROI cell noise
            # must not turn it into a stream of nominally different events.
            trigger_key = (trigger_kind, source, map_type)
            if trigger_key != self._trigger_key:
                self._record_trigger(
                    roi,
                    exposure,
                    source,
                    map_type,
                    trigger_kind,
                    sim_time_s,
                    position_xy,
                )
                self._trigger_key = trigger_key
                if mobility_relevant:
                    self._mobility_episode_served = True
        elif not should_request:
            self._trigger_key = None
        if self.state == "exhausted" and not should_request:
            self.state = "driving"
            self._region_key = None
            self.detail = "uncertainty cleared; ground autonomy resumed"
        if self.mode == "ground_only":
            if should_request:
                self.detail = f"{source} uncertainty exposure {exposure:.2f} recorded"
        elif self.mode == "greedy_uav":
            if persistent and now - self._last_request_s >= 10.0:
                self._write_request(
                    roi,
                    exposure,
                    hold=False,
                    source=source,
                    map_type=map_type,
                    decision_relevant=decision_relevant,
                    action_relevant=action_relevant,
                    mobility_relevant=mobility_relevant,
                    sim_time_s=sim_time_s,
                    position_xy=position_xy,
                )
        elif persistent:
            assert roi is not None
            key = self._key(roi, source, map_type)
            if self.state == "exhausted" and key != self._region_key:
                self.state = "driving"
            if self.state == "driving" and now - self._last_request_s >= 2.0:
                count = self._region_counts.get(key, 0)
                if count >= 2:
                    self.state = "exhausted"
                    self.detail = "two UAV maps did not clear this region"
                else:
                    self.state = "stopping"
                    self._region_key = key
                    self._region_counts[key] = count + 1
                    self._pending_roi = roi
                    self._pending_source = source
                    self._pending_map_type = map_type
                    self._pending_exposure = exposure
                    self._pending_decision_relevant = decision_relevant
                    self._pending_action_relevant = action_relevant
                    self._pending_mobility_relevant = mobility_relevant
                    self._pending_sim_time_s = sim_time_s
                    self._pending_position_xy = position_xy
                    if mobility_relevant:
                        self._mobility_episode_served = True
                    self._stopped_since = None
                    reason = (
                        "commanded motion stalled in an uncertain ROI"
                        if mobility_relevant
                        else (
                            "selected trajectory crossed uncertain evidence"
                            if action_relevant
                            else "uncertainty crossed the rollout threshold"
                        )
                    )
                    self.detail = f"{reason}; stopping to ask for help"
        if now - self._last_status_s >= 0.5:
            self._write_status(exposure)
            self._last_status_s = now

    def _write_request(
        self,
        roi: tuple[float, float, float, float],
        exposure: float,
        hold: bool,
        source: str,
        map_type: str,
        decision_relevant: bool = False,
        action_relevant: bool = False,
        mobility_relevant: bool = False,
        sim_time_s: float | None = None,
        position_xy: tuple[float, float] | None = None,
    ) -> None:
        roi = self._fixed_roi(roi)
        self._request_id = uuid.uuid4().hex
        self._request_count += 1
        self._last_request_s = self.clock()
        x0, y0, x1, y1 = roi
        payload = {
            "version": 1,
            "request_id": self._request_id,
            "request_number": self._request_count,
            "frame": "world",
            "source": source,
            "map_types": [map_type],
            "roi_xy": [[x0, y1], [x1, y1], [x1, y0], [x0, y0]],
            "uncertainty_exposure": exposure,
            "decision_relevant": decision_relevant,
            "action_relevant": action_relevant,
            "mobility_relevant": mobility_relevant,
            "sim_time_s": sim_time_s,
            "vehicle_xy": list(position_xy) if position_xy is not None else None,
            "hold_requested": hold,
            "result_path": str(self.map_path),
            "created_unix_s": time.time(),
        }
        self._atomic_json(self.request_path, payload)
        self._request_history.append(payload)
        self._atomic_json(
            self._request_history_path,
            {"version": 1, "requests": self._request_history},
        )
        print(
            f"UAV request {self._request_count}: {source} exposure "
            f"{exposure:.3f} at ROI ({x0:.2f}, {y0:.2f})-({x1:.2f}, {y1:.2f})",
            flush=True,
        )
        if self.request_handler is not None:
            try:
                # The producer receives the named product so the response
                # answers the source that was raised and nothing else.
                self.request_handler(roi, self._request_count, map_type)
            except Exception as error:
                self.detail = f"simulated UAV failed: {error}"
                return
        action = "Holding for" if hold else "Requested"
        self.detail = f"{action} UAV map {self._request_id}"

    def _record_trigger(
        self,
        roi: tuple[float, float, float, float],
        exposure: float,
        source: str,
        map_type: str,
        trigger_kind: str,
        sim_time_s: float | None,
        position_xy: tuple[float, float] | None,
    ) -> None:
        fixed_roi = self._fixed_roi(roi)
        x0, y0, x1, y1 = fixed_roi
        payload: dict[str, object] = {
            "trigger_number": len(self._trigger_history) + 1,
            "trigger_kind": trigger_kind,
            "source": source,
            "map_type": map_type,
            "uncertainty_exposure": exposure,
            "sim_time_s": sim_time_s,
            "vehicle_xy": list(position_xy) if position_xy is not None else None,
            "roi_xy": [[x0, y1], [x1, y1], [x1, y0], [x0, y0]],
            "created_unix_s": time.time(),
        }
        self._trigger_history.append(payload)
        self._atomic_json(
            self._trigger_history_path,
            {"version": 1, "triggers": self._trigger_history},
        )

    def _write_status(self, exposure: float) -> None:
        aerial = self.map_stack.aerial
        payload = {
            "version": 1,
            "mode": self.mode,
            "state": self.state,
            "hold_requested": self.hold,
            "request_id": self._request_id,
            "request_count": self._request_count,
            "uncertainty_threshold": self.uncertainty_threshold,
            "map_size_m": self.map_size_m,
            "uncertainty_exposure": exposure,
            "action_relevant": self._action_relevant,
            "mobility_relevant": self._mobility_relevant,
            "map_loaded": aerial is not None,
            "map_sequence": aerial.sequence if aerial is not None else None,
            "detail": self.detail,
            "updated_unix_s": time.time(),
        }
        self._atomic_json(self.status_path, payload)

    @staticmethod
    def _key(
        roi: tuple[float, float, float, float], source: str, map_type: str
    ) -> tuple[str, str, int, int]:
        x0, y0, x1, y1 = roi
        return (
            source,
            map_type,
            round((x0 + x1) / 10.0),
            round((y0 + y1) / 10.0),
        )

    def _fixed_roi(
        self, roi: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = roi
        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0
        half = self.map_size_m / 2.0
        return center_x - half, center_y - half, center_x + half, center_y + half

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
