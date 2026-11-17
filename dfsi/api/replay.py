"""Replay engine — streams a recorded scenario past the forecaster.

A replay walks a stored scenario's 10-second windows in order, scoring each
instant as it goes. It is the demo's clock: the UI's timeline, risk gauge and
evidence panel are all driven by the events this produces.

The replay only ever reads recorded data. There is no path from here to live
traffic, to Behemoth, or to Shared State.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from ..config import PATHS, STATE_ABOUT_TO_START, STATE_MONITORING_ONLY, Paths
from ..forecaster.alerting import AlertTracker
from ..forecaster.lifecycle import AlertLifecycle, LifecyclePolicy
from ..forecaster.serve import ForecasterService, RiskBands

log = logging.getLogger("dfsi.api.replay")

#: How many recent windows the agents may look back over.
EVIDENCE_LOOKBACK = 12


@dataclass(frozen=True)
class ReplayCatalogueEntry:
    replay_id: str
    scenario_type: str
    range_id: int
    has_onset: bool
    window_count: int

    def to_dict(self) -> dict:
        return {
            "replay_id": self.replay_id,
            "scenario_type": self.scenario_type,
            "range_id": self.range_id,
            "has_onset": self.has_onset,
            "window_count": self.window_count,
        }


def load_catalogue(paths: Paths = PATHS) -> list[ReplayCatalogueEntry]:
    path = paths.replay / "catalogue.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return [ReplayCatalogueEntry(**entry) for entry in raw]


def load_replay(replay_id: str, paths: Paths = PATHS) -> dict:
    """Load one stored replay, rejecting anything that is not a known id.

    `replay_id` arrives from an HTTP path parameter, so it is validated against
    the catalogue rather than interpolated into a path — otherwise
    `../../etc/passwd` would be a valid replay name.
    """
    known = {entry.replay_id for entry in load_catalogue(paths)}
    if replay_id not in known:
        raise KeyError(replay_id)
    path: Path = paths.replay / f"{replay_id}.json"
    if not path.exists():
        raise KeyError(replay_id)
    return json.loads(path.read_text())


@dataclass
class ReplayState:
    """Mutable progress of one replay, held per request."""

    replay_id: str
    scenario_type: str
    range_id: int
    instants: list[tuple[int, list[dict]]]
    cursor: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=EVIDENCE_LOOKBACK))
    first_warning_epoch: int | None = None
    onset_epoch: int | None = None
    bell_epoch: int | None = None
    #: Debounces the per-window verdict, created on the first step from the
    #: forecaster's own `min_consecutive` so the stream follows the same rule
    #: the evaluation was measured under.
    tracker: AlertTracker | None = None
    #: One alert that escalates and clears, built on the same signal as
    #: `tracker`. Additive: `dfsi_state` is unchanged and the lifecycle is
    #: reported alongside it.
    lifecycle: AlertLifecycle | None = None

    @property
    def total(self) -> int:
        return len(self.instants)

    @property
    def finished(self) -> bool:
        return self.cursor >= self.total


def build_state(replay: dict) -> ReplayState:
    """Group a replay's rows into per-instant buckets, ordered in time."""
    buckets: dict[int, list[dict]] = {}
    for row in replay["windows"]:
        buckets.setdefault(int(row["window_start_epoch"]), []).append(row)
    instants = [(epoch, buckets[epoch]) for epoch in sorted(buckets)]
    return ReplayState(
        replay_id=replay["replay_id"],
        scenario_type=replay["scenario_type"],
        range_id=int(replay["range_id"]),
        instants=instants,
    )


def step(state: ReplayState, forecaster: ForecasterService, threshold: float | None = None) -> dict:
    """Advance the replay by one 10-second window and return the event.

    `threshold` is the range's decision threshold from raw.json (`None`: the
    model's).
    """
    if state.finished:
        raise StopIteration
    epoch, rows = state.instants[state.cursor]
    state.cursor += 1

    forecast = forecaster.score_window(rows, threshold)
    state.history.append(rows)

    # A single window above threshold is not a warning: it takes
    # `min_consecutive` in a row, exactly as in the evaluation behind the
    # published head start and false-alarm figures. A window the forecaster
    # could not score is never "above", so it breaks the run.
    if state.tracker is None:
        state.tracker = AlertTracker(forecaster.min_consecutive)
    above = forecast.predicted_state == STATE_ABOUT_TO_START
    warned = state.tracker.update(above)

    # The lifecycle raises on the same debounce, so its first WARNING is the
    # window `first_warning_epoch` below records. It scores the risk itself, not
    # the verdict: escalation and hysteresis need the number, and `None` marks a
    # window that could not be scored.
    # Its warn and critical levels come from the range's threshold, so the
    # lifecycle and `warned` raise on the same window.
    if state.lifecycle is None:
        bands = forecaster.bands if threshold is None else RiskBands.for_threshold(threshold)
        state.lifecycle = AlertLifecycle(
            bands, LifecyclePolicy(raise_after=forecaster.min_consecutive)
        )
    # A window the evidence gate withheld has a high score but no traffic behind
    # it: it must not raise the alert that the per-window state has just declined.
    alert = state.lifecycle.update(epoch, forecast.risk_score, evidence=not forecast.gated)

    # Track when DFSI first warned and when the bell would actually ring, so
    # the UI can draw the lead-time gap that is the whole point of the project.
    if warned and state.first_warning_epoch is None:
        state.first_warning_epoch = epoch
    if any(int(row.get("is_onset", 0)) for row in rows) and state.onset_epoch is None:
        state.onset_epoch = epoch
    if any(int(row.get("declared", 0)) for row in rows) and state.bell_epoch is None:
        state.bell_epoch = epoch

    lead = (
        state.onset_epoch - state.first_warning_epoch
        if state.onset_epoch is not None and state.first_warning_epoch is not None
        else None
    )

    dfsi_state = (
        STATE_MONITORING_ONLY
        if forecast.degraded
        else (STATE_ABOUT_TO_START if warned else "QUIET")
    )
    # The gauge reads `forecast.predicted_state`, so it carries the debounced
    # state; the raw single-window verdict stays available as `above_threshold`.
    forecast_payload = forecast.to_dict()
    forecast_payload["above_threshold"] = above
    forecast_payload["predicted_state"] = dfsi_state

    aggregate = _aggregate(rows)
    return {
        "type": "window",
        "replay_id": state.replay_id,
        "scenario_type": state.scenario_type,
        "cursor": state.cursor,
        "total": state.total,
        "window_start_epoch": epoch,
        "traffic": aggregate,
        "forecast": forecast_payload,
        "shared_state": {
            # What the existing Java service would be showing right now.
            "over_threshold": bool(max(int(r.get("range_over_threshold", 0)) for r in rows)),
            "declared": bool(max(int(r.get("declared", 0)) for r in rows)),
            "bell_epoch": state.bell_epoch,
        },
        "dfsi_state": dfsi_state,
        "alert": alert.to_dict(),
        "first_warning_epoch": state.first_warning_epoch,
        "onset_epoch": state.onset_epoch,
        "lead_seconds": lead,
    }


def alert_summary(state: ReplayState) -> list[dict]:
    """Every alert raised so far in this replay; empty if none or not yet stepped."""
    return state.lifecycle.summary() if state.lifecycle is not None else []


def _aggregate(rows: list[dict]) -> dict:
    """Range-level traffic summary for the timeline chart."""
    total_pps = sum(float(r.get("total_pps", 0.0)) for r in rows)
    blocked_pps = sum(float(r.get("blocked_pps", 0.0)) for r in rows)
    policy_pps = sum(float(r.get("policy_pps", 0.0)) for r in rows)
    strongest = max(rows, key=lambda r: float(r.get("baseline_deviation_pps", 0.0)))
    return {
        "total_pps": round(total_pps, 1),
        "total_mbps": round(sum(float(r.get("total_mbps", 0.0)) for r in rows), 2),
        "blocked_pps": round(blocked_pps, 1),
        "policy_pps": round(policy_pps, 1),
        "blocked_share": round(blocked_pps / total_pps, 4) if total_pps else 0.0,
        "policy_utilisation": round(total_pps / policy_pps, 4) if policy_pps else 0.0,
        "dominant_traffic_type": strongest.get("traffic_type"),
        "baseline_deviation_pps": round(float(strongest.get("baseline_deviation_pps", 0.0)), 2),
        "country_hhi": round(float(strongest.get("country_hhi", 0.0)), 4),
        "per_traffic_type": [
            {
                "traffic_type": r.get("traffic_type"),
                "total_pps": round(float(r.get("total_pps", 0.0)), 1),
                "blocked_pps": round(float(r.get("blocked_pps", 0.0)), 1),
                "baseline_deviation_pps": round(float(r.get("baseline_deviation_pps", 0.0)), 2),
            }
            for r in rows
        ],
    }


def run_stats(state: ReplayState) -> dict:
    """What the wall and the bell did over the windows streamed so far.

    Lets two runs of the same attack under different policies be compared:
    how much was dropped at the wall, the most that still reached the range,
    and when (relative to the episode's start) the bell rang. The wall and
    bell figures cover the whole episode — the stream stops shortly after the
    bell, and a run whose bell rings earlier would otherwise cover less of the
    attack. The warning is DFSI's, so it is as far as the stream got.
    """
    if not state.instants:
        return {}
    start = state.instants[0][0]
    window = state.instants[1][0] - start if len(state.instants) > 1 else 10
    total = blocked = 0.0
    peak_passed = 0.0
    declared_windows = 0
    first_declared: int | None = None
    for epoch, rows in state.instants:
        window_total = sum(float(r.get("total_pps", 0.0)) for r in rows)
        window_blocked = sum(float(r.get("blocked_pps", 0.0)) for r in rows)
        total += window_total
        blocked += window_blocked
        peak_passed = max(peak_passed, window_total - window_blocked)
        if any(int(r.get("declared", 0)) for r in rows):
            declared_windows += 1
            if first_declared is None:
                first_declared = epoch
    return {
        "blocked_share_total": round(blocked / total, 4) if total else 0.0,
        "peak_passed_pps": round(peak_passed, 1),
        "seconds_declared": declared_windows * window,
        "bell_after_seconds": (first_declared - start) if first_declared is not None else None,
        "warning_after_seconds": (
            state.first_warning_epoch - start if state.first_warning_epoch is not None else None
        ),
    }
