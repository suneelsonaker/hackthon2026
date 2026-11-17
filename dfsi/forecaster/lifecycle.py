"""Alert lifecycle — one alert that escalates and clears, not a flickering light.

`alerting.AlertTracker` answers "is a warning raised right now?" window by
window. Fed a score that hovers around the decision threshold it flips on and
off every few windows, and an operator who sees a warning appear, vanish and
reappear learns to stop looking. This turns the same signal into the lifecycle
a human actually experiences:

    QUIET -> WATCH -> WARNING -> CRITICAL -> COOLING -> QUIET
                         ^------------------------|   (a breach while cooling
                                                      re-opens the same alert)

* **raise** — `raise_after` consecutive windows at or above the warn line. This
  is deliberately the debounce the evaluation was measured under, so the first
  WARNING lands on the very window the published head start is credited to.
* **escalate** — `escalate_after` consecutive windows at or above the critical
  line; **de-escalate** after `deescalate_after` windows clearly below it.
* **hysteresis** — once open, an alert is held until the score falls below
  `clear_ratio` x the warn line, not merely below it. Ordinary noise around the
  threshold no longer closes and reopens it.
* **cooldown** — after `clear_after` windows below that line the alert is
  COOLING for `cooldown_windows` more. A breach in that time **re-opens the same
  alert** (same id) instead of raising a new one, so one incident is one alert.

The evidence gate (`gate`) applies here too. A window whose score is above the
line but whose traffic gives no support (`evidence=False`) is a breach nobody
can corroborate, so it can neither raise, escalate nor re-open an alert. It does
not clear one either: an open alert is held by its score like any other window.
Without this the lifecycle would open alerts on exactly the windows the gate
withholds, and the dashboard would say "quiet" while an alert is open.

A window the forecaster could not score is *unknown*, and unknown never means
all-clear. It breaks a run toward raising or escalating, because a gap cannot
confirm consecutive breaches (the same rule as `AlertTracker`), but it never
advances clearing: an open alert is not cleared, and a cooldown does not run
down, on missing data, because a gap cannot confirm recovery either.

Pure logic, no I/O; one instance per range.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from .serve import RiskBands


class AlertState(StrEnum):
    QUIET = "QUIET"
    #: Elevated but no alert is open: informational, nothing to acknowledge.
    WATCH = "WATCH"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    #: The alert has cleared and is being held open for a while, so a relapse
    #: re-opens it instead of raising a fresh one.
    COOLING = "COOLING"


OPEN_STATES = frozenset({AlertState.WARNING, AlertState.CRITICAL, AlertState.COOLING})


@dataclass(frozen=True)
class LifecyclePolicy:
    """The timing of the lifecycle, in whole 10-second windows."""

    #: Consecutive windows at or above warn to raise. Keep equal to the model's
    #: `min_consecutive` so the lifecycle and the evaluation agree on when.
    raise_after: int = 2
    #: Consecutive windows at or above critical to escalate.
    escalate_after: int = 2
    #: Windows clearly below critical to step back down to WARNING.
    deescalate_after: int = 3
    #: An open alert is held while the score stays at or above this share of the
    #: warn line; below it counts toward clearing.
    clear_ratio: float = 0.85
    #: Windows below that line (60s at the default) before the alert cools.
    clear_after: int = 6
    #: Windows spent COOLING before the alert closes (120s at the default).
    cooldown_windows: int = 12

    def __post_init__(self) -> None:
        for name in ("raise_after", "escalate_after", "deescalate_after", "clear_after"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        if self.cooldown_windows < 0:
            raise ValueError(f"cooldown_windows must not be negative, got {self.cooldown_windows}")
        if not 0.0 < self.clear_ratio <= 1.0:
            raise ValueError(f"clear_ratio must be in (0, 1], got {self.clear_ratio}")


@dataclass(frozen=True)
class AlertStatus:
    """The lifecycle after one window."""

    state: AlertState
    #: Which alert is open, or `None`. Stable across escalation, cooling and relapse.
    alert_seq: int | None
    #: What changed on this window: raised / escalated / deescalated / cooling /
    #: reopened / cleared, or `None`.
    event: str | None
    opened_epoch: int | None
    peak_score: float | None
    #: Consecutive windows spent in `state`.
    windows_in_state: int
    #: This window could not be scored; nothing was advanced.
    degraded: bool = False

    @property
    def is_open(self) -> bool:
        return self.state in OPEN_STATES

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "alert_seq": self.alert_seq,
            "event": self.event,
            "opened_epoch": self.opened_epoch,
            "peak_score": None if self.peak_score is None else round(self.peak_score, 4),
            "windows_in_state": self.windows_in_state,
            "is_open": self.is_open,
            "degraded": self.degraded,
        }


class AlertLifecycle:
    """The lifecycle of one range's alerts. Feed it one window at a time."""

    def __init__(self, bands: RiskBands, policy: LifecyclePolicy | None = None) -> None:
        self.bands = bands
        self.policy = policy or LifecyclePolicy()
        self.state = AlertState.QUIET
        self._seq = 0
        self._windows_in_state = 0
        self._warn_run = 0
        self._crit_run = 0
        self._down_run = 0
        self._clear_run = 0
        self._cooldown_left = 0
        self._alerts: list[dict] = []

    # ------------------------------------------------------------------ #
    @property
    def is_open(self) -> bool:
        return self.state in OPEN_STATES

    @property
    def _current(self) -> dict | None:
        return self._alerts[-1] if self.is_open else None

    def update(self, epoch: int, score: float | None, evidence: bool = True) -> AlertStatus:
        """Advance one window. `score=None` means the window could not be scored.

        `evidence=False` marks a score above threshold that the traffic does not
        back (the evidence gate withheld it): it cannot raise, escalate or
        re-open an alert, and it does not count toward clearing one either.
        """
        if score is None or not math.isfinite(score):
            # Breaks a run toward raising/escalating; never advances clearing.
            self._warn_run = self._crit_run = 0
            return self._status(None, degraded=True)

        bands, policy = self.bands, self.policy
        above_warn = evidence and score >= bands.warn
        above_crit = evidence and score >= bands.critical
        holds = score >= bands.warn * policy.clear_ratio
        event: str | None = None
        previous = self.state
        current = self._current
        report_on = current  # the alert this window's status describes
        if current is not None:
            current["peak_score"] = max(current["peak_score"], score)

        if self.state in (AlertState.QUIET, AlertState.WATCH):
            self._warn_run = self._warn_run + 1 if above_warn else 0
            self._crit_run = self._crit_run + 1 if above_crit else 0
            if self._warn_run >= policy.raise_after:
                self._open(epoch, score)
                event = "raised"
            else:
                self.state = AlertState.WATCH if score >= bands.watch else AlertState.QUIET

        elif self.state is AlertState.WARNING:
            self._crit_run = self._crit_run + 1 if above_crit else 0
            self._clear_run = 0 if holds else self._clear_run + 1
            if self._crit_run >= policy.escalate_after:
                self._escalate()
                event = "escalated"
            elif self._clear_run >= policy.clear_after:
                self._cool()
                event = "cooling"

        elif self.state is AlertState.CRITICAL:
            self._down_run = (
                self._down_run + 1 if score < bands.critical * policy.clear_ratio else 0
            )
            self._clear_run = 0 if holds else self._clear_run + 1
            if self._clear_run >= policy.clear_after:
                self._cool()
                event = "cooling"
            elif self._down_run >= policy.deescalate_after:
                self.state = AlertState.WARNING
                self._crit_run = 0
                self._down_run = 0
                event = "deescalated"

        else:  # COOLING
            self._warn_run = self._warn_run + 1 if above_warn else 0
            self._crit_run = self._crit_run + 1 if above_crit else 0
            if self._warn_run >= policy.raise_after:
                self._reopen()
                event = "reopened"
            else:
                self._cooldown_left -= 1
                # Not while a breach is already building: closing now would throw
                # away its first window and delay a relapse's re-open by one.
                if self._cooldown_left <= 0 and self._warn_run == 0:
                    self._close(epoch)
                    self.state = AlertState.WATCH if score >= bands.watch else AlertState.QUIET
                    event = "cleared"

        if self.state is previous:
            self._windows_in_state += 1
        else:
            self._windows_in_state = 1
        # The window that closes an alert still reports which alert it closed.
        return self._status(event, record=report_on if event == "cleared" else None)

    # ----------------------------------------------------------- transitions
    def _open(self, epoch: int, score: float) -> None:
        self._seq += 1
        self._alerts.append(
            {
                "alert_seq": self._seq,
                "opened_epoch": epoch,
                "closed_epoch": None,
                "peak_score": score,
                "peak_state": AlertState.WARNING.value,
                "escalations": 0,
                "reopened": 0,
            }
        )
        self.state = AlertState.WARNING
        self._clear_run = self._down_run = 0
        if self._crit_run >= self.policy.escalate_after:
            self._escalate(count=False)

    def _escalate(self, count: bool = True) -> None:
        self.state = AlertState.CRITICAL
        self._down_run = 0
        self._clear_run = 0
        record = self._alerts[-1]
        record["peak_state"] = AlertState.CRITICAL.value
        if count:
            record["escalations"] += 1

    def _cool(self) -> None:
        self.state = AlertState.COOLING
        self._cooldown_left = self.policy.cooldown_windows
        self._warn_run = self._crit_run = self._down_run = self._clear_run = 0

    def _reopen(self) -> None:
        self._alerts[-1]["reopened"] += 1
        self._clear_run = self._down_run = 0
        if self._crit_run >= self.policy.escalate_after:
            self._escalate(count=False)
        else:
            self.state = AlertState.WARNING

    def _close(self, epoch: int) -> None:
        self._alerts[-1]["closed_epoch"] = epoch
        self._warn_run = self._crit_run = self._down_run = self._clear_run = 0
        self._cooldown_left = 0

    # ---------------------------------------------------------------- output
    def _status(
        self, event: str | None, degraded: bool = False, record: dict | None = None
    ) -> AlertStatus:
        current = record or self._current
        return AlertStatus(
            state=self.state,
            alert_seq=current["alert_seq"] if current else None,
            event=event,
            opened_epoch=current["opened_epoch"] if current else None,
            peak_score=current["peak_score"] if current else None,
            windows_in_state=self._windows_in_state,
            degraded=degraded,
        )

    def summary(self) -> list[dict]:
        """Every alert so far, oldest first; the last may still be open."""
        out = []
        for index, record in enumerate(self._alerts):
            is_last = index == len(self._alerts) - 1
            out.append(
                {
                    **record,
                    "peak_score": round(record["peak_score"], 4),
                    "open": record["closed_epoch"] is None and is_last and self.is_open,
                }
            )
        return out


def run_lifecycle(
    scores: Sequence[float | None],
    bands: RiskBands,
    policy: LifecyclePolicy | None = None,
    start_epoch: int = 0,
    window_seconds: int = 10,
) -> list[AlertStatus]:
    """Feed a whole score series through a fresh lifecycle, for evaluation and tests."""
    lifecycle = AlertLifecycle(bands, policy)
    return [
        lifecycle.update(start_epoch + index * window_seconds, score)
        for index, score in enumerate(scores)
    ]
