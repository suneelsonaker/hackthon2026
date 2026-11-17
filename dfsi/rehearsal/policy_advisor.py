"""Policy advisor — concrete threshold suggestions backed by a week of traffic.

The rehearsal room's "tighten the vector threshold" option used to be a fixed
assumption: remove 55% of the attack, lose 2% of legitimate traffic, on every
range and every attack. This module replaces that with numbers measured on the
range's own history:

1. Read the previous `HISTORY_DAYS` days of recorded windows for the range,
   ending where the incident's episode begins. Fewer than a full week of data
   means no suggestion at all — a threshold learned from a partial week has not
   seen a weekend, or a weekly peak.
2. Drop every hour in which Shared State's start threshold was crossed.
   Incidents are not normal behaviour, and an envelope learned from them would
   wave the next attack through.
3. Measure each traffic type's legitimate envelope — a high quantile of what
   it carried in *clean* hours, where the policy dropped nothing on that type —
   and suggest a policy just above it, never looser than today's.
4. Backtest the suggestion over every non-incident hour of that week,
   including the ones that brushed the policy line (big flash crowds, small
   attacks that never rang the bell): how many windows, and what share of
   packets, it would have dropped.
5. Pick the suspect vector(s): types running well above their normal for
   this hour of the day. Only those are changed — and a suspect that cannot be
   tightened without dropping last week's traffic is reported, not forced.

The rehearsal room then simulates the suggestion on the incident itself (see
`simulator.simulate_threshold_plan`).

Read-only. It reads recorded windows and returns numbers; it never writes a
policy. It also never reads the generator's ground-truth columns (scenario
type, attack vector) — a production advisor would not have them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from ..config import PATHS, WINDOW_SECONDS, SharedStateThresholds, thresholds_for

#: The suggestion must be backed by at least this much history.
HISTORY_DAYS = 7
#: Share of the week's 10-second windows that must actually be present.
MIN_COVERAGE = 0.95
#: The legitimate envelope: this quantile of the week's quiet traffic.
ENVELOPE_QUANTILE = 0.999
#: Headroom above the envelope, so ordinary growth does not hit the new line.
SAFETY_MARGIN = 1.20
#: Only suggest a change that tightens the policy by at least this much.
MIN_CHANGE = 0.05
#: A traffic type running at this multiple of its normal for the hour is a
#: suspect vector.
SUSPECT_RATIO = 1.5
#: If nothing reaches SUSPECT_RATIO, the most elevated type is the suspect
#: only if it is at least this far above normal.
WEAK_SUSPECT_RATIO = 1.1

#: The only columns this module reads. Observable in production — no labels.
HISTORY_COLUMNS: tuple[str, ...] = (
    "range_id",
    "window_start_epoch",
    "traffic_type",
    "total_pps",
    "total_mbps",
    "blocked_pps",
    "blocked_mbps",
    "policy_pps",
    "policy_mbps",
)

_WINDOWS_PER_DAY = 86_400 // WINDOW_SECONDS


@dataclass
class TypeSuggestion:
    """The suggestion for one traffic type on the range."""

    traffic_type: str
    current_policy_pps: float
    current_policy_mbps: float
    legit_median_pps: float
    legit_envelope_pps: float
    legit_max_pps: float
    suggested_policy_pps: float
    suggested_policy_mbps: float
    #: Relative change to the policy, e.g. -0.62 for 62% tighter. 0 if kept.
    change_pct: float
    #: The suggestion actually tightens this type's policy.
    changes: bool
    #: Traffic now, as a multiple of this type's past-week median for the hour.
    ratio_to_normal_now: float
    #: Running well above normal right now: a suspect vector.
    suspect: bool
    #: Included in the simulated change (suspect and can be tightened).
    apply: bool
    #: Over the week's non-incident hours, at the suggested line.
    backtest_windows_blocked: int
    backtest_hours_affected: int
    backtest_legit_lost_pct: float
    note: str | None = None


@dataclass
class PolicyPlan:
    """Per-type threshold suggestions for one range at one decision point."""

    available: bool
    range_id: int
    reason: str | None = None
    history_start_epoch: int | None = None
    history_end_epoch: int | None = None
    history_days: float = 0.0
    coverage: float = 0.0
    excluded_incident_hours: int = 0
    per_type: list[TypeSuggestion] = field(default_factory=list)
    #: (traffic_type, is_weekend, hour_of_day) -> median legitimate pps, used
    #: by the simulator to split incident traffic. Not part of the payload.
    _reference: dict[tuple[str, bool, int], float] = field(default_factory=dict, repr=False)

    @property
    def applied_thresholds(self) -> dict[str, float]:
        """Traffic type -> suggested policy pps, for the types the change covers."""
        return {s.traffic_type: s.suggested_policy_pps for s in self.per_type if s.apply}

    def legit_reference(self, traffic_type: str, epoch: int) -> float | None:
        """Median legitimate pps on this type at the same hour, over the past week."""
        return self._reference.get((traffic_type, _is_weekend(epoch), _hour(epoch)))

    def describe_change(self) -> str:
        """One line per applied type, plain integers so agents can quote them."""
        parts = [
            f"{s.traffic_type} policy {s.current_policy_pps:.0f} -> {s.suggested_policy_pps:.0f} pps"
            for s in self.per_type
            if s.apply
        ]
        return "; ".join(parts) if parts else "no change"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.pop("_reference", None)
        payload["history_required_days"] = HISTORY_DAYS
        payload["method"] = {
            "envelope_quantile": ENVELOPE_QUANTILE,
            "safety_margin": SAFETY_MARGIN,
            "min_change": MIN_CHANGE,
            "excludes": "every hour in which the start threshold was crossed",
        }
        payload["change"] = self.describe_change()
        return payload


def _hour(epoch):
    """UTC hour of day; works on an int or an array."""
    return (epoch % 86_400) // 3_600


def _is_weekend(epoch):
    """Saturday or Sunday (UTC). Day 0 of the Unix epoch was a Thursday."""
    return (epoch // 86_400 + 3) % 7 >= 5


def _nice_ceil(value: float) -> float:
    """Round up to two significant figures: 21_437 -> 22_000."""
    if value <= 0:
        return 0.0
    scale = 10 ** (math.floor(math.log10(value)) - 1)
    return float(math.ceil(value / scale) * scale)


@lru_cache(maxsize=64)
def _read_history(range_id: int, start_epoch: int, end_epoch: int) -> pd.DataFrame:
    return pd.read_parquet(
        PATHS.raw_windows,
        columns=list(HISTORY_COLUMNS),
        filters=[
            ("range_id", "==", int(range_id)),
            ("window_start_epoch", ">=", int(start_epoch)),
            ("window_start_epoch", "<", int(end_epoch)),
        ],
    )


def load_history(range_id: int, end_epoch: int, days: int = HISTORY_DAYS) -> pd.DataFrame:
    """The range's recorded windows for the `days` before `end_epoch`."""
    if not PATHS.raw_windows.exists():
        return pd.DataFrame(columns=list(HISTORY_COLUMNS))
    return _read_history(int(range_id), int(end_epoch - days * 86_400), int(end_epoch))


def _incident_hours(history: pd.DataFrame, thresholds: SharedStateThresholds) -> set[int]:
    """Hours (epoch // 3600) in which any traffic type crossed the start threshold."""
    over = (
        (history["blocked_pps"] >= history["policy_pps"] * thresholds.start_threshold_ratio)
        & (history["blocked_pps"] >= thresholds.start_floor_kpps * 1_000.0)
        & (history["blocked_mbps"] >= thresholds.start_floor_mbps)
    )
    return set((history.loc[over, "window_start_epoch"] // 3_600).astype(int).tolist())


def suggest_policy(
    range_id: int,
    episode_start_epoch: int,
    current_rows: Sequence[dict],
    environment: str | None = None,
    history: pd.DataFrame | None = None,
) -> PolicyPlan:
    """Build the per-type suggestion for a range at its decision window.

    `current_rows` are the range's rows at the decision window (one per
    traffic type); they supply today's policy and the traffic right now.
    `history` may be passed directly (tests); otherwise it is read from the
    recorded windows that precede the episode.
    """
    thresholds = thresholds_for(environment)
    if history is None:
        history = load_history(range_id, episode_start_epoch)
    history = history[history["window_start_epoch"] < episode_start_epoch]

    expected = HISTORY_DAYS * _WINDOWS_PER_DAY
    present = int(history["window_start_epoch"].nunique()) if len(history) else 0
    coverage = present / expected
    if coverage < MIN_COVERAGE:
        days = present / _WINDOWS_PER_DAY
        return PolicyPlan(
            available=False,
            range_id=int(range_id),
            reason=(
                f"only {days:.1f} days of recorded traffic precede this incident; "
                f"a threshold suggestion needs {HISTORY_DAYS}"
            ),
            history_days=round(days, 2),
            coverage=round(coverage, 4),
        )

    incidents = _incident_hours(history, thresholds)
    hours = (history["window_start_epoch"] // 3_600).astype(int)
    non_incident = history[~hours.isin(incidents)]

    by_type_now = {str(row["traffic_type"]): row for row in current_rows}
    suggestions: list[TypeSuggestion] = []
    reference: dict[tuple[str, bool, int], float] = {}

    for traffic_type, group in sorted(non_incident.groupby("traffic_type")):
        now = by_type_now.get(traffic_type)
        if now is None:
            continue
        group_hours = (group["window_start_epoch"] // 3_600).astype(int)
        # Clean hours: the policy dropped nothing on this type. The envelope
        # and the hourly reference are learned from these only.
        dirty = set(group_hours[group["blocked_pps"] > 0].tolist())
        clean = group[~group_hours.isin(dirty)]
        if clean.empty:
            clean = group
        clean_pps = clean["total_pps"].to_numpy(dtype=float)

        epochs = clean["window_start_epoch"].to_numpy()
        keys = pd.DataFrame(
            {"weekend": _is_weekend(epochs), "hour": _hour(epochs), "pps": clean_pps}
        )
        for (weekend, hour), median in keys.groupby(["weekend", "hour"])["pps"].median().items():
            reference[(traffic_type, bool(weekend), int(hour))] = float(median)

        current_pps = float(now["policy_pps"])
        current_mbps = float(now["policy_mbps"])
        envelope = float(np.quantile(clean_pps, ENVELOPE_QUANTILE))
        candidate = _nice_ceil(envelope * SAFETY_MARGIN)
        change = candidate / current_pps - 1.0 if current_pps > 0 else 0.0
        changes = change <= -MIN_CHANGE
        suggested = candidate if changes else current_pps
        scale = suggested / current_pps if current_pps > 0 else 1.0

        # Backtest over every non-incident window, clean or not: the cost of
        # the new line includes the hours that brushed today's.
        all_pps = group["total_pps"].to_numpy(dtype=float)
        extra = np.clip(all_pps - suggested, 0.0, None) - np.clip(all_pps - current_pps, 0.0, None)
        extra = np.clip(extra, 0.0, None)
        hit = extra > 0
        hit_hours = set(group_hours[hit].tolist())

        normal_now = reference.get(
            (
                traffic_type,
                _is_weekend(int(now["window_start_epoch"])),
                int(_hour(int(now["window_start_epoch"]))),
            )
        )
        ratio_now = float(now["total_pps"]) / normal_now if normal_now else 1.0

        suggestions.append(
            TypeSuggestion(
                traffic_type=traffic_type,
                current_policy_pps=round(current_pps, 1),
                current_policy_mbps=round(current_mbps, 2),
                legit_median_pps=round(float(np.median(clean_pps)), 1),
                legit_envelope_pps=round(envelope, 1),
                legit_max_pps=round(float(all_pps.max()), 1),
                suggested_policy_pps=round(suggested, 1),
                suggested_policy_mbps=round(current_mbps * scale, 2),
                change_pct=round(change if changes else 0.0, 4),
                changes=bool(changes),
                ratio_to_normal_now=round(ratio_now, 2),
                suspect=False,
                apply=False,
                backtest_windows_blocked=int(hit.sum()),
                backtest_hours_affected=len(hit_hours),
                backtest_legit_lost_pct=round(100.0 * extra.sum() / all_pps.sum(), 4)
                if all_pps.sum() > 0
                else 0.0,
            )
        )

    # Suspect vectors: well above their normal for this hour. If none is
    # clearly elevated yet, the most elevated one — provided it is above normal.
    suspects = [s for s in suggestions if s.ratio_to_normal_now >= SUSPECT_RATIO]
    if not suspects and suggestions:
        top = max(suggestions, key=lambda s: s.ratio_to_normal_now)
        if top.ratio_to_normal_now >= WEAK_SUSPECT_RATIO:
            suspects = [top]
    for s in suspects:
        s.suspect = True
        if s.changes:
            s.apply = True
        else:
            s.note = (
                f"suspect vector, but last week's clean {s.traffic_type} traffic peaked at "
                f"{s.legit_envelope_pps:.0f} pps; a line with {SAFETY_MARGIN:g}x headroom "
                f"would not be tighter than today's {s.current_policy_pps:.0f} pps"
            )

    return PolicyPlan(
        available=True,
        range_id=int(range_id),
        history_start_epoch=int(history["window_start_epoch"].min()),
        history_end_epoch=int(history["window_start_epoch"].max()) + WINDOW_SECONDS,
        history_days=round(present / _WINDOWS_PER_DAY, 2),
        coverage=round(coverage, 4),
        excluded_incident_hours=len(incidents),
        per_type=suggestions,
        _reference=reference,
    )
