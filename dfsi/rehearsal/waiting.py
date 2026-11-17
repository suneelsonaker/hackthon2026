"""Rehearsal room — what does waiting cost?

"Hold — keep watching" is always on the menu, and the Advisor is told that doing
nothing for another minute is frequently right. Until now that was a sentence,
not a number. This puts a price on it: for each real option, what would
deciding 30, 60 or 90 seconds from now have cost, and what would it have saved.

Waiting is simulated, not guessed. The same option is replayed over the *same*
horizon, from the same decision instant, with its engagement delayed by the
wait. Holding the horizon fixed is what keeps the figures comparable: deciding
later and scoring only the windows after the later instant would shrink the
denominator and make waiting look better or worse for no real reason.

Both sides are reported, because both are real:

* **cost** — attack traffic that gets through while the policy is not yet in
  force, extra seconds over the start threshold, and a later containment;
* **saving** — legitimate traffic not dropped, because the policy is in force
  for less of the incident.

**"Time to contain" is the shakiest of those numbers, and the report says so.**
The simulator defines it as the first 30-second clear stretch after the incident
*first crosses* the threshold under that policy. Engage a policy before the
incident's first crossing and that crossing may never happen under it, so the
clock starts at a later relapse; engage it after, and the early crossing counts
and the clock starts at once. The result is not monotonic in engagement time: on
real replays a rate-limit delayed by 90s reported containment 610s *sooner*,
while still relapsing twenty times. Waiting cannot make containment earlier, so a
negative difference is reported as "not comparable" (`containment_comparable:
false`, no delta) instead of as a number. Seconds over the threshold is the
dependable cost and is always reported.

Nothing here decides anything or touches production; it is arithmetic on the
same recorded traffic and the same stated assumptions as the rest of the drill.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from ..config import WINDOW_SECONDS
from .options import simulate_option
from .simulator import CANDIDATES, PolicyCandidate, SimulationOutcome

#: Waits worth showing an operator who is deciding inside a two-minute horizon.
DEFAULT_WAITS: tuple[int, ...] = (30, 60, 90)

#: Longest wait accepted. Beyond this the question is no longer "wait and see".
MAX_WAIT_SECONDS = 300


def validate_waits(waits: Sequence[int]) -> tuple[int, ...]:
    """Sorted, unique waits that are whole windows. Raises `ValueError` otherwise."""
    cleaned: set[int] = set()
    for wait in waits:
        if isinstance(wait, bool) or not isinstance(wait, int):
            raise ValueError(f"wait must be a whole number of seconds, got {wait!r}")
        if wait <= 0 or wait > MAX_WAIT_SECONDS:
            raise ValueError(f"wait must be between 1 and {MAX_WAIT_SECONDS} seconds, got {wait}")
        if wait % WINDOW_SECONDS:
            raise ValueError(f"wait must be a multiple of the {WINDOW_SECONDS}s window, got {wait}")
        cleaned.add(wait)
    if not cleaned:
        raise ValueError("at least one wait is required")
    return tuple(sorted(cleaned))


def _outcome(
    windows: Sequence[dict],
    decision_epoch: int,
    candidate: PolicyCandidate,
    extra_delay: int,
    environment: str | None,
    policy_plan,
) -> SimulationOutcome:
    """The option, engaged `extra_delay` seconds later than it would have been."""
    delayed = replace(
        candidate, engagement_delay_seconds=candidate.engagement_delay_seconds + extra_delay
    )
    return simulate_option(windows, decision_epoch, delayed, environment, policy_plan)


def _row(
    wait: int,
    now: SimulationOutcome,
    later: SimulationOutcome,
    last_epoch: int,
    decision_epoch: int,
) -> dict:
    """One option, one wait: the outcome and how it differs from deciding now."""
    extra_to_contain: int | None = None
    containment_comparable = True
    if now.time_to_contain_seconds is not None and later.time_to_contain_seconds is not None:
        difference = later.time_to_contain_seconds - now.time_to_contain_seconds
        if difference < 0:
            # Waiting cannot make containment earlier: the two clocks started at
            # different crossings, so the difference means nothing (see module doc).
            containment_comparable = False
        else:
            extra_to_contain = difference
    attack_lost = round(now.attack_packets_blocked_pct - later.attack_packets_blocked_pct, 2)
    collateral_saved = round(now.legitimate_packets_lost_pct - later.legitimate_packets_lost_pct, 2)
    extra_over = later.seconds_over_threshold - now.seconds_over_threshold
    no_longer_contains = (
        now.time_to_contain_seconds is not None and later.time_to_contain_seconds is None
    )
    return {
        "wait_seconds": wait,
        "attack_packets_blocked_pct": later.attack_packets_blocked_pct,
        "legitimate_packets_lost_pct": later.legitimate_packets_lost_pct,
        "seconds_over_threshold": later.seconds_over_threshold,
        "time_to_contain_seconds": later.time_to_contain_seconds,
        # Cost of waiting.
        "extra_seconds_over_threshold": extra_over,
        "extra_seconds_to_contain": extra_to_contain,
        "containment_comparable": containment_comparable,
        "no_longer_contains": no_longer_contains,
        "attack_blocked_pct_points_lost": attack_lost,
        # What waiting saves.
        "collateral_pct_points_avoided": collateral_saved,
        # The option would only engage after the last recorded window, so within
        # the recording it never acts and every figure collapses to "no intervention".
        "beyond_recorded_traffic": decision_epoch + later.engagement_delay_seconds > last_epoch,
    }


def cost_of_waiting(
    windows: Sequence[dict],
    decision_epoch: int,
    waits: Sequence[int] = DEFAULT_WAITS,
    environment: str | None = None,
    policy_plan=None,
) -> dict:
    """Price waiting for every option except `hold`, against deciding right now.

    `windows` and `decision_epoch` are exactly what `run_drill` takes. Raises
    `ValueError` for invalid `waits`, or when no window sits at or after the
    decision (as `simulate` does).
    """
    waits = validate_waits(waits)
    last_epoch = max(int(row["window_start_epoch"]) for row in windows)

    hold = simulate_option(windows, decision_epoch, CANDIDATES[0], environment)
    nothing_to_contain = hold.seconds_over_threshold == 0

    options: dict[str, dict] = {}
    for candidate in CANDIDATES:
        if candidate.key == "hold":
            continue
        now = _outcome(windows, decision_epoch, candidate, 0, environment, policy_plan)
        rows = [
            _row(
                wait,
                now,
                _outcome(windows, decision_epoch, candidate, wait, environment, policy_plan),
                last_epoch,
                decision_epoch,
            )
            for wait in waits
        ]
        options[candidate.key] = {
            "policy_name": candidate.name,
            "decide_now": {
                "attack_packets_blocked_pct": now.attack_packets_blocked_pct,
                "legitimate_packets_lost_pct": now.legitimate_packets_lost_pct,
                "seconds_over_threshold": now.seconds_over_threshold,
                "time_to_contain_seconds": now.time_to_contain_seconds,
            },
            "waiting": rows,
        }

    notes = [
        "Each wait replays the same incident over the same horizon with the "
        "option's engagement delayed, so the figures are comparable with deciding now."
    ]
    if nothing_to_contain:
        notes.append(
            "Even with no intervention this never crosses the start threshold, "
            "so there is nothing to contain: waiting costs nothing here."
        )
    if any(not row["containment_comparable"] for o in options.values() for row in o["waiting"]):
        notes.append(
            "For some waits the time to contain is not comparable with deciding now: "
            "the incident crosses the threshold before the delayed policy engages, so "
            "the two containment clocks start at different crossings. Judge those "
            "rows by seconds over the threshold, which is always comparable."
        )
    return {
        "simulation_only": True,
        "decision_epoch": decision_epoch,
        "waits_seconds": list(waits),
        "incident_crosses_threshold": not nothing_to_contain,
        "hold_seconds_over_threshold": hold.seconds_over_threshold,
        "options": options,
        "notes": notes,
    }
