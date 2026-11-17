"""Rehearsal room — simulate one option exactly the way the drill does.

`run_drill` simulates "tighten the vector threshold" from concrete per-type
thresholds learned from the past week when a plan is available, and from the
stated suppression/collateral assumption otherwise. Anything that re-runs
options (pricing a wait, sampling the assumptions) has to make the same choice,
or its "decide now" row would disagree with the drill sitting next to it.
"""

from __future__ import annotations

from collections.abc import Sequence

from .simulator import (
    PolicyCandidate,
    SimulationOutcome,
    simulate,
    simulate_threshold_plan,
)


def uses_plan(candidate: PolicyCandidate, policy_plan) -> bool:
    """True when this option rests on measured thresholds, not stated assumptions."""
    return (
        policy_plan is not None
        and getattr(policy_plan, "available", False)
        and candidate.key == "tighten_vector"
    )


def simulate_option(
    windows: Sequence[dict],
    decision_epoch: int,
    candidate: PolicyCandidate,
    environment: str | None = None,
    policy_plan=None,
) -> SimulationOutcome:
    """Replay `windows` under `candidate`, as `run_drill` would for that option."""
    if uses_plan(candidate, policy_plan):
        return simulate_threshold_plan(windows, decision_epoch, policy_plan, candidate, environment)
    return simulate(windows, decision_epoch, candidate, environment)
