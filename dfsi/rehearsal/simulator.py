"""Rehearsal room — the practice drill.

Replays a scenario against four fixed candidate responses and reports the
trade-offs. Nothing here touches production: the simulator only re-derives
what *would* have been blocked or passed under each candidate policy, using
the already-recorded traffic. It never calls Behemoth, never writes Shared
State, and never emits a configuration change.

The four candidates are fixed by design. An open-ended policy search would be
both unexplainable and unsafe to put in front of an operator mid-incident; a
short fixed menu is something a human can actually choose between in the 120
seconds DFSI buys them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from ..config import WINDOW_SECONDS, thresholds_for


#: Effectiveness of each candidate against attack traffic, and its cost to
#: legitimate traffic. These are stated assumptions of the simulation, not
#: measurements — they are surfaced in the UI alongside every result so an
#: operator can see what the drill assumed.
@dataclass(frozen=True)
class PolicyCandidate:
    key: str
    name: str
    description: str
    #: Fraction of attack traffic this candidate removes once engaged.
    attack_suppression: float
    #: Fraction of legitimate traffic lost as collateral.
    legitimate_collateral: float
    #: Seconds between the decision and the policy taking effect.
    engagement_delay_seconds: int
    #: Operational cost signal shown to the human (not a currency).
    operational_cost: str
    #: Blast-radius note. PPNG-6760: a /24 block can catch Teams/Zoom traffic.
    blast_radius: str
    reversible_seconds: int


CANDIDATES: tuple[PolicyCandidate, ...] = (
    PolicyCandidate(
        key="hold",
        name="Hold — keep watching",
        description=(
            "Change nothing. Let the existing policy and the duration-based "
            "alarm take their course. The honest baseline for every drill."
        ),
        attack_suppression=0.0,
        legitimate_collateral=0.0,
        engagement_delay_seconds=0,
        operational_cost="none",
        blast_radius="none",
        reversible_seconds=0,
    ),
    PolicyCandidate(
        key="tighten_vector",
        name="Tighten the vector threshold",
        description=(
            "Lower the policy threshold on the suspect traffic type only, to "
            "just above its legitimate peak over the past week. Narrow and "
            "quick; it drops whatever exceeds the new line, attack or not."
        ),
        attack_suppression=0.55,
        legitimate_collateral=0.02,
        engagement_delay_seconds=10,
        operational_cost="low",
        blast_radius="one traffic type on one range",
        reversible_seconds=10,
    ),
    PolicyCandidate(
        key="rate_limit_sources",
        name="Rate-limit concentrated sources",
        description=(
            "Apply a per-source-region rate limit against the concentrated "
            "country mix. Effective against reflection, poor against a "
            "well-distributed flood or a genuine flash crowd."
        ),
        attack_suppression=0.78,
        legitimate_collateral=0.06,
        engagement_delay_seconds=20,
        operational_cost="medium",
        blast_radius="all traffic from the top source regions",
        reversible_seconds=30,
    ),
    PolicyCandidate(
        key="full_scrub",
        name="Divert the range to full scrubbing",
        description=(
            "Send the whole range through Behemoth scrubbing. Strongest "
            "suppression and the widest blast radius — PPNG-6760 shows a /24 "
            "diversion can also catch Teams and Zoom traffic."
        ),
        attack_suppression=0.95,
        legitimate_collateral=0.11,
        engagement_delay_seconds=30,
        operational_cost="high",
        blast_radius="every service on the range, including unrelated ones",
        reversible_seconds=120,
    ),
)

CANDIDATES_BY_KEY = {candidate.key: candidate for candidate in CANDIDATES}


@dataclass
class SimulationOutcome:
    """What one candidate would have achieved on this replay."""

    policy_key: str
    policy_name: str
    attack_packets_blocked_pct: float
    legitimate_packets_lost_pct: float
    peak_blocked_share: float
    seconds_over_threshold: int
    time_to_contain_seconds: int | None
    operational_cost: str
    blast_radius: str
    reversible_seconds: int
    engagement_delay_seconds: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _split_row(row: dict) -> tuple[float, float]:
    """Split one traffic-type row's packets into legitimate and attack parts.

    The replay stores total and blocked rates but not ground truth, so the
    attack component is reconstructed from how far the row sits above its own
    hour-of-week baseline — the same quantity `baseline_deviation` is built
    from. A row that is not strongly abnormal is treated as entirely
    legitimate, which is the conservative direction: it makes every candidate
    look *less* effective rather than more.
    """
    total = float(row.get("total_pps", 0.0))
    deviation = float(row.get("baseline_deviation_pps", 0.0))
    utilisation = float(row.get("policy_utilisation", 0.0))
    if deviation > 3.0 and utilisation > 0.5:
        # Attribute the excess over the baseline-implied level to the attack.
        share = min(0.95, max(0.0, 1.0 - 1.0 / max(utilisation, 1e-6)))
        share = max(share, min(0.9, deviation / (deviation + 12.0)))
    else:
        share = 0.0
    return total * (1.0 - share), total * share


@dataclass
class _Trajectory:
    """Per-window result of replaying the horizon under one policy."""

    epochs: list[int] = field(default_factory=list)
    over: list[bool] = field(default_factory=list)
    attack_total: float = 0.0
    attack_removed: float = 0.0
    legit_total: float = 0.0
    legit_lost: float = 0.0
    peak_blocked_share: float = 0.0

    @property
    def seconds_over(self) -> int:
        return sum(WINDOW_SECONDS for flag in self.over if flag)

    @property
    def ever_over(self) -> bool:
        return any(self.over)


def _replay(
    by_epoch: dict[int, list[dict]],
    epochs: Sequence[int],
    engage_at: int,
    suppression: float,
    collateral: float,
    thresholds,
) -> _Trajectory:
    """Replay the post-decision horizon under one (suppression, collateral).

    Blocking is evaluated *per traffic type*, exactly as Behemoth and Shared
    State do it. Summing the policy line across traffic types would hide a
    single-vector flood behind the range's total headroom.
    """
    trajectory = _Trajectory()
    for epoch in epochs:
        rows = by_epoch[epoch]
        engaged = epoch >= engage_at
        active_suppression = suppression if engaged else 0.0
        active_collateral = collateral if engaged else 0.0

        window_blocked = 0.0
        window_surviving = 0.0
        window_over = False

        for row in rows:
            legit, attack = _split_row(row)
            trajectory.legit_total += legit
            trajectory.attack_total += attack
            trajectory.attack_removed += attack * active_suppression
            trajectory.legit_lost += legit * active_collateral

            surviving = max(
                attack * (1.0 - active_suppression) + legit * (1.0 - active_collateral),
                0.0,
            )
            policy_pps = float(row.get("policy_pps", 0.0))
            blocked = max(0.0, surviving - policy_pps)
            window_blocked += blocked
            window_surviving += surviving

            if (
                policy_pps > 0.0
                and blocked >= policy_pps * thresholds.start_threshold_ratio
                and blocked >= thresholds.start_floor_kpps * 1_000.0
            ):
                window_over = True

        share = window_blocked / window_surviving if window_surviving > 0 else 0.0
        trajectory.peak_blocked_share = max(trajectory.peak_blocked_share, share)
        trajectory.epochs.append(epoch)
        trajectory.over.append(window_over)
    return trajectory


def _time_to_contain(
    candidate: _Trajectory, hold: _Trajectory, decision_epoch: int, engage_at: int
) -> int | None:
    """Seconds from the decision until the incident is back under threshold.

    Three distinct cases, which a single loop over "clear windows" conflates:

    * the incident never crosses the threshold even without intervention —
      there is nothing to contain, so containment is undefined (`None`);
    * the candidate keeps it from ever crossing — contained at the moment the
      policy engages;
    * the candidate pushes an in-progress incident back under and it stays
      under for 30 seconds — contained at the start of that clear run.

    A candidate that never gets it back under threshold returns `None`.
    """
    if not hold.ever_over:
        return None
    if not candidate.ever_over:
        return max(0, engage_at - decision_epoch)

    clear_run = 0
    crossed = False
    required = 3  # 30 seconds under threshold counts as held
    for epoch, is_over in zip(candidate.epochs, candidate.over, strict=True):
        if is_over:
            crossed = True
            clear_run = 0
            continue
        # Clear windows only count once the incident has actually crossed the
        # threshold under this policy, and only after the policy is engaged.
        # Otherwise the quiet lead-in before an onset reads as containment.
        if not crossed or epoch < engage_at:
            continue
        clear_run += 1
        if clear_run >= required:
            start_of_run = epoch - (required - 1) * WINDOW_SECONDS
            return max(0, start_of_run - decision_epoch)
    return None


def simulate(
    windows: Sequence[dict],
    decision_epoch: int,
    candidate: PolicyCandidate,
    environment: str | None = None,
) -> SimulationOutcome:
    """Replay `windows` under `candidate`, engaged at `decision_epoch`.

    Only windows at or after the decision are evaluated: the drill asks what
    the *remainder* of the incident would look like, and crediting a policy
    for traffic that had already passed would overstate every candidate.
    """
    thresholds = thresholds_for(environment)
    engage_at = decision_epoch + candidate.engagement_delay_seconds

    by_epoch: dict[int, list[dict]] = {}
    for row in windows:
        by_epoch.setdefault(int(row["window_start_epoch"]), []).append(row)

    horizon = [epoch for epoch in sorted(by_epoch) if epoch >= decision_epoch]
    if not horizon:
        raise ValueError(f"no windows at or after decision epoch {decision_epoch}")

    result = _replay(
        by_epoch,
        horizon,
        engage_at,
        candidate.attack_suppression,
        candidate.legitimate_collateral,
        thresholds,
    )
    # "Hold" is the counterfactual every other candidate is measured against.
    hold = (
        result
        if candidate.key == "hold"
        else _replay(by_epoch, horizon, engage_at, 0.0, 0.0, thresholds)
    )

    attack_total = result.attack_total
    attack_removed = result.attack_removed
    legit_total = result.legit_total
    legit_lost = result.legit_lost
    peak_blocked_share = result.peak_blocked_share
    seconds_over = result.seconds_over
    contain = _time_to_contain(result, hold, decision_epoch, engage_at)

    notes: list[str] = []
    if not hold.ever_over:
        notes.append(
            "Even with no intervention this never crosses the start threshold, "
            "so there is nothing to contain — only collateral to avoid."
        )
    elif contain is None:
        notes.append("This option does not get the incident back under threshold.")
    if candidate.key == "hold":
        notes.append("Baseline: this is what happens if the operator does nothing.")
    if candidate.legitimate_collateral >= 0.1:
        notes.append("High collateral: legitimate services on this range would notice.")
    if candidate.key == "full_scrub":
        notes.append("See PPNG-6760 — a /24 diversion can also catch Teams/Zoom traffic.")
    if attack_total <= 0.0:
        notes.append(
            "No attack component detected in this replay; suppression figures "
            "are therefore zero and the collateral column is the whole story."
        )

    return SimulationOutcome(
        policy_key=candidate.key,
        policy_name=candidate.name,
        attack_packets_blocked_pct=round(
            100.0 * attack_removed / attack_total if attack_total > 0 else 0.0, 2
        ),
        legitimate_packets_lost_pct=round(
            100.0 * legit_lost / legit_total if legit_total > 0 else 0.0, 2
        ),
        peak_blocked_share=round(peak_blocked_share, 4),
        seconds_over_threshold=seconds_over,
        time_to_contain_seconds=contain,
        operational_cost=candidate.operational_cost,
        blast_radius=candidate.blast_radius,
        reversible_seconds=candidate.reversible_seconds,
        engagement_delay_seconds=candidate.engagement_delay_seconds,
        notes=notes,
    )


def simulate_threshold_plan(
    windows: Sequence[dict],
    decision_epoch: int,
    plan,
    candidate: PolicyCandidate | None = None,
    environment: str | None = None,
) -> SimulationOutcome:
    """Replay the incident with the plan's per-type policy thresholds.

    `plan` is a `policy_advisor.PolicyPlan`. A lower policy line makes the wall
    drop more of whatever exceeds it; it cannot tell attack packets from
    legitimate ones, so the extra drop is shared in proportion to each in the
    window. The legitimate part of a window is estimated from the same hour on
    the past week (`plan.legit_reference`), not from the incident itself.

    Figures are incremental to today's policy, like every other candidate:
    "hold" is zero by definition.
    """
    candidate = candidate or CANDIDATES_BY_KEY["tighten_vector"]
    thresholds = thresholds_for(environment)
    engage_at = decision_epoch + candidate.engagement_delay_seconds
    new_policy = plan.applied_thresholds

    by_epoch: dict[int, list[dict]] = {}
    for row in windows:
        by_epoch.setdefault(int(row["window_start_epoch"]), []).append(row)
    horizon = [epoch for epoch in sorted(by_epoch) if epoch >= decision_epoch]
    if not horizon:
        raise ValueError(f"no windows at or after decision epoch {decision_epoch}")

    hold = _replay(by_epoch, horizon, engage_at, 0.0, 0.0, thresholds)
    result = _Trajectory()
    for epoch in horizon:
        engaged = epoch >= engage_at
        window_blocked = 0.0
        window_total = 0.0
        window_over = False
        for row in by_epoch[epoch]:
            traffic_type = str(row.get("traffic_type"))
            total = float(row.get("total_pps", 0.0))
            policy_now = float(row.get("policy_pps", 0.0))
            reference = plan.legit_reference(traffic_type, epoch)
            legit = min(total, reference) if reference is not None else total
            attack = max(0.0, total - legit)
            result.legit_total += legit
            result.attack_total += attack

            line = new_policy.get(traffic_type, policy_now) if engaged else policy_now
            blocked_now = max(0.0, total - policy_now)
            blocked_new = max(0.0, total - line)
            extra = max(0.0, blocked_new - blocked_now)
            attack_share = attack / total if total > 0 else 0.0
            result.attack_removed += extra * attack_share
            result.legit_lost += extra * (1.0 - attack_share)

            window_blocked += blocked_new
            window_total += total
            # Over threshold, as Shared State would see it, only while today's
            # policy is still in force. Once the new line engages, everything
            # above it is dropped before it reaches the protected range.
            if not engaged and (
                policy_now > 0.0
                and blocked_now >= policy_now * thresholds.start_threshold_ratio
                and blocked_now >= thresholds.start_floor_kpps * 1_000.0
            ):
                window_over = True
        share = window_blocked / window_total if window_total > 0 else 0.0
        result.peak_blocked_share = max(result.peak_blocked_share, share)
        result.epochs.append(epoch)
        result.over.append(window_over)

    contain = _time_to_contain(result, hold, decision_epoch, engage_at)
    notes = [
        f"Suggested from the past {plan.history_days:g} days of this range's traffic: "
        f"{plan.describe_change()}."
    ]
    for suggestion in plan.per_type:
        if suggestion.apply:
            notes.append(
                f"Over that week, the new {suggestion.traffic_type} line would have dropped "
                f"{suggestion.backtest_legit_lost_pct:g}% of legitimate packets "
                f"({suggestion.backtest_windows_blocked} windows)."
            )
    if not new_policy:
        notes.append(
            "No traffic type can be tightened without dropping last week's legitimate peaks."
        )
    if not hold.ever_over:
        notes.append(
            "Even with no intervention this never crosses the start threshold, "
            "so there is nothing to contain — only collateral to avoid."
        )
    if result.attack_total <= 0.0:
        notes.append(
            "Traffic sits inside last week's normal range; no attack component is estimated."
        )

    return SimulationOutcome(
        policy_key=candidate.key,
        policy_name=candidate.name,
        attack_packets_blocked_pct=round(
            100.0 * result.attack_removed / result.attack_total if result.attack_total > 0 else 0.0,
            2,
        ),
        legitimate_packets_lost_pct=round(
            100.0 * result.legit_lost / result.legit_total if result.legit_total > 0 else 0.0, 2
        ),
        peak_blocked_share=round(result.peak_blocked_share, 4),
        seconds_over_threshold=result.seconds_over,
        time_to_contain_seconds=contain,
        operational_cost=candidate.operational_cost,
        blast_radius=candidate.blast_radius,
        reversible_seconds=candidate.reversible_seconds,
        engagement_delay_seconds=candidate.engagement_delay_seconds,
        notes=notes,
    )


def run_drill(
    windows: Sequence[dict],
    decision_epoch: int,
    environment: str | None = None,
    policy_plan=None,
) -> dict:
    """Simulate all four candidates and return a comparison table.

    With an available `policy_plan` (see `policy_advisor.suggest_policy`), the
    "tighten the vector threshold" row is simulated from concrete per-type
    thresholds learned from the past week, instead of from the fixed
    suppression/collateral assumption.

    The returned payload is explicitly marked as a simulation and carries its
    own assumptions, because it will be rendered next to real traffic data.
    """
    use_plan = policy_plan is not None and getattr(policy_plan, "available", False)
    outcomes = []
    for candidate in CANDIDATES:
        if use_plan and candidate.key == "tighten_vector":
            outcome = simulate_threshold_plan(
                windows, decision_epoch, policy_plan, candidate, environment
            )
        else:
            outcome = simulate(windows, decision_epoch, candidate, environment)
        outcomes.append(outcome.to_dict())

    assumptions = []
    for candidate in CANDIDATES:
        entry = {
            "policy_key": candidate.key,
            "basis": "stated assumption",
            "attack_suppression": candidate.attack_suppression,
            "legitimate_collateral": candidate.legitimate_collateral,
            "engagement_delay_seconds": candidate.engagement_delay_seconds,
        }
        if use_plan and candidate.key == "tighten_vector":
            entry.update(
                {
                    "basis": "past-week thresholds",
                    "attack_suppression": None,
                    "legitimate_collateral": None,
                }
            )
        assumptions.append(entry)

    return {
        "simulation_only": True,
        "decision_epoch": decision_epoch,
        "environment": thresholds_for(environment).environment,
        "candidates": outcomes,
        "assumptions": assumptions,
        "policy_plan": policy_plan.to_dict() if policy_plan is not None else None,
        "disclaimer": (
            "Modelled outcomes on recorded traffic. No policy was applied, no "
            "traffic was blocked, and no customer configuration was changed."
        ),
    }
