"""Deterministic offline Explainers.

Used when no `ANTHROPIC_API_KEY` is configured, when the API is unreachable,
or when Claude refuses. Rule-based, so the demo always runs and so there is
always a reference answer to compare the model against.

These rules obey exactly the same constraints as the agents: they quote only
values taken from the evidence bundle, they never compute a risk score of
their own, and they only ever propose one of the four fixed candidates.
"""

from __future__ import annotations

from ..rehearsal.simulator import CANDIDATES, CANDIDATES_BY_KEY
from .tools import EvidenceBundle


def _value(bundle: EvidenceBundle, feature: str, default: float = 0.0) -> float:
    for item in bundle.evidence:
        if item.get("feature") == feature:
            return float(item.get("value", default))
    if bundle.recent_windows:
        latest = bundle.recent_windows[-1]
        if feature in latest and latest[feature] is not None:
            return float(latest[feature])
    return default


def investigate(bundle: EvidenceBundle) -> dict:
    """Attack or flash crowd? Decided by the documented discriminators."""
    hhi = _value(bundle, "country_hhi")
    hhi_shift = _value(bundle, "country_hhi_delta_60s")
    growth_60 = _value(bundle, "growth_pps_60s")
    deviation = _value(bundle, "baseline_deviation_pps")
    persistence = _value(bundle, "anomaly_persistence")
    history = _value(bundle, "account_attack_history")
    blocked_delta = _value(bundle, "blocked_share_delta_60s")

    cited: list[dict] = []

    def cite(feature: str, note: str) -> None:
        for item in bundle.evidence:
            if item.get("feature") == feature:
                cited.append({**item, "why": note})
                return
        cited.append({"feature": feature, "label": feature, "why": note})

    attack_signals = 0
    crowd_signals = 0
    reasons: list[str] = []
    counter: list[str] = []

    if hhi >= 0.5:
        attack_signals += 2
        reasons.append("source traffic is concentrated in a few countries")
        cite("country_hhi", "concentration consistent with reflection or a botnet")
    elif hhi <= 0.3:
        crowd_signals += 1
        counter.append("source mix is broad, which fits organic demand")
        cite("country_hhi", "broad source mix argues for a legitimate surge")

    if hhi_shift >= 0.1:
        attack_signals += 1
        reasons.append("the source mix narrowed over the last minute")
        cite("country_hhi_delta_60s", "narrowing source mix")
    elif hhi_shift <= -0.05:
        crowd_signals += 1
        counter.append("the source mix broadened, which attacks rarely do")

    if deviation >= 6.0:
        attack_signals += 1
        reasons.append("the packet rate is far above normal for this hour of the week")
        cite("baseline_deviation_pps", "well outside the hour-of-week baseline")
    if persistence >= 6:
        attack_signals += 1
        reasons.append("abnormal windows have persisted rather than passing")
        cite("anomaly_persistence", "sustained abnormality, not a single spike")
    if blocked_delta >= 0.05:
        attack_signals += 1
        reasons.append("the blocked share is climbing")
        cite("blocked_share_delta_60s", "blocking is already increasing")
    if history >= 1:
        attack_signals += 1
        reasons.append("this account was attacked in the past few hours")
        cite("account_attack_history", "recent attack episodes on this account")
    if growth_60 >= 1.0 and hhi <= 0.35:
        crowd_signals += 2
        counter.append("fast growth from a diverse source mix is the flash-crowd shape")
        cite("growth_pps_60s", "rapid but broadly-sourced growth")

    if attack_signals >= crowd_signals + 2:
        reading = "attack"
        headline = "Reading: an attack is beginning"
    elif crowd_signals >= attack_signals + 1:
        reading = "flash_crowd"
        headline = "Reading: this looks like a legitimate surge"
    else:
        reading = "ambiguous"
        headline = "Reading: ambiguous — evidence points both ways"

    summary_parts = []
    if reasons:
        summary_parts.append("For an attack: " + "; ".join(reasons[:3]) + ".")
    if counter:
        summary_parts.append("Against: " + "; ".join(counter[:2]) + ".")
    if not summary_parts:
        summary_parts.append("No strong signal in either direction from the current evidence.")

    return {
        "agent": "investigator",
        "engine": "offline-rules",
        "range_id": bundle.range_id,
        "account_id": bundle.account_id,
        "reading": reading,
        "headline": headline,
        "summary": " ".join(summary_parts),
        "evidence_cited": cited[:6],
        "contrary_evidence": counter[:3],
        "requires_human_decision": True,
    }


def advise(bundle: EvidenceBundle, reading: str | None = None) -> dict:
    """Lay out the four options; recommend based on simulated trade-offs."""
    simulation = (bundle.simulation or {}).get("candidates", [])
    by_key = {row["policy_key"]: row for row in simulation}
    plan = (bundle.simulation or {}).get("policy_plan") or {}

    options: list[dict] = []
    for candidate in CANDIDATES:
        row = by_key.get(candidate.key, {})
        options.append(
            {
                "policy_key": candidate.key,
                "name": candidate.name,
                "what_it_buys": (
                    f"removes {row['attack_packets_blocked_pct']}% of the attack traffic"
                    if row.get("attack_packets_blocked_pct") is not None
                    else candidate.description
                )
                + (
                    f"; sets {plan['change']} (from the past {plan['history_days']:g} days)"
                    if candidate.key == "tighten_vector"
                    and plan.get("available")
                    and plan.get("change") not in (None, "no change")
                    else ""
                ),
                "what_it_costs": (
                    f"loses {row['legitimate_packets_lost_pct']}% of legitimate traffic"
                    if row.get("legitimate_packets_lost_pct") is not None
                    else f"operational cost: {candidate.operational_cost}"
                ),
                "blast_radius": candidate.blast_radius,
                "engagement_delay_seconds": candidate.engagement_delay_seconds,
                "reversible_seconds": candidate.reversible_seconds,
                "time_to_contain_seconds": row.get("time_to_contain_seconds"),
            }
        )

    # Pick the cheapest option that still contains the incident; fall back to
    # holding when nothing contains it or the evidence is thin.
    recommended = "hold"
    rationale = (
        "The evidence does not yet justify touching live policy. Keep watching "
        "for another minute — the forecast horizon gives room to wait."
    )
    if reading == "attack" and simulation:
        containing = [
            row
            for row in simulation
            if row["policy_key"] != "hold" and row.get("time_to_contain_seconds") is not None
        ]
        if containing:
            best = min(
                containing,
                key=lambda row: (
                    row["legitimate_packets_lost_pct"],
                    row["time_to_contain_seconds"],
                ),
            )
            recommended = best["policy_key"]
            rationale = (
                f"{CANDIDATES_BY_KEY[recommended].name} contains the incident with "
                f"the least collateral of the options that work. Blast radius: "
                f"{CANDIDATES_BY_KEY[recommended].blast_radius}."
            )
        else:
            recommended = "full_scrub"
            rationale = (
                "No narrower option contains this in the drill. Full scrubbing "
                "is the only effective choice, and its blast radius covers every "
                "service on the range."
            )
    elif reading == "flash_crowd":
        rationale = (
            "The evidence fits a legitimate surge. Applying a mitigation here "
            "would drop real customer traffic for no benefit."
        )

    return {
        "agent": "advisor",
        "engine": "offline-rules",
        "range_id": bundle.range_id,
        "account_id": bundle.account_id,
        "headline": f"Recommendation for a human: {CANDIDATES_BY_KEY[recommended].name}",
        "summary": rationale,
        "options": options,
        "recommended_policy_key": recommended,
        "evidence_cited": bundle.evidence[:4],
        "requires_human_decision": True,
    }
