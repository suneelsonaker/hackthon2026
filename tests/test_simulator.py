"""Rehearsal-room semantics.

The containment column is the one an operator will act on, so these tests pin
down the three cases that a naive "count the clear windows" implementation
conflates, plus the per-traffic-type threshold evaluation.
"""

from __future__ import annotations

import pytest
from conftest import EPOCH, POLICY_PPS, attack_windows, make_row, quiet_windows

from dfsi.rehearsal.simulator import (
    CANDIDATES,
    CANDIDATES_BY_KEY,
    run_drill,
    simulate,
)


def test_four_fixed_candidates_in_cost_order():
    """The menu is fixed by design, and `hold` must be first."""
    assert [c.key for c in CANDIDATES] == [
        "hold",
        "tighten_vector",
        "rate_limit_sources",
        "full_scrub",
    ]
    suppressions = [c.attack_suppression for c in CANDIDATES]
    collaterals = [c.legitimate_collateral for c in CANDIDATES]
    assert suppressions == sorted(suppressions), "stronger options must come later"
    assert collaterals == sorted(collaterals), "cost must rise with suppression"


def test_hold_removes_nothing_and_costs_nothing(attack_replay):
    outcome = simulate(attack_replay, EPOCH, CANDIDATES_BY_KEY["hold"])
    assert outcome.attack_packets_blocked_pct == 0.0
    assert outcome.legitimate_packets_lost_pct == 0.0


def test_hold_never_contains_an_incident(attack_replay):
    """`hold` changes nothing, so it cannot be credited with containment."""
    outcome = simulate(attack_replay, EPOCH, CANDIDATES_BY_KEY["hold"])
    assert outcome.seconds_over_threshold > 0, "this replay must cross the threshold"
    assert outcome.time_to_contain_seconds is None


def test_nothing_to_contain_on_a_quiet_replay(quiet_replay):
    """Containment is undefined when the incident never crosses the line.

    Reporting 0s here would read as "contained instantly", which is the
    opposite of the truth: there was nothing to contain.
    """
    for candidate in CANDIDATES:
        outcome = simulate(quiet_replay, EPOCH, candidate)
        assert outcome.time_to_contain_seconds is None, candidate.key
        assert outcome.seconds_over_threshold == 0, candidate.key


def test_strong_suppression_contains_what_weak_suppression_does_not(attack_replay):
    weak = simulate(attack_replay, EPOCH, CANDIDATES_BY_KEY["tighten_vector"])
    strong = simulate(attack_replay, EPOCH, CANDIDATES_BY_KEY["full_scrub"])
    assert strong.attack_packets_blocked_pct > weak.attack_packets_blocked_pct
    assert strong.time_to_contain_seconds is not None
    assert strong.seconds_over_threshold <= weak.seconds_over_threshold


def test_containment_is_never_before_the_policy_engages(attack_replay):
    """A candidate cannot contain anything before it takes effect."""
    for candidate in CANDIDATES:
        outcome = simulate(attack_replay, EPOCH, candidate)
        if outcome.time_to_contain_seconds is None:
            continue
        assert outcome.time_to_contain_seconds >= candidate.engagement_delay_seconds


def test_single_vector_flood_is_not_hidden_by_range_headroom():
    """Blocking is evaluated per traffic type, as Behemoth and Shared State do.

    The UDP vector alone is far over its own policy line, while the range's
    three policy lines summed together would leave headroom. Summing would
    report zero seconds over threshold — the bug this test exists to catch.
    """
    windows: list[dict] = []
    for i in range(12):
        epoch = EPOCH + i * 10
        windows.append(
            make_row(
                epoch=epoch,
                traffic_type="udp",
                total_pps=POLICY_PPS * 2.2,
                blocked_pps=POLICY_PPS * 1.2,
                baseline_deviation_pps=20.0,
            )
        )
        for traffic_type in ("tcp_syn", "http"):
            windows.append(make_row(epoch=epoch, traffic_type=traffic_type, total_pps=1_000.0))

    outcome = simulate(windows, EPOCH, CANDIDATES_BY_KEY["hold"])
    assert outcome.seconds_over_threshold > 0


def test_deciding_early_beats_deciding_late():
    """The whole value proposition, expressed as a test.

    Acting at the first warning engages the policy before the flood crosses the
    threshold, so the incident never crosses at all. Acting 200 seconds later —
    roughly when the duration-based alarm would have fired — leaves time over
    the threshold that the head start would have avoided.
    """
    windows = attack_windows(count=24)
    early = simulate(windows, EPOCH, CANDIDATES_BY_KEY["full_scrub"])
    late = simulate(windows, EPOCH + 200, CANDIDATES_BY_KEY["full_scrub"])
    assert early.seconds_over_threshold == 0
    assert late.seconds_over_threshold > early.seconds_over_threshold


def test_decision_epoch_after_the_replay_is_an_error():
    with pytest.raises(ValueError, match="no windows at or after"):
        simulate(quiet_windows(), EPOCH + 100_000, CANDIDATES_BY_KEY["hold"])


def test_drill_is_marked_simulation_only_and_states_its_assumptions(attack_replay):
    result = run_drill(attack_replay, EPOCH)
    assert result["simulation_only"] is True
    assert len(result["candidates"]) == 4
    assert len(result["assumptions"]) == 4
    assert "No policy was applied" in result["disclaimer"]
    for assumption in result["assumptions"]:
        assert assumption["policy_key"] in CANDIDATES_BY_KEY
        assert "attack_suppression" in assumption
        assert "legitimate_collateral" in assumption


def test_full_scrub_always_carries_its_blast_radius_warning(attack_replay):
    """PPNG-6760 is the reason this column exists; it must not be droppable."""
    outcome = simulate(attack_replay, EPOCH, CANDIDATES_BY_KEY["full_scrub"])
    assert "PPNG-6760" in " ".join(outcome.notes)
    assert "every service on the range" in outcome.blast_radius
