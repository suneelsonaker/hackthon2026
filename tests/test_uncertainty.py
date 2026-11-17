"""How much does the drill's comparison depend on its stated assumptions?

The bands come from replaying each option under assumptions drawn from a spread
around their stated values. The load-bearing property is that with no spread
the bands collapse onto the plain drill exactly: anything else would mean the
sampler is measuring something other than the drill it sits beside.
"""

from __future__ import annotations

import pytest
from conftest import EPOCH, attack_windows, quiet_windows

from dfsi.rehearsal.policy_advisor import PolicyPlan, TypeSuggestion, _hour, _is_weekend
from dfsi.rehearsal.simulator import CANDIDATES_BY_KEY, SimulationOutcome, simulate
from dfsi.rehearsal.uncertainty import (
    MAX_DRAWS,
    NONE_CONTAIN,
    AssumptionSpread,
    assumption_uncertainty,
    cheapest_containing,
)

DECISION = EPOCH + 50
FLOOD = attack_windows(count=40, onset_index=6)
ZERO = AssumptionSpread(suppression_sd=0.0, collateral_sigma=0.0, delay_jitter_windows=0)
REAL_OPTIONS = ("tighten_vector", "rate_limit_sources", "full_scrub")


def run(**kwargs) -> dict:
    kwargs.setdefault("draws", 60)
    return assumption_uncertainty(FLOOD, DECISION, **kwargs)


def width(report: dict, key: str, metric: str) -> float:
    band = report["options"][key]["bands"][metric]
    return band["p90"] - band["p10"]


# ------------------------------------------------------------ the contract


def test_no_spread_collapses_onto_the_plain_drill_exactly():
    report = run(spread=ZERO, draws=5)
    for key in REAL_OPTIONS:
        plain = simulate(FLOOD, DECISION, CANDIDATES_BY_KEY[key])
        for metric in (
            "attack_packets_blocked_pct",
            "legitimate_packets_lost_pct",
            "seconds_over_threshold",
            "peak_blocked_share",
        ):
            band = report["options"][key]["bands"][metric]
            assert (
                band["p10"]
                == band["p50"]
                == band["p90"]
                == pytest.approx(getattr(plain, metric), abs=1e-3)
            )
        assert report["options"][key]["p_contains"] == (
            1.0 if plain.time_to_contain_seconds is not None else 0.0
        )


def test_the_same_seed_gives_the_same_bands_and_a_different_seed_does_not():
    assert run(seed=1) == run(seed=1)
    assert run(seed=1) != run(seed=2)


def test_bands_are_ordered_and_hold_is_not_sampled():
    report = run()
    assert set(report["options"]) == set(REAL_OPTIONS)
    for option in report["options"].values():
        for metric, band in option["bands"].items():
            if band is not None:
                assert band["p10"] <= band["p50"] <= band["p90"], metric


def test_a_wider_spread_widens_the_bands():
    narrow = run(
        spread=AssumptionSpread(suppression_sd=0.02, collateral_sigma=0.05, delay_jitter_windows=0)
    )
    wide = run(
        spread=AssumptionSpread(suppression_sd=0.25, collateral_sigma=0.6, delay_jitter_windows=0)
    )
    for key in REAL_OPTIONS:
        assert width(wide, key, "attack_packets_blocked_pct") > width(
            narrow, key, "attack_packets_blocked_pct"
        )
        assert width(wide, key, "legitimate_packets_lost_pct") > width(
            narrow, key, "legitimate_packets_lost_pct"
        )


def test_the_stated_outcome_sits_inside_the_bands():
    report = run(draws=200)
    for option in report["options"].values():
        band = option["bands"]["attack_packets_blocked_pct"]
        assert band["p10"] <= option["stated"]["attack_packets_blocked_pct"] <= band["p90"]


# ------------------------------------------------------ who wins, how often


def test_win_shares_are_a_distribution():
    report = run()
    assert set(report["p_best"]) == {*REAL_OPTIONS, NONE_CONTAIN}
    assert sum(report["p_best"].values()) == pytest.approx(1.0, abs=0.01)
    assert report["leader"] == max(report["p_best"], key=report["p_best"].get)
    assert report["leader_share"] == report["p_best"][report["leader"]]


def test_a_recommendation_that_always_wins_is_called_robust():
    # With the assumptions fixed, one option wins every draw.
    report = run(spread=ZERO, draws=5)
    assert report["leader_share"] == 1.0
    assert report["recommendation_is_robust"] is True


def test_a_recommendation_that_flips_is_not_called_robust():
    """Under the default spread, scrub contains this flood only about two draws in three."""
    report = run(draws=200)
    assert report["leader_share"] < 0.8
    assert report["recommendation_is_robust"] is False


def outcome(key: str, lost: float, contain: int | None) -> SimulationOutcome:
    return SimulationOutcome(
        policy_key=key,
        policy_name=key,
        attack_packets_blocked_pct=50.0,
        legitimate_packets_lost_pct=lost,
        peak_blocked_share=0.1,
        seconds_over_threshold=0,
        time_to_contain_seconds=contain,
        operational_cost="low",
        blast_radius="x",
        reversible_seconds=0,
        engagement_delay_seconds=0,
    )


def test_the_cheapest_option_that_contains_wins():
    """Built by hand: on the toy flood only one option contains, so min == max there."""
    outcomes = {
        "hold": outcome("hold", 0.0, None),
        "tighten_vector": outcome("tighten_vector", 2.0, None),  # cheapest, but does not contain
        "rate_limit_sources": outcome("rate_limit_sources", 6.0, 40),
        "full_scrub": outcome("full_scrub", 11.0, 30),
    }
    assert cheapest_containing(outcomes) == "rate_limit_sources"


def test_equal_collateral_is_broken_by_faster_containment():
    outcomes = {
        "hold": outcome("hold", 0.0, None),
        "tighten_vector": outcome("tighten_vector", 5.0, 60),
        "rate_limit_sources": outcome("rate_limit_sources", 5.0, 30),
        "full_scrub": outcome("full_scrub", 5.0, 90),
    }
    assert cheapest_containing(outcomes) == "rate_limit_sources"


def test_hold_is_never_chosen_even_if_it_looks_free_and_contained():
    outcomes = {
        "hold": outcome("hold", 0.0, 0),
        "full_scrub": outcome("full_scrub", 11.0, 30),
    }
    assert cheapest_containing(outcomes) == "full_scrub"


def test_no_option_that_contains_means_none_contain():
    outcomes = {
        "hold": outcome("hold", 0.0, None),
        "tighten_vector": outcome("tighten_vector", 2.0, None),
        "full_scrub": outcome("full_scrub", 11.0, None),
    }
    assert cheapest_containing(outcomes) == NONE_CONTAIN


def test_the_win_rule_is_the_offline_advisors_rule_on_a_flood_several_options_contain():
    """Not a copy that can drift: the same pick the shipped Advisor makes."""
    from dfsi.agents import fallback
    from dfsi.agents.tools import EvidenceBundle
    from dfsi.rehearsal.simulator import CANDIDATES, run_drill

    moderate = attack_windows(count=40, onset_index=6, peak_pps=80_000.0)
    outcomes = {c.key: simulate(moderate, DECISION, c) for c in CANDIDATES}
    containing = [
        k for k, o in outcomes.items() if k != "hold" and o.time_to_contain_seconds is not None
    ]
    assert len(containing) >= 2, "fixture must make the choice non-trivial"

    bundle = EvidenceBundle(
        range_id=1,
        account_id=1,
        window_start_epoch=DECISION,
        risk_score=0.9,
        risk_band="critical",
        model_version="t",
        feature_version="t",
        horizon_seconds=120,
        simulation=run_drill(moderate, DECISION),
    )
    advised = fallback.advise(bundle, "attack")["recommended_policy_key"]
    assert cheapest_containing(outcomes) == advised


def test_a_clear_cut_incident_has_a_robust_recommendation_under_the_default_spread():
    moderate = attack_windows(count=40, onset_index=6, peak_pps=60_000.0)
    report = assumption_uncertainty(moderate, DECISION, draws=100)
    assert report["leader"] == "tighten_vector"
    assert report["recommendation_is_robust"] is True


# ------------------------------------------------------ nothing to contain


def test_when_nothing_crosses_the_threshold_there_is_no_winner_to_report():
    report = assumption_uncertainty(quiet_windows(count=30), EPOCH + 50, draws=20)
    assert report["p_best"] is None and report["leader"] is None
    assert report["recommendation_is_robust"] is None
    assert any("nothing to contain" in note for note in report["notes"])
    for option in report["options"].values():
        assert option["p_contains"] == 0.0


# --------------------------------------------------------- the past-week plan


def make_plan() -> PolicyPlan:
    udp = TypeSuggestion(
        traffic_type="udp",
        current_policy_pps=40_000.0,
        current_policy_mbps=0.0,
        legit_median_pps=10_000.0,
        legit_envelope_pps=14_000.0,
        legit_max_pps=15_000.0,
        suggested_policy_pps=20_000.0,
        suggested_policy_mbps=0.0,
        change_pct=-0.5,
        changes=True,
        ratio_to_normal_now=8.0,
        suspect=True,
        apply=True,
        backtest_windows_blocked=0,
        backtest_hours_affected=0,
        backtest_legit_lost_pct=0.0,
    )
    return PolicyPlan(
        available=True,
        range_id=90110001,
        history_days=7.0,
        per_type=[udp],
        _reference={
            ("udp", _is_weekend(EPOCH + dt), _hour(EPOCH + dt)): 10_000.0
            for dt in range(0, 7200, 3600)
        },
    )


def test_an_option_driven_by_the_past_week_is_not_sampled():
    report = assumption_uncertainty(FLOOD, DECISION, policy_plan=make_plan(), draws=40)
    tighten = report["options"]["tighten_vector"]
    assert tighten["basis"].startswith("past-week thresholds")
    for band in tighten["bands"].values():
        if band is not None:
            assert band["p10"] == band["p50"] == band["p90"]
    assert report["options"]["full_scrub"]["basis"] == "stated assumptions, sampled"


def test_the_plan_driven_stated_outcome_is_the_one_the_drill_reports():
    from dfsi.rehearsal.simulator import run_drill

    plan = make_plan()
    drill = {c["policy_key"]: c for c in run_drill(FLOOD, DECISION, None, plan)["candidates"]}
    report = assumption_uncertainty(FLOOD, DECISION, policy_plan=plan, draws=10)
    stated = report["options"]["tighten_vector"]["stated"]
    assert (
        stated["attack_packets_blocked_pct"]
        == drill["tighten_vector"]["attack_packets_blocked_pct"]
    )
    assert (
        stated["legitimate_packets_lost_pct"]
        == drill["tighten_vector"]["legitimate_packets_lost_pct"]
    )


# ----------------------------------------------------------------- inputs


@pytest.mark.parametrize("draws", [0, -1, MAX_DRAWS + 1])
def test_a_draw_count_out_of_range_is_rejected(draws):
    with pytest.raises(ValueError):
        assumption_uncertainty(FLOOD, DECISION, draws=draws)


@pytest.mark.parametrize(
    "bad", [dict(suppression_sd=-0.1), dict(collateral_sigma=-1), dict(delay_jitter_windows=-1)]
)
def test_a_negative_spread_is_rejected(bad):
    with pytest.raises(ValueError):
        AssumptionSpread(**bad)


def test_the_caveat_says_these_are_not_confidence_intervals():
    assert "not confidence intervals" in run(draws=5)["caveat"]


# ------------------------- what "robust" does and does not cover (the leader's own assumptions)


def test_a_robust_leader_driven_by_the_past_week_says_its_own_assumptions_were_not_varied():
    """With a plan, "tighten" rests on measured thresholds and has nothing to sample, so
    "robust" only means no sampled option beats it. The report must say so."""
    report = assumption_uncertainty(FLOOD, DECISION, policy_plan=make_plan(), draws=60)
    assert report["leader"] == "tighten_vector" and report["recommendation_is_robust"] is True
    assert report["leader_assumptions_sampled"] is False
    assert any("not sampled" in note and "does not test" in note for note in report["notes"])


def test_a_sampled_leader_carries_no_such_caveat():
    moderate = attack_windows(count=40, onset_index=6, peak_pps=60_000.0)
    report = assumption_uncertainty(moderate, DECISION, draws=100)
    assert report["leader_assumptions_sampled"] is True
    assert not any("not sampled" in note for note in report["notes"])


def test_there_is_nothing_to_say_about_sampling_when_there_is_no_leader():
    report = assumption_uncertainty(quiet_windows(count=30), EPOCH + 50, draws=20)
    assert report["leader"] is None and report["leader_assumptions_sampled"] is None
