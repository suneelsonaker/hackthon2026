"""The offline Explainers: each documented discriminator moves the reading the right way.

The rule engine is the reference answer the model is compared against and the
fail-safe when Claude is unavailable, so every branch is pinned here.
"""

from __future__ import annotations

import pytest
from conftest import ACCOUNT_ID, EPOCH, RANGE_ID

from dfsi.agents import fallback
from dfsi.agents.tools import EvidenceBundle
from dfsi.rehearsal.simulator import CANDIDATES_BY_KEY

#: Neutral values: no discriminator fires on these.
NEUTRAL = {
    "country_hhi": 0.4,
    "country_hhi_delta_60s": 0.0,
    "growth_pps_60s": 0.0,
    "baseline_deviation_pps": 1.0,
    "anomaly_persistence": 0,
    "account_attack_history": 0,
    "blocked_share_delta_60s": 0.0,
}


def bundle(
    evidence: dict | None = None, window: dict | None = None, simulation=None
) -> EvidenceBundle:
    values = {**NEUTRAL, **(evidence or {})}
    return EvidenceBundle(
        range_id=RANGE_ID,
        account_id=ACCOUNT_ID,
        window_start_epoch=EPOCH,
        risk_score=0.8,
        risk_band="warn",
        model_version="m/1",
        feature_version="f/1",
        horizon_seconds=120,
        evidence=[{"feature": k, "value": v} for k, v in values.items() if k not in (window or {})],
        recent_windows=[{"window_start_epoch": EPOCH, **(window or {})}],
        simulation=simulation,
        tenant_account_id=ACCOUNT_ID,
    )


def test_neutral_evidence_is_ambiguous_with_no_strong_signal():
    out = fallback.investigate(bundle())
    assert out["reading"] == "ambiguous"
    assert out["summary"] == "No strong signal in either direction from the current evidence."
    assert out["requires_human_decision"] is True


@pytest.mark.parametrize(
    "evidence, reason",
    [
        ({"country_hhi": 0.7}, "concentrated in a few countries"),
        ({"country_hhi_delta_60s": 0.2}, "narrowed over the last minute"),
        ({"baseline_deviation_pps": 8.0}, "far above normal"),
        ({"anomaly_persistence": 9}, "persisted"),
        ({"blocked_share_delta_60s": 0.1}, "blocked share is climbing"),
        ({"account_attack_history": 2}, "attacked in the past few hours"),
    ],
)
def test_each_attack_discriminator_is_reported_and_cited(evidence, reason):
    out = fallback.investigate(bundle(evidence))
    assert reason in out["summary"]
    (feature,) = evidence
    assert feature in [c["feature"] for c in out["evidence_cited"]]


def test_enough_attack_signals_read_as_an_attack():
    out = fallback.investigate(
        bundle({"country_hhi": 0.7, "baseline_deviation_pps": 8.0, "anomaly_persistence": 9})
    )
    assert out["reading"] == "attack" and out["headline"].endswith("an attack is beginning")


def test_fast_diverse_growth_reads_as_a_flash_crowd():
    out = fallback.investigate(
        bundle({"country_hhi": 0.25, "growth_pps_60s": 2.0, "country_hhi_delta_60s": -0.1})
    )
    assert out["reading"] == "flash_crowd"
    assert "flash-crowd shape" in " ".join(out["contrary_evidence"])
    assert "broadened" in " ".join(out["contrary_evidence"])


def test_a_value_missing_from_evidence_is_read_from_the_latest_window():
    out = fallback.investigate(bundle(window={"country_hhi": 0.9}))
    assert "concentrated" in out["summary"]
    cited = next(c for c in out["evidence_cited"] if c["feature"] == "country_hhi")
    assert "value" not in cited  # no evidence item to copy a value from


# ------------------------------------------------------------------ advise


def drill(**overrides) -> dict:
    rows = {
        "hold": dict(
            attack_packets_blocked_pct=0.0,
            legitimate_packets_lost_pct=0.0,
            time_to_contain_seconds=None,
        ),
        "tighten_vector": dict(
            attack_packets_blocked_pct=80.0,
            legitimate_packets_lost_pct=1.5,
            time_to_contain_seconds=40,
        ),
        "rate_limit_sources": dict(
            attack_packets_blocked_pct=70.0,
            legitimate_packets_lost_pct=4.0,
            time_to_contain_seconds=60,
        ),
        "full_scrub": dict(
            attack_packets_blocked_pct=99.0,
            legitimate_packets_lost_pct=9.0,
            time_to_contain_seconds=30,
        ),
    }
    for key, change in overrides.items():
        rows[key].update(change)
    return {"candidates": [{"policy_key": k, **v} for k, v in rows.items()]}


def test_without_a_reading_the_advice_is_to_hold():
    out = fallback.advise(bundle(simulation=drill()), reading=None)
    assert out["recommended_policy_key"] == "hold"
    assert {o["policy_key"] for o in out["options"]} == set(CANDIDATES_BY_KEY)


def test_on_an_attack_the_least_collateral_containing_option_wins():
    out = fallback.advise(bundle(simulation=drill()), reading="attack")
    assert out["recommended_policy_key"] == "tighten_vector"
    assert "least collateral" in out["summary"]


def test_on_an_attack_nothing_narrow_contains_so_full_scrub_with_its_blast_radius():
    sim = drill(
        tighten_vector={"time_to_contain_seconds": None},
        rate_limit_sources={"time_to_contain_seconds": None},
        full_scrub={"time_to_contain_seconds": None},
    )
    out = fallback.advise(bundle(simulation=sim), reading="attack")
    assert out["recommended_policy_key"] == "full_scrub"
    assert "blast radius" in out["summary"]


def test_a_flash_crowd_is_never_mitigated():
    out = fallback.advise(bundle(simulation=drill()), reading="flash_crowd")
    assert out["recommended_policy_key"] == "hold"
    assert "legitimate surge" in out["summary"]


def test_without_a_drill_options_fall_back_to_the_stated_descriptions():
    out = fallback.advise(bundle(), reading="attack")
    assert out["recommended_policy_key"] == "hold"
    by_key = {o["policy_key"]: o for o in out["options"]}
    assert by_key["full_scrub"]["what_it_buys"] == CANDIDATES_BY_KEY["full_scrub"].description
    assert by_key["full_scrub"]["what_it_costs"].startswith("operational cost:")


def test_an_available_threshold_plan_is_quoted_on_tighten_vector():
    sim = {
        **drill(),
        "policy_plan": {"available": True, "change": "udp 40000 → 12000 pps", "history_days": 7},
    }
    out = fallback.advise(bundle(simulation=sim), reading="attack")
    tighten = next(o for o in out["options"] if o["policy_key"] == "tighten_vector")
    assert (
        "udp 40000 → 12000 pps" in tighten["what_it_buys"]
        and "past 7 days" in tighten["what_it_buys"]
    )
