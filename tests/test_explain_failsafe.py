"""Explanations fail safe when the forecaster could not score the window.

README, "Safety model": any failure in the prediction or explanation path
returns `MONITORING_ONLY` — explicitly "unknown", never "low risk" or "all
clear". A degraded forecast carries no risk score. Before this was enforced,
`/api/explain` ignored that and ran both agents over the raw traffic rows
anyway, returning state `EXPLAINED`, a reading and a recommended policy — a
validated-looking briefing about a window the model had said nothing about.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import ACCOUNT_ID, EPOCH, RANGE_ID, attack_windows, make_services

from dfsi.agents.orchestrator import ExplainerOrchestrator
from dfsi.agents.tools import EvidenceBundle
from dfsi.config import STATE_MONITORING_ONLY
from dfsi.forecaster.serve import ForecasterService
from dfsi.rehearsal.simulator import run_drill

DECISION = EPOCH + 100


def make_bundle(risk_score: float | None) -> EvidenceBundle:
    return EvidenceBundle(
        range_id=RANGE_ID,
        account_id=ACCOUNT_ID,
        window_start_epoch=DECISION,
        risk_score=risk_score,
        risk_band="critical" if risk_score is not None else "unknown",
        model_version="test/v1",
        feature_version="features-test",
        horizon_seconds=120,
        evidence=[{"feature": "country_hhi", "label": "concentration", "value": 0.68}]
        if risk_score is not None
        else [],
        recent_windows=[r for r in attack_windows() if r["window_start_epoch"] <= DECISION],
        # With a drill the offline Advisor quotes figures from it; without one it
        # falls back to the candidates' prose, which is a separate concern.
        simulation=run_drill(attack_windows(), DECISION),
        tenant_account_id=ACCOUNT_ID,
    )


@pytest.fixture
def offline(monkeypatch) -> ExplainerOrchestrator:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return ExplainerOrchestrator(api_key=None)


def test_an_unscored_window_is_monitoring_only_not_explained(offline):
    briefing = offline.explain(make_bundle(risk_score=None))

    assert briefing["state"] == STATE_MONITORING_ONLY
    assert briefing["requires_human_decision"] is True
    assert "no risk score" in briefing["withheld_reason"]
    for agent in ("investigator", "advisor"):
        result = briefing[agent]
        assert result["validated"] is False
        assert result["output"]["state"] == STATE_MONITORING_ONLY
        assert result["output"]["recommended_policy_key"] is None
        assert result["output"]["options"] == []


def test_the_withheld_briefing_never_says_low_risk(offline):
    briefing = offline.explain(make_bundle(risk_score=None))
    text = " ".join(
        str(briefing[a]["output"].get(k, ""))
        for a in ("investigator", "advisor")
        for k in ("headline", "summary")
    ).lower()
    assert "unknown" in text
    assert "low risk" not in text and "all clear" not in text


def test_no_agent_runs_and_no_model_is_called_for_an_unscored_window(offline, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("an agent ran on an unscored window")

    monkeypatch.setattr(offline, "_run", forbidden)
    monkeypatch.setattr(offline, "_call_claude", forbidden)

    briefing = offline.explain(make_bundle(risk_score=None))

    assert briefing["engine"] == "withheld"
    assert briefing["investigator"]["tool_calls"] == []  # nothing was read, nothing sent


def test_a_scored_window_is_still_explained(offline):
    """Guard against over-correcting: a healthy forecast must still brief."""
    briefing = offline.explain(make_bundle(risk_score=0.82))
    assert briefing["state"] == "EXPLAINED"
    assert briefing["investigator"]["validated"] and briefing["advisor"]["validated"]
    assert "withheld_reason" not in briefing


def test_a_score_of_zero_is_a_real_score_not_a_missing_one(offline):
    """`not risk_score` would have treated 0.0 as degraded."""
    briefing = offline.explain(make_bundle(risk_score=0.0))
    assert "withheld_reason" not in briefing


# ---------------------------------------------------- through the API layer


def test_a_down_forecaster_yields_monitoring_only_end_to_end(tmp_path):
    """The real path: no model on disk -> degraded forecast -> API bundle -> explain."""
    api = make_services(
        tmp_path,
        forecaster=ForecasterService(Path("/nonexistent/forecaster.pkl")),
        orchestrator=ExplainerOrchestrator(api_key=None),
    )
    # The shape `load_replay` returns, not the smallest dict that happens to work.
    replay = {
        "replay_id": "sustained_attack__test",
        "scenario_type": "sustained_attack",
        "range_id": RANGE_ID,
        "windows": attack_windows(),
    }

    bundle = api.build_bundle(replay, DECISION, include_simulation=True)
    assert bundle.risk_score is None  # the forecaster said nothing

    briefing = api.orchestrator.explain(bundle)
    assert briefing["state"] == STATE_MONITORING_ONLY
    assert briefing["advisor"]["output"]["recommended_policy_key"] is None
