"""The freshness, tenant-scope and tool-use checks must be able to fail.

All three were wired so they could not: freshness compared the evidence epoch
with itself, tenant scope ran after the orchestrator had stamped the expected
ids onto the output, and a refused tool call was never recorded, so the
tool-use check had nothing to find. A check that cannot fail is worse than no
check, because the validator strip in the UI shows it as passed.

These tests drive the real orchestrator (offline engine, no network) so they
exercise the wiring, not just the individual check functions.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from conftest import ACCOUNT_ID, EPOCH, RANGE_ID, attack_windows

from dfsi.agents.orchestrator import ExplainerOrchestrator
from dfsi.agents.tools import (
    INVESTIGATOR_TOOLS,
    EvidenceBundle,
    ReadOnlyToolbox,
)
from dfsi.agents.validator import ValidationReport, check_tenant_scope, validate
from dfsi.config import STATE_MONITORING_ONLY

DECISION = EPOCH + 100


def make_bundle(**overrides) -> EvidenceBundle:
    rows = [r for r in attack_windows() if r["window_start_epoch"] <= DECISION]
    fields = dict(
        range_id=RANGE_ID,
        account_id=ACCOUNT_ID,
        window_start_epoch=DECISION,
        risk_score=0.8,
        risk_band="critical",
        model_version="test/v1",
        feature_version="features-test",
        horizon_seconds=120,
        evidence=[{"feature": "country_hhi", "label": "concentration", "value": 0.68}],
        recent_windows=rows,
        tenant_account_id=ACCOUNT_ID,
        as_of_epoch=DECISION,
    )
    fields.update(overrides)
    return EvidenceBundle(**fields)


@pytest.fixture
def offline(monkeypatch) -> ExplainerOrchestrator:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return ExplainerOrchestrator(api_key=None)


# --------------------------------------------------------------- freshness


def test_a_current_bundle_passes_freshness(offline):
    result = offline.run_investigator(make_bundle())
    assert result.validator_checks["freshness"] is True
    assert result.validated


def test_evidence_older_than_the_limit_is_withheld(offline):
    """Live use: opt in to a wall clock and stale evidence is refused."""
    stale = make_bundle(as_of_epoch=None)
    orchestrator = ExplainerOrchestrator(api_key=None, clock=lambda: DECISION + 600)
    result = orchestrator.run_investigator(stale)
    assert not result.validated
    assert result.validator_checks["freshness"] is False
    assert "old" in " ".join(result.validator_failures)


def test_the_bundles_own_clock_wins_over_the_wall_clock():
    bundle = make_bundle(as_of_epoch=DECISION)
    orchestrator = ExplainerOrchestrator(api_key=None, clock=lambda: DECISION + 10_000)
    assert orchestrator.run_investigator(bundle).validator_checks["freshness"] is True


def test_recorded_epochs_are_not_judged_against_todays_date(offline):
    """No clock configured (the replay-only default): a bundle that carries none
    is judged against its own forecast epoch, not rejected as years old."""
    result = offline.run_investigator(make_bundle(as_of_epoch=None))
    assert result.validator_checks["freshness"] is True
    assert result.validated


def test_windows_after_the_decision_are_evidence_from_the_future(offline):
    """A regression in bundle assembly must not let an agent see ahead."""
    leaky = make_bundle(recent_windows=attack_windows())  # runs past DECISION
    result = offline.run_investigator(leaky)
    assert not result.validated
    assert result.validator_checks["freshness"] is False
    assert "future" in " ".join(result.validator_failures)


def test_the_look_ahead_guard_works_without_any_clock_too(offline):
    """The part of freshness that needs no clock must not depend on one."""
    leaky = make_bundle(as_of_epoch=None, recent_windows=attack_windows())
    result = offline.run_investigator(leaky)
    assert not result.validated
    assert "future" in " ".join(result.validator_failures)


def test_the_newest_row_not_just_the_forecast_epoch_is_the_evidence_age():
    bundle = make_bundle(window_start_epoch=EPOCH)  # forecast stamp lags the rows
    assert bundle.newest_evidence_epoch() == DECISION


# ------------------------------------------------------------------ tenant


def test_the_expected_ids_are_not_stamped_before_validation(offline, monkeypatch):
    """Stamping first handed the tenant check the values it was verifying."""
    received: dict = {}
    real_validate = validate

    def spy(output, **kwargs):
        received.update(output)
        return real_validate(output, **kwargs)

    bare = {"headline": "h", "summary": "s", "options": [{"policy_key": "hold"}]}
    monkeypatch.setattr("dfsi.agents.orchestrator.validate", spy)
    monkeypatch.setattr("dfsi.agents.fallback.advise", lambda bundle, reading=None: dict(bare))

    result = offline.run_advisor(make_bundle(), reading="attack")

    assert "range_id" not in received and "account_id" not in received
    assert result.validated
    # Added only once the checks passed, so downstream consumers still get them.
    assert result.output["range_id"] == RANGE_ID
    assert result.output["account_id"] == ACCOUNT_ID
    assert result.output["requires_human_decision"] is True


@pytest.mark.parametrize(
    "text",
    [
        "This mirrors the pattern on range 90110002 last week.",
        "Account 999999 has seen the same traffic.",
        "Compare with range #90110002 and account: 123456.",
    ],
)
def test_prose_about_another_tenant_is_rejected(text):
    report = ValidationReport(passed=True)
    check_tenant_scope({"summary": text}, ACCOUNT_ID, RANGE_ID, report)
    assert not report.passed
    assert report.checks["tenant_scope"] is False


@pytest.mark.parametrize(
    "text",
    [
        f"Range {RANGE_ID} is climbing.",
        f"Account {ACCOUNT_ID} has no prior episodes.",
        "Traffic is in the range 2000 to 9000 packets per second.",  # a count, not an id
        "The account history is short.",
    ],
)
def test_prose_about_the_tenant_under_review_is_accepted(text):
    report = ValidationReport(passed=True)
    check_tenant_scope({"summary": text}, ACCOUNT_ID, RANGE_ID, report)
    assert report.passed, report.failures


def test_a_bundle_spanning_tenants_is_refused_before_any_agent_runs(offline):
    foreign = copy.deepcopy(make_bundle().recent_windows[-1])
    foreign["range_id"] = 90110002
    foreign["account_id"] = 999999
    bundle = make_bundle()
    bundle.recent_windows.append(foreign)

    called = []
    offline._call_claude = lambda **kw: called.append(kw)  # would be the egress point
    offline._client = SimpleNamespace()  # pretend a model is configured

    result = offline.run_investigator(bundle)

    assert not result.validated
    assert result.engine == "withheld"
    assert result.validator_checks == {"tenant_scope": False}
    assert "90110002" in result.validator_failures[0]
    assert called == [], "the model must never see another tenant's rows"


def test_a_single_tenant_bundle_has_no_scope_violations():
    assert make_bundle().scope_violations() == []


# ---------------------------------------------------------------- tool use


def test_a_refused_tool_call_is_recorded():
    toolbox = ReadOnlyToolbox(make_bundle())
    toolbox.record_denied("get_simulation_results", {"x": 1})
    assert toolbox.calls == [
        {"tool": "get_simulation_results", "arguments": {"x": 1}, "denied": True}
    ]


def test_an_attempt_outside_the_role_fails_the_agent(offline):
    """End to end through `_execute_tool`, the path Claude's tool loop uses."""
    bundle = make_bundle()
    toolbox = ReadOnlyToolbox(bundle)
    block = SimpleNamespace(name="get_simulation_results", id="toolu_1", input={})

    reply = offline._execute_tool(block, toolbox, INVESTIGATOR_TOOLS)

    assert reply["is_error"] is True  # refused ...
    report = validate(
        {"headline": "h", "summary": "s", "reading": "attack"},
        required_fields=("reading", "headline", "summary"),
        expected_range_id=RANGE_ID,
        expected_account_id=ACCOUNT_ID,
        citable_numbers=bundle.citable_numbers(),
        evidence_epoch=bundle.newest_evidence_epoch(),
        now_epoch=DECISION,
        tool_calls=toolbox.calls,
        allowed_tools=INVESTIGATOR_TOOLS,
    )
    assert not report.passed  # ... and still counted against the agent
    assert report.checks["tool_use"] is False
    assert "attempted" in report.failures[0]


def test_a_permitted_tool_call_is_not_flagged(offline):
    bundle = make_bundle()
    toolbox = ReadOnlyToolbox(bundle)
    block = SimpleNamespace(name="get_forecast_evidence", id="toolu_2", input={})
    reply = offline._execute_tool(block, toolbox, INVESTIGATOR_TOOLS)
    assert "is_error" not in reply
    assert all(not call.get("denied") for call in toolbox.calls)


def test_a_stale_forecast_fails_in_a_replay_even_beside_fresh_rows(offline):
    """Replay path: "now" is the decision instant. A briefing on a forecast from
    ten minutes earlier is stale, whatever the newest traffic row says."""
    stale = make_bundle(window_start_epoch=DECISION - 600, as_of_epoch=DECISION)
    result = offline.run_investigator(stale)
    assert result.validator_checks["freshness"] is False
    assert "forecast is 600s old" in " ".join(result.validator_failures)


def test_even_one_window_from_the_future_fails_freshness(offline):
    ahead = make_bundle(window_start_epoch=DECISION + 10, as_of_epoch=DECISION)
    result = offline.run_investigator(ahead)
    assert result.validator_checks["freshness"] is False
    assert "10s in the future" in " ".join(result.validator_failures)


def test_an_output_that_breaks_the_validator_is_withheld_not_a_500(offline, monkeypatch):
    def broken(*_, **__):
        raise TypeError("unexpected shape")

    monkeypatch.setattr("dfsi.agents.orchestrator.validate", broken)
    result = offline.run_investigator(make_bundle())
    assert not result.validated
    assert result.output["state"] == STATE_MONITORING_ONLY
    assert "could not check this output (TypeError)" in result.validator_failures[0]
