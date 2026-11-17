"""Validator rejection cases.

The validator is the only safety component written in code rather than in a
prompt, so it is the only one that can be tested exhaustively. Each test here
is a specific way an agent could mislead an operator mid-incident.
"""

from __future__ import annotations

from conftest import ACCOUNT_ID, EPOCH, RANGE_ID

from dfsi.agents.validator import (
    MAX_EVIDENCE_AGE_SECONDS,
    ValidationReport,
    check_allowed_actions,
    check_evidence,
    check_freshness,
    check_tenant_scope,
    check_tool_use,
    monitoring_only_notice,
    validate,
)
from dfsi.config import STATE_MONITORING_ONLY

REQUIRED = ("reading", "headline", "summary")

#: A clean output: in scope, no invented numbers, no action claims.
GOOD_OUTPUT = {
    "reading": "attack",
    "headline": "Packet rate is climbing well above this hour's normal level",
    "summary": (
        "Source concentration has tightened while the packet rate climbs. "
        "The pattern matches an attack beginning rather than a legitimate surge."
    ),
    "evidence_cited": [{"feature": "country_hhi", "value": 0.68, "why": "concentrated"}],
    "contrary_evidence": ["This account has no prior attack episodes."],
    "range_id": RANGE_ID,
    "account_id": ACCOUNT_ID,
}

CITABLE = {"0.68", "68", "18", "18.0", str(RANGE_ID), str(ACCOUNT_ID)}


def run(output: dict, **overrides) -> ValidationReport:
    kwargs = {
        "required_fields": REQUIRED,
        "expected_range_id": RANGE_ID,
        "expected_account_id": ACCOUNT_ID,
        "citable_numbers": CITABLE,
        "evidence_epoch": EPOCH,
        "now_epoch": EPOCH,
    }
    kwargs.update(overrides)
    return validate(output, **kwargs)


def test_clean_output_passes():
    report = run(GOOD_OUTPUT)
    assert report.passed, report.failures
    assert all(report.checks.values())


def test_missing_required_field_is_rejected():
    report = run({**GOOD_OUTPUT, "headline": ""})
    assert not report.passed
    assert report.checks["output_format"] is False


def test_output_longer_than_the_limit_is_rejected():
    report = run({**GOOD_OUTPUT, "summary": "x" * 5_000})
    assert not report.passed
    assert report.checks["output_format"] is False


# ------------------------------------------------------------------ tenant


def test_output_about_another_range_is_rejected():
    report = ValidationReport(passed=True)
    check_tenant_scope({"range_id": 99999999}, ACCOUNT_ID, RANGE_ID, report)
    assert not report.passed
    assert "99999999" in report.failures[0]


def test_output_about_another_account_is_rejected():
    report = ValidationReport(passed=True)
    check_tenant_scope({"range_id": RANGE_ID, "account_id": 999999}, ACCOUNT_ID, RANGE_ID, report)
    assert not report.passed
    assert "999999" in report.failures[0]


# --------------------------------------------------------------- freshness


def test_stale_evidence_is_rejected():
    report = ValidationReport(passed=True)
    check_freshness(EPOCH, EPOCH + MAX_EVIDENCE_AGE_SECONDS + 10, report)
    assert not report.passed
    assert "old" in report.failures[0]


def test_evidence_from_the_future_is_rejected():
    """A forecast dated ahead of now means the replay clock is wrong."""
    report = ValidationReport(passed=True)
    check_freshness(EPOCH + 600, EPOCH, report)
    assert not report.passed
    assert "future" in report.failures[0]


def test_evidence_one_window_ahead_is_tolerated():
    """Window boundaries make a 10-second skew normal, not an error."""
    report = ValidationReport(passed=True)
    check_freshness(EPOCH + 10, EPOCH, report)
    assert report.passed


# ---------------------------------------------------------------- evidence


def test_agent_inventing_a_risk_percentage_is_rejected():
    """The single most dangerous failure mode in the design."""
    report = ValidationReport(passed=True)
    check_evidence(
        {"summary": "I assess the risk at 87% over the next two minutes."}, CITABLE, report
    )
    assert not report.passed
    assert "risk figure" in report.failures[0]


def test_agent_inventing_a_risk_decimal_is_rejected():
    report = ValidationReport(passed=True)
    check_evidence({"summary": "Probability is 0.91 for this range."}, CITABLE, report)
    assert not report.passed
    assert "risk figure" in report.failures[0]


def test_agent_claiming_to_calculate_risk_is_rejected():
    report = ValidationReport(passed=True)
    check_evidence(
        {"summary": "I calculate the probability from the growth curve."}, CITABLE, report
    )
    assert not report.passed


def test_number_absent_from_the_bundle_is_rejected():
    report = ValidationReport(passed=True)
    check_evidence({"summary": "Traffic reached 742000 packets per second."}, CITABLE, report)
    assert not report.passed
    assert "not present in the evidence bundle" in report.failures[0]


def test_numbers_present_in_the_bundle_pass():
    report = ValidationReport(passed=True)
    check_evidence(
        {
            "summary": "Source concentration is 0.68 and deviation is 18.",
            "evidence_cited": [{"feature": "country_hhi", "value": 0.68}],
        },
        CITABLE,
        report,
    )
    assert report.passed, report.failures


def test_citing_no_evidence_warns_but_does_not_fail():
    """A vague answer is a quality problem, not a safety one."""
    report = ValidationReport(passed=True)
    check_evidence({"summary": "Something looks unusual."}, CITABLE, report)
    assert report.passed
    assert report.warnings


# --------------------------------------------------------- allowed actions


def test_claiming_an_action_was_taken_is_rejected():
    for text in (
        "I have applied the scrubbing policy.",
        "Traffic has been blocked on the suspect vector.",
        "I will divert the range to scrubbing.",
        "The policy was updated automatically.",
    ):
        report = ValidationReport(passed=True)
        check_allowed_actions({"summary": text}, report)
        assert not report.passed, text
        assert "simulation-only" in report.failures[0]


def test_unknown_policy_option_is_rejected():
    report = ValidationReport(passed=True)
    check_allowed_actions({"options": [{"policy_key": "null_route_the_prefix"}]}, report)
    assert not report.passed
    assert "null_route_the_prefix" in report.failures[0]


def test_unknown_recommended_policy_is_rejected():
    report = ValidationReport(passed=True)
    check_allowed_actions({"recommended_policy_key": "call_the_customer"}, report)
    assert not report.passed


def test_describing_the_four_options_passes():
    report = ValidationReport(passed=True)
    check_allowed_actions(
        {
            "summary": "Scrubbing would remove the most attack traffic.",
            "options": [
                {"policy_key": "hold"},
                {"policy_key": "tighten_vector"},
                {"policy_key": "rate_limit_sources"},
                {"policy_key": "full_scrub"},
            ],
            "recommended_policy_key": "tighten_vector",
        },
        report,
    )
    assert report.passed, report.failures


# --------------------------------------------------------------- tool scope


def test_agent_reaching_outside_its_role_is_rejected():
    report = ValidationReport(passed=True)
    check_tool_use(
        [{"tool": "get_forecast_evidence"}, {"tool": "get_simulation_results"}],
        ["get_forecast_evidence", "get_recent_traffic"],
        report,
    )
    assert not report.passed
    assert "get_simulation_results" in report.failures[0]


def test_agent_within_its_role_passes():
    report = ValidationReport(passed=True)
    check_tool_use(
        [{"tool": "get_forecast_evidence"}],
        ["get_forecast_evidence", "get_recent_traffic"],
        report,
    )
    assert report.passed


# --------------------------------------------------------------- fail-safe


def test_failure_notice_says_unknown_not_safe():
    """The whole point of failing safe: never 'low risk'."""
    notice = monitoring_only_notice("validation failed", ["evidence check failed"])
    assert notice["state"] == STATE_MONITORING_ONLY
    assert notice["recommended_policy_key"] is None
    assert notice["options"] == []
    assert notice["requires_human_decision"] is True
    text = (notice["headline"] + " " + notice["summary"]).lower()
    assert "unknown" in text
    assert "low risk" not in text
    assert "all clear" not in text
