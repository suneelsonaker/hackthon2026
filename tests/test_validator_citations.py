"""The two places the validator used to be lenient, now pinned down.

1. Structural numbers (horizon, window, start duration) are tolerated only as
   *durations*. The same digits as a rate, a size or a percentage must be in
   the bundle, or the output is rejected.
2. Structured citations are checked against the feature they name, so a real
   number attributed to the wrong feature fails.
"""

from __future__ import annotations

import pytest
from conftest import ACCOUNT_ID, EPOCH, RANGE_ID

from dfsi.agents.tools import EvidenceBundle
from dfsi.agents.validator import (
    STRUCTURAL_SECONDS,
    ValidationReport,
    check_citations,
    check_evidence,
    validate,
)
from dfsi.config import HORIZON_SECONDS, THRESHOLDS_BY_ENV, WINDOW_SECONDS

CITABLE = {"0.82", "4200", "1.9"}


def evidence_report(text: str, citable: set[str] = CITABLE) -> ValidationReport:
    report = ValidationReport(passed=True)
    check_evidence({"summary": text, "evidence_cited": [{"feature": "x"}]}, citable, report)
    return report


# ------------------------------------------------- structural durations


def test_the_structural_set_is_derived_from_config():
    assert HORIZON_SECONDS in STRUCTURAL_SECONDS
    assert WINDOW_SECONDS in STRUCTURAL_SECONDS
    for env in THRESHOLDS_BY_ENV.values():
        assert env.start_duration_seconds in STRUCTURAL_SECONDS


@pytest.mark.parametrize(
    "text",
    [
        "The forecast looks 120 seconds ahead.",
        "One 10s window at a time.",
        "Shared State waits out a 300-second start duration in prod.",
        "Doing nothing for another 60 seconds is often right.",
        "Within 30 sec the bell rings.",
    ],
)
def test_structural_durations_with_a_unit_pass(text):
    assert evidence_report(text).passed, text


@pytest.mark.parametrize(
    "text, figure",
    [
        ("Expect 300 Mbps at the edge.", "300"),
        ("Traffic will reach 120 pps per source.", "120"),
        ("About 60% of packets are spoofed.", "60"),
        ("Roughly 30 kpps more than usual.", "30"),
        ("The flood is 10 times the baseline.", "10"),
    ],
)
def test_the_same_digits_as_a_measurement_fail(text, figure):
    report = evidence_report(text)
    assert not report.passed
    assert figure in report.failures[0]


@pytest.mark.parametrize(
    "text", ["Four options: 4 are on the table.", "2 windows in a row.", "0 of them"]
)
def test_bare_small_counts_pass(text):
    assert evidence_report(text).passed


@pytest.mark.parametrize("text", ["4% of traffic is lost", "3 pps above normal", "2x the baseline"])
def test_small_numbers_with_a_unit_must_be_cited(text):
    assert not evidence_report(text).passed


def test_a_cited_figure_with_a_unit_still_passes():
    assert evidence_report("Peak was 4200 pps, source concentration 0.82.").passed


# ------------------------------------------------- structured citations

VALUES = {"country_hhi": [0.82, 0.79], "blocked_share": [0.12], "growth_pps_60s": []}


def citations_report(*items: dict) -> ValidationReport:
    report = ValidationReport(passed=True)
    check_citations({"evidence_cited": list(items)}, VALUES, report)
    return report


def test_a_citation_at_its_value_passes():
    assert citations_report({"feature": "country_hhi", "value": 0.82}).passed


def test_rounding_for_prose_is_tolerated():
    assert citations_report({"feature": "country_hhi", "value": 0.8}).passed is False
    assert citations_report({"feature": "country_hhi", "value": 0.821}).passed


def test_a_real_number_attributed_to_the_wrong_feature_fails():
    report = citations_report({"feature": "blocked_share", "value": 0.82})
    assert not report.passed and "blocked_share" in report.failures[0]


def test_an_unknown_feature_fails():
    report = citations_report({"feature": "attacker_ip_count", "value": 3})
    assert not report.passed and "attacker_ip_count" in report.failures[0]


def test_a_citation_without_a_feature_fails():
    assert not citations_report({"value": 0.82}).passed


def test_a_non_numeric_value_fails():
    assert not citations_report({"feature": "country_hhi", "value": "high"}).passed


def test_a_feature_without_a_value_or_without_known_values_passes():
    assert citations_report({"feature": "country_hhi", "why": "narrow mix"}).passed
    assert citations_report({"feature": "growth_pps_60s", "value": 9.0}).passed


def test_free_text_citations_are_left_to_the_prose_scan():
    report = ValidationReport(passed=True)
    check_citations({"evidence_cited": ["narrow source mix"]}, VALUES, report)
    assert report.passed


def test_nan_never_matches():
    report = ValidationReport(passed=True)
    check_citations(
        {"evidence_cited": [{"feature": "x", "value": float("nan")}]}, {"x": [1.0]}, report
    )
    assert not report.passed


# ------------------------------------------------- end to end through validate()


def bundle(**overrides) -> EvidenceBundle:
    fields = dict(
        range_id=RANGE_ID,
        account_id=ACCOUNT_ID,
        window_start_epoch=EPOCH,
        risk_score=0.71,
        risk_band="warn",
        model_version="m/1",
        feature_version="f/1",
        horizon_seconds=HORIZON_SECONDS,
        evidence=[
            {"feature": "country_hhi", "value": 0.82},
            {"feature": "total_pps", "value": None},
        ],
        recent_windows=[{"window_start_epoch": EPOCH, "total_pps": 4200.0, "ok": True}],
        simulation=None,
        tenant_account_id=ACCOUNT_ID,
        as_of_epoch=EPOCH,
    )
    fields.update(overrides)
    return EvidenceBundle(**fields)


def test_citable_values_collects_evidence_and_window_columns():
    values = bundle().citable_values()
    assert values["country_hhi"] == [0.82]
    assert values["total_pps"] == [4200.0]  # the None evidence value adds nothing
    assert values["ok"] == []  # booleans are not figures


def test_validate_runs_the_citation_check_when_given_values():
    b = bundle()
    output = {
        "headline": "Reading: attack",
        "summary": "Source concentration is 0.82.",
        "evidence_cited": [{"feature": "total_pps", "value": 0.82, "why": "misattributed"}],
    }
    common = dict(
        required_fields=("headline", "summary"),
        expected_range_id=RANGE_ID,
        expected_account_id=ACCOUNT_ID,
        citable_numbers=b.citable_numbers(),
        evidence_epoch=EPOCH,
        now_epoch=EPOCH,
    )
    assert validate(output, **common).passed  # the prose scan alone is satisfied
    report = validate(output, citable_values=b.citable_values(), **common)
    assert not report.passed and report.checks["citations"] is False


@pytest.mark.parametrize("text", ["300Mbps at the edge", "a 10x jump", "peaks of 9000pps"])
def test_a_unit_glued_to_the_digits_does_not_hide_a_figure(text):
    assert not evidence_report(text).passed


def test_feature_names_with_digits_are_not_figures():
    assert evidence_report("growth_pps_60s and country_hhi_delta_60s both rose.").passed


@pytest.mark.parametrize("text", ["Traffic is 1e6 pps", "about 2.5E+3 Mbps"])
def test_scientific_notation_is_refused(text):
    report = evidence_report(text)
    assert not report.passed and "scientific notation" in report.failures[0]


@pytest.mark.parametrize(
    "text", ["three thousand Mbps", "about twelve times normal", "hundreds of sources"]
)
def test_figures_in_words_are_refused(text):
    report = evidence_report(text)
    assert not report.passed and "in words" in report.failures[0]


def test_small_counts_in_words_are_fine():
    assert evidence_report("The two agents weigh four options.").passed


def test_epoch_columns_are_not_citable_figures():
    b = bundle(recent_windows=[{"window_start_epoch": EPOCH, "total_pps": 4200.0}])
    assert str(EPOCH) not in b.citable_numbers()
    assert "4200" in b.citable_numbers()


@pytest.mark.parametrize("text", ["Peak hit 4M pps", "about 3k more", "a 4K-fold jump", "2 fold"])
def test_a_magnitude_suffix_turns_a_small_count_into_a_figure(text):
    assert not evidence_report(text).passed
