"""Which numbers may an agent quote?

The validator's evidence check rejects any figure in an explanation that is not
traceable to the evidence bundle. That is only useful if it accepts a *faithful*
quote: a correct explanation that is rejected turns the whole briefing into
`MONITORING_ONLY`, which operators learn to ignore.

The check extracts numbers from prose without their sign, so the bundle has to
offer every rendering of a value — magnitude, rounded, percentage — and must
survive the awkward values real traffic produces (negatives, NaN).
"""

from __future__ import annotations

import math

import pytest
from conftest import ACCOUNT_ID, RANGE_ID

from dfsi.agents.orchestrator import ExplainerOrchestrator
from dfsi.agents.prompts import ADVISOR_SYSTEM
from dfsi.agents.tools import EvidenceBundle, numeric_forms
from dfsi.agents.validator import ValidationReport, check_evidence


def bundle_with(*, evidence=(), windows=()) -> EvidenceBundle:
    return EvidenceBundle(
        range_id=RANGE_ID,
        account_id=ACCOUNT_ID,
        window_start_epoch=0,
        risk_score=0.7,
        risk_band="warn",
        model_version="test/v1",
        feature_version="features-test",
        horizon_seconds=120,
        evidence=[dict(item) for item in evidence],
        recent_windows=[dict(row) for row in windows],
    )


def verdict(text: str, bundle: EvidenceBundle) -> ValidationReport:
    report = ValidationReport(passed=True)
    check_evidence({"summary": text, "evidence_cited": [{}]}, bundle.citable_numbers(), report)
    return report


# ------------------------------------------------------------ numeric_forms


def test_negative_value_is_citable_by_its_magnitude():
    forms = numeric_forms(-0.4231)
    assert "0.4231" in forms
    assert "-0.4231" not in forms  # the validator never sees a sign


@pytest.mark.parametrize("rendering", ["0.4231", "0.423", "0.42", "42.31", "42.3", "42"])
def test_rounded_and_percentage_renderings_are_citable(rendering):
    assert rendering in numeric_forms(0.4231)


@pytest.mark.parametrize("too_coarse", ["0.4", "0", "40"])
def test_a_rounding_more_than_5_percent_off_is_not_a_quote(too_coarse):
    assert too_coarse not in numeric_forms(0.4231)


def test_small_values_no_longer_donate_a_nearby_integer():
    assert "7" not in numeric_forms(7.4)  # 5.4% off
    assert "7.4" in numeric_forms(7.4)
    assert "57" in numeric_forms(57.3)  # 0.5% off: a faithful rounding


@pytest.mark.parametrize("invented", ["0.43", "0.4232", "0.5", "43", "41.9"])
def test_a_different_value_is_not_citable(invented):
    assert invented not in numeric_forms(0.4231)


def test_trailing_zeros_are_trimmed_but_integers_are_left_alone():
    forms = numeric_forms(100.0)
    assert "100" in forms
    assert "10000" in forms  # 100 as a percentage of 1
    assert "1" not in forms  # "100" must not collapse to "1"


def test_large_identifiers_survive_unrounded():
    assert str(RANGE_ID) in numeric_forms(RANGE_ID)


@pytest.mark.parametrize(
    "value", [None, True, False, "0.5", [0.5], float("nan"), math.inf, -math.inf]
)
def test_non_numbers_and_non_finite_values_yield_nothing(value):
    assert numeric_forms(value) == set()


# ------------------------------------------------- through the validator


NEGATIVE = [{"feature": "blocked_share_delta_60s", "value": -0.4231}]


@pytest.mark.parametrize(
    "text",
    [
        "Blocked share fell by -0.4231 over the last minute.",
        "Blocked share changed by 0.4231 over the last minute.",
        "Blocked share dropped about 42% in a minute.",
        "Blocked share moved 0.42 in the last minute.",
    ],
)
def test_a_faithful_quote_of_a_negative_value_passes(text):
    report = verdict(text, bundle_with(evidence=NEGATIVE))
    assert report.passed, report.failures


def test_an_invented_figure_is_still_rejected():
    report = verdict("Blocked share fell by 0.57 in a minute.", bundle_with(evidence=NEGATIVE))
    assert not report.passed
    assert "0.57" in report.failures[0]


def test_values_in_recent_windows_are_citable():
    bundle = bundle_with(windows=[{"traffic_type": "udp", "growth_pps_60s": -0.18}])
    assert verdict("Packet rate is down 18% on the minute.", bundle).passed


def test_a_nan_in_the_evidence_does_not_break_the_bundle():
    """`int(round(nan))` used to raise, taking the whole explain request down."""
    bundle = bundle_with(
        evidence=[{"feature": "country_hhi", "value": float("nan")}],
        windows=[{"traffic_type": "udp", "growth_pps_60s": math.inf, "total_pps": 1200.0}],
    )
    citable = bundle.citable_numbers()  # must not raise
    assert "1200" in citable
    assert "nan" not in citable and "inf" not in citable


# ------------------------------------------- reference names are not figures


@pytest.mark.parametrize(
    "text",
    [
        "A /24 diversion can also catch Teams and Zoom traffic (see PPNG-6760).",
        "Full scrubbing covers the whole /24, per PPNG-6760.",
        "Tracked as PPNG-6760 and CVE-2024-3094.",
    ],
)
def test_ticket_ids_and_cidr_prefixes_are_not_figures(text):
    report = verdict(text, bundle_with())
    assert report.passed, report.failures


def test_the_advisors_own_prompt_wording_passes_validation():
    """The prompt tells the Advisor to cite exactly this; it must not be punished for it."""
    sentence = (
        "A diversion of a whole range can also catch unrelated services on it "
        "(see PPNG-6760: a /24 block has caught Teams and Zoom traffic)."
    )
    assert "PPNG-6760" in ADVISOR_SYSTEM and "/24" in ADVISOR_SYSTEM  # still what we ask for
    assert verdict(sentence, bundle_with()).passed


def test_a_real_figure_beside_a_reference_is_still_rejected():
    report = verdict("PPNG-6760 shows 742000 packets per second.", bundle_with())
    assert not report.passed
    assert "742000" in report.failures[0]
    assert "6760" not in report.failures[0]  # the identifier is not what failed


def test_a_slash_number_that_is_not_a_prefix_cannot_hide_a_figure():
    report = verdict("Traffic reached 5/87 of capacity.", bundle_with())
    assert not report.passed
    assert "87" in report.failures[0]


def test_an_invented_risk_figure_is_still_caught_next_to_a_reference():
    report = verdict("Given PPNG-6760 the risk is 87% for this range.", bundle_with())
    assert not report.passed
    assert "risk figure" in report.failures[0]


def test_the_offline_advisor_validates_when_no_drill_is_attached(monkeypatch):
    """Without a drill it describes each option in prose, PPNG-6760 included.

    That used to fail validation, so any `include_simulation=false` request
    ended in MONITORING_ONLY.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    bundle = bundle_with(evidence=[{"feature": "country_hhi", "value": 0.68}])
    result = ExplainerOrchestrator(api_key=None).run_advisor(bundle, reading="attack")
    assert result.validated, result.validator_failures
