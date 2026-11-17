"""Lead-time arithmetic.

"Warning lead time" is the headline number in the pitch, so the way it is
counted has to be beyond argument: a warning after the onset is a miss, a
single-window spike is not a warning, and the fixed-threshold baseline can
never score a positive lead.

Two of these tests exist because the measurement was wrong once in each
direction. Scoring the incumbent on the threshold *crossing* handed it the
start duration as a free head start; averaging already-underway attacks into
the lead time halved DFSI's median with a number that measures the episode
boundary rather than the forecaster.
"""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import EPOCH

from dfsi.forecaster.metrics import (
    ScenarioOutcome,
    evaluate,
    fixed_threshold_baseline,
    lead_time_histogram,
    scenario_outcomes,
)


def scored_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def scenario(
    scenario_id: str,
    scenario_type: str,
    scores: list[float],
    onset_index: int | None,
    debounce_windows: int = 3,
    in_progress: bool = False,
) -> list[dict]:
    """One scenario's worth of scored windows, one row per window.

    Models the real sequence Shared State walks through: the range crosses the
    start threshold `debounce_windows` before the declaration, stays over it,
    and only then does the bell ring. `in_progress` opens the episode already
    over threshold, which is the left-censored case.
    """
    crossing = 0 if in_progress else max(1, (onset_index or 0) - debounce_windows + 1)
    return [
        {
            "scenario_id": scenario_id,
            "scenario_type": scenario_type,
            "window_start_epoch": EPOCH + i * 10,
            "risk_score": score,
            "is_onset": int(onset_index is not None and i == onset_index),
            "range_over_threshold": float(onset_index is not None and i >= crossing),
            "declared": float(onset_index is not None and i >= onset_index),
        }
        for i, score in enumerate(scores)
    ]


# ------------------------------------------------------------------- lead


def test_lead_is_onset_minus_first_warning():
    outcome = ScenarioOutcome(
        scenario_id="s",
        scenario_type="gradual_buildup",
        has_onset=True,
        onset_epoch=EPOCH + 200,
        first_warning_epoch=EPOCH + 80,
    )
    assert outcome.lead_seconds == 120.0
    assert outcome.warned is True


def test_no_warning_means_no_lead():
    outcome = ScenarioOutcome(
        scenario_id="s",
        scenario_type="sustained_attack",
        has_onset=True,
        onset_epoch=EPOCH + 100,
        first_warning_epoch=None,
    )
    assert outcome.lead_seconds is None
    assert outcome.warned is False


def test_no_onset_means_no_lead():
    """A warning on a quiet scenario is a false alarm, not a lead time."""
    outcome = ScenarioOutcome(
        scenario_id="s",
        scenario_type="normal",
        has_onset=False,
        onset_epoch=None,
        first_warning_epoch=EPOCH + 50,
    )
    assert outcome.lead_seconds is None
    assert outcome.warned is True


def test_warning_after_the_onset_is_negative_and_counted_as_a_miss():
    frame = scored_frame(scenario("late", "sustained_attack", [0.0] * 6 + [0.9] * 6, onset_index=3))
    result = evaluate(frame, "risk_score", 0.5)
    assert result.attacks_caught == 0.0, "a late warning must not count as caught"
    assert result.late_warning_count == 1
    assert result.median_lead_seconds == 0.0


# --------------------------------------------------------------- warnings


def test_single_window_spike_is_not_a_warning():
    """Noise an operator would learn to ignore must not count as a warning."""
    frame = scored_frame(
        scenario("spiky", "normal", [0.1, 0.9, 0.1, 0.1, 0.1, 0.1], onset_index=None)
    )
    outcomes = scenario_outcomes(frame, "risk_score", 0.5, min_consecutive=2)
    assert outcomes[0].first_warning_epoch is None
    assert evaluate(frame, "risk_score", 0.5).false_alarm_rate == 0.0


def test_two_consecutive_windows_is_a_warning_raised_at_the_second():
    frame = scored_frame(
        scenario("real", "gradual_buildup", [0.1, 0.9, 0.9, 0.1, 0.1, 0.1], onset_index=5)
    )
    outcomes = scenario_outcomes(frame, "risk_score", 0.5, min_consecutive=2)
    # The run starts at index 1 but is only known to be a run at index 2, the
    # first moment an operator could be told. Crediting index 1 would overstate
    # the head start by a window nobody could have acted in.
    assert outcomes[0].first_warning_epoch == EPOCH + 20
    assert outcomes[0].lead_seconds == 30.0


def test_traffic_types_collapse_to_the_range_level():
    """One vector over threshold is enough; the range-level max is the signal."""
    rows = []
    for traffic_type, scores in (("udp", [0.9, 0.9]), ("http", [0.0, 0.0])):
        for i, score in enumerate(scores):
            rows.append(
                {
                    "scenario_id": "multi",
                    "scenario_type": "sustained_attack",
                    "window_start_epoch": EPOCH + i * 10,
                    "traffic_type": traffic_type,
                    "risk_score": score,
                    "is_onset": 0,
                    "range_over_threshold": 0.0,
                }
            )
    outcomes = scenario_outcomes(scored_frame(rows), "risk_score", 0.5, min_consecutive=2)
    assert outcomes[0].first_warning_epoch == EPOCH + 10  # the run completes at window 2


# ------------------------------------------------------------ aggregation


def test_false_alarm_rate_counts_only_quiet_scenarios():
    frame = scored_frame(
        scenario("quiet-a", "normal", [0.9, 0.9] + [0.1] * 4, onset_index=None)
        + scenario("quiet-b", "normal", [0.1] * 6, onset_index=None)
        + scenario("attack", "gradual_buildup", [0.9, 0.9] + [0.1] * 4, onset_index=5)
    )
    result = evaluate(frame, "risk_score", 0.5)
    assert result.n_scenarios == 3
    assert result.n_attack_scenarios == 1
    assert result.false_alarm_rate == pytest.approx(0.5)
    assert result.attacks_caught == pytest.approx(1.0)


def test_mlflow_metric_names_are_prefixed():
    frame = scored_frame(scenario("a", "normal", [0.1] * 6, onset_index=None))
    metrics = evaluate(frame, "risk_score", 0.5).as_mlflow_metrics("dfsi")
    assert "dfsi_false_alarm_rate" in metrics
    assert "dfsi_attacks_caught" in metrics
    assert all(isinstance(value, float) for value in metrics.values())


# --------------------------------------------------------------- baseline


def test_fixed_threshold_baseline_cannot_warn_early():
    """The incumbent reacts to blocking already happening — lead time is zero.

    This is the comparison the whole project rests on, so it is asserted
    rather than assumed.
    """
    frame = scored_frame(scenario("attack", "gradual_buildup", [0.0] * 12, onset_index=6))
    frame["baseline"] = fixed_threshold_baseline(frame)
    result = evaluate(frame, "baseline", 0.5, min_consecutive=1)
    assert result.attacks_caught == 0.0
    assert result.median_lead_seconds == 0.0


def test_baseline_is_not_credited_with_the_start_duration():
    """The incumbent must not be paid for its own debounce.

    Scoring the threshold *crossing* rather than the declaration makes the
    bell look like it has foresight worth the entire start duration — 20s on
    the scenario set, 290s in production. It is the gap DFSI exists to fill,
    so it cannot be handed to the incumbent.
    """
    frame = scored_frame(
        scenario("attack", "gradual_buildup", [0.0] * 12, onset_index=6, debounce_windows=3)
    )
    # The crossing precedes the declaration by the start duration less the
    # crossing window itself, which already counts towards that duration.
    crossing = frame[frame["range_over_threshold"] > 0]["window_start_epoch"].min()
    declaration = frame[frame["declared"] > 0]["window_start_epoch"].min()
    assert declaration - crossing == 20

    # So scoring the crossing would manufacture a 20-second head start.
    frame["crossing_score"] = frame["range_over_threshold"]
    mismeasured = evaluate(frame, "crossing_score", 0.5, min_consecutive=1)
    assert mismeasured.median_lead_seconds == 20.0

    frame["baseline"] = fixed_threshold_baseline(frame)
    honest = evaluate(frame, "baseline", 0.5, min_consecutive=1)
    assert honest.median_lead_seconds == 0.0


def test_baseline_refuses_to_score_without_the_declaration_column():
    """A silent fallback here would reintroduce the debounce credit."""
    frame = scored_frame(scenario("attack", "gradual_buildup", [0.0] * 8, onset_index=4))
    with pytest.raises(KeyError, match="declaration, not the threshold crossing"):
        fixed_threshold_baseline(frame.drop(columns=["declared"]))


# ------------------------------------------------------- left-censored attacks


def test_attack_already_under_way_is_excluded_from_lead_time():
    """An episode that opens mid-attack has no runway to forecast from.

    A `recovery` replay starts over threshold and declares two windows later.
    No forecaster can beat that, so counting it would report the episode
    boundary as DFSI's head start.
    """
    frame = scored_frame(
        scenario("recovery", "recovery", [0.9] * 8, onset_index=2, in_progress=True)
        + scenario("buildup", "gradual_buildup", [0.0] * 6 + [0.9] * 10, onset_index=12)
    )
    result = evaluate(frame, "risk_score", 0.5)

    assert result.n_attack_scenarios == 2
    assert result.n_forecastable_scenarios == 1
    assert result.n_in_progress_scenarios == 1
    # 50s, from the buildup alone — not the 20s average of both. (The warning
    # is raised at window 7, the second of the run, and the bell rings at 12.)
    assert result.median_lead_seconds == 50.0
    assert result.attacks_caught == 1.0


def test_already_under_way_attacks_are_still_reported_as_detected():
    """Excluded from lead time is not the same as ignored."""
    frame = scored_frame(
        scenario("seen", "recovery", [0.9] * 8, onset_index=2, in_progress=True)
        + scenario("missed", "recovery", [0.0] * 8, onset_index=2, in_progress=True)
    )
    result = evaluate(frame, "risk_score", 0.5)
    assert result.n_forecastable_scenarios == 0
    assert result.in_progress_detected == pytest.approx(0.5)
    # No forecastable attack means no head-start claim to make.
    assert result.median_lead_seconds == 0.0
    assert result.attacks_caught == 0.0


def test_histogram_separates_already_under_way_from_a_real_miss():
    outcomes = [
        ScenarioOutcome("a", "recovery", True, EPOCH + 20, None, in_progress_at_start=True),
        ScenarioOutcome("b", "sustained_attack", True, EPOCH + 400, None),
    ]
    histogram = lead_time_histogram(outcomes)
    assert histogram["already under way"] == 1
    assert histogram["no useful warning"] == 1


def test_leads_beyond_the_horizon_are_counted_separately():
    """A 300s warning is real but comes from the trend, not the 120s label."""
    frame = scored_frame(
        scenario("long", "gradual_buildup", [0.9] * 2 + [0.1] * 38, onset_index=39)
    )
    result = evaluate(frame, "risk_score", 0.5)
    assert result.median_lead_seconds == 380.0  # raised at window 1, bell at window 39
    assert result.leads_beyond_horizon == 1


def test_histogram_buckets_useless_warnings_separately():
    """A quiet scenario is not a missed warning — it had nothing to warn about.

    Only the sustained attack that went unwarned belongs in the failure
    bucket; lumping the 70% of scenarios that are simply normal in with it
    would make correct silence look like the dominant failure mode.
    """
    outcomes = [
        ScenarioOutcome("a", "gradual_buildup", True, EPOCH + 200, EPOCH + 80),
        ScenarioOutcome("b", "sustained_attack", True, EPOCH + 100, None),
        ScenarioOutcome("c", "normal", False, None, None),
    ]
    histogram = lead_time_histogram(outcomes, bucket_seconds=30)
    assert histogram["120-150s"] == 1
    assert histogram["no useful warning"] == 1
    assert histogram["no attack"] == 1
