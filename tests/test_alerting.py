"""One debounce rule across the evaluator, the live stream and the API default.

The evaluation behind the published head start and false-alarm rate requires
`min_consecutive` windows in a row above the decision threshold before it
counts a warning. The live replay stream used to warn on the first window above
threshold, so the dashboard raised alarms the evaluation would have discarded
and the demo did not behave like the numbers quoted about it.

The parity tests are the point of this file: whatever the scores, the stream
and the evaluator must agree on *when* the warning was raised.
"""

from __future__ import annotations

import random

import pandas as pd
import pytest
from conftest import ACCOUNT_ID, EPOCH, RANGE_ID, TRAFFIC_TYPES, make_row, make_services

from dfsi.api import replay as replay_engine
from dfsi.config import STATE_ABOUT_TO_START, STATE_MONITORING_ONLY
from dfsi.forecaster.alerting import AlertTracker, first_alert_index
from dfsi.forecaster.metrics import scenario_outcomes
from dfsi.forecaster.serve import Forecast, RiskBands

THRESHOLD = 0.5


# --------------------------------------------------------------- the rule


@pytest.mark.parametrize(
    ("above", "needed", "expected"),
    [
        ([], 2, None),
        ([False, False, False], 2, None),
        ([True, False, True, False], 2, None),  # isolated spikes never add up
        ([False, True, True], 2, 2),  # raised at the window that completes the run
        ([True, True], 2, 1),
        ([True], 1, 0),
        ([True, False, True, True, True], 3, 4),
        ([False, True, False, True, True], 2, 4),  # a broken run starts over
    ],
)
def test_first_alert_index(above, needed, expected):
    assert first_alert_index(above, needed) == expected


@pytest.mark.parametrize("bad", [0, -1])
def test_a_run_length_below_one_is_rejected(bad):
    with pytest.raises(ValueError):
        first_alert_index([True], bad)
    with pytest.raises(ValueError):
        AlertTracker(bad)


def test_tracker_is_active_while_the_run_continues_and_drops_when_it_breaks():
    tracker = AlertTracker(min_consecutive=2)
    flags = [False, True, True, True, False, True, True]
    assert [tracker.update(flag) for flag in flags] == [
        False,
        False,
        True,
        True,
        False,
        False,
        True,
    ]


def test_tracker_agrees_with_the_batch_function_on_random_sequences():
    rng = random.Random(7)
    for _ in range(300):
        needed = rng.randint(1, 4)
        flags = [rng.random() < 0.4 for _ in range(rng.randint(0, 40))]
        tracker = AlertTracker(needed)
        streamed = next((i for i, flag in enumerate(flags) if tracker.update(flag)), None)
        assert streamed == first_alert_index(flags, needed), (needed, flags)


# ----------------------------------------------------------- the live stream


class ScriptedForecaster:
    """Stands in for `ForecasterService`: one fixed score per window.

    `None` models a window the forecaster could not score (a degraded forecast).
    """

    def __init__(self, scores, min_consecutive: int = 2, threshold: float = THRESHOLD):
        self.min_consecutive = min_consecutive
        self.decision_threshold = threshold
        self.bands = RiskBands.for_threshold(threshold)
        self._by_epoch = {EPOCH + i * 10: score for i, score in enumerate(scores)}

    def score_window(self, rows: list[dict], threshold: float | None = None) -> Forecast:
        epoch = int(rows[0]["window_start_epoch"])
        score = self._by_epoch[epoch]
        threshold = self.decision_threshold if threshold is None else threshold
        common = dict(
            window_start_epoch=epoch,
            range_id=RANGE_ID,
            account_id=ACCOUNT_ID,
            horizon_seconds=120,
            model_version="test/v1",
            feature_version="features-test",
            decision_threshold=threshold,
        )
        if score is None:
            return Forecast(
                risk_score=None,
                risk_band="unknown",
                predicted_state=STATE_MONITORING_ONLY,
                degraded=True,
                degraded_reason="scripted",
                **common,
            )
        return Forecast(
            risk_score=score,
            risk_band="warn",
            predicted_state=STATE_ABOUT_TO_START if score >= threshold else "QUIET",
            **common,
        )


def make_replay(count: int, onset_index: int | None = None) -> dict:
    rows = []
    for i in range(count):
        for traffic_type in TRAFFIC_TYPES:
            rows.append(
                make_row(
                    epoch=EPOCH + i * 10,
                    traffic_type=traffic_type,
                    is_onset=int(onset_index is not None and i == onset_index),
                    declared=int(onset_index is not None and i >= onset_index),
                )
            )
    return {
        "replay_id": "scripted",
        "scenario_type": "gradual_buildup",
        "range_id": RANGE_ID,
        "windows": rows,
    }


def stream(scores, min_consecutive: int = 2, onset_index: int | None = None):
    forecaster = ScriptedForecaster(scores, min_consecutive)
    state = replay_engine.build_state(make_replay(len(scores), onset_index))
    events = []
    while not state.finished:
        events.append(replay_engine.step(state, forecaster))
    return state, events


def test_a_single_window_above_threshold_is_not_a_warning():
    state, events = stream([0.1, 0.9, 0.1, 0.1])

    assert state.first_warning_epoch is None
    assert all(e["dfsi_state"] != STATE_ABOUT_TO_START for e in events)
    # The gauge reads forecast.predicted_state, so it must not flash either ...
    assert all(e["forecast"]["predicted_state"] != STATE_ABOUT_TO_START for e in events)
    # ... while the raw verdict stays visible for anyone who wants it.
    assert [e["forecast"]["above_threshold"] for e in events] == [False, True, False, False]


def test_two_in_a_row_warn_at_the_second_and_stay_warned():
    state, events = stream([0.1, 0.9, 0.9, 0.9, 0.1])

    assert [e["dfsi_state"] for e in events] == [
        "QUIET",
        "QUIET",
        STATE_ABOUT_TO_START,
        STATE_ABOUT_TO_START,
        "QUIET",
    ]
    assert [e["forecast"]["predicted_state"] for e in events] == [e["dfsi_state"] for e in events]
    assert state.first_warning_epoch == EPOCH + 20


def test_a_window_the_forecaster_could_not_score_breaks_the_run():
    state, events = stream([0.9, None, 0.9, 0.9])

    assert events[1]["dfsi_state"] == STATE_MONITORING_ONLY
    assert events[2]["dfsi_state"] == "QUIET"  # run restarted after the gap
    assert state.first_warning_epoch == EPOCH + 30


def test_a_run_length_of_one_warns_immediately():
    state, _ = stream([0.1, 0.9], min_consecutive=1)
    assert state.first_warning_epoch == EPOCH + 10


def test_the_lead_reported_to_the_dashboard_is_measured_from_the_alert():
    state, events = stream([0.1, 0.9, 0.9, 0.9, 0.9, 0.9], onset_index=5)
    assert state.onset_epoch == EPOCH + 50
    assert events[-1]["lead_seconds"] == 30  # raised at +20, bell at +50


# -------------------------------------------------- the stream vs the evaluator


def evaluator_first_warning(scores, min_consecutive: int) -> int | None:
    rows = [
        {
            "scenario_id": "scripted",
            "scenario_type": "gradual_buildup",
            "window_start_epoch": EPOCH + i * 10,
            "traffic_type": traffic_type,
            "risk_score": score,
            "is_onset": 0,
        }
        for i, score in enumerate(scores)
        for traffic_type in TRAFFIC_TYPES
    ]
    outcome = scenario_outcomes(pd.DataFrame(rows), "risk_score", THRESHOLD, min_consecutive)[0]
    return outcome.first_warning_epoch


def test_the_stream_and_the_evaluator_agree_on_when_the_warning_was_raised():
    """The invariant whose violation this change fixes."""
    rng = random.Random(20260101)
    for _ in range(150):
        needed = rng.randint(1, 4)
        scores = [round(rng.random(), 2) for _ in range(rng.randint(5, 40))]
        state, _ = stream(scores, min_consecutive=needed)
        assert state.first_warning_epoch == evaluator_first_warning(scores, needed), (
            needed,
            scores,
        )


# ------------------------------------------------------- the API default epoch


@pytest.fixture
def api(tmp_path):
    """Builds services around a scripted forecaster; `api(scores)` -> DfsiServices."""

    def build(scores, min_consecutive: int = 2):
        replay = make_replay(len(scores))
        return make_services(
            tmp_path,
            forecaster=ScriptedForecaster(scores, min_consecutive),
            load_stored_replay=lambda replay_id: replay,
        )

    return build


def resolve(api, monkeypatch, scores, min_consecutive: int = 2):
    return api(scores, min_consecutive).resolve_decision_epoch("scripted", None)[1]


def test_the_default_decision_epoch_is_the_window_the_dashboard_warned_at(api, monkeypatch):
    assert resolve(api, monkeypatch, [0.1, 0.9, 0.9, 0.9]) == EPOCH + 20


def test_an_isolated_spike_does_not_set_the_default_decision_epoch(api, monkeypatch):
    scores = [0.1, 0.9, 0.1, 0.1]
    assert resolve(api, monkeypatch, scores) == EPOCH + 30  # falls back to the last window
