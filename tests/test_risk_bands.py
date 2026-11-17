"""The risk band shown must never contradict the state raised.

The decision threshold is calibrated per model (0.65 on the shipped one, 0.55
after a retrain at a different false-alarm budget), but the bands were fixed at
0.35 / 0.60 / 0.80. With a threshold of 0.65 a score of 0.64 therefore read
"warn" while the state stayed `QUIET`, and the next retrain could move the
threshold anywhere. Bands are now derived from the model's own threshold, with
`warn` starting exactly at it.
"""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest
from conftest import EPOCH, make_row

from dfsi.config import FEATURE_VERSION, STATE_ABOUT_TO_START, STATE_MONITORING_ONLY
from dfsi.forecaster.serve import ForecasterService, RiskBands
from dfsi.memory.features import FEATURE_COLUMNS

THRESHOLDS = [0.05, 0.25, 0.4, 0.55, 0.65, 0.8, 0.95]


class StubModel:
    """Scores every row the same, so a test picks the score it wants."""

    def __init__(self, score: float) -> None:
        self.score = score

    def predict_proba(self, frame):
        return np.tile([1.0 - self.score, self.score], (len(frame), 1))


def service(tmp_path, *, threshold: float, score: float) -> ForecasterService:
    path = tmp_path / "forecaster.pkl"
    payload = {
        "model": StubModel(score),
        "features": list(FEATURE_COLUMNS),
        "feature_version": FEATURE_VERSION,
        "decision_threshold": threshold,
        "min_consecutive": 2,
        "model_version": "1",
        "mlflow_run_id": "",
        "feature_importance": {},
    }
    path.write_bytes(pickle.dumps(payload))
    return ForecasterService(path)


# ------------------------------------------------------------------ the bands


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_bands_are_ordered_and_inside_the_unit_interval(threshold):
    bands = RiskBands.for_threshold(threshold)
    assert 0.0 < bands.watch < bands.warn < bands.critical < 1.0


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_warn_begins_exactly_at_the_decision_threshold(threshold):
    bands = RiskBands.for_threshold(threshold)
    assert bands.warn == threshold
    assert bands.band(threshold) in ("warn", "critical")
    assert bands.band(math.nextafter(threshold, 0.0)) in ("calm", "watch")


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_the_band_agrees_with_the_threshold_for_every_score(threshold):
    """Warn-or-above if and only if the window counts as above threshold."""
    bands = RiskBands.for_threshold(threshold)
    for step in range(0, 101):
        score = step / 100
        in_warning_band = bands.band(score) in ("warn", "critical")
        assert in_warning_band == (score >= threshold), (threshold, score)


def test_the_case_that_used_to_contradict():
    """Threshold 0.65: 0.62 and 0.64 were 'warn' under the fixed scale."""
    bands = RiskBands.for_threshold(0.65)
    assert bands.band(0.62) == "watch"
    assert bands.band(0.64) == "watch"
    assert bands.band(0.66) == "warn"


def test_the_bands_follow_a_retrained_threshold():
    shipped, retrained = RiskBands.for_threshold(0.65), RiskBands.for_threshold(0.40)
    assert shipped.band(0.5) == "watch"  # below 0.65
    assert retrained.band(0.5) == "warn"  # above 0.40
    assert shipped.warn != retrained.warn


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.5, float("nan"), float("inf")])
def test_a_threshold_the_bands_cannot_be_built_from_is_rejected(bad):
    with pytest.raises(ValueError):
        RiskBands.for_threshold(bad)


# ---------------------------------------------------- through the real service


@pytest.mark.parametrize("threshold", [0.4, 0.65])
@pytest.mark.parametrize("score", [0.0, 0.2, 0.39, 0.4, 0.62, 0.64, 0.65, 0.66, 0.9, 1.0])
def test_a_forecast_never_shows_a_warning_band_while_quiet(tmp_path, threshold, score):
    svc = service(tmp_path, threshold=threshold, score=score)
    forecast = svc.score_window([make_row(epoch=EPOCH)])

    assert forecast.degraded is False
    warned_state = forecast.predicted_state == STATE_ABOUT_TO_START
    warned_band = forecast.risk_band in ("warn", "critical")
    assert warned_band == warned_state, (
        threshold,
        score,
        forecast.risk_band,
        forecast.predicted_state,
    )


def test_the_service_bands_come_from_the_loaded_threshold(tmp_path):
    svc = service(tmp_path, threshold=0.65, score=0.5)
    assert svc.bands == RiskBands.for_threshold(0.65)


def test_a_model_with_a_corrupt_threshold_degrades_instead_of_scoring(tmp_path):
    svc = service(tmp_path, threshold=1.7, score=0.9)
    forecast = svc.score_window([make_row(epoch=EPOCH)])

    assert svc.ready is False
    assert "decision threshold" in (svc.load_error or "")
    assert forecast.degraded and forecast.predicted_state == STATE_MONITORING_ONLY
    assert forecast.risk_band == "unknown" and forecast.risk_score is None
