"""The counterfactual and the evidence gate together.

A window the gate withholds has a score above threshold but raises no alert, so
there is nothing to calm: explaining how to "calm" it would describe an alert
that does not exist. The counterfactual is for windows that actually alert.
"""

from __future__ import annotations

import pickle

import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from test_evidence_attribution import FEATURES, row_for, training_data

from dfsi.config import FEATURE_VERSION, STATE_ABOUT_TO_START
from dfsi.forecaster.serve import ForecasterService
from dfsi.forecaster.train import quiet_reference

GATE = 3.0
ALERT = dict(growth_pps_60s=1.9, country_hhi=0.8)


@pytest.fixture(scope="module")
def trained():
    frame, label = training_data()
    model = HistGradientBoostingClassifier(max_iter=60, learning_rate=0.1, random_state=0)
    return model.fit(frame, label), frame, label


def service(tmp_path, trained, gate=GATE) -> ForecasterService:
    model, frame, label = trained
    payload = {
        "model": model,
        "features": FEATURES,
        "feature_version": FEATURE_VERSION,
        "decision_threshold": 0.5,
        "min_consecutive": 2,
        "model_version": "1",
        "mlflow_run_id": "",
        "feature_importance": {},
        "feature_reference": quiet_reference(frame[FEATURES], label),
        "evidence_gate_deviation": gate,
    }
    path = tmp_path / "forecaster.pkl"
    path.write_bytes(pickle.dumps(payload))
    return ForecasterService(path)


def forecast(svc, deviation):
    return svc.score_window([row_for(**ALERT, baseline_deviation_pps_max_hour=deviation)])


def test_a_gated_window_has_no_counterfactual(tmp_path, trained):
    result = forecast(service(tmp_path, trained), deviation=0.5)
    assert result.risk_score >= 0.5 and result.gated is True
    assert result.predicted_state == "QUIET"
    assert result.counterfactual is None


def test_a_window_that_alerts_still_has_one(tmp_path, trained):
    result = forecast(service(tmp_path, trained), deviation=9.0)
    assert result.predicted_state == STATE_ABOUT_TO_START and result.gated is False
    assert result.counterfactual is not None


def test_without_a_gate_it_follows_the_score_as_before(tmp_path, trained):
    result = forecast(service(tmp_path, trained, gate=None), deviation=0.5)
    assert result.predicted_state == STATE_ABOUT_TO_START and result.counterfactual is not None
