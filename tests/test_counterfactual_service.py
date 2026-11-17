"""The counterfactual on a real (small) trained model, through `ForecasterService`.

The search itself is checked against an analytic model in `test_counterfactual`.
Here the point is the wiring and the round trip: apply what the service says
would calm an alert, score again, and the alert must actually be gone.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from test_evidence_attribution import DRIVER_A, DRIVER_B, FEATURES, row_for, training_data

from dfsi.config import FEATURE_VERSION
from dfsi.forecaster.serve import ForecasterService
from dfsi.forecaster.train import quiet_reference

THRESHOLD = 0.5


@pytest.fixture(scope="module")
def trained():
    frame, label = training_data()
    model = HistGradientBoostingClassifier(max_iter=60, learning_rate=0.1, random_state=0)
    model.fit(frame, label)
    return model, frame, label


def needs_both_data(n: int = 8000, seed: int = 5):
    """Label = 0.5*A + B > 1: neither driver alone can bring a strong alert back."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({name: rng.normal(0.0, 1.0, n) for name in FEATURES})
    frame[DRIVER_A] = rng.uniform(0.0, 2.0, n)
    frame[DRIVER_B] = rng.uniform(0.0, 1.0, n)
    margin = 0.5 * frame[DRIVER_A] + frame[DRIVER_B] - 1.0
    label = (rng.random(n) < 1.0 / (1.0 + np.exp(-8.0 * margin))).astype(int)
    return frame, label


@pytest.fixture(scope="module")
def needs_both():
    frame, label = needs_both_data()
    model = HistGradientBoostingClassifier(max_iter=60, learning_rate=0.1, random_state=0)
    return model.fit(frame, label), frame, label


def service_for(tmp_path, trained, *, with_reference: bool = True) -> ForecasterService:
    model, frame, label = trained
    payload = {
        "model": model,
        "features": FEATURES,
        "feature_version": FEATURE_VERSION,
        "decision_threshold": THRESHOLD,
        "min_consecutive": 2,
        "model_version": "1",
        "mlflow_run_id": "",
        "feature_importance": {},
    }
    if with_reference:
        payload["feature_reference"] = quiet_reference(frame[FEATURES], label)
    path = tmp_path / "forecaster.pkl"
    path.write_bytes(pickle.dumps(payload))
    return ForecasterService(path)


#: Growth is far out and the source mix is high: a clear alert driven by both.
ALERT = dict(growth_pps_60s=1.9, country_hhi=0.8)


def forecast(service, **features):
    return service.score_window([row_for(**features)])


# ----------------------------------------------------------------- when present


def test_an_alerting_window_carries_a_counterfactual(tmp_path, trained):
    result = forecast(service_for(tmp_path, trained), **ALERT)
    assert result.risk_score >= THRESHOLD
    assert result.counterfactual is not None
    assert result.counterfactual["threshold"] == THRESHOLD
    assert result.counterfactual["score"] == pytest.approx(result.risk_score, abs=1e-3)


def test_a_quiet_window_has_none(tmp_path, trained):
    result = forecast(service_for(tmp_path, trained), growth_pps_60s=0.1, country_hhi=0.1)
    assert result.risk_score < THRESHOLD
    assert result.counterfactual is None


def test_it_is_part_of_the_serialised_forecast(tmp_path, trained):
    payload = forecast(service_for(tmp_path, trained), **ALERT).to_dict()
    assert json.loads(json.dumps(payload))["counterfactual"]["joint"]["clears"] in (True, False)
    quiet = forecast(service_for(tmp_path, trained), growth_pps_60s=0.1).to_dict()
    assert quiet["counterfactual"] is None


# -------------------------------------------------------------------- round trip


def test_applying_the_joint_reset_actually_clears_the_alert(tmp_path, trained):
    service = service_for(tmp_path, trained)
    alert = forecast(service, **ALERT)
    joint = alert.counterfactual["joint"]
    assert joint["clears"] is True

    calmed = dict(ALERT)
    for item in joint["features"]:
        calmed[item["feature"]] = item["quiet_value"]
    after = forecast(service, **calmed)

    assert after.risk_score < THRESHOLD
    assert after.risk_score == pytest.approx(joint["score_if_reset"], abs=1e-3)
    assert after.predicted_state == "QUIET"


def test_an_alert_that_needs_both_drivers_is_reported_as_needing_both(tmp_path, needs_both):
    """Neither driver alone clears it; together they do, and nothing smaller does."""
    service = service_for(tmp_path, needs_both)
    both = dict(growth_pps_60s=1.9, country_hhi=0.95)
    alert = forecast(service, **both)
    assert alert.risk_score >= THRESHOLD

    cf = alert.counterfactual
    assert all(not s["clears_alone"] for s in cf["single_feature"])
    assert cf["joint"]["clears"] is True
    assert {f["feature"] for f in cf["joint"]["features"]} == {DRIVER_A, DRIVER_B}

    # Minimality: dropping the last driver of the set leaves the alert standing.
    first, last = cf["joint"]["features"]
    assert (
        forecast(service, **{**both, first["feature"]: first["quiet_value"]}).risk_score
        >= THRESHOLD
    )
    # And resetting the whole set clears it.
    calmed = {**both, first["feature"]: first["quiet_value"], last["feature"]: last["quiet_value"]}
    assert forecast(service, **calmed).risk_score < THRESHOLD


def test_a_single_feature_value_that_is_reported_does_clear_the_alert(tmp_path, trained):
    service = service_for(tmp_path, trained)
    singles = forecast(service, growth_pps_60s=1.95, country_hhi=0.4).counterfactual[
        "single_feature"
    ]
    clearing = [s for s in singles if s["clears_alone"]]
    assert clearing, "this alert is driven by growth alone, so one driver should suffice"
    top = clearing[0]
    moved = {"growth_pps_60s": 1.95, "country_hhi": 0.4, top["feature"]: top["required_value"]}
    assert forecast(service, **moved).risk_score < THRESHOLD
    assert top["quiet_value"] <= top["required_value"] <= top["current"]


def test_only_features_pushing_risk_up_are_tried(tmp_path, trained):
    """A feature holding the score down is not a driver of the alert."""
    service = service_for(tmp_path, trained)
    result = forecast(service, growth_pps_60s=1.95, country_hhi=0.05)
    assert result.counterfactual is not None
    tried = {s["feature"] for s in result.counterfactual["single_feature"]}
    assert DRIVER_B not in tried and DRIVER_A in tried


# ----------------------------------------------------------- fail soft, fail safe


def test_a_model_without_a_quiet_reference_returns_none_and_keeps_the_forecast(tmp_path, trained):
    result = forecast(service_for(tmp_path, trained, with_reference=False), **ALERT)
    assert result.risk_score >= THRESHOLD and result.predicted_state == "ABOUT_TO_START"
    assert result.counterfactual is None


def test_a_failure_while_explaining_never_costs_the_forecast(tmp_path, trained):
    service = service_for(tmp_path, trained)
    real = service._model.predict_proba
    calls = {"n": 0}

    def flaky(frame):
        calls["n"] += 1
        # call 1 scores, call 2 attributes, call 3 is the counterfactual's first batch
        if calls["n"] == 3:
            raise RuntimeError("boom")
        return real(frame)

    service._model.predict_proba = flaky
    result = forecast(service, **ALERT)

    assert result.degraded is False and result.risk_score >= THRESHOLD
    assert result.evidence  # attribution (call 2) still worked
    assert result.counterfactual is None


def test_a_degraded_forecast_has_no_counterfactual():
    degraded = ForecasterService(pytest.importorskip("pathlib").Path("/nonexistent/f.pkl"))
    result = degraded.score_window([row_for(**ALERT)])
    assert result.degraded and result.counterfactual is None


def test_it_does_not_change_the_score_or_the_evidence(tmp_path, trained):
    with_ref = service_for(tmp_path, trained)
    a = forecast(with_ref, **ALERT)
    again = forecast(with_ref, **ALERT)
    assert a.risk_score == again.risk_score and a.evidence == again.evidence
    # The input row is not mutated by the search.
    row = row_for(**ALERT)
    before = dict(row)
    with_ref.score_window([row])
    assert row == before


def test_the_extra_model_calls_are_bounded(tmp_path, trained):
    service = service_for(tmp_path, trained)
    real = service._model.predict_proba
    sizes = []

    def counting(frame):
        sizes.append(len(frame))
        return real(frame)

    service._model.predict_proba = counting
    forecast(service, **ALERT)
    # scoring + attribution + three single-feature grids + one joint batch
    assert len(sizes) <= 1 + 1 + 3 + 1
    assert max(sizes) <= 41
