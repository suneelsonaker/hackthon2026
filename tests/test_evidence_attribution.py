"""The evidence behind a score must be about *that* score.

`ForecasterService.build_evidence` used to rank features by global permutation
importance, so every alert showed the same handful of features whatever had
actually driven it, and the Investigator reasoned from a list that did not
describe the case in front of it. It now attributes each score by putting one
feature at a time back to its typical quiet value and measuring the drop.

A real (small) gradient-boosted model is trained here on data where the label
depends on exactly two features, additively and with noise. The noise matters:
a perfectly separable label makes the model saturate at 0 and 1, and when one
feature alone is sufficient, occluding the other changes nothing, which would
make every assertion below degenerate.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from conftest import ACCOUNT_ID, EPOCH, RANGE_ID, make_row
from sklearn.ensemble import HistGradientBoostingClassifier

from dfsi.agents.tools import EvidenceBundle
from dfsi.config import FEATURE_VERSION
from dfsi.forecaster.serve import EVIDENCE_LABELS, ForecasterService
from dfsi.forecaster.train import quiet_reference
from dfsi.memory.features import FEATURE_COLUMNS

FEATURES = list(FEATURE_COLUMNS)
#: The only two features the synthetic label depends on.
DRIVER_A = "growth_pps_60s"
DRIVER_B = "country_hhi"


def training_data(n: int = 8000, seed: int = 11) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({name: rng.normal(0.0, 1.0, n) for name in FEATURES})
    frame[DRIVER_A] = rng.uniform(0.0, 2.0, n)
    frame[DRIVER_B] = rng.uniform(0.0, 1.0, n)
    margin = (frame[DRIVER_A] - 1.0) + (frame[DRIVER_B] - 0.5)
    probability = 1.0 / (1.0 + np.exp(-6.0 * margin))
    return frame, (rng.random(n) < probability).astype(int)


@pytest.fixture(scope="module")
def trained():
    frame, label = training_data()
    model = HistGradientBoostingClassifier(max_iter=60, learning_rate=0.1, random_state=0)
    model.fit(frame, label)
    reference = quiet_reference(frame[FEATURES], label)
    return model, frame, label, reference


def save(tmp_path, model, frame, label, *, with_reference: bool = True) -> ForecasterService:
    payload = {
        "model": model,
        "features": FEATURES,
        "feature_version": FEATURE_VERSION,
        "decision_threshold": 0.5,
        "min_consecutive": 2,
        "model_version": "1",
        "mlflow_run_id": "",
        # A deliberately misleading global ranking: a feature the model ignores
        # tops it, so a test can tell per-alert attribution from the old ordering.
        "feature_importance": {"blocked_share": 0.9, DRIVER_A: 0.2, DRIVER_B: 0.1},
    }
    if with_reference:
        payload["feature_reference"] = quiet_reference(frame[FEATURES], label)
    path = tmp_path / "forecaster.pkl"
    path.write_bytes(pickle.dumps(payload))
    return ForecasterService(path)


def row_for(**overrides) -> dict:
    """A replay-shaped row with every model feature present."""
    row = make_row(epoch=EPOCH)
    row.update({name: 0.0 for name in FEATURES})
    row.update(overrides)
    return row


def forecast_for(service: ForecasterService, **features):
    forecast = service.score_window([row_for(**features)])
    assert not forecast.degraded
    return forecast


def by_feature(forecast) -> dict[str, dict]:
    return {item["feature"]: item for item in forecast.evidence}


# ---------------------------------------------------------------- reference


def test_the_quiet_reference_is_the_median_of_the_quiet_windows_only(trained):
    _, frame, label, reference = trained
    assert reference[DRIVER_A] == pytest.approx(frame.loc[label == 0, DRIVER_A].median())
    # Not the all-window median: the onset windows would pull it up.
    assert reference[DRIVER_A] < frame[DRIVER_A].median()
    assert set(reference) == set(FEATURES)


def test_a_training_split_with_no_quiet_window_is_rejected(trained):
    _, frame, _, _ = trained
    with pytest.raises(ValueError, match="no quiet windows"):
        quiet_reference(frame[FEATURES].head(10), np.ones(10, dtype=int))


# --------------------------------------------------------------- attribution


def test_the_driver_of_this_alert_leads_the_evidence(trained, tmp_path):
    model, frame, label, _ = trained
    service = save(tmp_path, model, frame, label)

    forecast = forecast_for(service, growth_pps_60s=1.8, country_hhi=0.9)

    assert forecast.evidence[0]["feature"] == DRIVER_A
    assert forecast.evidence[0]["contribution"] > 0.3
    # A feature the model ignores contributes (almost) nothing.
    assert abs(by_feature(forecast)["baseline_deviation_pps"]["contribution"]) < 0.05


def test_two_different_alerts_get_different_evidence(trained, tmp_path):
    """The property global importance could not give."""
    model, frame, label, _ = trained
    service = save(tmp_path, model, frame, label)

    # Growth is far out; the source mix is about where it usually is.
    growth_driven = forecast_for(service, growth_pps_60s=1.6, country_hhi=0.45)
    # The reverse: sources are very concentrated; growth is only a little up.
    source_driven = forecast_for(service, growth_pps_60s=0.8, country_hhi=0.98)

    assert growth_driven.evidence[0]["feature"] == DRIVER_A
    assert source_driven.evidence[0]["feature"] == DRIVER_B
    assert growth_driven.evidence[0]["contribution"] > 0.2
    assert source_driven.evidence[0]["contribution"] > 0.2


def test_the_ranking_is_not_the_global_importance_ordering(trained, tmp_path):
    model, frame, label, _ = trained
    service = save(tmp_path, model, frame, label)
    forecast = forecast_for(service, growth_pps_60s=1.8, country_hhi=0.9)
    # The stored global ranking puts blocked_share first; this alert is not about it.
    assert forecast.evidence[0]["feature"] != "blocked_share"


def test_a_contribution_is_the_score_drop_when_the_feature_is_reset(trained, tmp_path):
    model, frame, label, reference = trained
    service = save(tmp_path, model, frame, label)
    row = row_for(growth_pps_60s=1.8, country_hhi=0.9)

    forecast = service.score_window([row])
    reset = pd.DataFrame(
        [{**{name: row[name] for name in FEATURES}, DRIVER_A: reference[DRIVER_A]}]
    )
    expected = forecast.risk_score - model.predict_proba(reset)[0, 1]
    reported = by_feature(forecast)[DRIVER_A]["contribution"]

    assert reported > 0.2  # a real effect, not 0 == 0
    assert reported == pytest.approx(expected, abs=1e-3)


def test_a_feature_holding_risk_down_has_a_negative_contribution(trained, tmp_path):
    model, frame, label, _ = trained
    service = save(tmp_path, model, frame, label)
    # Growth is up, but the source mix is unusually diverse: the source mix is
    # what is holding this score down, which is exactly contrary evidence.
    evidence = by_feature(forecast_for(service, growth_pps_60s=1.2, country_hhi=0.0))

    assert evidence[DRIVER_A]["contribution"] > 0.05
    assert evidence[DRIVER_B]["contribution"] < -0.1


def test_evidence_is_at_most_top_n_and_only_labelled_features(trained, tmp_path):
    model, frame, label, _ = trained
    forecast = forecast_for(
        save(tmp_path, model, frame, label), growth_pps_60s=1.8, country_hhi=0.9
    )
    assert len(forecast.evidence) <= 6
    assert {item["feature"] for item in forecast.evidence} <= set(EVIDENCE_LABELS)


# ------------------------------------------------------------ compatibility


def test_a_model_saved_without_a_reference_keeps_the_old_ordering(trained, tmp_path):
    model, frame, label, _ = trained
    service = save(tmp_path, model, frame, label, with_reference=False)

    forecast = forecast_for(service, growth_pps_60s=1.8, country_hhi=0.9)

    assert all("contribution" not in item for item in forecast.evidence)
    assert forecast.evidence[0]["feature"] == "blocked_share"  # by global importance, as before


def test_an_attribution_failure_falls_back_instead_of_losing_the_forecast(trained, tmp_path):
    model, frame, label, _ = trained
    service = save(tmp_path, model, frame, label)
    real_predict = service._model.predict_proba
    calls = {"n": 0}

    def flaky(batch):
        calls["n"] += 1
        if calls["n"] == 2:  # scoring (call 1) succeeds; the attribution batch fails
            raise RuntimeError("boom")
        return real_predict(batch)

    service._model.predict_proba = flaky
    forecast = service.score_window([row_for(growth_pps_60s=1.8, country_hhi=0.9)])

    assert forecast.degraded is False and forecast.risk_score is not None
    assert forecast.evidence and all("contribution" not in item for item in forecast.evidence)


def test_a_non_finite_value_is_left_out_of_the_evidence(trained, tmp_path):
    model, frame, label, _ = trained
    service = save(tmp_path, model, frame, label)
    forecast = forecast_for(
        service, growth_pps_60s=1.8, country_hhi=0.9, growth_mbps_60s=float("nan")
    )
    assert "growth_mbps_60s" not in by_feature(forecast)


# ------------------------------------------------- what the agents may quote


def test_every_figure_the_evidence_tool_shows_is_citable():
    bundle = EvidenceBundle(
        range_id=RANGE_ID,
        account_id=ACCOUNT_ID,
        window_start_epoch=EPOCH,
        risk_score=0.8,
        risk_band="warn",
        model_version="t",
        feature_version="t",
        horizon_seconds=120,
        evidence=[
            {
                "feature": "country_hhi",
                "value": 0.68,
                "importance": 0.012345,
                "contribution": 0.2345,
            }
        ],
    )
    assert {"0.68", "0.2345", "0.0123"} <= bundle.citable_numbers()
