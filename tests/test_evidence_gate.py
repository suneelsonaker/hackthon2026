"""Context alone does not raise an alert.

The forecaster leans on context (recent attack history on the account) as well
as traffic, so an ordinary flash crowd on an account attacked last week could
alert with the traffic still inside its normal range. The gate requires the
traffic to be abnormal for the hour before a score above threshold becomes an
alert.

What these tests pin down: the rule itself; that training, calibration and live
scoring make *the same decision*; that the gate only ever withholds and never
acts on missing data; and that the band shown can still never contradict the
state raised.
"""

from __future__ import annotations

import pickle
import random

import numpy as np
import pandas as pd
import pytest
from conftest import EPOCH, make_row
from test_risk_bands import StubModel

from dfsi.api import replay as replay_engine
from dfsi.config import FEATURE_VERSION, STATE_ABOUT_TO_START, STATE_MONITORING_ONLY
from dfsi.forecaster.gate import EVIDENCE_FEATURE, apply_gate, has_traffic_evidence
from dfsi.forecaster.serve import ForecasterService
from dfsi.forecaster.train import TrainConfig, _describe_gate, calibrate_threshold
from dfsi.memory.features import FEATURE_COLUMNS

GATE = 3.0
THRESHOLD = 0.6


# ------------------------------------------------------------------- the rule


def test_no_gate_means_no_requirement():
    assert has_traffic_evidence(0.0, None) is True


@pytest.mark.parametrize(
    ("deviation", "expected"),
    [(5.0, True), (3.01, True), (3.0, False), (1.2, False), (0.0, False), (-8.0, False)],
)
def test_the_traffic_must_be_more_abnormal_than_the_bar(deviation, expected):
    """Strictly more: at the bar is not enough. A *drop* is abnormal but is not
    evidence of an attack, which adds traffic."""
    assert has_traffic_evidence(deviation, GATE) is expected


@pytest.mark.parametrize("unknown", [None, float("nan"), float("inf"), float("-inf")])
def test_unknown_evidence_never_withholds_an_alert(unknown):
    """Not knowing is not a reason to stay silent, any more than to say all-clear."""
    assert has_traffic_evidence(unknown, GATE) is True


def scored(scores, deviations) -> pd.DataFrame:
    return pd.DataFrame({"risk_score": scores, EVIDENCE_FEATURE: deviations})


def test_apply_gate_zeroes_only_the_windows_without_evidence():
    frame = scored([0.9, 0.9, 0.9, 0.2], [5.0, 1.0, float("nan"), 0.5])
    assert apply_gate(frame, "risk_score", GATE).tolist() == [0.9, 0.0, 0.9, 0.0]


def test_apply_gate_without_a_bar_returns_the_scores_unchanged():
    frame = scored([0.9, 0.2], [0.0, 0.0])
    assert apply_gate(frame, "risk_score", None).tolist() == [0.9, 0.2]


def test_apply_gate_does_not_touch_the_input_frame():
    frame = scored([0.9], [0.0])
    apply_gate(frame, "risk_score", GATE)
    assert frame["risk_score"].tolist() == [0.9]


def test_a_missing_evidence_column_gates_nothing():
    assert apply_gate(pd.DataFrame({"risk_score": [0.9]}), "risk_score", GATE).tolist() == [0.9]


def test_the_gate_never_raises_a_score():
    rng = random.Random(3)
    frame = scored([rng.random() for _ in range(500)], [rng.uniform(-5, 10) for _ in range(500)])
    gated = apply_gate(frame, "risk_score", GATE)
    assert (gated <= frame["risk_score"]).all()


# ---------------------------------------------------------- calibration and training


class CarrierModel:
    """Reads the score straight from a feature column, so a test sets it directly."""

    def predict_proba(self, frame):
        s = frame["growth_pps_60s"].to_numpy(dtype=float)
        return np.column_stack([1 - s, s])


def calibration_frame() -> pd.DataFrame:
    """Three scenarios: a real attack, a quiet one, and a quiet one that the
    *context* pushes to 0.9 for three windows while the traffic stays normal."""
    rows = []
    layout = {
        "attack": ("gradual_buildup", [0.1, 0.1, 0.9, 0.9, 0.9, 0.9], [0.2, 0.2, 6, 8, 9, 9], 5),
        "calm": ("normal", [0.1] * 6, [0.2] * 6, None),
        "hot_context": (
            "flash_crowd",
            [0.1, 0.9, 0.9, 0.9, 0.1, 0.1],
            [0.2, 1.0, 1.1, 1.3, 0.2, 0.2],
            None,
        ),
    }
    for sid, (kind, scores, deviations, onset) in layout.items():
        for i, (score, dev) in enumerate(zip(scores, deviations)):
            row = {name: 0.0 for name in FEATURE_COLUMNS}
            row.update(
                growth_pps_60s=score,
                scenario_id=sid,
                scenario_type=kind,
                window_start_epoch=EPOCH + i * 10,
                is_onset=int(onset is not None and i == onset),
                range_over_threshold=0.0,
            )
            row[EVIDENCE_FEATURE] = dev
            rows.append(row)
    return pd.DataFrame(rows)


def test_the_threshold_is_calibrated_on_the_raw_score_whatever_the_gate():
    """Sweeping the gated score made the choice lurch (0.85 vs 0.15 for a one-point
    budget change); the threshold belongs to the model, the gate sits on top."""
    frame = calibration_frame()
    base = TrainConfig(false_alarm_budget=0.03, min_consecutive=2)
    gated, gated_sweep = calibrate_threshold(CarrierModel(), frame, base)
    ungated, ungated_sweep = calibrate_threshold(
        CarrierModel(), frame, TrainConfig(**{**base.__dict__, "evidence_gate_deviation": None})
    )
    assert gated == ungated
    assert gated_sweep == ungated_sweep


def test_the_sweep_sees_the_context_driven_false_alarm_the_gate_exists_to_remove():
    """The raw sweep counts the flash crowd as a false alarm; the gate removes it later."""
    _, sweep = calibrate_threshold(
        CarrierModel(), calibration_frame(), TrainConfig(min_consecutive=2)
    )
    at_half = next(row for row in sweep if row["threshold"] == 0.5)
    assert at_half["false_alarm_rate"] == pytest.approx(0.5)  # 1 of 2 quiet scenarios alerts
    frame = calibration_frame()
    frame["risk_score"] = CarrierModel().predict_proba(frame)[:, 1]
    frame["alert_score"] = apply_gate(frame, "risk_score", GATE)
    from dfsi.forecaster.metrics import evaluate

    gated = evaluate(frame, "alert_score", 0.5, 2)
    assert gated.false_alarm_rate == 0.0 and gated.attacks_caught == 1.0


def test_the_report_says_how_many_alerting_windows_the_gate_withheld():
    frame = calibration_frame()
    frame["risk_score"] = CarrierModel().predict_proba(frame)[:, 1]
    frame["alert_score"] = apply_gate(frame, "risk_score", GATE)
    report = _describe_gate(frame, 0.6, TrainConfig())
    assert report["min_deviation"] == GATE and report["feature"] == EVIDENCE_FEATURE
    assert report["windows_above_threshold"] == 7  # 4 attack windows + 3 context windows
    assert report["windows_withheld"] == 3  # the context-driven ones only
    assert set(report["per_scenario_type_without_gate"]) == {
        "gradual_buildup",
        "normal",
        "flash_crowd",
    }


def test_the_default_bar_is_the_pipelines_own_abnormal_window_threshold():
    from dfsi.config import ANOMALY_DEVIATION_THRESHOLD

    assert TrainConfig().evidence_gate_deviation == ANOMALY_DEVIATION_THRESHOLD


# ------------------------------------------------------------------ live scoring


def service(tmp_path, *, score: float, gate: float | None = GATE, threshold: float = THRESHOLD):
    payload = {
        "model": StubModel(score),
        "features": list(FEATURE_COLUMNS),
        "feature_version": FEATURE_VERSION,
        "decision_threshold": threshold,
        "min_consecutive": 2,
        "model_version": "1",
        "mlflow_run_id": "",
        "feature_importance": {},
        "evidence_gate_deviation": gate,
    }
    path = tmp_path / "forecaster.pkl"
    path.write_bytes(pickle.dumps(payload))
    return ForecasterService(path)


def forecast(svc, deviation, traffic_types=("udp",)):
    rows = [
        make_row(epoch=EPOCH, traffic_type=t, baseline_deviation_pps_max_hour=deviation)
        for t in traffic_types
    ]
    return svc.score_window(rows)


def test_a_high_score_without_traffic_evidence_raises_no_alert_and_says_why(tmp_path):
    result = forecast(service(tmp_path, score=0.9), deviation=1.0)

    assert result.risk_score == pytest.approx(0.9)  # the model's number is untouched
    assert result.predicted_state == "QUIET"
    assert result.gated is True
    assert "not an all-clear" in result.gate_reason and "1.0" in result.gate_reason
    assert result.degraded is False


def test_a_high_score_with_traffic_evidence_raises_the_alert(tmp_path):
    result = forecast(service(tmp_path, score=0.9), deviation=7.0)
    assert result.predicted_state == STATE_ABOUT_TO_START
    assert result.gated is False and result.gate_reason is None


def test_a_gated_window_never_shows_a_warning_band(tmp_path):
    result = forecast(service(tmp_path, score=0.99), deviation=0.5)
    assert result.risk_band == "watch"  # not warn/critical: no alert is raised


def test_a_score_below_threshold_is_quiet_not_gated(tmp_path):
    result = forecast(service(tmp_path, score=0.2), deviation=0.0)
    assert result.predicted_state == "QUIET" and result.gated is False


def test_a_model_saved_before_the_gate_behaves_exactly_as_it_did(tmp_path):
    svc = service(tmp_path, score=0.9, gate=None)
    result = forecast(svc, deviation=0.0)
    assert result.predicted_state == STATE_ABOUT_TO_START and result.gated is False


@pytest.mark.parametrize("unknown", [None, float("nan")])
def test_unknown_traffic_evidence_still_alerts(tmp_path, unknown):
    result = forecast(service(tmp_path, score=0.9), deviation=unknown)
    assert result.predicted_state == STATE_ABOUT_TO_START and result.gated is False


def test_the_strongest_vector_decides_across_traffic_types(tmp_path):
    svc = service(tmp_path, score=0.9)
    rows = [
        make_row(epoch=EPOCH, traffic_type="udp", baseline_deviation_pps_max_hour=0.5),
        make_row(epoch=EPOCH, traffic_type="http", baseline_deviation_pps_max_hour=8.0),
    ]
    assert svc.score_window(rows).predicted_state == STATE_ABOUT_TO_START


def test_the_band_and_the_state_can_never_disagree(tmp_path):
    """The invariant from the risk-band change, now that a gate can withhold alerts."""
    for score in (0.1, 0.35, 0.59, 0.6, 0.61, 0.8, 0.99):
        for deviation in (None, -4.0, 0.0, 3.0, 3.1, 12.0):
            result = forecast(service(tmp_path, score=score), deviation)
            in_warning_band = result.risk_band in ("warn", "critical")
            assert in_warning_band == (result.predicted_state == STATE_ABOUT_TO_START), (
                score,
                deviation,
            )


def test_training_and_serving_make_the_same_decision(tmp_path):
    """Parity: `apply_gate` (calibration, metrics) and the service agree on every case."""
    rng = random.Random(11)
    for _ in range(120):
        score = round(rng.random(), 2)
        deviation = rng.choice([None, float("nan"), round(rng.uniform(-6, 12), 2), GATE])
        frame = scored([score], [np.nan if deviation is None else deviation])
        trained_would_alert = apply_gate(frame, "risk_score", GATE).iloc[0] >= THRESHOLD
        live = forecast(service(tmp_path, score=score), deviation).predicted_state
        assert trained_would_alert == (live == STATE_ABOUT_TO_START), (score, deviation)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_a_corrupt_gate_makes_the_model_unusable_rather_than_silently_wrong(tmp_path, bad):
    svc = service(tmp_path, score=0.9, gate=bad)
    assert svc.ready is False and "evidence gate" in (svc.load_error or "")
    assert svc.score_window([make_row(epoch=EPOCH)]).predicted_state == STATE_MONITORING_ONLY


def test_the_gate_is_in_the_serialised_forecast(tmp_path):
    payload = forecast(service(tmp_path, score=0.9), deviation=0.0).to_dict()
    assert payload["gated"] is True and payload["gate_reason"]
    quiet = forecast(service(tmp_path, score=0.1), deviation=0.0).to_dict()
    assert quiet["gated"] is False and quiet["gate_reason"] is None


# ----------------------------------------------------------------- the live stream


def stream_state(svc, deviations):
    rows = []
    for i, dev in enumerate(deviations):
        rows.append(make_row(epoch=EPOCH + i * 10, baseline_deviation_pps_max_hour=dev))
    replay = {
        "replay_id": "gated",
        "scenario_type": "flash_crowd",
        "range_id": rows[0]["range_id"],
        "windows": rows,
    }
    state = replay_engine.build_state(replay)
    events = []
    while not state.finished:
        events.append(replay_engine.step(state, svc))
    return state, events


def test_a_context_driven_stretch_never_warns_on_the_stream(tmp_path):
    state, events = stream_state(service(tmp_path, score=0.95), [0.5, 1.0, 1.2, 0.8, 1.1, 0.9])
    assert state.first_warning_epoch is None
    assert all(e["dfsi_state"] != STATE_ABOUT_TO_START for e in events)
    assert all(e["forecast"]["gated"] for e in events)


def test_the_same_scores_with_traffic_evidence_do_warn(tmp_path):
    state, _ = stream_state(service(tmp_path, score=0.95), [0.5, 6.0, 7.0, 8.0])
    assert state.first_warning_epoch == EPOCH + 20


def test_a_stretch_that_becomes_abnormal_warns_when_it_does(tmp_path):
    """Gating delays an alert until the traffic corroborates it; it does not drop it."""
    state, _ = stream_state(service(tmp_path, score=0.95), [0.5, 0.8, 1.0, 5.0, 6.0, 7.0])
    assert state.first_warning_epoch == EPOCH + 40


# ------------------------------------------------- the gate and the alert lifecycle agree


def test_a_context_driven_stretch_opens_no_lifecycle_alert(tmp_path):
    """The dashboard says QUIET; the lifecycle must not contradict it with an open alert."""
    state, events = stream_state(service(tmp_path, score=0.95), [0.5, 1.0, 1.2, 0.8, 1.1, 0.9])
    assert all(not e["alert"]["is_open"] for e in events)
    assert state.lifecycle.summary() == []
    assert {e["alert"]["state"] for e in events} == {"WATCH"}


def test_the_lifecycle_raises_on_the_same_window_the_dashboard_warns(tmp_path):
    state, events = stream_state(service(tmp_path, score=0.95), [0.5, 0.8, 1.0, 5.0, 6.0, 7.0])
    raised = next(e["window_start_epoch"] for e in events if e["alert"]["event"] == "raised")
    # Backed windows start at index 3; the run of two completes at index 4.
    assert raised == state.first_warning_epoch == EPOCH + 40


def test_an_open_alert_is_not_closed_by_a_stretch_without_traffic_evidence(tmp_path):
    deviations = [6.0, 7.0, 8.0] + [0.5] * 12  # a real attack, then deviation drops below the bar
    state, events = stream_state(service(tmp_path, score=0.95), deviations)
    assert all(e["alert"]["is_open"] for e in events[1:])
    assert len(state.lifecycle.summary()) == 1
