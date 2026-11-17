"""Every route, through HTTP, on an app built with injected fakes.

No Memory layer, no trained model, no network: the forecaster is scripted, the
replay loader returns a hand-built replay, and the explainers run offline. What
is under test is the HTTP contract — status codes, payload shapes, error
mapping, isolation — rather than the models behind it.
"""

from __future__ import annotations

import dataclasses
import importlib
import json

import pytest
from conftest import RANGE_ID, make_client, make_services
from fastapi.testclient import TestClient
from test_alerting import EPOCH, ScriptedForecaster, make_replay

from dfsi.api.app import create_app
from dfsi.api.replay import ReplayCatalogueEntry
from dfsi.api.services import DfsiServices
from dfsi.config import Paths
from dfsi.rehearsal.policy_advisor import PolicyPlan, TypeSuggestion

SCORES = [0.1, 0.2, 0.9, 0.9, 0.9, 0.9, 0.3, 0.2]
REPLAY = make_replay(len(SCORES), onset_index=5)


class FakeForecaster(ScriptedForecaster):
    ready = True
    load_error = None
    model_version = "test/v1"
    feature_version = "features-test"


def load(replay_id: str) -> dict:
    if replay_id != "scripted":
        raise KeyError(replay_id)
    return REPLAY


ENTRY = ReplayCatalogueEntry("scripted", "gradual_buildup", RANGE_ID, True, len(SCORES))


@pytest.fixture
def services(tmp_path) -> DfsiServices:
    return make_services(
        tmp_path,
        forecaster=FakeForecaster(SCORES),
        load_stored_replay=load,
        plan_policy=lambda *a: PolicyPlan(available=False, range_id=RANGE_ID, reason="test"),
        range_thresholds={},
        load_catalogue=lambda: [],
    )


@pytest.fixture
def client(services) -> TestClient:
    return TestClient(create_app(services))


def response_text(stream: list[dict]) -> str:
    return json.dumps(stream)


def events(response) -> list[dict]:
    return [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
    ]


# ------------------------------------------------------------------ status


def test_health_is_degraded_without_replays(client):
    body = client.get("/api/health").json()
    assert body["status"] == "degraded" and body["replays_available"] == 0
    assert body["forecaster_ready"] is True and body["agent_engine"] == "offline-rules"


def test_replays_is_503_until_memory_is_built(client):
    response = client.get("/api/replays")
    assert response.status_code == 503 and "memory" in response.json()["detail"].lower()


def test_with_replays_and_a_model_the_service_is_ok(services, client):
    services.load_catalogue = lambda: [ENTRY]
    assert client.get("/api/health").json()["status"] == "ok"
    body = client.get("/api/replays").json()
    assert body == {"replays": [ENTRY.to_dict()], "scenario_types": ["gradual_buildup"]}


def test_model_card_is_503_without_a_training_report(services, client, tmp_path):
    services.paths = Paths(home=tmp_path)
    assert client.get("/api/model-card").status_code == 503


def test_model_card_reads_the_training_report(services, client, tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "training_report.json").write_text(
        json.dumps(
            {
                "decision_threshold": 0.55,
                "evidence_gate": {"bar": 3.0, "per_scenario_type_without_gate": {}},
            }
        )
    )
    services.paths = Paths(home=tmp_path)
    body = client.get("/api/model-card").json()
    assert body["decision_threshold"] == 0.55
    assert body["evidence_gate"] == {"bar": 3.0}  # the bulky comparison is dropped
    # No sweep in this report, so calibration cannot be claimed either way.
    assert body["calibration"] is None
    assert any("synthetic" in item for item in body["known_limitations"])


def test_model_card_states_a_fallback_threshold_as_such(services, client, tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "training_report.json").write_text(
        json.dumps(
            {
                "decision_threshold": 0.55,
                "config": {"false_alarm_budget": 0.03},
                "threshold_sweep": [{"threshold": 0.95, "false_alarm_rate": 0.042}],
            }
        )
    )
    services.paths = Paths(home=tmp_path)
    calibration = client.get("/api/model-card").json()["calibration"]
    assert calibration["source"] == "fallback" and calibration["met_budget"] is False


# ------------------------------------------------------------- domain errors


@pytest.mark.parametrize(
    "method, path, body",
    [
        ("post", "/api/replay/stream", {"replay_id": "nope"}),
        ("post", "/api/drill", {"replay_id": "nope"}),
        ("post", "/api/explain", {"replay_id": "nope"}),
        (
            "post",
            "/api/decision",
            {
                "replay_id": "nope",
                "decision_epoch": EPOCH,
                "policy_key": "hold",
                "verdict": "approved",
            },
        ),
    ],
)
def test_an_unknown_replay_is_404_everywhere(client, method, path, body):
    response = getattr(client, method)(path, json=body)
    assert response.status_code == 404
    assert response.json() == {"detail": "unknown replay 'nope'"}


@pytest.mark.parametrize("path", ["/api/drill", "/api/explain"])
def test_an_epoch_that_is_not_a_window_is_400(client, path):
    response = client.post(path, json={"replay_id": "scripted", "decision_epoch": EPOCH + 5})
    assert response.status_code == 400 and "not a window" in response.json()["detail"]


def test_a_wait_that_is_not_whole_windows_is_422(client):
    response = client.post(
        "/api/drill",
        json={"replay_id": "scripted", "include_waiting": True, "waiting_seconds": [15]},
    )
    assert response.status_code == 422


def test_request_validation_is_422(client):
    assert client.post("/api/drill", json={}).status_code == 422
    assert client.get("/api/decisions?limit=0").status_code == 422
    assert client.get("/api/decisions?limit=501").status_code == 422


# ------------------------------------------------------------------- stream


def test_the_stream_opens_with_start_and_ends_with_summary(client):
    response = client.post("/api/replay/stream", json={"replay_id": "scripted", "speed_seconds": 0})
    assert response.headers["content-type"].startswith("text/event-stream")
    stream = events(response)
    assert stream[0]["type"] == "start" and stream[0]["total_windows"] == len(SCORES)
    assert stream[-1]["type"] == "summary"
    assert stream[-1]["first_warning_epoch"] == EPOCH + 30  # second window over: debounced


def test_a_failing_step_becomes_an_error_event_not_a_crash(client, monkeypatch):
    from dfsi.api.routes import replays

    def boom(*_):
        raise RuntimeError("scripted failure")

    monkeypatch.setattr(replays.replay_engine, "step", boom)
    stream = events(
        client.post("/api/replay/stream", json={"replay_id": "scripted", "speed_seconds": 0})
    )
    assert stream[-1] == {
        "type": "error",
        "message": "replay failed (RuntimeError); see the API log for details",
    }
    assert "scripted failure" not in response_text(stream)


# ---------------------------------------------------------- drill / explain


def test_drill_defaults_to_the_first_warned_window(client):
    body = client.post("/api/drill", json={"replay_id": "scripted"}).json()
    assert body["decision_epoch"] == EPOCH + 30
    assert {c["policy_key"] for c in body["candidates"]} == {
        "hold",
        "tighten_vector",
        "rate_limit_sources",
        "full_scrub",
    }
    assert body["simulation_only"] is True


def test_explain_returns_both_agents_with_validator_checks(client):
    body = client.post("/api/explain", json={"replay_id": "scripted"}).json()
    assert body["decision_epoch"] == EPOCH + 30
    for agent in ("investigator", "advisor"):
        checks = body[agent]["validator"]["checks"]
        assert body[agent]["validated"] is all(checks.values())
        assert {"tenant_scope", "evidence", "citations", "allowed_actions"} <= set(checks)


# ---------------------------------------------------------------- decisions


def decide(client, **fields):
    body = {
        "replay_id": "scripted",
        "decision_epoch": EPOCH + 30,
        "policy_key": "hold",
        "verdict": "approved",
    }
    body.update(fields)
    return client.post("/api/decision", json=body)


def test_a_decision_is_audited_and_read_back_newest_first(client, tmp_path):
    first = decide(client, note="first").json()
    decide(client, note="second", verdict="rejected")
    assert first["applied"] is False and first["policy_key"] == "hold"
    notes = [d["note"] for d in client.get("/api/decisions").json()["decisions"]]
    assert notes == ["second", "first"]
    assert len((tmp_path / "decisions.jsonl").read_text().splitlines()) == 2


def test_a_decision_at_an_epoch_outside_the_replay_is_400(client, tmp_path):
    response = decide(client, decision_epoch=-999)
    assert response.status_code == 400 and "not a window" in response.json()["detail"]
    assert not (tmp_path / "decisions.jsonl").exists()  # nothing was audited


def test_the_audit_states_the_servers_model_and_flags_a_stale_dashboard(client):
    honest = decide(client, model_version="test/v1").json()
    assert honest["model_version"] == "test/v1" and honest["client_model_version"] is None
    stale = decide(client, model_version="forged-v9").json()
    assert stale["model_version"] == "test/v1"  # never the client's word
    assert stale["client_model_version"] == "forged-v9"


@pytest.mark.parametrize(
    "fields, problem",
    [
        ({"verdict": "changed"}, "needs the recommended_policy_key"),
        (
            {"verdict": "approved", "policy_key": "full_scrub", "recommended_policy_key": "hold"},
            "use 'changed'",
        ),
    ],
)
def test_a_verdict_that_contradicts_the_pick_is_422(client, tmp_path, fields, problem):
    response = decide(client, **fields)
    assert response.status_code == 422 and problem in response.text
    assert not (tmp_path / "decisions.jsonl").exists()


def test_concurrent_approvals_each_record_the_version_they_actually_moved(services, client):
    """The reviewer's reproduction: a slow planner, two approvals at once."""
    import threading
    import time as clock

    lines = iter([11_000.0, 12_000.0])

    def slow_plan(*_):
        clock.sleep(0.2)
        suggestion = dataclasses.replace(tighten_udp(), suggested_policy_pps=next(lines))
        return PolicyPlan(available=True, range_id=RANGE_ID, per_type=[suggestion])

    services.plan_policy = slow_plan
    records: list[dict] = []
    threads = [
        threading.Thread(
            target=lambda: records.append(decide(client, policy_key="tighten_vector").json())
        )
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    first, second = sorted(records, key=lambda r: r["policy_version_after"])
    assert first["policy_version_after"] == second["policy_version_before"]  # a chain, no fork


def test_changed_records_which_recommendation_was_overridden(client):
    record = decide(
        client, verdict="changed", policy_key="full_scrub", recommended_policy_key="hold"
    ).json()
    assert record["verdict"] == "changed" and record["recommended_policy_key"] == "hold"


def test_changed_to_the_recommendation_itself_is_a_contradiction(client):
    response = decide(client, verdict="changed", policy_key="hold", recommended_policy_key="hold")
    assert response.status_code == 422 and "overrides the recommendation" in response.text


def test_an_approved_tighten_without_a_plan_changes_nothing(client):
    record = decide(client, policy_key="tighten_vector").json()
    assert record["applied"] is False
    assert record["policy_version_before"] == record["policy_version_after"]
    assert record["policy_change"]["basis"] == "stated assumption"


def test_an_approved_tighten_with_a_plan_updates_the_simulated_store(services, client):
    plan = PolicyPlan(available=True, range_id=RANGE_ID, per_type=[tighten_udp()])
    services.plan_policy = lambda *a: plan
    record = decide(client, policy_key="tighten_vector").json()
    assert record["applied"] is True
    assert record["policy_version_after"] != record["policy_version_before"]
    assert "simulated policy store" in record["note_to_auditor"]

    reset = client.post("/api/policies/reset").json()
    assert reset == client.get("/api/policies").json()


def tighten_udp() -> TypeSuggestion:
    fields = {f.name: 0.0 for f in dataclasses.fields(TypeSuggestion)}
    fields.update(
        traffic_type="udp",
        current_policy_pps=40_000.0,
        suggested_policy_pps=12_000.0,
        change_pct=-0.7,
        changes=True,
        suspect=True,
        apply=True,
    )
    return TypeSuggestion(**fields)


def test_risk_thresholds_reports_the_model_default_and_overrides(services, client):
    services.range_thresholds = {RANGE_ID: 0.7}
    body = client.get("/api/risk-thresholds").json()
    assert body == {
        "model_default": services.forecaster.decision_threshold,
        "ranges": {str(RANGE_ID): 0.7},
    }


# ------------------------------------------------------------------- wiring


def test_two_apps_never_share_state(tmp_path):
    a = make_client(tmp_path / "a", forecaster=FakeForecaster(SCORES), load_stored_replay=load)
    b = make_client(tmp_path / "b", forecaster=FakeForecaster(SCORES), load_stored_replay=load)
    decide(a)
    assert len(a.get("/api/decisions").json()["decisions"]) == 1
    assert b.get("/api/decisions").json()["decisions"] == []


def test_an_app_without_services_builds_them_once_on_first_use(monkeypatch, services):
    built = []

    def fake_build():
        built.append(1)
        return services

    monkeypatch.setattr("dfsi.api.deps.build_services", fake_build)
    client = TestClient(create_app())
    client.get("/api/risk-thresholds")
    client.get("/api/risk-thresholds")
    assert built == [1]


def test_cors_allows_only_the_configured_origins(monkeypatch, services):
    monkeypatch.setenv("DFSI_CORS_ORIGINS", "http://dash.example")
    client = TestClient(create_app(services))
    allowed = client.get("/api/risk-thresholds", headers={"Origin": "http://dash.example"})
    denied = client.get("/api/risk-thresholds", headers={"Origin": "http://evil.example"})
    assert allowed.headers.get("access-control-allow-origin") == "http://dash.example"
    assert "access-control-allow-origin" not in denied.headers


def test_the_module_level_app_is_importable_without_side_effects():
    app_module = importlib.import_module("dfsi.api.app")  # the module; `dfsi.api.app` is the app
    assert app_module.app.state.services is None  # nothing built at import time
