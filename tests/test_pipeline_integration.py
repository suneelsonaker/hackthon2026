"""End to end: Memory (PySpark) -> Forecaster (train) -> API, in a throwaway DFSI_HOME.

The unit tests build rows by hand so they run without a JDK. This one runs the
real build exactly as `scripts/run.sh` does, on a two-week synthetic history
(training on half its scenarios, as CI does, to fit a 16 GB runner), and then
serves a replay from what it built. It takes about five minutes and needs a JDK
11/17/21, so it is opt-in locally (`pytest -m integration`) and always runs in
CI.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        shutil.which("java") is None and not os.getenv("JAVA_HOME"), reason="needs a JDK"
    ),
]


def run_stage(home: Path, module: str, *args: str) -> None:
    env = {
        **os.environ,
        "DFSI_HOME": str(home),
        # Sized for a 16 GB CI runner unless the caller says otherwise.
        "DFSI_SPARK_DRIVER_MEMORY": os.getenv("DFSI_SPARK_DRIVER_MEMORY", "8g"),
        "DFSI_POLICY_DB": str(ROOT / "raw.json"),
        "MLFLOW_TRACKING_URI": (home / "mlruns").as_uri(),
        "PYTHONPATH": str(ROOT),
    }
    done = subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,  # a 4-vCPU runner is several times slower than a laptop
    )
    assert done.returncode == 0, f"{module} failed:\n{done.stdout[-3000:]}\n{done.stderr[-3000:]}"


@pytest.fixture(scope="module")
def built_home(tmp_path_factory) -> Path:
    home = tmp_path_factory.mktemp("dfsi_home")
    run_stage(home, "dfsi.memory.pipeline", "--days", "14")
    run_stage(home, "dfsi.forecaster.train", "--split", "scenario", "--sample-fraction", "0.5")
    return home


def test_memory_writes_features_replays_and_no_addresses(built_home):
    data = built_home / "data"
    assert any((data / "features").rglob("*.parquet"))
    catalogue = json.loads((data / "replay" / "catalogue.json").read_text())
    assert catalogue, "no replays written"
    for entry in catalogue:
        replay = (data / "replay" / f"{entry['replay_id']}.json").read_text()
        assert "src_ip" not in replay and "ip_address" not in replay


def test_training_writes_a_model_and_an_honest_report(built_home):
    assert (built_home / "models" / "forecaster.pkl").exists()
    report = json.loads((built_home / "models" / "training_report.json").read_text())
    assert 0.0 < report["decision_threshold"] < 1.0
    assert "scenario_id" not in report["features"] and "scenario_type" not in report["features"]
    assert report["fixed_threshold_baseline"]["median_lead_seconds"] in (0, 0.0, None)


def test_the_api_serves_a_replay_from_the_built_artefacts(built_home, tmp_path):
    from fastapi.testclient import TestClient

    from dfsi.agents.orchestrator import ExplainerOrchestrator
    from dfsi.api import replay as replay_engine
    from dfsi.api.app import create_app
    from dfsi.api.audit import AuditLog
    from dfsi.api.services import build_services
    from dfsi.config import Paths
    from dfsi.forecaster.serve import ForecasterService

    replay_dir = built_home / "data" / "replay"

    def load(replay_id: str) -> dict:
        path = replay_dir / f"{replay_id}.json"
        if not path.exists():
            raise KeyError(replay_id)
        return json.loads(path.read_text())

    def catalogue():
        return [
            replay_engine.ReplayCatalogueEntry(**entry)
            for entry in json.loads((replay_dir / "catalogue.json").read_text())
        ]

    services = build_services(
        Paths(home=built_home),
        forecaster=ForecasterService(built_home / "models" / "forecaster.pkl"),
        orchestrator=ExplainerOrchestrator(api_key=None),
        audit_log=AuditLog(tmp_path / "decisions.jsonl"),
        load_stored_replay=load,
        load_catalogue=catalogue,
    )
    client = TestClient(create_app(services))
    assert client.get("/api/health").json()["status"] == "ok"

    replay_id = catalogue()[0].replay_id
    response = client.post("/api/replay/stream", json={"replay_id": replay_id, "speed_seconds": 0})
    events = [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert events[0]["type"] == "start" and events[-1]["type"] == "summary"
    assert client.post("/api/drill", json={"replay_id": replay_id}).status_code == 200
    briefing = client.post("/api/explain", json={"replay_id": replay_id}).json()
    assert briefing["state"] in {"EXPLAINED", "MONITORING_ONLY"}
