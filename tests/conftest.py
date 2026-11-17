"""Shared fixtures.

The builders here produce replay-shaped rows by hand rather than loading the
generated Memory layer, so the behavioural tests run without PySpark, without a
JDK, and without a trained model. The few tests that genuinely need the built
artefacts skip when they are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dfsi.config import PATHS, STAGE_THRESHOLDS

EPOCH = 1_767_225_600
RANGE_ID = 90110001
ACCOUNT_ID = 421001
TRAFFIC_TYPES = ("udp", "tcp_syn", "http")

#: Comfortably above the stage start floor of 5 kpps, so a flood that clears
#: the policy line also clears the absolute floor.
POLICY_PPS = 40_000.0


def make_row(
    *,
    epoch: int,
    traffic_type: str = "udp",
    total_pps: float = 10_000.0,
    policy_pps: float = POLICY_PPS,
    baseline_deviation_pps: float = 0.2,
    country_hhi: float = 0.14,
    blocked_pps: float = 0.0,
    **overrides,
) -> dict:
    """One replay row: a single traffic type on one range at one instant."""
    row = {
        "scenario_id": "scenario-test-0001",
        "scenario_type": "gradual_buildup",
        "window_start_epoch": epoch,
        "range_id": RANGE_ID,
        "account_id": ACCOUNT_ID,
        "range_type": "ipv4",
        "traffic_type": traffic_type,
        "total_pps": total_pps,
        "total_mbps": total_pps * 1_100 * 8 / 1e6,
        "blocked_pps": blocked_pps,
        "blocked_mbps": blocked_pps * 1_100 * 8 / 1e6,
        "policy_pps": policy_pps,
        "policy_mbps": policy_pps * 1_100 * 8 / 1e6,
        "country_hhi": country_hhi,
        "account_attack_history": 0,
        "growth_pps_10s": 0.0,
        "growth_pps_60s": 0.0,
        "growth_mbps_10s": 0.0,
        "growth_mbps_60s": 0.0,
        "baseline_deviation_pps": baseline_deviation_pps,
        "baseline_deviation_mbps": baseline_deviation_pps,
        "baseline_deviation_pps_max_hour": baseline_deviation_pps,
        "country_hhi_delta_60s": 0.0,
        "blocked_share": blocked_pps / total_pps if total_pps else 0.0,
        "blocked_share_delta_10s": 0.0,
        "blocked_share_delta_60s": 0.0,
        "anomaly_persistence": 0,
        "policy_utilisation": total_pps / policy_pps if policy_pps else 0.0,
        "policy_utilisation_mbps": total_pps / policy_pps if policy_pps else 0.0,
        "range_over_threshold": 0,
        "declared": 0,
        "is_onset": 0,
        "label_onset_120s": 0,
    }
    row.update(overrides)
    return row


def quiet_windows(count: int = 12, start: int = EPOCH) -> list[dict]:
    """A range sitting well inside its policy on all three traffic types."""
    rows: list[dict] = []
    for i in range(count):
        epoch = start + i * 10
        for traffic_type in TRAFFIC_TYPES:
            rows.append(make_row(epoch=epoch, traffic_type=traffic_type))
    return rows


def attack_windows(
    count: int = 24,
    start: int = EPOCH,
    onset_index: int = 6,
    peak_pps: float = 300_000.0,
) -> list[dict]:
    """A single-vector UDP flood that ramps past the policy line on `udp`.

    The other two traffic types stay quiet throughout. That is deliberate: a
    simulator that sums the policy line across traffic types would let this
    flood hide behind the range's total headroom.
    """
    rows: list[dict] = []
    for i in range(count):
        epoch = start + i * 10
        if i < onset_index:
            attack_pps = 0.0
        else:
            ramp = min(1.0, (i - onset_index + 1) / 6.0)
            attack_pps = peak_pps * ramp
        udp_total = 10_000.0 + attack_pps
        blocked = max(0.0, udp_total - POLICY_PPS)
        over = int(blocked >= POLICY_PPS * STAGE_THRESHOLDS.start_threshold_ratio)
        rows.append(
            make_row(
                epoch=epoch,
                traffic_type="udp",
                total_pps=udp_total,
                blocked_pps=blocked,
                baseline_deviation_pps=0.2 if attack_pps == 0 else 18.0,
                country_hhi=0.14 if attack_pps == 0 else 0.68,
                range_over_threshold=over,
                is_onset=int(i == onset_index + 3),
                declared=int(i >= onset_index + 3),
            )
        )
        for traffic_type in ("tcp_syn", "http"):
            rows.append(make_row(epoch=epoch, traffic_type=traffic_type))
    return rows


@pytest.fixture
def quiet_replay() -> list[dict]:
    return quiet_windows()


@pytest.fixture
def attack_replay() -> list[dict]:
    return attack_windows()


@pytest.fixture
def memory_built() -> bool:
    return PATHS.features.exists() and (PATHS.replay / "catalogue.json").exists()


requires_memory = pytest.mark.skipif(
    not (PATHS.features.exists() and (PATHS.replay / "catalogue.json").exists()),
    reason="Memory layer not built; run `python -m dfsi.memory.pipeline`",
)

requires_model = pytest.mark.skipif(
    not (PATHS.models / "forecaster.pkl").exists(),
    reason="no trained model; run `python -m dfsi.forecaster.train`",
)


def make_services(tmp_path: Path, **overrides):
    """Production wiring with test-safe defaults, any collaborator overridable.

    The audit log goes to `tmp_path`, the explainers run offline, and the
    policy store is a fresh copy of the raw.json seed, so nothing one test
    approves can leak into another or into the real `data/decisions/`.
    """
    from dfsi.agents.orchestrator import ExplainerOrchestrator
    from dfsi.api.audit import AuditLog
    from dfsi.api.services import build_services

    overrides.setdefault("audit_log", AuditLog(tmp_path / "decisions.jsonl"))
    overrides.setdefault("orchestrator", ExplainerOrchestrator(api_key=None))
    return build_services(**overrides)


def make_client(tmp_path: Path, **overrides):
    """A `TestClient` on an app built from `make_services(tmp_path, **overrides)`."""
    from fastapi.testclient import TestClient

    from dfsi.api.app import create_app

    return TestClient(create_app(make_services(tmp_path, **overrides)))
