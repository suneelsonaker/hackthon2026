"""Policy suggestions backed by a week of traffic.

The suggestion is a concrete number an operator could approve, so these pin
what makes it safe to show: it needs a full week, it never learns from an
incident, it never loosens a policy, it only touches the suspect vector, and
its backtest shows the cost of the hours that brushed today's line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import (
    EPOCH,
    POLICY_PPS,
    RANGE_ID,
    TRAFFIC_TYPES,
    attack_windows,
    make_row,
    requires_memory,
    requires_model,
)

from dfsi.agents.orchestrator import ExplainerOrchestrator
from dfsi.agents.tools import EvidenceBundle
from dfsi.rehearsal.policy_advisor import (
    HISTORY_COLUMNS,
    HISTORY_DAYS,
    SAFETY_MARGIN,
    suggest_policy,
)
from dfsi.rehearsal.simulator import run_drill

BASE_PPS = 10_000.0
DAY = 86_400


def history(
    days: float = HISTORY_DAYS,
    end: int = EPOCH,
    hot_hours: dict[tuple[str, int], float] | None = None,
) -> pd.DataFrame:
    """`days` of 10s windows on all three types, ending at `end`.

    `hot_hours` maps (traffic_type, hours_before_end) to a total pps for that
    whole hour — a past incident, a flash crowd, a near miss.
    """
    rng = np.random.default_rng(0)
    epochs = np.arange(end - int(days * DAY), end, 10)
    frames = []
    for traffic_type in TRAFFIC_TYPES:
        total = BASE_PPS * (1.0 + rng.normal(0.0, 0.03, size=epochs.size))
        for (hot_type, hours_back), pps in (hot_hours or {}).items():
            if hot_type == traffic_type:
                start = end - hours_back * 3_600
                total[(epochs >= start) & (epochs < start + 3_600)] = pps
        blocked = np.clip(total - POLICY_PPS, 0.0, None)
        frames.append(
            pd.DataFrame(
                {
                    "range_id": RANGE_ID,
                    "window_start_epoch": epochs,
                    "traffic_type": traffic_type,
                    "total_pps": total,
                    "total_mbps": total * 1_100 * 8 / 1e6,
                    "blocked_pps": blocked,
                    "blocked_mbps": blocked * 1_100 * 8 / 1e6,
                    "policy_pps": POLICY_PPS,
                    "policy_mbps": POLICY_PPS * 1_100 * 8 / 1e6,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def current(udp_pps: float = BASE_PPS, http_pps: float = BASE_PPS) -> list[dict]:
    return [
        make_row(epoch=EPOCH, traffic_type="udp", total_pps=udp_pps),
        make_row(epoch=EPOCH, traffic_type="tcp_syn", total_pps=BASE_PPS),
        make_row(epoch=EPOCH, traffic_type="http", total_pps=http_pps),
    ]


def by_type(plan):
    return {s.traffic_type: s for s in plan.per_type}


# ------------------------------------------------------------ prerequisites


def test_a_partial_week_gives_no_suggestion():
    plan = suggest_policy(RANGE_ID, EPOCH, current(), history=history(days=6))
    assert not plan.available
    assert "needs 7" in plan.reason
    assert plan.per_type == []


def test_a_full_week_gives_a_suggestion():
    plan = suggest_policy(RANGE_ID, EPOCH, current(udp_pps=80_000), history=history())
    assert plan.available
    assert plan.history_days == pytest.approx(7.0)


def test_history_never_includes_the_incident_itself():
    """Windows at or after the episode start are not history."""
    frame = history(end=EPOCH + 3_600)  # an hour of the 'future' appended
    plan = suggest_policy(RANGE_ID, EPOCH, current(udp_pps=80_000), history=frame)
    assert plan.available
    assert plan.history_end_epoch <= EPOCH


def test_the_advisor_reads_no_ground_truth():
    assert not set(HISTORY_COLUMNS) & {"scenario_type", "scenario_id", "is_attack_vector"}


# ---------------------------------------------------------- the threshold


def test_suggestion_sits_above_last_weeks_peak_and_below_todays_policy():
    plan = suggest_policy(RANGE_ID, EPOCH, current(udp_pps=80_000), history=history())
    udp = by_type(plan)["udp"]
    assert udp.changes
    assert udp.suggested_policy_pps >= udp.legit_envelope_pps * SAFETY_MARGIN
    assert udp.suggested_policy_pps < udp.current_policy_pps
    assert udp.backtest_windows_blocked == 0


def test_a_past_incident_does_not_raise_the_envelope():
    """An attack hour last week is not normal traffic."""
    clean = by_type(suggest_policy(RANGE_ID, EPOCH, current(udp_pps=80_000), history=history()))
    hit = by_type(
        suggest_policy(
            RANGE_ID,
            EPOCH,
            current(udp_pps=80_000),
            history=history(hot_hours={("udp", 30): 300_000.0}),
        )
    )
    assert hit["udp"].legit_envelope_pps == pytest.approx(clean["udp"].legit_envelope_pps, rel=0.01)


def test_a_policy_is_never_loosened():
    """Last week's traffic near today's line: keep the line."""
    frame = history(hot_hours={("http", 50): POLICY_PPS * 0.95})
    plan = suggest_policy(RANGE_ID, EPOCH, current(http_pps=60_000), history=frame)
    http = by_type(plan)["http"]
    assert not http.changes
    assert http.suggested_policy_pps == http.current_policy_pps


def test_the_backtest_shows_the_cost_of_a_near_miss_hour():
    """An hour that brushed today's line (blocked, but never an incident) is
    left out of the envelope and kept in the backtest."""
    frame = history(hot_hours={("udp", 40): POLICY_PPS * 1.1})
    plan = suggest_policy(RANGE_ID, EPOCH, current(udp_pps=80_000), history=frame)
    udp = by_type(plan)["udp"]
    assert udp.changes
    assert udp.backtest_windows_blocked == 360
    assert udp.backtest_hours_affected == 1
    assert udp.backtest_legit_lost_pct > 0


# ------------------------------------------------------- which vector moves


def test_only_the_suspect_vector_is_changed():
    plan = suggest_policy(RANGE_ID, EPOCH, current(udp_pps=80_000), history=history())
    rows = by_type(plan)
    assert rows["udp"].suspect and rows["udp"].apply
    assert not rows["tcp_syn"].apply and not rows["http"].apply
    assert set(plan.applied_thresholds) == {"udp"}


def test_a_suspect_that_cannot_be_tightened_is_reported_not_forced():
    frame = history(hot_hours={("http", 50): POLICY_PPS * 0.95})
    plan = suggest_policy(RANGE_ID, EPOCH, current(http_pps=60_000), history=frame)
    http = by_type(plan)["http"]
    assert http.suspect and not http.apply
    assert http.note and "would not be tighter" in http.note
    assert plan.applied_thresholds == {}


def test_quiet_traffic_has_no_suspect():
    plan = suggest_policy(RANGE_ID, EPOCH, current(), history=history())
    assert not any(s.suspect for s in plan.per_type)
    assert plan.describe_change() == "no change"


# ------------------------------------------------------------- the drill


@pytest.fixture
def flood_plan():
    rows = attack_windows()
    onset = [r for r in rows if r["window_start_epoch"] == EPOCH + 10 * 10]
    return rows, suggest_policy(RANGE_ID, EPOCH, onset, history=history())


def test_drill_simulates_tighten_from_the_plan(flood_plan):
    rows, plan = flood_plan
    drill = run_drill(rows, EPOCH + 10 * 10, None, plan)
    tighten = next(c for c in drill["candidates"] if c["policy_key"] == "tighten_vector")
    hold = next(c for c in drill["candidates"] if c["policy_key"] == "hold")
    assert tighten["attack_packets_blocked_pct"] > hold["attack_packets_blocked_pct"]
    assert any("udp policy 40000 ->" in note for note in tighten["notes"])
    assert drill["policy_plan"]["available"]
    basis = {a["policy_key"]: a["basis"] for a in drill["assumptions"]}
    assert basis["tighten_vector"] == "past-week thresholds"
    assert basis["full_scrub"] == "stated assumption"


def test_without_history_the_drill_keeps_the_stated_assumption():
    rows = attack_windows()
    plan = suggest_policy(RANGE_ID, EPOCH, rows[:3], history=history(days=2))
    drill = run_drill(rows, EPOCH + 100, None, plan)
    basis = {a["policy_key"]: a["basis"] for a in drill["assumptions"]}
    assert basis["tighten_vector"] == "stated assumption"
    assert not drill["policy_plan"]["available"]


def test_the_offline_advisor_quotes_the_plan_and_passes_the_validator(flood_plan, monkeypatch):
    """Every number the advisor cites must be traceable to the bundle."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # never call the real API
    rows, plan = flood_plan
    decision = EPOCH + 10 * 10
    bundle = EvidenceBundle(
        range_id=RANGE_ID,
        account_id=rows[0]["account_id"],
        window_start_epoch=decision,
        risk_score=0.8,
        risk_band="critical",
        model_version="test",
        feature_version="test",
        horizon_seconds=120,
        recent_windows=[r for r in rows if r["window_start_epoch"] <= decision],
        simulation=run_drill(rows, decision, None, plan),
        tenant_account_id=rows[0]["account_id"],
    )
    result = ExplainerOrchestrator(api_key="").run_advisor(bundle, "attack")
    assert result.validated, result.validator_failures
    tighten = next(o for o in result.output["options"] if o["policy_key"] == "tighten_vector")
    assert "udp policy 40000 ->" in tighten["what_it_buys"]


# ------------------------------------------------- against the built world


@requires_memory
def test_every_demo_replay_has_a_week_of_history():
    import json

    from dfsi.config import PATHS

    for entry in json.loads((PATHS.replay / "catalogue.json").read_text()):
        replay = json.loads((PATHS.replay / f"{entry['replay_id']}.json").read_text())
        start = min(r["window_start_epoch"] for r in replay["windows"])
        rows = [r for r in replay["windows"] if r["window_start_epoch"] == start]
        plan = suggest_policy(replay["range_id"], start, rows)
        assert plan.available, (entry["replay_id"], plan.reason)
        assert plan.history_days >= HISTORY_DAYS * 0.95


@requires_memory
@requires_model
def test_an_approved_threshold_change_is_recorded_with_its_numbers(tmp_path):
    """The audit record states the exact thresholds a human signed off on."""
    import json

    from conftest import make_client

    from dfsi.config import PATHS

    # A private audit log and store, so an approved change cannot leak elsewhere.
    client = make_client(tmp_path)
    replay_id = next(
        e["replay_id"]
        for e in json.loads((PATHS.replay / "catalogue.json").read_text())
        if e["scenario_type"] == "sustained_attack"
    )
    drill = client.post("/api/drill", json={"replay_id": replay_id}).json()
    assert drill["policy_plan"]["available"]

    record = client.post(
        "/api/decision",
        json={
            "replay_id": replay_id,
            "decision_epoch": drill["decision_epoch"],
            "policy_key": "tighten_vector",
            "verdict": "approved",
        },
    ).json()
    # Applied to the simulated store exactly when the plan changes something.
    assert record["applied"] is bool(record["policy_change"]["per_type"])
    assert record["policy_change"]["basis"] == "past-week thresholds"
    assert record["policy_change"]["change"] == drill["policy_plan"]["change"]
    logged = (tmp_path / "decisions.jsonl").read_text().splitlines()
    assert json.loads(logged[-1])["policy_change"] == record["policy_change"]
