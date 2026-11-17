"""The simulated policy store, and replays re-run under it.

Approve a suggested threshold, replay the same attack, and the attack is
played under the new line: more dropped at the wall, the bell on the new
line, the forecaster's policy features recomputed. These pin that loop and
the store's bookkeeping (versions, history, reset).
"""

from __future__ import annotations

import json

import pytest
from conftest import (
    POLICY_PPS,
    RANGE_ID,
    attack_windows,
    make_client,
    requires_memory,
    requires_model,
)

from dfsi.config import PATHS, STAGE_THRESHOLDS
from dfsi.rehearsal.policy_store import PolicyStore, apply_policy


def seed(udp_pps: float = POLICY_PPS) -> dict:
    """A one-range policy db in raw.json's shape."""

    def line(pps: float, packet: int) -> dict:
        return {
            "base_pps": pps / 5,
            "max_threshold_pps": pps,
            "max_threshold_mbps": pps * packet * 8 / 1e6,
            "mean_packet_bytes": packet,
        }

    return {
        "accounts": [
            {
                "account_id": 421001,
                "ranges": [
                    {
                        "range_id": RANGE_ID,
                        "range_type": "INFRA_PROTECT_TOPOLOGY_TYPE_RANGE",
                        "traffic_types": {
                            "udp": line(udp_pps, 1100),
                            "tcp_syn": line(POLICY_PPS, 1100),
                            "http": line(POLICY_PPS, 1100),
                        },
                    }
                ],
            }
        ]
    }


# ----------------------------------------------------------------- the store


def test_store_reads_raw_json_shape():
    store = PolicyStore(seed())
    policy = store.range_policy(RANGE_ID)
    assert policy["udp"].pps == POLICY_PPS
    assert store.version == 1
    assert store.range_policy(1) is None


def test_a_change_is_versioned_and_recorded():
    store = PolicyStore(seed())
    version = store.apply_change(RANGE_ID, {"udp": 30_000.0}, {"operator": "alice"})
    assert version == 2 == store.version
    assert store.range_policy(RANGE_ID)["udp"].pps == 30_000.0
    entry = store.history[-1]
    assert entry["operator"] == "alice"
    assert entry["changes"][0]["from_pps"] == POLICY_PPS
    assert entry["changes"][0]["to_pps"] == 30_000.0
    assert store.range_version(RANGE_ID) == 2


def test_the_mbps_line_moves_with_the_pps_line():
    store = PolicyStore(seed())
    store.apply_change(RANGE_ID, {"udp": 30_000.0})
    assert store.range_policy(RANGE_ID)["udp"].mbps == pytest.approx(30_000 * 1100 * 8 / 1e6)


def test_a_change_to_an_unknown_range_or_type_is_refused():
    store = PolicyStore(seed())
    with pytest.raises(KeyError):
        store.apply_change(1, {"udp": 10.0})
    with pytest.raises(KeyError):
        store.apply_change(RANGE_ID, {"icmp": 10.0})
    with pytest.raises(ValueError):
        store.apply_change(RANGE_ID, {"udp": 0.0})
    assert store.version == 1


def test_reset_restores_the_seed_as_a_new_version():
    store = PolicyStore(seed())
    store.apply_change(RANGE_ID, {"udp": 30_000.0})
    version = store.reset()
    assert version == 3
    assert store.range_policy(RANGE_ID)["udp"].pps == POLICY_PPS
    assert store.history[-1]["action"] == "reset"


def test_a_missing_db_leaves_replays_as_recorded(tmp_path):
    store = PolicyStore.from_file(tmp_path / "missing.json")
    rows = attack_windows()
    assert apply_policy(rows, store.range_policy(RANGE_ID)) == rows


# --------------------------------------------------- a replay under a policy


def test_the_recorded_policy_leaves_rows_untouched():
    rows = attack_windows()
    assert apply_policy(rows, PolicyStore(seed()).range_policy(RANGE_ID)) == rows


def _first_declared(rows: list[dict]) -> int | None:
    declared = sorted(r["window_start_epoch"] for r in rows if r["declared"])
    return declared[0] if declared else None


def test_a_tighter_line_drops_more_and_rings_the_bell_no_later():
    rows = attack_windows()
    store = PolicyStore(seed())
    before = apply_policy(rows, store.range_policy(RANGE_ID), STAGE_THRESHOLDS, force=True)
    store.apply_change(RANGE_ID, {"udp": 20_000.0})
    after = apply_policy(rows, store.range_policy(RANGE_ID), STAGE_THRESHOLDS)

    def blocked(rs):
        return sum(r["blocked_pps"] for r in rs if r["traffic_type"] == "udp")

    assert blocked(after) > blocked(before)
    assert _first_declared(after) is not None
    assert _first_declared(before) is None or _first_declared(after) <= _first_declared(before)

    udp_after = [r for r in after if r["traffic_type"] == "udp"]
    assert all(r["policy_pps"] == 20_000.0 for r in udp_after)
    assert max(r["policy_utilisation"] for r in udp_after) > max(
        r["policy_utilisation"] for r in before if r["traffic_type"] == "udp"
    )


def test_traffic_itself_is_never_rewritten():
    rows = attack_windows()
    store = PolicyStore(seed())
    store.apply_change(RANGE_ID, {"udp": 20_000.0})
    after = apply_policy(rows, store.range_policy(RANGE_ID))
    for original, derived in zip(rows, after):
        for column in ("total_pps", "baseline_deviation_pps", "country_hhi", "growth_pps_60s"):
            assert derived[column] == original[column]


@requires_memory
def test_rederiving_under_the_recorded_policy_reproduces_the_pipeline():
    """The replay-side rules must be the feature pipeline's rules, exactly."""
    store = PolicyStore.from_file(PATHS.policy_db)
    fields = (
        "blocked_pps",
        "blocked_share",
        "blocked_share_delta_10s",
        "blocked_share_delta_60s",
        "policy_utilisation",
        "range_over_threshold",
        "declared",
        "is_onset",
        "label_onset_120s",
    )
    for entry in json.loads((PATHS.replay / "catalogue.json").read_text()):
        replay = json.loads((PATHS.replay / f"{entry['replay_id']}.json").read_text())
        derived = apply_policy(
            replay["windows"], store.range_policy(replay["range_id"]), force=True
        )
        for original, row in zip(replay["windows"], derived):
            for field in fields:
                assert row[field] == pytest.approx(original[field], abs=1e-9), (
                    entry["replay_id"],
                    field,
                )


# ------------------------------------------------- approve, then replay again


@pytest.fixture
def api_client(tmp_path):
    # A private audit log and a fresh copy of the seed store, so an approved
    # change cannot leak into other tests or the real data/decisions/.
    return make_client(tmp_path)


def _run(client, replay_id: str) -> dict:
    with client.stream(
        "POST", "/api/replay/stream", json={"replay_id": replay_id, "speed_seconds": 0}
    ) as response:
        events = [
            json.loads(line[5:]) for line in response.iter_lines() if line.startswith("data:")
        ]
    assert events[-1]["type"] == "summary"
    return events[-1]


def _approve_tighten(client, replay_id: str, verdict: str = "approved") -> dict:
    drill = client.post("/api/drill", json={"replay_id": replay_id}).json()
    return client.post(
        "/api/decision",
        json={
            "replay_id": replay_id,
            "decision_epoch": drill["decision_epoch"],
            "policy_key": "tighten_vector",
            "verdict": verdict,
        },
    ).json()


@pytest.fixture
def tightenable(api_client) -> str:
    """An attack replay whose drill suggests a tighter line.

    Picked from the catalogue rather than hardcoded: replay ids change whenever
    the synthetic set is regenerated.
    """
    for entry in api_client.get("/api/replays").json()["replays"]:
        if not entry["has_onset"]:
            continue
        plan = api_client.post("/api/drill", json={"replay_id": entry["replay_id"]}).json()[
            "policy_plan"
        ]
        if plan["change"] != "no change":
            return entry["replay_id"]
    pytest.skip("no replay in the catalogue has a tightening to approve")


@requires_memory
@requires_model
def test_an_approved_change_is_in_force_on_the_next_run(api_client, tightenable):
    before = _run(api_client, tightenable)
    record = _approve_tighten(api_client, tightenable)
    assert record["applied"], record
    assert record["policy_version_after"] == record["policy_version_before"] + 1

    after = _run(api_client, tightenable)
    assert after["policy_version"] == record["policy_version_after"]
    new_line = record["policy_change"]["per_type"][0]
    assert (
        after["policy"][new_line["traffic_type"]]["max_threshold_pps"]
        == new_line["suggested_policy_pps"]
    )
    assert after["blocked_share_total"] > before["blocked_share_total"]
    assert after["peak_passed_pps"] < before["peak_passed_pps"]
    assert after["bell_after_seconds"] <= before["bell_after_seconds"]

    # Already tightened: the next suggestion keeps the new line.
    plan = api_client.post("/api/drill", json={"replay_id": tightenable}).json()["policy_plan"]
    assert plan["change"] == "no change"


@requires_memory
@requires_model
def test_a_rejected_change_is_not_applied(api_client, tightenable):
    record = _approve_tighten(api_client, tightenable, verdict="rejected")
    assert not record["applied"]
    assert api_client.get("/api/policies").json()["version"] == 1


@requires_memory
@requires_model
def test_reset_puts_the_seed_back_in_force(api_client, tightenable):
    before = _run(api_client, tightenable)
    _approve_tighten(api_client, tightenable)
    api_client.post("/api/policies/reset")
    after = _run(api_client, tightenable)
    assert after["blocked_share_total"] == pytest.approx(before["blocked_share_total"])
    assert after["policy"] == before["policy"]


def test_a_change_to_the_current_line_is_not_a_change():
    store = PolicyStore.from_file(PATHS.policy_db)
    current = store.range_policy(RANGE_ID)["udp"].pps
    before = store.version
    assert store.apply_change(RANGE_ID, {"udp": current}) == before
    assert store.version == before
    assert store.snapshot()["history"] == []


def test_a_rejected_change_leaves_every_line_untouched():
    store = PolicyStore.from_file(PATHS.policy_db)
    before = {t: p.pps for t, p in store.range_policy(RANGE_ID).items()}
    with pytest.raises(KeyError):
        store.apply_change(RANGE_ID, {"udp": 1_234.0, "gre": 5.0})  # second type is unknown
    assert {t: p.pps for t, p in store.range_policy(RANGE_ID).items()} == before
    assert store.snapshot()["history"] == []
