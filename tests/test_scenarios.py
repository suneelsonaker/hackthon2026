"""The synthetic world must not hand the model the answer.

These run on the generator directly — no Spark, no built artefacts — and pin
the properties that stop a feature from being a disguised label.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from dfsi.config import STAGE_THRESHOLDS
from dfsi.memory.scenarios import (
    ATTACK_HISTORY_HOURS,
    ATTACK_SCENARIOS,
    DEFAULT_RANGES,
    TRAFFIC_TYPES,
    EpisodePlan,
    generate_episode,
    generate_episode_columns,
    plan_week,
)

EPOCH_BASE = 1_767_225_600


def _episode(scenario_type: str, seed: int, vector: str = "udp", hour: int = 13) -> pd.DataFrame:
    plan = EpisodePlan(
        scenario_id=f"test-{scenario_type}-{seed}",
        scenario_type=scenario_type,
        range_profile=DEFAULT_RANGES[0],
        start_epoch=EPOCH_BASE + 14 * 86_400 + hour * 3_600,
        attack_vector=vector,
        seed=seed,
    )
    return pd.DataFrame(generate_episode_columns(plan))


def _rings_the_bell(frame: pd.DataFrame) -> bool:
    """Shared State's rule: over the start threshold for the full start duration."""
    t = STAGE_THRESHOLDS
    over = (
        (frame.blocked_pps >= frame.policy_pps * t.start_threshold_ratio)
        & (frame.blocked_pps >= t.start_floor_kpps * 1_000)
        & (frame.blocked_mbps >= t.start_floor_mbps)
    )
    per_instant = over.groupby(frame.window_start_epoch).max().to_numpy().astype(int)
    run = np.convolve(per_instant, np.ones(t.start_duration_windows, dtype=int), "valid")
    return bool((run >= t.start_duration_windows).any())


# ------------------------------------------------------------- the plan


@pytest.fixture(scope="module")
def plans() -> list[EpisodePlan]:
    return plan_week(days=14, seed=11)


def test_plan_spans_whole_weeks(plans):
    assert sorted({p.week_index for p in plans}) == [0, 1]
    assert len(plans) == 14 * 24 * len(DEFAULT_RANGES)


def test_attack_history_counts_only_the_recent_past(plans):
    """Recomputed independently: same account, strictly earlier, inside the look-back."""
    attack_hours: dict[int, list[int]] = defaultdict(list)
    for plan in plans:
        if plan.scenario_type in ATTACK_SCENARIOS:
            hour = (plan.start_epoch - EPOCH_BASE) // 3_600
            attack_hours[plan.range_profile.account_id].append(hour)

    for plan in plans:
        hour = (plan.start_epoch - EPOCH_BASE) // 3_600
        expected = sum(
            1
            for past in attack_hours[plan.range_profile.account_id]
            if hour - ATTACK_HISTORY_HOURS <= past < hour
        )
        assert plan.account_attack_history == expected, plan.scenario_id


def test_attack_history_is_not_a_clock(plans):
    """A cumulative count only grows with time, so it encodes the date, not risk."""
    by_account: dict[int, list[int]] = defaultdict(list)
    for plan in sorted(plans, key=lambda p: p.start_epoch):
        by_account[plan.range_profile.account_id].append(plan.account_attack_history)
    for account, history in by_account.items():
        assert any(b < a for a, b in itertools.pairwise(history)), account


def test_attack_history_carries_some_signal(plans):
    """Campaigns make a recent attack predictive of the next one."""
    frame = pd.DataFrame(
        {
            "history": [p.account_attack_history for p in plans],
            "attack": [p.scenario_type in ATTACK_SCENARIOS for p in plans],
        }
    )
    quiet = frame[frame.history == 0].attack.mean()
    busy = frame[frame.history >= 3].attack.mean()
    assert busy > quiet + 0.1, (quiet, busy)


# ------------------------------------------------------------ the rows


def test_source_concentration_is_not_an_attack_label():
    """Regional crowds concentrate; distributed botnets do not.

    If HHI were a function of `attack_share` alone, no flash crowd window could
    ever look concentrated and no attacked window could look diverse.
    """
    crowd_max = [_episode("flash_crowd", seed).country_hhi.max() for seed in range(40)]
    attacked_min = []
    for seed in range(40):
        frame = _episode("sustained_attack", seed)
        hit = frame[frame.is_attack_vector & (frame.blocked_pps > 0)]
        if not hit.empty:
            attacked_min.append(hit.country_hhi.median())
    assert max(crowd_max) > 0.35, "no flash crowd ever looks concentrated"
    assert min(attacked_min) < 0.2, "no attack ever looks diverse"


@pytest.mark.parametrize(
    "scenario_type", ["gradual_buildup", "sustained_attack", "recovery", "pulse_wave"]
)
def test_single_vector_attacks_mark_exactly_one_vector(scenario_type):
    frame = _episode(scenario_type, seed=3, vector="tcp_syn")
    marked = set(frame.loc[frame.is_attack_vector, "traffic_type"])
    assert marked == {"tcp_syn"}


def test_carpet_bombing_spreads_across_every_vector():
    frame = _episode("carpet_bomb", seed=5)
    assert set(frame.loc[frame.is_attack_vector, "traffic_type"]) == set(TRAFFIC_TYPES)


def test_quiet_families_mark_no_vector():
    for scenario_type in ("normal", "flash_crowd"):
        assert not _episode(scenario_type, seed=2).is_attack_vector.any()


def test_flash_crowds_are_web_led():
    """HTTP takes the full surge; the others a share of it."""
    for seed in range(10):
        frame = _episode("flash_crowd", seed)
        peak = frame.groupby("traffic_type").total_pps.max()
        base = DEFAULT_RANGES[0].base_pps
        rise = {t: peak[t] / base[t] for t in TRAFFIC_TYPES}
        assert rise["http"] >= max(rise["udp"], rise["tcp_syn"]) - 0.25, rise


def test_attack_sizes_straddle_the_bell():
    """Some attacks never ring the bell; a world where every attack is huge is easy.

    Spread across the day, because the same attack that rings at the daily
    peak can stay under the line at night.
    """
    rang = [
        _rings_the_bell(_episode("sustained_attack", seed, hour=seed % 24)) for seed in range(120)
    ]
    share = float(np.mean(rang))
    assert 0.6 < share < 0.995, f"{share:.1%} of attacks ring the bell"


def test_list_and_columnar_generators_agree():
    plan = plan_week(days=1, seed=4)[7]
    rows = generate_episode(plan)
    columns = generate_episode_columns(plan)
    assert len(rows) == columns["window_start_epoch"].shape[0]
    assert rows[5]["total_pps"] == pytest.approx(columns["total_pps"][5])
    assert rows[0]["week_index"] == plan.week_index
