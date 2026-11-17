"""Memory, part 1 — the ship's logbook.

Generates synthetic, ANL-shaped 10-second traffic windows across the scenario
families DFSI is evaluated on:

    normal · flash_crowd · gradual_buildup · sustained_attack · recovery
    pulse_wave · carpet_bomb

Everything here is synthetic. No customer traffic, no raw IP addresses: source
information is reduced to a country-concentration index and a per-account
attack-history count, both computed chronologically so they can never see the
future.

Each *episode* is one contiguous hour for one protected range, holding rows for
every traffic type in that hour. Episodes are laid out over several weeks, so
an episode's hour-of-week baseline can be built from *earlier* weeks only (see
`features.compute_trailing_baselines`) and the evaluation can be split by time.
The episode id doubles as the scenario id used for splitting — it is never
given to the model.

Design rules that keep the synthetic world from handing the model the answer:

* **Source concentration is drawn per episode, independently of the label.**
  Some attacks come from a few reflector networks (concentrated), others from
  a distributed botnet (more diverse than normal traffic). Some flash crowds
  are regional (concentrated), others broad. Country HHI is a clue, not an
  oracle.
* **Neither family has a fixed shape across traffic types.** Flash crowds are
  HTTP-led and lift the other types by a random fraction; carpet bombing
  spreads an attack across every type at once.
* **Attack sizes span the threshold.** Peaks are log-uniform from just under
  to far over the policy line, so some attacks never ring the bell and some
  only ring it on a pulse.
* **Attack history means something.** Attacks cluster in per-account
  campaigns, and `account_attack_history` counts attack episodes on the
  account in the previous 6 hours — not a counter that only grows with time.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import PATHS, SCENARIO_SET_VERSION, WINDOW_SECONDS

TRAFFIC_TYPES: tuple[str, ...] = ("udp", "tcp_syn", "http")
SCENARIO_TYPES: tuple[str, ...] = (
    "normal",
    "flash_crowd",
    "gradual_buildup",
    "sustained_attack",
    "recovery",
    "pulse_wave",
    "carpet_bomb",
)
# Scenario families that contain attack traffic. Whether blocking above the
# start threshold actually occurs depends on the attack's size.
ATTACK_SCENARIOS: frozenset[str] = frozenset(
    {"gradual_buildup", "sustained_attack", "recovery", "pulse_wave", "carpet_bomb"}
)
#: Attack families that spread across every traffic type rather than one.
MULTI_VECTOR_SCENARIOS: frozenset[str] = frozenset({"carpet_bomb"})

# Mean packet size per traffic type, in bytes. UDP reflection carries large
# packets; SYN floods are tiny; HTTP sits in between.
_MEAN_PACKET_BYTES = {"udp": 1100, "tcp_syn": 64, "http": 700}

# Policy headroom: the configured policy threshold is this multiple of the
# range's normal baseline rate, so ordinary peaks never reach it.
_POLICY_HEADROOM = 5.0

WINDOWS_PER_EPISODE = 360  # one hour of 10s windows
EPOCH_BASE = 1_767_225_600  # 2026-01-01T00:00:00Z, a Thursday
WEEK_SECONDS = 7 * 86_400


@dataclass(frozen=True)
class RangeProfile:
    """A protected IP range and its steady-state traffic characteristics."""

    range_id: int
    account_id: int
    range_type: str
    base_pps: dict[str, float]

    def policy_pps(self, traffic_type: str) -> float:
        return self.base_pps[traffic_type] * _POLICY_HEADROOM

    def policy_mbps(self, traffic_type: str) -> float:
        return float(_pps_to_mbps(self.policy_pps(traffic_type), traffic_type))


def _pps_to_mbps(pps: float | np.ndarray, traffic_type: str) -> float | np.ndarray:
    return pps * _MEAN_PACKET_BYTES[traffic_type] * 8.0 / 1_000_000.0


def load_ranges(path: Path) -> tuple[RangeProfile, ...]:
    """Every range in raw.json, in file order (the order seeds the generator)."""
    return tuple(
        RangeProfile(
            range_id=int(rng["range_id"]),
            account_id=int(account["account_id"]),
            range_type=str(rng["range_type"]),
            base_pps={t: float(spec["base_pps"]) for t, spec in rng["traffic_types"].items()},
        )
        for account in json.loads(path.read_text())["accounts"]
        for rng in account["ranges"]
    )


DEFAULT_RANGES: tuple[RangeProfile, ...] = load_ranges(PATHS.policy_db)

# Base frequency of each scenario family. Heavily skewed towards normal, which
# is what production looks like. Attack families are then reweighted by the
# account's campaign state (see `plan_week`).
_SCENARIO_WEIGHTS = {
    "normal": 0.62,
    "flash_crowd": 0.10,
    "gradual_buildup": 0.07,
    "sustained_attack": 0.07,
    "recovery": 0.05,
    "pulse_wave": 0.04,
    "carpet_bomb": 0.05,
}

#: How often each account comes under a campaign, relative to the default.
_ACCOUNT_TARGETING = {421_001: 1.4, 558_777: 1.0, 612_430: 0.6}
#: Daily chance that a campaign starts against an account (times targeting).
_CAMPAIGN_DAILY_RATE = 0.5
#: Campaign length, in hours (inclusive bounds).
_CAMPAIGN_HOURS = (6, 18)
#: Attack-family weight multiplier inside and outside a campaign.
_CAMPAIGN_ATTACK_MULTIPLIER = 3.0
_QUIET_ATTACK_MULTIPLIER = 0.3
#: Look-back for `account_attack_history`, in hours. Short, so the count
#: tracks whether a campaign is live rather than saturating.
ATTACK_HISTORY_HOURS = 6


@dataclass
class EpisodePlan:
    """Chronological placement of one scenario episode."""

    scenario_id: str
    scenario_type: str
    range_profile: RangeProfile
    start_epoch: int
    attack_vector: str
    seed: int
    #: Past-only: attack episodes on this account in the previous
    #: `ATTACK_HISTORY_HOURS` hours, not counting concurrent episodes.
    account_attack_history: int = 0
    n_windows: int = WINDOWS_PER_EPISODE

    @property
    def end_epoch(self) -> int:
        return self.start_epoch + self.n_windows * WINDOW_SECONDS

    @property
    def week_index(self) -> int:
        return int((self.start_epoch - EPOCH_BASE) // WEEK_SECONDS)

    @property
    def attack_vectors(self) -> tuple[str, ...]:
        """Traffic types that carry attack packets in this episode."""
        if self.scenario_type not in ATTACK_SCENARIOS:
            return ()
        if self.scenario_type in MULTI_VECTOR_SCENARIOS:
            return TRAFFIC_TYPES
        return (self.attack_vector,)


def _diurnal(epoch: np.ndarray) -> np.ndarray:
    """Daily + weekly demand shape, as a multiplier around 1.0."""
    seconds_of_day = epoch % 86_400
    day_of_week = ((epoch // 86_400) + 3) % 7  # EPOCH_BASE is a Thursday
    daily = 1.0 + 0.45 * np.sin(2 * np.pi * (seconds_of_day - 21_600) / 86_400)
    weekend = np.where(day_of_week >= 5, 0.72, 1.0)
    return daily * weekend


def _crowd_bump(scenario_type: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Extra legitimate demand from a flash crowd, as a multiple of normal.

    A product launch / news event: a smooth bump that is large but usually
    stays under the policy threshold. This is the false-alarm trap for DFSI.
    Zero for every other scenario family.
    """
    if scenario_type != "flash_crowd":
        return np.zeros(n)
    peak = rng.uniform(2.0, 3.6)  # extra demand at the crest, x normal
    centre = rng.integers(120, 260)
    width = rng.integers(35, 95)
    idx = np.arange(n)
    return peak * np.exp(-0.5 * ((idx - centre) / width) ** 2)


def _crowd_type_weights(scenario_type: str, rng: np.random.Generator) -> dict[str, float]:
    """How much of the flash crowd lands on each traffic type.

    Crowds are web-led: HTTP takes the full bump, the others a random share.
    So a flash crowd is neither "all three types together" nor "one type" by
    construction — the count of rising types is not a free tell.
    """
    if scenario_type != "flash_crowd":
        return {t: 0.0 for t in TRAFFIC_TYPES}
    return {
        "http": 1.0,
        "tcp_syn": float(rng.uniform(0.25, 1.0)),
        "udp": float(rng.uniform(0.0, 0.7)),
    }


def _attack_profiles(plan: EpisodePlan, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Attack packets-per-second per traffic type, per window.

    Timing is drawn once per episode so a multi-vector attack starts on every
    vector together; size is drawn per vector.
    """
    idx = np.arange(n, dtype=float)
    profiles = {t: np.zeros(n) for t in TRAFFIC_TYPES}
    scenario_type = plan.scenario_type
    if scenario_type not in ATTACK_SCENARIOS:
        return profiles

    if scenario_type == "gradual_buildup":
        # Slow probing ramp: the case a fixed threshold notices far too late.
        onset = int(rng.integers(60, 150))
        ramp_end = onset + int(rng.integers(40, 180))
        shape = np.clip((idx - onset) / max(1, ramp_end - onset), 0.0, 1.0) ** 1.6
    elif scenario_type in ("sustained_attack", "carpet_bomb"):
        onset = int(rng.integers(80, 150))
        rise = int(rng.integers(2, 14))
        shape = np.clip((idx - onset) / rise, 0.0, 1.0)
    elif scenario_type == "recovery":
        # Episode opens mid-attack and decays to nothing.
        decay_end = int(rng.integers(150, 230))
        shape = np.clip(1.0 - idx / decay_end, 0.0, 1.0) ** 1.3
    else:  # pulse_wave
        # Hit-and-release bursts. Long enough pulses ring a short start
        # duration; none ring a long one.
        onset = int(rng.integers(80, 140))
        stop = onset + int(rng.integers(120, 220))
        on_windows = int(rng.integers(3, 9))
        off_windows = int(rng.integers(3, 13))
        phase = (idx - onset) % (on_windows + off_windows)
        shape = ((idx >= onset) & (idx < stop) & (phase < on_windows)).astype(float)

    for vector in plan.attack_vectors:
        base = plan.range_profile.base_pps[vector]
        if scenario_type in MULTI_VECTOR_SCENARIOS:
            # Spread thin: each vector alone may sit under its policy line.
            peak = base * np.exp(rng.uniform(np.log(2.5), np.log(10.0)))
        else:
            # Log-uniform from a near-miss to far over the line.
            peak = base * np.exp(rng.uniform(np.log(4.5), np.log(30.0)))
        # Attacks are not smooth: pulse the carrier.
        carrier = 1.0 + 0.12 * np.sin(2 * np.pi * idx / rng.uniform(5, 14))
        jitter = 1.0 + rng.normal(0.0, 0.05, size=n)
        profiles[vector] = np.clip(peak * shape * carrier * jitter, 0.0, None)
    return profiles


@dataclass(frozen=True)
class _SourceMix:
    """Per-episode country concentration of each traffic component."""

    normal: float
    crowd: float
    attack: float


def _draw_source_mix(scenario_type: str, rng: np.random.Generator) -> _SourceMix:
    """Draw the episode's source concentrations, independently of the label.

    HHI of the source-country mix, in [0, 1]. Derived from country counts
    only — no addresses are generated or stored.
    """
    normal = float(rng.uniform(0.11, 0.17))
    # A regional event concentrates demand; broad, global demand spreads it.
    regional = rng.random() < 0.45
    crowd = float(rng.uniform(0.32, 0.65) if regional else rng.uniform(0.10, 0.22))
    if rng.random() < 0.55:
        attack = float(rng.uniform(0.40, 0.78))  # a few reflector networks
    else:
        attack = float(rng.uniform(0.05, 0.16))  # distributed botnet / spoofed
    return _SourceMix(normal=normal, crowd=crowd, attack=attack)


def _country_hhi(
    mix: _SourceMix,
    crowd_share: np.ndarray,
    attack_share: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Traffic-weighted blend of each component's concentration.

    Every component moves the index in proportion to its share of the
    window's traffic, so a regional flash crowd raises HHI just as a
    reflection flood does, and a botnet can lower it.
    """
    normal_share = np.clip(1.0 - crowd_share - attack_share, 0.0, 1.0)
    blended = mix.normal * normal_share + mix.crowd * crowd_share + mix.attack * attack_share
    noise = rng.normal(0.0, 0.012, size=crowd_share.shape[0])
    return np.clip(blended + noise, 0.02, 0.98)


def generate_episode_columns(plan: EpisodePlan) -> dict[str, np.ndarray]:
    """Materialise one episode as columns of ANL-shaped window rows.

    Columnar so a multi-week logbook can be built without millions of dicts.
    Rows are ordered traffic type first, then time.
    """
    rng = np.random.default_rng(plan.seed)
    n = plan.n_windows
    epoch = plan.start_epoch + np.arange(n) * WINDOW_SECONDS
    demand = _diurnal(epoch)
    bump = _crowd_bump(plan.scenario_type, n, rng)
    crowd_weights = _crowd_type_weights(plan.scenario_type, rng)
    attacks = _attack_profiles(plan, n, rng)
    mix = _draw_source_mix(plan.scenario_type, rng)
    attack_vectors = set(plan.attack_vectors)

    columns: dict[str, list[np.ndarray]] = {}

    def put(name: str, values: np.ndarray) -> None:
        columns.setdefault(name, []).append(values)

    for traffic_type in TRAFFIC_TYPES:
        base = plan.range_profile.base_pps[traffic_type]
        noise = 1.0 + rng.normal(0.0, 0.035, size=n)
        normal_pps = base * demand * noise
        crowd_pps = normal_pps * bump * crowd_weights[traffic_type]
        attack = attacks[traffic_type]
        total_pps = np.clip(normal_pps + crowd_pps + attack, 1.0, None)

        policy_pps = plan.range_profile.policy_pps(traffic_type)
        policy_mbps = plan.range_profile.policy_mbps(traffic_type)
        # Behemoth ("the wall") drops whatever exceeds the configured policy.
        blocked_pps = np.clip(total_pps - policy_pps, 0.0, None)

        hhi = _country_hhi(mix, crowd_pps / total_pps, attack / total_pps, rng)

        put("traffic_type", np.full(n, traffic_type, dtype=object))
        put("window_start_epoch", epoch.astype(np.int64))
        put("total_pps", total_pps)
        put("total_mbps", np.asarray(_pps_to_mbps(total_pps, traffic_type)))
        put("blocked_pps", blocked_pps)
        put("blocked_mbps", np.asarray(_pps_to_mbps(blocked_pps, traffic_type)))
        put("policy_pps", np.full(n, policy_pps))
        put("policy_mbps", np.full(n, policy_mbps))
        put("country_hhi", hhi)
        put("is_attack_vector", np.full(n, traffic_type in attack_vectors, dtype=bool))

    out = {name: np.concatenate(parts) for name, parts in columns.items()}
    rows = out["window_start_epoch"].shape[0]
    out.update(
        {
            "scenario_id": np.full(rows, plan.scenario_id, dtype=object),
            "scenario_type": np.full(rows, plan.scenario_type, dtype=object),
            "scenario_set_version": np.full(rows, SCENARIO_SET_VERSION, dtype=object),
            "range_id": np.full(rows, plan.range_profile.range_id, dtype=np.int64),
            "account_id": np.full(rows, plan.range_profile.account_id, dtype=np.int64),
            "range_type": np.full(rows, plan.range_profile.range_type, dtype=object),
            "account_attack_history": np.full(rows, plan.account_attack_history, dtype=np.int32),
            "week_index": np.full(rows, plan.week_index, dtype=np.int32),
        }
    )
    return out


def generate_episode(plan: EpisodePlan) -> list[dict]:
    """Materialise one episode as a list of ANL-shaped window rows."""
    columns = generate_episode_columns(plan)
    names = list(columns)
    count = columns["window_start_epoch"].shape[0]
    return [{name: _python_scalar(columns[name][i]) for name in names} for i in range(count)]


def _python_scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _campaign_hours(
    accounts: Sequence[int], days: int, rng: np.random.Generator
) -> dict[int, set[int]]:
    """Absolute hour indices during which each account is under a campaign."""
    hours: dict[int, set[int]] = {}
    for account in accounts:
        active: set[int] = set()
        rate = _CAMPAIGN_DAILY_RATE * _ACCOUNT_TARGETING.get(account, 1.0)
        for day in range(days):
            if rng.random() < rate:
                start = day * 24 + int(rng.integers(0, 24))
                length = int(rng.integers(_CAMPAIGN_HOURS[0], _CAMPAIGN_HOURS[1] + 1))
                active.update(range(start, start + length))
        hours[account] = active
    return hours


def plan_week(
    ranges: Sequence[RangeProfile] = DEFAULT_RANGES,
    days: int = 28,
    seed: int = 20260101,
    weights: dict[str, float] | None = None,
) -> list[EpisodePlan]:
    """Lay out `days` of hourly episodes per range, in chronological order.

    `account_attack_history` is filled in as the plan is walked forward, so it
    only ever counts episodes that already happened. Episodes on sibling
    ranges in the same hour are concurrent, not past, and do not count.
    """
    rng = np.random.default_rng(seed)
    weights = weights or _SCENARIO_WEIGHTS
    types = list(weights)
    base = np.array([weights[t] for t in types], dtype=float)
    is_attack = np.array([t in ATTACK_SCENARIOS for t in types])

    accounts = sorted({r.account_id for r in ranges})
    campaigns = _campaign_hours(accounts, days, rng)
    attack_hours: dict[int, list[int]] = {account: [] for account in accounts}

    plans: list[EpisodePlan] = []
    for day in range(days):
        for hour in range(24):
            absolute_hour = day * 24 + hour
            start = EPOCH_BASE + absolute_hour * 3_600
            attacked_this_hour: list[int] = []
            for profile in ranges:
                account = profile.account_id
                multiplier = (
                    _CAMPAIGN_ATTACK_MULTIPLIER
                    if absolute_hour in campaigns[account]
                    else _QUIET_ATTACK_MULTIPLIER
                )
                probs = base * np.where(is_attack, multiplier, 1.0)
                probs = probs / probs.sum()
                scenario_type = str(rng.choice(types, p=probs))
                vector = str(rng.choice(TRAFFIC_TYPES))
                history = sum(
                    1
                    for past in attack_hours[account]
                    if absolute_hour - ATTACK_HISTORY_HOURS <= past < absolute_hour
                )
                plans.append(
                    EpisodePlan(
                        scenario_id=f"{profile.range_id}-d{day:02d}h{hour:02d}",
                        scenario_type=scenario_type,
                        range_profile=profile,
                        start_epoch=start,
                        attack_vector=vector,
                        seed=int(rng.integers(0, 2**31 - 1)),
                        account_attack_history=history,
                    )
                )
                if scenario_type in ATTACK_SCENARIOS:
                    attacked_this_hour.append(account)
            for account in attacked_this_hour:
                attack_hours[account].append(absolute_hour)
    return plans


def iter_rows(plans: Sequence[EpisodePlan]) -> Iterator[dict]:
    for plan in plans:
        yield from generate_episode(plan)
