"""Rehearsal room — how much does the comparison depend on the guesses?

The drill compares four responses using three numbers per option that are
*stated assumptions*, not measurements: how much attack traffic it removes, how
much legitimate traffic it costs, and how long it takes to engage. A table of
single figures hides that. "Rate-limit removes 78%" reads like a fact.

This replays each option many times with those assumptions drawn from a stated
spread around their central values, and reports ranges instead of points:

* **bands** — p10 / p50 / p90 of each outcome across the draws;
* **p_contains** — how often the option gets the incident back under threshold;
* **p_best** — how often it is the cheapest option that does, which is the
  Advisor's own recommendation rule. If one option wins nearly every time, the
  recommendation survives the guesses; if the lead flips, the operator should
  know that before trusting it.

The spread is itself a judgement. It is not calibrated against real incidents,
and the bands are not confidence intervals; they answer "if the assumptions
were this wrong, would the answer change?", not "how likely is each outcome?".
Setting every spread to zero reproduces the plain drill exactly.

Seeded and deterministic: the same inputs give the same bands.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from ..config import WINDOW_SECONDS
from .options import simulate_option, uses_plan
from .simulator import CANDIDATES, PolicyCandidate, SimulationOutcome

DEFAULT_DRAWS = 200
MAX_DRAWS = 2_000
DEFAULT_SEED = 20260101

#: A recommendation that wins at least this share of the draws is called robust.
ROBUST_SHARE = 0.8

#: Outcomes summarised as bands, in the order they are reported.
METRICS: tuple[str, ...] = (
    "attack_packets_blocked_pct",
    "legitimate_packets_lost_pct",
    "seconds_over_threshold",
    "peak_blocked_share",
)

NONE_CONTAIN = "none_contain"

#: `basis` of an option whose assumptions are drawn from the spread.
SAMPLED_BASIS = "stated assumptions, sampled"


@dataclass(frozen=True)
class AssumptionSpread:
    """How far each stated assumption may plausibly be from the truth.

    Defaults are deliberately wide: the figures behind the candidates are round
    numbers chosen for a drill, not fitted to anything.
    """

    #: Standard deviation of attack suppression, as an absolute share (0.10 = 10 points).
    suppression_sd: float = 0.10
    #: Log-normal sigma on legitimate collateral; 0.35 puts 2% roughly in 1-4%.
    collateral_sigma: float = 0.35
    #: Engagement delay may land this many whole windows early or late.
    delay_jitter_windows: int = 1

    def __post_init__(self) -> None:
        if self.suppression_sd < 0 or self.collateral_sigma < 0 or self.delay_jitter_windows < 0:
            raise ValueError("spreads must not be negative")

    def to_dict(self) -> dict:
        return {
            "suppression_sd": self.suppression_sd,
            "collateral_sigma": self.collateral_sigma,
            "delay_jitter_seconds": self.delay_jitter_windows * WINDOW_SECONDS,
        }


def _draw(
    candidate: PolicyCandidate, spread: AssumptionSpread, rng: np.random.Generator
) -> PolicyCandidate:
    """One plausible version of `candidate`'s assumptions."""
    suppression = candidate.attack_suppression
    if spread.suppression_sd > 0:
        suppression = float(np.clip(rng.normal(suppression, spread.suppression_sd), 0.0, 1.0))
    collateral = candidate.legitimate_collateral
    if spread.collateral_sigma > 0:
        collateral = float(
            np.clip(collateral * rng.lognormal(0.0, spread.collateral_sigma), 0.0, 1.0)
        )
    delay = candidate.engagement_delay_seconds
    if spread.delay_jitter_windows > 0:
        jitter = int(rng.integers(-spread.delay_jitter_windows, spread.delay_jitter_windows + 1))
        delay = max(0, delay + jitter * WINDOW_SECONDS)
    return replace(
        candidate,
        attack_suppression=suppression,
        legitimate_collateral=collateral,
        engagement_delay_seconds=delay,
    )


def _band(values: Sequence[float]) -> dict[str, float]:
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    return {"p10": round(float(p10), 4), "p50": round(float(p50), 4), "p90": round(float(p90), 4)}


def cheapest_containing(outcomes: dict[str, SimulationOutcome]) -> str:
    """The Advisor's rule: of the options that contain the incident, the one that
    costs legitimate traffic least (faster containment breaks ties).

    `hold` is not an intervention and is never a candidate for this. Returns
    `NONE_CONTAIN` when no option gets the incident back under threshold.
    """
    containing = [
        (o.legitimate_packets_lost_pct, o.time_to_contain_seconds, index, key)
        for index, (key, o) in enumerate(outcomes.items())
        if key != "hold" and o.time_to_contain_seconds is not None
    ]
    return min(containing)[3] if containing else NONE_CONTAIN


def _summarise(outcomes: list[SimulationOutcome]) -> dict:
    contained = [
        o.time_to_contain_seconds for o in outcomes if o.time_to_contain_seconds is not None
    ]
    summary: dict[str, dict[str, float] | None] = {
        metric: _band([getattr(o, metric) for o in outcomes]) for metric in METRICS
    }
    summary["time_to_contain_seconds"] = _band(contained) if contained else None
    return {"bands": summary, "p_contains": round(len(contained) / len(outcomes), 4)}


def assumption_uncertainty(
    windows: Sequence[dict],
    decision_epoch: int,
    environment: str | None = None,
    policy_plan=None,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
    spread: AssumptionSpread | None = None,
) -> dict:
    """Replay every option under `draws` plausible versions of its assumptions.

    `windows`, `decision_epoch`, `environment` and `policy_plan` are exactly what
    `run_drill` takes. When a past-week `policy_plan` drives "tighten the vector
    threshold", that option rests on measured thresholds, not on the stated
    assumptions, so it is not sampled and its band collapses to a point.
    """
    if not 1 <= draws <= MAX_DRAWS:
        raise ValueError(f"draws must be between 1 and {MAX_DRAWS}, got {draws}")
    spread = spread or AssumptionSpread()
    rng = np.random.default_rng(seed)

    # The stated (central) outcome of each option, for comparison with the bands.
    stated = {
        c.key: simulate_option(windows, decision_epoch, c, environment, policy_plan)
        for c in CANDIDATES
    }

    per_draw: list[dict[str, SimulationOutcome]] = []
    for _ in range(draws):
        outcomes: dict[str, SimulationOutcome] = {}
        for candidate in CANDIDATES:
            if candidate.key == "hold" or uses_plan(candidate, policy_plan):
                outcomes[candidate.key] = stated[candidate.key]
            else:
                outcomes[candidate.key] = simulate_option(
                    windows, decision_epoch, _draw(candidate, spread, rng), environment
                )
        per_draw.append(outcomes)

    options: dict[str, dict] = {}
    for candidate in CANDIDATES:
        if candidate.key == "hold":
            continue
        entry = _summarise([draw[candidate.key] for draw in per_draw])
        entry["policy_name"] = candidate.name
        entry["stated"] = {metric: getattr(stated[candidate.key], metric) for metric in METRICS} | {
            "time_to_contain_seconds": stated[candidate.key].time_to_contain_seconds
        }
        entry["basis"] = (
            "past-week thresholds (not sampled)"
            if uses_plan(candidate, policy_plan)
            else SAMPLED_BASIS
        )
        options[candidate.key] = entry

    hold = stated["hold"]
    notes = []
    if hold.seconds_over_threshold == 0:
        # Nothing to contain, so "which option contains it cheapest" has no answer;
        # reporting "no option contains it" at 100% would read as failure.
        p_best = leader = leader_share = robust = leader_sampled = None
        notes.append(
            "Even with no intervention this never crosses the start threshold, "
            "so there is nothing to contain: the bands show collateral only."
        )
    else:
        wins = [cheapest_containing(draw) for draw in per_draw]
        p_best = {key: round(wins.count(key) / draws, 4) for key in [*options, NONE_CONTAIN]}
        leader = max(p_best, key=lambda key: p_best[key])
        leader_share = p_best[leader]
        robust = leader_share >= ROBUST_SHARE
        # Robustness is measured over the *sampled* assumptions. If the leader is
        # the option driven by past-week thresholds, it has none of its own to
        # vary, so "robust" only says no sampled option beats it: a weaker claim
        # than it sounds, and the reader should be told which one they are getting.
        leader_sampled = options[leader]["basis"] == SAMPLED_BASIS if leader in options else None
        if robust and leader_sampled is False:
            notes.append(
                "The leading option rests on past-week thresholds, which are not sampled. "
                "Robust here means no sampled option beats it under the spread; it does "
                "not test the leader's own assumptions."
            )

    return {
        "simulation_only": True,
        "decision_epoch": decision_epoch,
        "draws": draws,
        "seed": seed,
        "spread": spread.to_dict(),
        "options": options,
        "p_best": p_best,
        "leader": leader,
        "leader_share": leader_share,
        "recommendation_is_robust": robust,
        #: Whether the leader's own assumptions were varied. `False` means the
        #: robustness claim covers its rivals only; `None` when there is no leader.
        "leader_assumptions_sampled": leader_sampled,
        "notes": notes,
        "caveat": (
            "The spread is a judgement, not a measurement. These bands show how far "
            "the comparison moves if the stated assumptions are off; they are not "
            "confidence intervals and carry no probability of any real outcome."
        ),
    }
