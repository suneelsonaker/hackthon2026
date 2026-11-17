"""What would calm this alert?

The evidence behind an alert says what is pushing the score up. An operator
deciding whether to act also wants the converse: how far would those drivers
have to come back for the alert to stop? "Packet-rate deviation would have to
fall from 18 to about 7" is something a person can check against the traffic
in front of them; "contribution 0.85" is not.

Two views, both answered by asking the model, not by guessing:

* **single feature** — for each top driver alone, the smallest move back towards
  its typical quiet value after which the score is below the decision
  threshold, if any such move exists. Often none does: when several drivers
  agree, fixing one is not enough, and saying so is the honest answer;
* **joint** — the smallest set of top drivers which, reset together to their
  quiet values, brings the score below threshold.

This is a sensitivity analysis of the model, not an instruction and not a causal
claim about the traffic. Features are correlated (packet rate, policy
utilisation and blocked share move together), so moving one in isolation can
describe a window that never occurs. Treat it as "which signals is the model
leaning on, and how hard", and read the joint view as the more realistic one.

Pure and model-agnostic: it takes a `predict` function and plain numbers.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np

#: Points along the way from the current value to the quiet value.
DEFAULT_GRID = 41
#: How many top drivers get a single-feature search.
DEFAULT_SINGLE_TOP = 3
#: Most drivers reset together in the joint view.
DEFAULT_MAX_SET = 5

NOTE = (
    "A what-if on the model, not an instruction and not a causal claim: features "
    "are correlated, so a single-feature move can describe a window that never "
    "occurs. The joint view is the more realistic one."
)


def _round(value: float) -> float:
    return round(float(value), 4)


def find_counterfactual(
    predict: Callable[[np.ndarray], np.ndarray],
    names: Sequence[str],
    values: Sequence[float],
    reference: Mapping[str, float],
    drivers: Sequence[str],
    labels: Mapping[str, str],
    threshold: float,
    score: float,
    *,
    grid: int = DEFAULT_GRID,
    single_top: int = DEFAULT_SINGLE_TOP,
    max_set: int = DEFAULT_MAX_SET,
) -> dict | None:
    """Which moves back towards normal would put `score` below `threshold`?

    `predict` maps an (n, len(names)) array to n scores. `values` is the current
    feature vector in `names` order. `drivers` are the features to try, strongest
    first. Returns `None` when there is nothing to explain (the score is already
    below the threshold) or when no driver can be tried.
    """
    if score < threshold:
        return None
    if grid < 2 or single_top < 1 or max_set < 1:
        raise ValueError("grid must be at least 2 and single_top and max_set at least 1")

    index = {name: i for i, name in enumerate(names)}
    base = np.asarray(values, dtype=float)
    tried = [
        name
        for name in drivers
        if name in index
        and name in reference
        and math.isfinite(base[index[name]])
        and math.isfinite(reference[name])
    ]
    if not tried:
        return None

    def describe(name: str) -> dict:
        return {
            "feature": name,
            "label": labels.get(name, name),
            "current": _round(base[index[name]]),
            "quiet_value": _round(reference[name]),
        }

    # ---- single feature: walk each top driver back towards its quiet value ----
    alphas = np.linspace(0.0, 1.0, grid)
    single = []
    for name in tried[:single_top]:
        column = index[name]
        batch = np.tile(base, (grid, 1))
        batch[:, column] = base[column] + alphas * (reference[name] - base[column])
        scores = np.asarray(predict(batch), dtype=float)
        cleared = np.flatnonzero(scores < threshold)
        entry = describe(name)
        if cleared.size:
            first = int(cleared[0])
            entry.update(
                {
                    "clears_alone": True,
                    "required_value": _round(batch[first, column]),
                    "fraction_of_move": _round(alphas[first]),
                    "score_there": _round(scores[first]),
                }
            )
        else:
            entry.update(
                {
                    "clears_alone": False,
                    "required_value": None,
                    "fraction_of_move": None,
                    # Even fully back to quiet it stays above threshold.
                    "score_there": _round(scores[-1]),
                }
            )
        single.append(entry)

    # ---- joint: reset the strongest drivers together, one more at a time ----
    candidates = tried[:max_set]
    batch = np.tile(base, (len(candidates), 1))
    for k in range(len(candidates)):
        for earlier in candidates[: k + 1]:
            batch[k, index[earlier]] = reference[earlier]
    joint_scores = np.asarray(predict(batch), dtype=float)
    cleared = np.flatnonzero(joint_scores < threshold)
    if cleared.size:
        k = int(cleared[0])
        joint = {
            "clears": True,
            "features": [describe(name) for name in candidates[: k + 1]],
            "score_if_reset": _round(joint_scores[k]),
        }
    else:
        joint = {
            "clears": False,
            "features": [describe(name) for name in candidates],
            # Even with all of them back to quiet, the model still alerts.
            "score_if_reset": _round(joint_scores[-1]),
        }

    return {
        "threshold": _round(threshold),
        "score": _round(score),
        "single_feature": single,
        "joint": joint,
        "note": NOTE,
    }
