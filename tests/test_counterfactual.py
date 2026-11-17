"""What would calm this alert?

The search is model-agnostic, so it is tested against an analytic model whose
answers can be worked out by hand:

    score = sigmoid(a + b - 3)        (a third feature, `c`, is ignored)

with every quiet value 0. The decision threshold is 0.5, i.e. the alert stops
when a + b < 3. That makes "does one driver suffice?", "do two?" and "can it
never clear?" all computable on paper.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dfsi.forecaster.counterfactual import NOTE, find_counterfactual

NAMES = ["a", "b", "c"]
REFERENCE = {"a": 0.0, "b": 0.0, "c": 0.0}
LABELS = {"a": "feature a", "b": "feature b", "c": "feature c"}
THRESHOLD = 0.5


def sigmoid_model(rows: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(rows[:, 0] + rows[:, 1] - 3.0)))


def search(values, drivers=("a", "b", "c"), predict=sigmoid_model, **kwargs):
    score = float(predict(np.array([values], dtype=float))[0])
    return find_counterfactual(
        predict, NAMES, values, REFERENCE, list(drivers), LABELS, THRESHOLD, score, **kwargs
    )


def single(result: dict, name: str) -> dict:
    return next(e for e in result["single_feature"] if e["feature"] == name)


# ----------------------------------------------------------- one driver suffices


def test_the_smallest_move_that_clears_the_alert_is_found():
    """a=6, b=0.5: z=3.5. The alert stops once a < 2.5, i.e. 58.3% of the way back."""
    result = search([6.0, 0.5, 9.0])
    a = single(result, "a")
    assert a["clears_alone"] is True
    assert a["fraction_of_move"] == pytest.approx(0.6)  # first grid point past 0.5833
    assert a["required_value"] == pytest.approx(2.4)
    assert a["score_there"] < THRESHOLD
    assert a["current"] == 6.0 and a["quiet_value"] == 0.0


def test_the_required_value_really_does_clear_it_and_one_step_earlier_does_not():
    result = search([6.0, 0.5, 9.0])
    a = single(result, "a")
    just_enough = np.array([[a["required_value"], 0.5, 9.0]])
    one_step_short = np.array([[a["required_value"] + 6.0 / 40, 0.5, 9.0]])
    assert sigmoid_model(just_enough)[0] < THRESHOLD <= sigmoid_model(one_step_short)[0]


def test_a_driver_that_cannot_clear_it_alone_says_so():
    result = search([6.0, 0.5, 9.0])
    b = single(result, "b")  # b is already small; zeroing it leaves z = 3
    assert b["clears_alone"] is False
    assert b["required_value"] is None and b["fraction_of_move"] is None
    assert b["score_there"] >= THRESHOLD


# --------------------------------------------------------- neither alone, both together


def test_two_drivers_that_must_both_come_back_are_reported_jointly():
    """a=4, b=3: z=4. Zeroing either alone gives z=0 -> score 0.5, not below."""
    result = search([4.0, 3.0, 9.0], drivers=("a", "b"))
    assert single(result, "a")["clears_alone"] is False
    assert single(result, "b")["clears_alone"] is False
    joint = result["joint"]
    assert joint["clears"] is True
    assert [f["feature"] for f in joint["features"]] == ["a", "b"]
    assert joint["score_if_reset"] < THRESHOLD


def test_the_joint_set_is_the_smallest_prefix_that_clears():
    # One driver is enough here, so the set must not pad itself with the second.
    result = search([6.0, 0.5, 9.0], drivers=("a", "b"))
    assert [f["feature"] for f in result["joint"]["features"]] == ["a"]


def test_the_joint_order_follows_the_drivers_given_strongest_first():
    result = search([4.0, 3.0, 9.0], drivers=("b", "a"))
    assert [f["feature"] for f in result["joint"]["features"]] == ["b", "a"]


# --------------------------------------------------------------- it cannot clear


def test_when_nothing_clears_it_the_answer_is_no_not_a_guess():
    always_high = lambda rows: np.full(len(rows), 0.9)  # noqa: E731
    result = search([4.0, 3.0, 9.0], predict=always_high)
    assert result["joint"]["clears"] is False
    assert result["joint"]["score_if_reset"] == pytest.approx(0.9)
    assert all(not e["clears_alone"] for e in result["single_feature"])


def test_an_irrelevant_driver_never_clears_anything():
    result = search([4.0, 3.0, 9.0], drivers=("c",))
    assert single(result, "c")["clears_alone"] is False
    assert result["joint"]["clears"] is False


# ----------------------------------------------------------------- nothing to explain


def test_a_score_below_the_threshold_has_nothing_to_explain():
    assert search([0.5, 0.5, 0.5]) is None


def test_a_score_exactly_at_the_threshold_is_above_it():
    """Matches the service: a window counts as above at score >= threshold."""
    assert search([1.5, 1.5, 0.0]) is not None  # z = 0 -> score 0.5


def test_no_usable_driver_returns_none():
    assert search([6.0, 0.5, 9.0], drivers=()) is None
    assert search([6.0, 0.5, 9.0], drivers=("not_a_feature",)) is None
    assert search([float("nan"), 0.5, 9.0], drivers=("a",)) is None


# ----------------------------------------------------------------- how it works


def test_it_does_not_modify_its_inputs():
    values = np.array([6.0, 0.5, 9.0])
    before = values.copy()
    score = float(sigmoid_model(values[None, :])[0])
    find_counterfactual(
        sigmoid_model, NAMES, values, REFERENCE, ["a", "b"], LABELS, THRESHOLD, score
    )
    assert np.array_equal(values, before)


def test_the_model_is_called_in_batches_not_once_per_point():
    calls = []

    def counting(rows):
        calls.append(len(rows))
        return sigmoid_model(rows)

    search([6.0, 0.5, 9.0], predict=counting, grid=41, single_top=3)
    # one batch per single-feature search, plus one for the joint view, plus the score above
    assert len(calls) == 1 + 3 + 1
    assert max(calls) == 41


def test_the_result_is_plain_json_and_carries_the_caveat():
    result = search([4.0, 3.0, 9.0], drivers=("a", "b"))
    assert json.loads(json.dumps(result)) == result
    assert result["note"] == NOTE and "not an instruction" in NOTE


@pytest.mark.parametrize("bad", [dict(grid=1), dict(single_top=0), dict(max_set=0)])
def test_nonsense_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        search([6.0, 0.5, 9.0], **bad)
