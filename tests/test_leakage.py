"""Anti-leakage guarantees.

A forecaster that can see the future scores beautifully and is worthless. The
pitch commits to three specific rules, and each has a test here:

* split by whole scenario, never by neighbouring windows;
* `scenario_id` is never a model input;
* no future-derived value reaches the feature vector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from conftest import requires_memory

from dfsi.config import PATHS
from dfsi.forecaster.train import (
    FORBIDDEN_INPUTS,
    WEEK_COLUMN,
    assert_no_leakage,
    split_by_scenario,
    split_by_time,
)
from dfsi.memory.features import FEATURE_COLUMNS, GROUP_COLUMN, LABEL_COLUMN


def test_feature_list_is_clean():
    assert_no_leakage(list(FEATURE_COLUMNS))


def test_label_is_not_a_feature():
    assert LABEL_COLUMN not in FEATURE_COLUMNS
    assert LABEL_COLUMN in FORBIDDEN_INPUTS


def test_scenario_id_is_not_a_feature():
    """It identifies the episode, so it would let the model memorise outcomes."""
    assert GROUP_COLUMN not in FEATURE_COLUMNS
    assert GROUP_COLUMN in FORBIDDEN_INPUTS


def test_scenario_type_is_not_a_feature():
    """The scenario type *is* the answer; it cannot be an input."""
    assert "scenario_type" not in FEATURE_COLUMNS
    assert "scenario_type" in FORBIDDEN_INPUTS


@pytest.mark.parametrize(
    "column",
    ["is_onset", "declared", "range_over_threshold", "blocked_pps", "sustained_windows"],
)
def test_outcome_derived_columns_are_forbidden(column):
    """Each of these is a consequence of the onset, not a precursor of it."""
    assert column in FORBIDDEN_INPUTS
    assert column not in FEATURE_COLUMNS


def test_assert_no_leakage_rejects_a_forbidden_column():
    with pytest.raises(ValueError, match="future-derived or identifying"):
        assert_no_leakage([*FEATURE_COLUMNS, "is_onset"])


def test_assert_no_leakage_names_every_offender():
    with pytest.raises(ValueError) as excinfo:
        assert_no_leakage([*FEATURE_COLUMNS, "is_onset", "declared"])
    message = str(excinfo.value)
    assert "is_onset" in message and "declared" in message


# ------------------------------------------------------------------ split


def synthetic_frame(n_scenarios: int = 20, windows: int = 8) -> pd.DataFrame:
    """Half attack scenarios, half quiet, so both pools can be split."""
    rows = []
    for s in range(n_scenarios):
        has_onset = s % 2 == 0
        for w in range(windows):
            rows.append(
                {
                    GROUP_COLUMN: f"scenario-{s:04d}",
                    LABEL_COLUMN: int(has_onset and w == windows - 2),
                    "window_start_epoch": w * 10,
                }
            )
    return pd.DataFrame(rows)


def test_split_keeps_whole_scenarios_on_one_side():
    """Neighbouring windows are near-identical; splitting between them leaks."""
    frame = synthetic_frame()
    train, test = split_by_scenario(frame, test_size=0.3, random_state=7)
    overlap = set(train[GROUP_COLUMN]) & set(test[GROUP_COLUMN])
    assert overlap == set()
    assert len(train) + len(test) == len(frame)


def test_split_puts_attacks_on_both_sides():
    """A test set with no onsets cannot measure recall."""
    frame = synthetic_frame()
    train, test = split_by_scenario(frame, test_size=0.3, random_state=7)
    assert train[LABEL_COLUMN].max() == 1
    assert test[LABEL_COLUMN].max() == 1


def test_split_is_reproducible():
    frame = synthetic_frame()
    first, _ = split_by_scenario(frame, test_size=0.3, random_state=7)
    second, _ = split_by_scenario(frame, test_size=0.3, random_state=7)
    assert sorted(first[GROUP_COLUMN].unique()) == sorted(second[GROUP_COLUMN].unique())


def weekly_frame(weeks=(1, 2, 3, 4), per_week: int = 6) -> pd.DataFrame:
    frame = synthetic_frame(n_scenarios=len(weeks) * per_week)
    scenario_week = {
        f"scenario-{s:04d}": weeks[s // per_week] for s in range(len(weeks) * per_week)
    }
    frame[WEEK_COLUMN] = frame[GROUP_COLUMN].map(scenario_week)
    return frame


def test_time_split_tests_on_the_latest_week_only():
    fit, calib, test = split_by_time(weekly_frame(), test_weeks=1, calibration_weeks=1)
    assert set(test[WEEK_COLUMN]) == {4}
    assert set(calib[WEEK_COLUMN]) == {3}
    assert set(fit[WEEK_COLUMN]) == {1, 2}


def test_time_split_never_trains_on_the_future():
    fit, calib, test = split_by_time(weekly_frame(), test_weeks=1, calibration_weeks=1)
    assert fit[WEEK_COLUMN].max() < calib[WEEK_COLUMN].min()
    assert calib[WEEK_COLUMN].max() < test[WEEK_COLUMN].min()
    groups = [set(part[GROUP_COLUMN]) for part in (fit, calib, test)]
    assert not (groups[0] & groups[1] or groups[1] & groups[2] or groups[0] & groups[2])


def test_time_split_refuses_too_few_weeks():
    with pytest.raises(ValueError, match="needs 3 featurised weeks"):
        split_by_time(weekly_frame(weeks=(1, 2)), test_weeks=1, calibration_weeks=1)


def test_split_bookkeeping_is_forbidden_as_input():
    for column in (WEEK_COLUMN, "source_week_max", "is_attack_vector"):
        assert column in FORBIDDEN_INPUTS


# -------------------------------------------------- the built feature table


@requires_memory
def test_built_feature_table_has_no_forbidden_input_in_the_vector():
    # The schema alone: no need to load ~20M rows to read column names.
    columns = set(pq.read_schema(next(PATHS.features.glob("*.parquet"))).names)
    assert_no_leakage(list(FEATURE_COLUMNS))
    for column in FEATURE_COLUMNS:
        assert column in columns, column


@requires_memory
def test_built_features_are_finite():
    """A NaN or infinity here would silently become a split point in the trees."""
    frame = pd.read_parquet(PATHS.features, columns=list(FEATURE_COLUMNS))
    values = frame.to_numpy(dtype=float)
    assert np.isfinite(values).all()


@requires_memory
def test_label_is_rare_and_present():
    """Onsets are a fraction of a percent; a label that is common is a bug."""
    frame = pd.read_parquet(PATHS.features, columns=[LABEL_COLUMN])
    rate = frame[LABEL_COLUMN].mean()
    assert 0.0 < rate < 0.05, f"positive rate {rate:.4%} is implausible"


@requires_memory
def test_every_labelled_window_sits_inside_the_horizon_before_an_onset():
    """The label's defining property, asserted directly.

    A scenario can contain several onset runs (an attack that ebbs and
    resumes), so the invariant is against *any* onset, not just the first:
    every labelled window must sit strictly between 1 and 12 windows before
    one. A window at or after every onset would mean the label describes the
    present; a window more than 120 seconds ahead would mean it is not a
    120-second forecast.
    """
    frame = pd.read_parquet(
        PATHS.features, columns=[GROUP_COLUMN, "window_start_epoch", "is_onset", LABEL_COLUMN]
    )
    onsets = {
        scenario_id: sorted(set(group))
        for scenario_id, group in frame[frame["is_onset"] == 1].groupby(GROUP_COLUMN)[
            "window_start_epoch"
        ]
    }
    labelled = frame[frame[LABEL_COLUMN] == 1][
        [GROUP_COLUMN, "window_start_epoch"]
    ].drop_duplicates()
    assert not labelled.empty, "no positive labels at all"

    leads: list[int] = []
    for scenario_id, group in labelled.groupby(GROUP_COLUMN):
        scenario_onsets = onsets.get(scenario_id, [])
        assert scenario_onsets, f"{scenario_id} is labelled but has no onset"
        for epoch in group["window_start_epoch"]:
            ahead = [o - epoch for o in scenario_onsets if 0 < o - epoch <= 120]
            assert ahead, (
                f"{scenario_id} window {epoch} is labelled but no onset falls "
                "in the next 120 seconds"
            )
            leads.append(min(ahead))

    assert min(leads) >= 10, "a label cannot point at the current window"
    assert max(leads) <= 120, "a label cannot reach past the forecast horizon"


@requires_memory
def test_declared_requires_the_full_start_duration():
    """The bell waits out the start duration; the label must respect that.

    This is the arithmetic that broke once: the duration frame is counted in
    *windows*, and the feature table has one row per traffic type, so a
    row-counted frame would divide every duration by three.
    """
    from dfsi.config import thresholds_for

    required = thresholds_for().start_duration_windows
    per_instant = pd.read_parquet(
        PATHS.features,
        columns=[GROUP_COLUMN, "window_start_epoch", "range_over_threshold", "declared"],
    ).drop_duplicates()

    for _, group in per_instant.groupby(GROUP_COLUMN):
        group = group.sort_values("window_start_epoch")
        over = group["range_over_threshold"].to_numpy()
        declared = group["declared"].to_numpy()
        for i in np.flatnonzero(declared == 1):
            window = over[max(0, i - required + 1) : i + 1]
            assert window.sum() >= required, (
                "declared without the full start duration over threshold"
            )


# ------------------------------------------- past-only baselines and labels


@requires_memory
def test_every_baseline_is_built_from_earlier_weeks_only():
    """The leak this replaced: one week of data meant each (range, hour of
    week) had exactly one episode, so its baseline was the episode itself —
    future windows included."""
    from dfsi.config import BASELINE_LOOKBACK_WEEKS

    baselines = pd.read_parquet(
        PATHS.baselines, columns=["week_index", "source_week_min", "source_week_max"]
    )
    assert not baselines.empty
    assert (baselines["source_week_max"] < baselines["week_index"]).all()
    assert (baselines["source_week_min"] >= baselines["week_index"] - BASELINE_LOOKBACK_WEEKS).all()


@requires_memory
def test_featurised_windows_never_see_their_own_week_in_the_baseline():
    frame = pd.read_parquet(PATHS.features, columns=["week_index", "source_week_max"])
    joined = frame.dropna(subset=["source_week_max"])
    assert len(joined) > 0.99 * len(frame), "most windows should have a baseline"
    assert (joined["source_week_max"] < joined["week_index"]).all()


@requires_memory
def test_the_warm_up_week_never_reaches_the_feature_table():
    raw_weeks = set(pd.read_parquet(PATHS.raw_windows, columns=["week_index"])["week_index"])
    feature_weeks = set(pd.read_parquet(PATHS.features, columns=["week_index"])["week_index"])
    assert min(raw_weeks) not in feature_weeks
    assert feature_weeks == raw_weeks - {min(raw_weeks)}


@requires_memory
def test_positive_labels_sit_on_a_vector_that_carries_the_onset():
    """Per-type label: a quiet vector's row is never asked to predict an onset
    its own features cannot see."""
    columns = [
        GROUP_COLUMN,
        "traffic_type",
        "window_start_epoch",
        "is_onset",
        "over_threshold",
        LABEL_COLUMN,
    ]
    frame = pd.read_parquet(PATHS.features, columns=columns)
    carriers = frame[(frame["is_onset"] == 1) & (frame["over_threshold"] == 1)][
        [GROUP_COLUMN, "traffic_type", "window_start_epoch"]
    ].rename(columns={"window_start_epoch": "onset_epoch"})
    positives = frame[frame[LABEL_COLUMN] == 1][
        [GROUP_COLUMN, "traffic_type", "window_start_epoch"]
    ]
    assert not positives.empty

    paired = positives.merge(carriers, on=[GROUP_COLUMN, "traffic_type"], how="left")
    paired["ahead"] = paired["onset_epoch"] - paired["window_start_epoch"]
    paired["ok"] = (paired["ahead"] > 0) & (paired["ahead"] <= 120)
    supported = paired.groupby([GROUP_COLUMN, "traffic_type", "window_start_epoch"])["ok"].any()
    assert supported.all(), supported[~supported].head()


@requires_memory
def test_demo_replays_come_from_the_held_out_week():
    """The demo must not show the model a scenario it was trained on."""
    import json

    from dfsi.config import TEST_WEEKS

    feature_weeks = sorted(
        set(pd.read_parquet(PATHS.features, columns=["week_index"])["week_index"])
    )
    test_weeks = set(feature_weeks[-TEST_WEEKS:])
    catalogue = json.loads((PATHS.replay / "catalogue.json").read_text())
    assert catalogue
    for entry in catalogue:
        replay = json.loads((PATHS.replay / f"{entry['replay_id']}.json").read_text())
        day = int(replay["scenario_id"].split("-d")[1][:2])
        assert day // 7 in test_weeks, entry["replay_id"]
