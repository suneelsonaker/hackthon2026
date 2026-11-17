"""One alert that escalates and clears, not a flickering light.

The lifecycle is a state machine over the risk score, so it is tested as one:
every transition, the hysteresis that stops flapping, the cooldown that turns a
relapse into the same alert, and the rule that an unscorable window never
clears anything. The first WARNING must also land on exactly the window the
published head start is credited to (`alerting.first_alert_index`), which is
checked against random score series, gaps included.
"""

from __future__ import annotations

import itertools
import random

import pytest

from dfsi.forecaster.alerting import AlertTracker, first_alert_index
from dfsi.forecaster.lifecycle import (
    AlertLifecycle,
    AlertState,
    LifecyclePolicy,
    run_lifecycle,
)
from dfsi.forecaster.serve import RiskBands

#: watch 0.325 / warn 0.65 / critical 0.825; the alert is held down to 0.5525.
BANDS = RiskBands.for_threshold(0.65)
LOW, WATCH, HELD, WARN, CRIT = 0.1, 0.4, 0.6, 0.7, 0.9


def run(scores, **policy):
    return run_lifecycle(scores, BANDS, LifecyclePolicy(**policy))


def states(statuses) -> list[str]:
    return [s.state.value for s in statuses]


def events(statuses) -> list[tuple[int, str]]:
    return [(i, s.event) for i, s in enumerate(statuses) if s.event]


# ------------------------------------------------------------------ raising


def test_a_quiet_range_stays_quiet_and_has_no_alert():
    out = run([LOW] * 10)
    assert set(states(out)) == {"QUIET"}
    assert all(s.alert_seq is None and not s.is_open for s in out)


def test_elevated_but_below_warn_is_watch_not_an_alert():
    out = run([WATCH] * 4)
    assert set(states(out)) == {"WATCH"}
    assert all(s.alert_seq is None for s in out)


def test_a_single_spike_never_raises():
    out = run([LOW, CRIT, LOW, LOW])
    assert not any(s.is_open for s in out)


def test_the_run_is_raised_at_the_window_that_completes_it():
    out = run([LOW, WARN, WARN, WARN])
    assert events(out) == [(2, "raised")]
    assert states(out) == ["QUIET", "WATCH", "WARNING", "WARNING"]
    assert out[2].alert_seq == 1 and out[2].opened_epoch == 20


def test_a_sustained_critical_run_raises_straight_to_critical():
    out = run([CRIT, CRIT, CRIT])
    assert events(out) == [(1, "raised")]
    assert out[1].state is AlertState.CRITICAL


# ------------------------------------------------------ parity with the debounce


def test_the_first_alert_is_the_window_the_published_metrics_credit():
    """The invariant: lifecycle and evaluation agree on when, gaps included."""
    rng = random.Random(20260101)
    for _ in range(400):
        needed = rng.randint(1, 4)
        scores = [
            None if rng.random() < 0.1 else round(rng.random(), 2)
            for _ in range(rng.randint(0, 40))
        ]
        out = run_lifecycle(scores, BANDS, LifecyclePolicy(raise_after=needed))
        first_open = next((i for i, s in enumerate(out) if s.event == "raised"), None)
        flags = [score is not None and score >= BANDS.warn for score in scores]
        assert first_open == first_alert_index(flags, needed), (needed, scores)


# ---------------------------------------------------------------- hysteresis


#: Pairs of breaching windows separated by one dip: a score hovering at the threshold.
HOVERING = [WARN, WARN] + [HELD, WARN, WARN] * 6


def test_noise_around_the_threshold_is_one_alert_not_many():
    out = run(HOVERING)
    assert [e for _, e in events(out)] == ["raised"]
    assert all(s.is_open for s in out[1:])


def test_the_bare_debounce_flickers_on_the_same_series():
    """What the lifecycle exists to fix: the same scores, one light blinking."""
    tracker = AlertTracker(2)
    active = [tracker.update(score >= BANDS.warn) for score in HOVERING]
    drops = sum(1 for before, now in itertools.pairwise(active) if before and not now)
    assert drops >= 6


def test_an_open_alert_is_held_while_the_score_sits_between_the_hold_line_and_warn():
    """Hysteresis proper: a long stretch just below warn does not start clearing."""
    out = run([WARN, WARN] + [HELD] * 20, clear_after=6)
    assert all(s.state is AlertState.WARNING for s in out[1:])


def test_below_the_hold_line_the_same_stretch_does_clear():
    below = BANDS.warn * LifecyclePolicy().clear_ratio - 0.05  # clearly under the hold line
    out = run([WARN, WARN] + [below] * 6, clear_after=6)
    assert out[-1].event == "cooling"


def test_a_single_good_window_does_not_start_clearing():
    out = run([WARN, WARN, LOW, WARN, LOW, WARN])
    assert all(s.state in (AlertState.WARNING,) for s in out[1:])


# ------------------------------------------------------------------ clearing


def test_an_alert_cools_after_clear_after_windows_below_the_hold_line():
    out = run([WARN, WARN] + [LOW] * 7, clear_after=6)
    assert events(out) == [(1, "raised"), (7, "cooling")]
    assert out[6].state is AlertState.WARNING and out[7].state is AlertState.COOLING


def test_it_closes_after_the_cooldown_and_the_closing_window_names_the_alert():
    out = run([WARN, WARN] + [LOW] * 6 + [LOW] * 12, clear_after=6, cooldown_windows=12)
    assert [e for _, e in events(out)] == ["raised", "cooling", "cleared"]
    closed = out[-1]
    assert closed.event == "cleared" and closed.state is AlertState.QUIET
    assert closed.alert_seq == 1  # says which alert it closed...
    assert closed.is_open is False  # ...and that nothing is open now
    assert out[-2].state is AlertState.COOLING


def test_a_relapse_during_cooldown_reopens_the_same_alert():
    series = [WARN, WARN] + [LOW] * 6 + [LOW] * 3 + [WARN, WARN]
    out = run(series, clear_after=6, cooldown_windows=12)
    assert [e for _, e in events(out)] == ["raised", "cooling", "reopened"]
    assert out[-1].state is AlertState.WARNING
    assert out[-1].alert_seq == 1  # one incident, one alert


def test_a_breach_after_the_alert_closed_is_a_new_alert():
    series = [WARN, WARN] + [LOW] * 6 + [LOW] * 12 + [WARN, WARN]
    out = run(series, clear_after=6, cooldown_windows=12)
    assert [e for _, e in events(out)] == ["raised", "cooling", "cleared", "raised"]
    assert out[-1].alert_seq == 2


def test_cooldown_does_not_close_while_a_relapse_is_building():
    # One window of cooldown, and the first breach window lands exactly on it.
    series = [WARN, WARN] + [LOW] * 6 + [WARN, WARN]
    out = run(series, clear_after=6, cooldown_windows=1)
    assert [e for _, e in events(out)] == ["raised", "cooling", "reopened"]


def test_the_closing_window_goes_to_watch_if_the_score_is_still_elevated():
    series = [WARN, WARN] + [LOW] * 6 + [LOW] * 11 + [WATCH]
    out = run(series, clear_after=6, cooldown_windows=12)
    assert out[-1].event == "cleared" and out[-1].state is AlertState.WATCH


# ---------------------------------------------------------------- escalation


def test_escalation_needs_consecutive_critical_windows():
    assert "CRITICAL" not in states(run([WARN, WARN, CRIT, WARN, CRIT, WARN]))
    out = run([WARN, WARN, CRIT, CRIT])
    assert events(out) == [(1, "raised"), (3, "escalated")]
    assert out[3].state is AlertState.CRITICAL


def test_it_steps_back_down_after_deescalate_after_windows_clearly_below_critical():
    out = run([CRIT, CRIT] + [HELD] * 3, deescalate_after=3)
    assert events(out) == [(1, "raised"), (4, "deescalated")]
    assert out[4].state is AlertState.WARNING and out[3].state is AlertState.CRITICAL


def test_a_critical_alert_that_drops_to_nothing_steps_down_then_cools():
    out = run([CRIT, CRIT] + [LOW] * 6, deescalate_after=3, clear_after=6)
    assert [e for _, e in events(out)] == ["raised", "deescalated", "cooling"]


def test_peak_state_and_escalations_are_recorded():
    lifecycle = AlertLifecycle(BANDS, LifecyclePolicy())
    for i, score in enumerate([WARN, WARN, CRIT, CRIT]):
        lifecycle.update(i * 10, score)
    (alert,) = lifecycle.summary()
    assert alert["peak_state"] == "CRITICAL" and alert["escalations"] == 1


# -------------------------------------------------------- unknown is not all-clear


def test_a_gap_never_clears_an_open_alert():
    out = run([WARN, WARN] + [None] * 100)
    assert all(s.state is AlertState.WARNING for s in out[1:])
    assert all(s.degraded for s in out[2:]) and not out[1].degraded


def test_a_gap_does_not_run_the_cooldown_down():
    out = run([WARN, WARN] + [LOW] * 6 + [None] * 100, clear_after=6, cooldown_windows=12)
    assert out[-1].state is AlertState.COOLING
    assert "cleared" not in [e for _, e in events(out)]


def test_a_gap_does_not_advance_clearing_either():
    # 5 windows toward clearing, a gap, then 1 more: clearing needs 6 *scored* windows.
    out = run([WARN, WARN] + [LOW] * 5 + [None] * 50 + [LOW], clear_after=6)
    assert out[-1].event == "cooling"  # the 6th scored window, gap or not
    assert all(s.state is AlertState.WARNING for s in out[2 : 2 + 5 + 50])


def test_a_gap_breaks_a_run_toward_raising():
    assert not any(s.is_open for s in run([WARN, None, WARN]))
    assert any(s.is_open for s in run([WARN, WARN, None]))


def test_a_gap_in_a_quiet_range_does_not_raise_or_clear_anything():
    out = run([None] * 5)
    assert set(states(out)) == {"QUIET"} and all(s.degraded for s in out)


def test_non_finite_scores_are_unknown_not_numbers():
    out = run([WARN, float("nan"), WARN])
    assert not any(s.is_open for s in out)


# ------------------------------------------------------------------- records


def test_the_peak_score_is_tracked_across_the_alert():
    out = run([WARN, 0.72, 0.95, 0.6, 0.6])
    assert out[-1].peak_score == pytest.approx(0.95)
    assert out[-1].opened_epoch == 10


def test_summary_lists_every_alert_and_flags_the_open_one():
    series = [WARN, WARN] + [LOW] * 6 + [LOW] * 12 + [WARN, WARN]
    lifecycle = AlertLifecycle(BANDS, LifecyclePolicy(clear_after=6, cooldown_windows=12))
    for i, score in enumerate(series):
        lifecycle.update(i * 10, score)
    first, second = lifecycle.summary()
    assert first["open"] is False and first["closed_epoch"] is not None
    assert second["open"] is True and second["closed_epoch"] is None
    assert [first["alert_seq"], second["alert_seq"]] == [1, 2]


def test_windows_in_state_counts_consecutive_windows():
    out = run([LOW, LOW, LOW])
    assert [s.windows_in_state for s in out] == [1, 2, 3]


def test_to_dict_is_plain_json():
    import json

    status = run([WARN, WARN])[-1]
    payload = json.loads(json.dumps(status.to_dict()))
    assert payload["state"] == "WARNING" and payload["is_open"] is True


# --------------------------------------------------------------------- policy


@pytest.mark.parametrize(
    "bad",
    [
        dict(raise_after=0),
        dict(escalate_after=0),
        dict(deescalate_after=0),
        dict(clear_after=0),
        dict(cooldown_windows=-1),
        dict(clear_ratio=0.0),
        dict(clear_ratio=1.2),
    ],
)
def test_an_invalid_policy_is_rejected(bad):
    with pytest.raises(ValueError):
        LifecyclePolicy(**bad)


# ------------------------------------------------ the evidence gate (a breach with no traffic behind it)


def run_with_evidence(scores, evidence, **policy):
    lifecycle = AlertLifecycle(BANDS, LifecyclePolicy(**policy))
    return [lifecycle.update(i * 10, s, e) for i, (s, e) in enumerate(zip(scores, evidence))]


def test_a_breach_the_traffic_does_not_back_cannot_raise_an_alert():
    out = run_with_evidence([CRIT] * 6, [False] * 6)
    assert not any(s.is_open for s in out)
    assert set(states(out)) == {"WATCH"}  # elevated, nothing to acknowledge


def test_the_same_breach_with_evidence_raises():
    assert run_with_evidence([CRIT] * 3, [True] * 3)[-1].is_open


def test_a_breach_must_be_backed_on_every_window_of_the_run():
    """A run needs `raise_after` consecutive *backed* windows, not just high ones."""
    out = run_with_evidence([WARN, WARN, WARN, WARN], [True, False, True, False])
    assert not any(s.is_open for s in out)


def test_unbacked_windows_do_not_clear_an_open_alert():
    """An open alert is held by its score, like any window; the gate only withholds."""
    out = run_with_evidence([WARN, WARN] + [HELD] * 20, [True, True] + [False] * 20, clear_after=6)
    assert all(s.state is AlertState.WARNING for s in out[1:])


def test_an_unbacked_critical_window_cannot_escalate():
    out = run_with_evidence([WARN, WARN, CRIT, CRIT, CRIT], [True, True, False, False, False])
    assert "CRITICAL" not in states(out)
    out = run_with_evidence([WARN, WARN, CRIT, CRIT], [True, True, True, True])
    assert states(out)[-1] == "CRITICAL"


def test_an_unbacked_relapse_does_not_reopen_an_alert():
    series = [WARN, WARN] + [LOW] * 6 + [WARN, WARN, WARN]
    out = run_with_evidence(series, [True] * 8 + [False] * 3, clear_after=6, cooldown_windows=12)
    assert "reopened" not in [e for _, e in events(out)]
    assert out[-1].state is AlertState.COOLING


def test_evidence_defaults_to_present():
    """Callers that predate the gate behave exactly as before."""
    plain = AlertLifecycle(BANDS, LifecyclePolicy())
    assert [plain.update(i * 10, s).state for i, s in enumerate([WARN, WARN, WARN])] == states(
        run([WARN] * 3)
    )


def test_the_first_alert_still_matches_the_debounce_once_evidence_is_required():
    rng = random.Random(7)
    for _ in range(300):
        needed = rng.randint(1, 4)
        n = rng.randint(0, 40)
        scores = [None if rng.random() < 0.1 else round(rng.random(), 2) for _ in range(n)]
        backed = [rng.random() < 0.7 for _ in range(n)]
        lifecycle = AlertLifecycle(BANDS, LifecyclePolicy(raise_after=needed))
        out = [lifecycle.update(i * 10, s, b) for i, (s, b) in enumerate(zip(scores, backed))]
        first = next((i for i, s in enumerate(out) if s.event == "raised"), None)
        flags = [s is not None and s >= BANDS.warn and b for s, b in zip(scores, backed)]
        assert first == first_alert_index(flags, needed), (needed, scores, backed)
