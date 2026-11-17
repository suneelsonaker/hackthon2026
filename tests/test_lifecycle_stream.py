"""The lifecycle on the live stream.

It is reported *alongside* the existing per-window state, never instead of it:
`dfsi_state` keeps meaning what it meant, and the lifecycle adds the part an
operator actually wants, which is one alert per incident. The invariant that
ties it to the published numbers is that the stream's first alert is the very
window the evaluation credits the head start to.
"""

from __future__ import annotations

import json
import random

from conftest import make_client
from test_alerting import EPOCH, ScriptedForecaster, make_replay

from dfsi.api import replay as replay_engine
from dfsi.config import STATE_ABOUT_TO_START

#: Against a 0.65 threshold: warn 0.65, critical 0.825, held down to 0.5525.
THRESHOLD = 0.65
WARN, HELD, LOW = 0.7, 0.6, 0.1
#: Pairs of breaching windows with one dip between them: a score hovering at threshold.
HOVERING = [WARN, WARN] + [HELD, WARN, WARN] * 5


def stream(scores, min_consecutive: int = 2, onset_index: int | None = None):
    forecaster = ScriptedForecaster(scores, min_consecutive, THRESHOLD)
    state = replay_engine.build_state(make_replay(len(scores), onset_index))
    events = []
    while not state.finished:
        events.append(replay_engine.step(state, forecaster))
    return state, events


def test_every_window_event_carries_the_alert():
    _, events = stream([LOW, LOW, WARN, WARN])
    assert all("alert" in e for e in events)
    assert [e["alert"]["state"] for e in events] == ["QUIET", "QUIET", "WATCH", "WARNING"]
    assert events[-1]["alert"]["event"] == "raised" and events[-1]["alert"]["is_open"] is True


def test_the_per_window_state_is_unchanged_by_the_lifecycle():
    _, events = stream(HOVERING)
    # dfsi_state still drops on every dip, exactly as before...
    assert "QUIET" in [e["dfsi_state"] for e in events[2:]]
    # ...while the lifecycle holds one alert open through them.
    assert all(e["alert"]["is_open"] for e in events[1:])
    assert {e["alert"]["alert_seq"] for e in events[1:]} == {1}


def test_the_first_alert_is_the_window_the_dashboard_credits_the_lead_to():
    """The invariant, on random series, gaps included."""
    rng = random.Random(20260102)
    for _ in range(150):
        needed = rng.randint(1, 4)
        scores = [
            None if rng.random() < 0.1 else round(rng.random(), 2)
            for _ in range(rng.randint(5, 40))
        ]
        state, events = stream(scores, min_consecutive=needed)
        first_alert = next(
            (e["window_start_epoch"] for e in events if e["alert"]["event"] == "raised"), None
        )
        assert first_alert == state.first_warning_epoch, (needed, scores)


def test_a_window_the_forecaster_could_not_score_is_flagged_and_clears_nothing():
    _state, events = stream([WARN, WARN, None, None, None])
    assert [e["alert"]["degraded"] for e in events] == [False, False, True, True, True]
    assert all(e["alert"]["state"] == "WARNING" for e in events[1:])
    assert events[-1]["dfsi_state"] == "MONITORING_ONLY"


def test_the_summary_lists_one_alert_per_incident():
    scores = [WARN, WARN] + [LOW] * 6 + [LOW] * 12 + [WARN, WARN] + [WARN] * 2
    state, _ = stream(scores)
    first, second = replay_engine.alert_summary(state)
    assert (first["alert_seq"], first["open"]) == (1, False)
    assert (second["alert_seq"], second["open"]) == (2, True)


def test_no_alerts_is_an_empty_summary_not_an_error():
    state, _ = stream([LOW] * 5)
    assert replay_engine.alert_summary(state) == []
    unstarted = replay_engine.build_state(make_replay(3))
    assert replay_engine.alert_summary(unstarted) == []


def test_the_alert_state_values_are_the_documented_vocabulary():
    _, events = stream([LOW, 0.4, WARN, WARN, 0.99, 0.99] + [LOW] * 20)
    seen = {e["alert"]["state"] for e in events}
    assert seen <= {"QUIET", "WATCH", "WARNING", "CRITICAL", "COOLING"}
    assert {"WATCH", "WARNING", "CRITICAL", "COOLING"} <= seen
    assert STATE_ABOUT_TO_START not in seen  # a different vocabulary, deliberately


# ------------------------------------------------------------ over the wire


class StreamForecaster(ScriptedForecaster):
    model_version = "test/v1"
    feature_version = "features-test"
    ready = True


def test_the_sse_summary_event_carries_the_alerts(tmp_path):
    scores = [LOW, WARN, WARN, WARN] + [LOW] * 8
    client = make_client(
        tmp_path,
        forecaster=StreamForecaster(scores, threshold=THRESHOLD),
        load_stored_replay=lambda replay_id: make_replay(len(scores)),
    )

    response = client.post("/api/replay/stream", json={"replay_id": "scripted", "speed_seconds": 0})
    events = [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
    ]

    assert events[0]["type"] == "start" and events[-1]["type"] == "summary"
    windows = [e for e in events if e["type"] == "window"]
    assert len(windows) == len(scores) and all("alert" in w for w in windows)
    (alert,) = events[-1]["alerts"]
    assert alert["alert_seq"] == 1 and alert["opened_epoch"] == EPOCH + 20
