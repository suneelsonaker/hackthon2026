"""`/api/drill` carries the rehearsal extras only when asked, and never changes the drill.

The uncertainty bands and the cost of waiting are opt-in: they add work and
payload, and a client that does not ask must get exactly the response it always
got. Driven through the service layer with the replay loader injected, so no
built Memory layer is needed.
"""

from __future__ import annotations

import pytest
from conftest import EPOCH, RANGE_ID, attack_windows, make_services
from pydantic import ValidationError

from dfsi.api.errors import InvalidRequest
from dfsi.api.schemas import DrillRequest
from dfsi.rehearsal.policy_advisor import PolicyPlan
from dfsi.rehearsal.uncertainty import MAX_DRAWS

DECISION = EPOCH + 50


@pytest.fixture
def api(tmp_path):
    replay = {
        "replay_id": "sustained_attack__test",
        "scenario_type": "sustained_attack",
        "range_id": RANGE_ID,
        "windows": attack_windows(count=40, onset_index=6),
    }
    return make_services(
        tmp_path,
        load_stored_replay=lambda replay_id: replay,
        plan_policy=lambda *args: PolicyPlan(available=False, range_id=RANGE_ID, reason="test"),
    )


def drill(api, **fields) -> dict:
    return api.drill(
        DrillRequest(replay_id="sustained_attack__test", decision_epoch=DECISION, **fields)
    )


def test_by_default_the_response_has_neither_extra(api):
    result = drill(api)
    assert "uncertainty" not in result and "cost_of_waiting" not in result
    assert len(result["candidates"]) == 4


def test_the_extras_never_change_the_drill_itself(api):
    plain = drill(api)
    full = drill(api, include_uncertainty=True, uncertainty_draws=10, include_waiting=True)
    for key in ("candidates", "assumptions", "decision_epoch", "simulation_only"):
        assert full[key] == plain[key]


def test_uncertainty_is_included_when_asked(api):
    result = drill(api, include_uncertainty=True, uncertainty_draws=25)
    assert result["uncertainty"]["draws"] == 25
    assert set(result["uncertainty"]["options"]) == {
        "tighten_vector",
        "rate_limit_sources",
        "full_scrub",
    }
    assert "cost_of_waiting" not in result


def test_waiting_is_included_when_asked_with_the_default_waits(api):
    result = drill(api, include_waiting=True)
    assert result["cost_of_waiting"]["waits_seconds"] == [30, 60, 90]
    assert "uncertainty" not in result


def test_custom_waits_are_honoured(api):
    result = drill(api, include_waiting=True, waiting_seconds=[20, 40])
    assert result["cost_of_waiting"]["waits_seconds"] == [20, 40]


def test_a_wait_that_is_not_whole_windows_is_a_client_error(api):
    with pytest.raises(InvalidRequest) as caught:
        drill(api, include_waiting=True, waiting_seconds=[25])
    assert caught.value.status_code == 422
    assert "multiple" in caught.value.detail


def test_the_waits_are_ignored_unless_waiting_is_requested(api):
    assert "cost_of_waiting" not in drill(api, waiting_seconds=[25])


@pytest.mark.parametrize("draws", [0, MAX_DRAWS + 1])
def test_the_draw_count_is_bounded_by_the_request_schema(draws):
    with pytest.raises(ValidationError):
        DrillRequest(replay_id="x", include_uncertainty=True, uncertainty_draws=draws)


def test_the_schema_cap_matches_the_modules_cap():
    assert DrillRequest(replay_id="x", uncertainty_draws=MAX_DRAWS).uncertainty_draws == MAX_DRAWS


def test_too_many_waits_are_rejected_by_the_schema():
    with pytest.raises(ValidationError):
        DrillRequest(replay_id="x", waiting_seconds=[10, 20, 30, 40, 50, 60, 70])
