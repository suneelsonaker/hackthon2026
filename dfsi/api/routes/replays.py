"""The trigger: stream a scenario replay past the forecaster as server-sent events."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ...config import HORIZON_SECONDS
from .. import replay as replay_engine
from ..deps import Services
from ..schemas import ReplayRequest
from ..services import DfsiServices

log = logging.getLogger("dfsi.api")

router = APIRouter(prefix="/api", tags=["replay"])


@router.post("/replay/stream")
async def stream_replay(request: ReplayRequest, services: Services) -> StreamingResponse:
    """One `window` event per 10-second window, then a `summary` with the head start.

    The replay is loaded before the response starts, so an unknown id is a
    plain 404 rather than an error event inside a 200 stream.
    """
    replay = services.load_replay(request.replay_id)
    return StreamingResponse(
        _events(services, replay, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _events(
    services: DfsiServices, replay: dict, request: ReplayRequest
) -> AsyncIterator[str]:
    forecaster = services.forecaster
    state = replay_engine.build_state(replay)
    threshold = services.threshold_for(state.range_id)
    policy_version, policy = replay["policy_version"], replay["policy"]

    yield _sse(
        {
            "type": "start",
            "replay_id": state.replay_id,
            "scenario_type": state.scenario_type,
            "range_id": state.range_id,
            "total_windows": state.total,
            "horizon_seconds": HORIZON_SECONDS,
            "model_version": forecaster.model_version,
            "feature_version": forecaster.feature_version,
            "forecaster_ready": forecaster.ready,
            "policy_version": policy_version,
            "policy": policy,
            "decision_threshold": threshold,
        }
    )
    windows_after_bell = 0
    try:
        while not state.finished:
            event = await asyncio.to_thread(replay_engine.step, state, forecaster, threshold)
            yield _sse(event)
            if state.bell_epoch is not None:
                windows_after_bell += 1
                stop_after = request.stop_after_bell_windows
                if stop_after and windows_after_bell >= stop_after:
                    break
            if request.speed_seconds:
                await asyncio.sleep(request.speed_seconds)
    except asyncio.CancelledError:
        # The client navigated away. The replay holds no external resources,
        # so there is nothing to clean up — but never swallow a cancellation.
        raise
    except Exception as exc:
        log.exception("replay %s failed", state.replay_id)
        # The detail stays in the server log: an exception message can carry
        # paths or internals that have no business in a browser.
        yield _sse(
            {
                "type": "error",
                "message": f"replay failed ({type(exc).__name__}); see the API log for details",
            }
        )
        return

    yield _sse(
        {
            "type": "summary",
            "replay_id": state.replay_id,
            "scenario_type": state.scenario_type,
            "first_warning_epoch": state.first_warning_epoch,
            "onset_epoch": state.onset_epoch,
            "bell_epoch": state.bell_epoch,
            "lead_seconds": (
                state.onset_epoch - state.first_warning_epoch
                if state.onset_epoch is not None and state.first_warning_epoch is not None
                else None
            ),
            "windows_streamed": state.cursor,
            "policy_version": policy_version,
            "policy": policy,
            **replay_engine.run_stats(state),
            # One entry per incident, not per flicker: when each alert opened
            # and closed, how high it peaked, whether it escalated.
            "alerts": replay_engine.alert_summary(state),
        }
    )


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"
