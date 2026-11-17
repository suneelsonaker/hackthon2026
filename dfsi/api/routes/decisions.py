"""Human decisions, the simulated policy store and the per-range thresholds."""

from __future__ import annotations

from fastapi import APIRouter, Query

from ..deps import Services
from ..schemas import DecisionRecord, DecisionRequest

router = APIRouter(prefix="/api", tags=["decisions"])

#: Most audit records one request may read.
MAX_DECISIONS = 500


@router.post("/decision", response_model=DecisionRecord)
def record_decision(request: DecisionRequest, services: Services) -> DecisionRecord:
    """Audit a decision; an approved threshold change updates the simulated store only."""
    return services.record_decision(request)


@router.get("/decisions")
def list_decisions(
    services: Services, limit: int = Query(default=50, ge=1, le=MAX_DECISIONS)
) -> dict:
    """The newest audit records, newest first."""
    return {"decisions": services.audit_log.tail(limit)}


@router.get("/policies")
def list_policies(services: Services) -> dict:
    """The simulated policy store: current lines per range and type, and history."""
    return services.policy_store.snapshot()


@router.post("/policies/reset")
def reset_policies(services: Services) -> dict:
    """Restore the store to its seed (raw.json)."""
    return services.reset_policies()


@router.get("/risk-thresholds")
def list_risk_thresholds(services: Services) -> dict:
    """The model's decision threshold, and each range's `risk_threshold` from raw.json."""
    return {
        "model_default": services.forecaster.decision_threshold,
        "ranges": {str(k): v for k, v in sorted(services.range_thresholds.items())},
    }
