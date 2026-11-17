"""The rehearsal room and the Explainer agents."""

from __future__ import annotations

from fastapi import APIRouter

from ..deps import Services
from ..schemas import DrillRequest, ExplainRequest

router = APIRouter(prefix="/api", tags=["rehearsal"])


@router.post("/drill")
def drill(request: DrillRequest, services: Services) -> dict:
    """Compare the four candidate responses on a replay. Simulation only."""
    return services.drill(request)


@router.post("/explain")
def explain(request: ExplainRequest, services: Services) -> dict:
    """Run the Investigator and Advisor, each checked by the code validator."""
    return services.explain(request)
