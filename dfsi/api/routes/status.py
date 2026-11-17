"""Service status, the replay catalogue and the model card."""

from __future__ import annotations

import json

from fastapi import APIRouter

from ...config import FEATURE_VERSION, HORIZON_SECONDS, thresholds_for
from ...forecaster.calibration import calibration_summary
from ..deps import Services
from ..errors import NotBuilt
from ..schemas import HealthResponse, ModelCardResponse

router = APIRouter(prefix="/api", tags=["status"])

KNOWN_LIMITATIONS = [
    "Trained and evaluated on synthetic traffic only. Real-world performance is unmeasured.",
    "Evaluated on one held-out synthetic week that follows every week the "
    "model was fitted or calibrated on. One week is a small sample; treat "
    "the headline figures as estimates, not guarantees.",
    "An alert needs traffic that is abnormal for the hour (the evidence gate), "
    "so context alone, such as recent attack history on the account, cannot "
    "raise one. An attack whose first minutes stay inside the normal range "
    "alerts only once its deviation passes the bar. The bar was set from the "
    "pipeline's own abnormal-window threshold and its effect measured on "
    "synthetic data only; withheld windows are reported, not hidden.",
    "Recovery scenarios open mid-attack, so there is no runway to forecast "
    "from. They are excluded from the head-start figures and reported "
    "separately as `in_progress_detected` — a detection rate, not a warning.",
    "The head start is measured against the moment Shared State would "
    "declare, not the moment traffic crosses the policy threshold. On a "
    "production start duration of 300s the gap between those two is larger "
    "than it is on this scenario set.",
    "A flash crowd large enough to cross the policy threshold is a true onset "
    "by DFSI's label definition, even though no attacker is involved.",
    "The rehearsal room's suppression and collateral figures are stated "
    "assumptions, not measurements of Behemoth's actual behaviour.",
    "Source information is country-mix concentration only; an attack that is "
    "evenly distributed across many countries loses that signal.",
    "Approved threshold changes apply to DFSI's simulated, in-memory policy "
    "store (seeded from raw.json), not to Behemoth or any customer "
    "configuration. Restarting the API, or resetting the store, restores the seed.",
]


@router.get("/health", response_model=HealthResponse)
def health(services: Services) -> HealthResponse:
    """Service, model and agent status. `degraded` when no model or no replays."""
    forecaster = services.forecaster
    catalogue = services.load_catalogue()
    return HealthResponse(
        status="ok" if forecaster.ready and catalogue else "degraded",
        forecaster_ready=forecaster.ready,
        forecaster_error=forecaster.load_error,
        model_version=forecaster.model_version,
        feature_version=forecaster.feature_version,
        horizon_seconds=HORIZON_SECONDS,
        agent_engine=services.orchestrator.engine,
        replays_available=len(catalogue),
        environment=thresholds_for().environment,
    )


@router.get("/replays")
def list_replays(services: Services) -> dict:
    """The scenario catalogue: the dashboard's trigger menu."""
    catalogue = services.load_catalogue()
    if not catalogue:
        raise NotBuilt(
            "no replays available; run `python -m dfsi.memory.pipeline` to build the Memory layer"
        )
    return {
        "replays": [entry.to_dict() for entry in catalogue],
        "scenario_types": sorted({entry.scenario_type for entry in catalogue}),
    }


@router.get("/model-card", response_model=ModelCardResponse)
def model_card(services: Services) -> ModelCardResponse:
    """What the model is, how it scored, and what it cannot do."""
    path = services.paths.models / "training_report.json"
    if not path.exists():
        raise NotBuilt("no training report; run `python -m dfsi.forecaster.train`")
    report = json.loads(path.read_text())
    forecaster = services.forecaster
    return ModelCardResponse(
        model_version=forecaster.model_version,
        feature_version=report.get("feature_version", FEATURE_VERSION),
        horizon_seconds=report.get("horizon_seconds", HORIZON_SECONDS),
        decision_threshold=report.get("decision_threshold", forecaster.decision_threshold),
        features=report.get("features", []),
        headline_metrics=report.get("dfsi", {}),
        baseline_metrics=report.get("fixed_threshold_baseline", {}),
        per_scenario_type=report.get("per_scenario_type", {}),
        feature_importance=report.get("feature_importance", {}),
        evidence_gate=_gate_summary(report.get("evidence_gate")),
        calibration=report.get("calibration") or _derived_calibration(report),
        headline_metrics_without_gate=report.get("dfsi_without_gate"),
        known_limitations=KNOWN_LIMITATIONS,
    )


def _derived_calibration(report: dict) -> dict | None:
    """For reports written before training recorded it: re-derive from the sweep."""
    sweep = report.get("threshold_sweep")
    budget = (report.get("config") or {}).get("false_alarm_budget")
    threshold = report.get("decision_threshold")
    if not sweep or budget is None or threshold is None:
        return None
    return calibration_summary(sweep, budget, threshold)


def _gate_summary(gate: dict | None) -> dict | None:
    """The gate's setting and effect, without the bulky per-type comparison."""
    if gate is None:
        return None
    return {k: v for k, v in gate.items() if k != "per_scenario_type_without_gate"}
