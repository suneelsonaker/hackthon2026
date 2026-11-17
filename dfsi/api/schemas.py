"""Request and response models for the DFSI API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ReplayRequest(BaseModel):
    """Trigger a scenario replay."""

    replay_id: str = Field(..., min_length=1, max_length=120)
    #: Wall-clock seconds between streamed windows. 0.25 keeps a 60-minute
    #: scenario inside a demo slot; 0 streams as fast as the client reads.
    speed_seconds: float = Field(default=0.25, ge=0.0, le=5.0)
    #: Stop automatically once the bell would have rung, so the demo lands on
    #: the comparison rather than running on into the recovery.
    stop_after_bell_windows: int = Field(default=18, ge=0, le=120)


class DrillRequest(BaseModel):
    """Run the rehearsal room on a replay, deciding at a given instant."""

    replay_id: str = Field(..., min_length=1, max_length=120)
    decision_epoch: int | None = Field(
        default=None,
        description=(
            "Instant the operator is deciding at. Defaults to the first window DFSI warned about."
        ),
    )
    environment: Literal["stage", "prod"] | None = None
    #: Replay every option under plausible versions of its stated assumptions
    #: and report ranges and win shares. Costs a fraction of a second, so it is
    #: opt-in rather than slowing every drill.
    include_uncertainty: bool = False
    uncertainty_draws: int = Field(default=200, ge=1, le=2_000)
    #: Price holding off: what deciding later would cost and save per option.
    include_waiting: bool = False
    waiting_seconds: list[int] = Field(default_factory=lambda: [30, 60, 90], max_length=6)


class ExplainRequest(BaseModel):
    """Ask the two Explainer agents to brief the operator."""

    replay_id: str = Field(..., min_length=1, max_length=120)
    decision_epoch: int | None = None
    include_simulation: bool = True


PolicyKey = Literal["hold", "tighten_vector", "rate_limit_sources", "full_scrub"]


class DecisionRequest(BaseModel):
    """Record a human's decision. An approved threshold change is applied to
    DFSI's simulated policy store; nothing real is ever changed.

    Verdicts: `approved` takes the Advisor's recommendation; `changed` overrides
    it with a different option (`recommended_policy_key` says which one was
    overridden); `rejected` declines `policy_key` and changes nothing.
    """

    replay_id: str = Field(..., min_length=1, max_length=120)
    decision_epoch: int
    policy_key: PolicyKey
    verdict: Literal["approved", "rejected", "changed"]
    operator: str = Field(default="demo-operator", min_length=1, max_length=64)
    note: str = Field(default="", max_length=500)
    #: The model version the dashboard was showing. The audit record states the
    #: server's own version; this is kept only to flag a stale dashboard.
    model_version: str = Field(default="unknown", max_length=120)
    #: The Advisor's pick, when there was one. Required context for `changed`.
    recommended_policy_key: PolicyKey | None = None

    @model_validator(mode="after")
    def _verdict_matches_the_recommendation(self) -> DecisionRequest:
        """The verdict must agree with what was picked, or the audit record lies."""
        picked, recommended = self.policy_key, self.recommended_policy_key
        if self.verdict == "changed":
            if recommended is None:
                raise ValueError("verdict 'changed' needs the recommended_policy_key it overrides")
            if picked == recommended:
                raise ValueError(
                    f"verdict 'changed' overrides the recommendation, but {picked!r} is it"
                )
        if self.verdict == "approved" and recommended is not None and picked != recommended:
            raise ValueError(
                f"verdict 'approved' takes the recommendation {recommended!r}, "
                f"but policy_key is {picked!r}; use 'changed'"
            )
        return self


class DecisionRecord(BaseModel):
    recorded_at_epoch: int
    replay_id: str
    decision_epoch: int
    policy_key: str
    policy_name: str
    verdict: str
    operator: str
    note: str
    #: The model the server was running when the decision was recorded.
    model_version: str
    feature_version: str
    #: What the dashboard said it was showing, when that differs from the
    #: server: the operator decided against a stale view. None when they agree.
    client_model_version: str | None = None
    #: The Advisor's pick this decision approved, or overrode for `changed`.
    recommended_policy_key: str | None = None
    #: For a threshold change: the concrete per-type thresholds the human
    #: signed off on, and the past-week backtest behind them.
    policy_change: dict[str, Any] | None = None
    #: True when an approved threshold change was written to DFSI's simulated
    #: policy store. Never means a real policy or customer config changed.
    applied: bool = False
    policy_version_before: int | None = None
    policy_version_after: int | None = None
    note_to_auditor: str = "Decision recorded for audit. Nothing was applied."


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    forecaster_ready: bool
    forecaster_error: str | None
    model_version: str
    feature_version: str
    horizon_seconds: int
    agent_engine: str
    replays_available: int
    environment: str


class ModelCardResponse(BaseModel):
    """What the model is, how it was measured, and where it falls short."""

    model_version: str
    feature_version: str
    horizon_seconds: int
    decision_threshold: float
    features: list[str]
    headline_metrics: dict[str, Any]
    baseline_metrics: dict[str, Any]
    per_scenario_type: dict[str, Any]
    feature_importance: dict[str, float]
    #: What an alert additionally requires (traffic evidence), the same model's
    #: figures with the requirement off, and how many windows it withheld.
    #: `None` for a model trained before the gate existed.
    evidence_gate: dict[str, Any] | None = None
    headline_metrics_without_gate: dict[str, Any] | None = None
    #: Whether `decision_threshold` met the false-alarm budget on calibration
    #: (`source: "calibrated"`) or is the configured fallback (`"fallback"`).
    calibration: dict[str, Any] | None = None
    known_limitations: list[str]
