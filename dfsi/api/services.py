"""The service layer: everything the API does, with no HTTP in it.

`DfsiServices` owns the long-lived collaborators (forecaster, explainers,
policy store, per-range thresholds, audit log) and the use cases the routes
expose. Routes receive one through FastAPI's dependency injection, so a test
builds a `DfsiServices` with fakes and hands it to `create_app` instead of
patching module globals.

Errors are raised as `dfsi.api.errors` types; `app.py` maps them to status
codes in one place.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..agents.orchestrator import ExplainerOrchestrator
from ..agents.tools import EvidenceBundle
from ..config import PATHS, STATE_ABOUT_TO_START, Paths
from ..forecaster.alerting import AlertTracker
from ..forecaster.range_thresholds import load_range_thresholds
from ..forecaster.serve import ForecasterService
from ..rehearsal.policy_advisor import PolicyPlan, suggest_policy
from ..rehearsal.policy_store import PolicyStore, apply_policy
from ..rehearsal.simulator import CANDIDATES_BY_KEY, run_drill
from ..rehearsal.uncertainty import assumption_uncertainty
from ..rehearsal.waiting import cost_of_waiting
from . import replay as replay_engine
from .audit import AuditLog
from .errors import InvalidEpoch, InvalidRequest, UnknownReplay
from .schemas import DecisionRecord, DecisionRequest, DrillRequest, ExplainRequest

log = logging.getLogger("dfsi.api")

#: (range_id, episode_start_epoch, current_rows, environment) -> PolicyPlan
PolicyPlanner = Callable[[int, int, list[dict], str | None], PolicyPlan]

#: Verdicts under which an approved threshold change is written to the store.
_APPLYING_VERDICTS = frozenset({"approved", "changed"})


@dataclass
class DfsiServices:
    """The API's collaborators and use cases. One per running app."""

    forecaster: ForecasterService
    orchestrator: ExplainerOrchestrator
    policy_store: PolicyStore
    audit_log: AuditLog
    #: Per-range `risk_threshold` from raw.json; ranges not listed follow the model.
    range_thresholds: dict[int, float] = field(default_factory=dict)
    load_stored_replay: Callable[[str], dict] = replay_engine.load_replay
    load_catalogue: Callable[[], list[replay_engine.ReplayCatalogueEntry]] = (
        replay_engine.load_catalogue
    )
    plan_policy: PolicyPlanner = suggest_policy
    clock: Callable[[], float] = time.time
    #: Where generated artefacts (training report) are read from.
    paths: Paths = PATHS
    #: One decision at a time: reading the current version, planning and
    #: writing must be one step, or two concurrent approvals each record a
    #: "before" the other has already moved.
    _decision_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------------ #
    # Shared lookups
    # ------------------------------------------------------------------ #
    def threshold_for(self, range_id: int) -> float:
        """The decision threshold a range is read against."""
        return self.range_thresholds.get(int(range_id), self.forecaster.decision_threshold)

    def load_replay(self, replay_id: str) -> dict:
        """A stored replay, re-derived under the policy currently in the store.

        Every use case goes through this, so the stream, the drill, the agents
        and the decision all see the same policy.
        """
        try:
            stored = self.load_stored_replay(replay_id)
        except KeyError:
            raise UnknownReplay(replay_id) from None
        range_id = int(stored["range_id"])
        policy = self.policy_store.range_policy(range_id)
        replay = dict(stored)
        replay["windows"] = apply_policy(stored["windows"], policy)
        replay["policy_version"] = self.policy_store.range_version(range_id)
        replay["policy"] = (
            {
                t: {"max_threshold_pps": p.pps, "max_threshold_mbps": round(p.mbps, 3)}
                for t, p in sorted(policy.items())
            }
            if policy
            else None
        )
        return replay

    def policy_plan(
        self, replay: dict, decision_epoch: int, environment: str | None = None
    ) -> PolicyPlan:
        """Per-type threshold suggestion from the week of traffic before this episode."""
        windows = replay["windows"]
        episode_start = min(int(row["window_start_epoch"]) for row in windows)
        current = [row for row in windows if int(row["window_start_epoch"]) == decision_epoch]
        try:
            return self.plan_policy(int(replay["range_id"]), episode_start, current, environment)
        except Exception as exc:
            log.exception("policy suggestion failed for %s", replay.get("replay_id"))
            return PolicyPlan(
                available=False,
                range_id=int(replay["range_id"]),
                # The detail is in the log; responses and audit records carry
                # only the kind of failure.
                reason=f"history could not be read ({type(exc).__name__})",
            )

    def resolve_decision_epoch(
        self, replay_id: str, decision_epoch: int | None
    ) -> tuple[dict, int]:
        """Load a replay and resolve the instant the operator is deciding at.

        When unspecified, this is the first window DFSI warned about — the
        moment the operator would actually have been looking at the dashboard.
        Falling back to the last window keeps quiet scenarios usable.
        """
        replay = self.load_replay(replay_id)
        buckets: dict[int, list[dict]] = {}
        for row in replay["windows"]:
            buckets.setdefault(int(row["window_start_epoch"]), []).append(row)
        epochs = sorted(buckets)

        if decision_epoch is not None:
            if decision_epoch not in buckets:
                raise InvalidEpoch(f"epoch {decision_epoch} is not a window in this replay")
            return replay, decision_epoch

        # The same debounce the stream applies, so "the first window DFSI warned
        # about" is the window the dashboard actually raised the warning at.
        tracker = AlertTracker(self.forecaster.min_consecutive)
        threshold = self.threshold_for(int(replay["range_id"]))
        for epoch in epochs:
            forecast = self.forecaster.score_window(buckets[epoch], threshold)
            if tracker.update(forecast.predicted_state == STATE_ABOUT_TO_START):
                return replay, epoch
        return replay, epochs[-1]

    # ------------------------------------------------------------------ #
    # Rehearsal room
    # ------------------------------------------------------------------ #
    def drill(self, request: DrillRequest) -> dict[str, Any]:
        """Run the four candidate responses, plus the opt-in uncertainty and waiting."""
        replay, decision_epoch = self.resolve_decision_epoch(
            request.replay_id, request.decision_epoch
        )
        windows = replay["windows"]
        plan = self.policy_plan(replay, decision_epoch, request.environment)
        result = run_drill(windows, decision_epoch, request.environment, plan)
        result["replay_id"] = request.replay_id
        result["scenario_type"] = replay["scenario_type"]
        if request.include_uncertainty:
            result["uncertainty"] = assumption_uncertainty(
                windows, decision_epoch, request.environment, plan, draws=request.uncertainty_draws
            )
        if request.include_waiting:
            try:
                result["cost_of_waiting"] = cost_of_waiting(
                    windows, decision_epoch, request.waiting_seconds, request.environment, plan
                )
            except ValueError as exc:  # a wait that is not whole windows is the caller's mistake
                raise InvalidRequest(str(exc)) from exc
        return result

    # ------------------------------------------------------------------ #
    # Explainers
    # ------------------------------------------------------------------ #
    def explain(self, request: ExplainRequest) -> dict[str, Any]:
        replay, decision_epoch = self.resolve_decision_epoch(
            request.replay_id, request.decision_epoch
        )
        bundle = self.build_bundle(replay, decision_epoch, request.include_simulation)
        briefing = self.orchestrator.explain(bundle)
        briefing["replay_id"] = request.replay_id
        briefing["scenario_type"] = replay["scenario_type"]
        briefing["decision_epoch"] = decision_epoch
        return briefing

    def build_bundle(
        self, replay: dict, decision_epoch: int, include_simulation: bool
    ) -> EvidenceBundle:
        """Assemble exactly what the agents are allowed to see.

        Only windows at or before `decision_epoch` are included: an agent that
        can see the future would explain the incident beautifully and uselessly.
        """
        rows = [r for r in replay["windows"] if int(r["window_start_epoch"]) <= decision_epoch]
        if not rows:
            raise InvalidEpoch(f"no windows at or before epoch {decision_epoch}")
        current = [r for r in rows if int(r["window_start_epoch"]) == decision_epoch]
        if not current:
            raise InvalidEpoch(f"epoch {decision_epoch} is not a window boundary")

        forecast = self.forecaster.score_window(
            current, self.threshold_for(int(replay["range_id"]))
        )
        lookback = set(
            sorted({int(r["window_start_epoch"]) for r in rows})[-replay_engine.EVIDENCE_LOOKBACK :]
        )
        recent = [r for r in rows if int(r["window_start_epoch"]) in lookback]
        simulation = (
            run_drill(
                replay["windows"], decision_epoch, None, self.policy_plan(replay, decision_epoch)
            )
            if include_simulation
            else None
        )
        return EvidenceBundle(
            range_id=forecast.range_id,
            account_id=forecast.account_id,
            window_start_epoch=decision_epoch,
            risk_score=forecast.risk_score,
            risk_band=forecast.risk_band,
            model_version=forecast.model_version,
            feature_version=forecast.feature_version,
            horizon_seconds=forecast.horizon_seconds,
            evidence=forecast.evidence,
            recent_windows=recent,
            simulation=simulation,
            tenant_account_id=forecast.account_id,
            # A replay's "now" is the instant the operator is deciding at; any
            # window after it would be evidence from the future.
            as_of_epoch=decision_epoch,
        )

    # ------------------------------------------------------------------ #
    # Human decision (simulated, audited)
    # ------------------------------------------------------------------ #
    def record_decision(self, request: DecisionRequest) -> DecisionRecord:
        """Record a human decision, and apply an approved threshold change.

        Approving "tighten the vector threshold" writes the suggested per-type
        lines into DFSI's simulated policy store, so the next run of any replay
        on that range is played under them. Every other decision is recorded
        only. Nothing reaches Behemoth, Shared State or a customer's
        configuration. Every record is traceable to model and policy version.
        """
        with self._decision_lock:
            return self._record_decision(request)

    def reset_policies(self) -> dict[str, Any]:
        """Restore the seed, serialised with decisions so none straddles a reset."""
        with self._decision_lock:
            self.policy_store.reset()
            return self.policy_store.snapshot()

    def _record_decision(self, request: DecisionRequest) -> DecisionRecord:
        # Same validation as the drill and the agents: the epoch must be a window
        # of this replay, or the audit record would cite an instant that never was.
        replay, _ = self.resolve_decision_epoch(request.replay_id, request.decision_epoch)
        candidate = CANDIDATES_BY_KEY[request.policy_key]
        version_before = replay["policy_version"]
        version_after = version_before
        policy_change: dict[str, Any] | None = None
        applied = False

        if candidate.key == "tighten_vector":
            # Recomputed here rather than trusted from the client, so the audit
            # record states exactly which thresholds the human signed off on.
            plan = self.policy_plan(replay, request.decision_epoch)
            policy_change = _describe_change(plan)
            new_lines = plan.applied_thresholds if plan.available else {}
            if request.verdict in _APPLYING_VERDICTS and new_lines:
                version_after = self.policy_store.apply_change(
                    int(replay["range_id"]),
                    new_lines,
                    provenance={
                        "replay_id": request.replay_id,
                        "decision_epoch": request.decision_epoch,
                        "operator": request.operator,
                        "verdict": request.verdict,
                        "basis": "past-week thresholds",
                    },
                )
                applied = True

        record = DecisionRecord(
            recorded_at_epoch=int(self.clock()),
            replay_id=request.replay_id,
            decision_epoch=request.decision_epoch,
            policy_key=candidate.key,
            policy_name=candidate.name,
            verdict=request.verdict,
            operator=request.operator,
            note=request.note,
            model_version=self.forecaster.model_version,
            feature_version=self.forecaster.feature_version,
            client_model_version=(
                request.model_version
                if request.model_version not in ("unknown", self.forecaster.model_version)
                else None
            ),
            recommended_policy_key=request.recommended_policy_key,
            policy_change=policy_change,
            applied=applied,
            policy_version_before=version_before,
            policy_version_after=version_after,
            note_to_auditor=_note_to_auditor(applied, version_after),
        )
        self.audit_log.append(record.model_dump())
        return record


def _describe_change(plan: PolicyPlan) -> dict[str, Any]:
    """The audit-record view of a threshold plan: basis, summary, per-type lines."""
    return {
        "basis": "past-week thresholds" if plan.available else "stated assumption",
        "change": plan.describe_change() if plan.available else None,
        "reason": plan.reason,
        "per_type": [
            {
                "traffic_type": s.traffic_type,
                "current_policy_pps": s.current_policy_pps,
                "suggested_policy_pps": s.suggested_policy_pps,
                "backtest_legit_lost_pct": s.backtest_legit_lost_pct,
            }
            for s in plan.per_type
            if s.apply
        ],
    }


def _note_to_auditor(applied: bool, version_after: int | None) -> str:
    if applied:
        return (
            f"Applied to DFSI's simulated policy store (version {version_after}). "
            "Replays on this range now run under the new lines. No real policy, "
            "Behemoth rule or customer configuration was changed."
        )
    return (
        "Decision recorded for audit. Only an approved threshold change is "
        "applied, and only to DFSI's simulated policy store; nothing changed."
    )


def build_services(paths: Paths = PATHS, **overrides: Any) -> DfsiServices:
    """The production wiring, with any collaborator replaceable by keyword.

    `build_services(forecaster=fake)` is how tests and alternative deployments
    swap one piece without touching the rest.
    """
    overrides.setdefault("paths", paths)
    overrides.setdefault(
        "load_stored_replay", functools.partial(replay_engine.load_replay, paths=paths)
    )
    overrides.setdefault(
        "load_catalogue", functools.partial(replay_engine.load_catalogue, paths=paths)
    )
    defaults: dict[str, Any] = {
        # The model and the report the model card reads come from the same home.
        "forecaster": lambda: ForecasterService(paths.models / "forecaster.pkl"),
        "orchestrator": lambda: ExplainerOrchestrator(),
        "policy_store": lambda: PolicyStore.from_file(paths.policy_db),
        "audit_log": lambda: AuditLog(paths.decisions / "decisions.jsonl"),
        "range_thresholds": lambda: load_range_thresholds(paths.policy_db),
    }
    built = {
        name: overrides.pop(name) if name in overrides else make()
        for name, make in defaults.items()
    }
    return DfsiServices(**built, **overrides)
