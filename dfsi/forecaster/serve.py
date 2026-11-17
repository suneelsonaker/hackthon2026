"""Forecaster — live scoring for the replay path.

Loads the trained model once and scores incoming 10-second windows. Two
guarantees matter here:

1. **Fail safe.** If the model is missing, stale, or raises, the scorer returns
   a `MONITORING_ONLY` forecast — never a low risk score. A broken lookout who
   reports "all clear" is worse than no lookout.
2. **Provenance.** Every forecast carries the model version, feature version
   and the exact feature values it was computed from, so the agents can quote
   evidence and the audit log can reconstruct the decision.
"""

from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import (
    FEATURE_VERSION,
    HORIZON_SECONDS,
    PATHS,
    STATE_ABOUT_TO_START,
    STATE_MONITORING_ONLY,
)
from ..memory.features import FEATURE_COLUMNS
from .counterfactual import find_counterfactual
from .gate import EVIDENCE_FEATURE, has_traffic_evidence

log = logging.getLogger("dfsi.forecaster.serve")

#: Features worth surfacing as human-readable evidence, with a short label.
EVIDENCE_LABELS: dict[str, str] = {
    "growth_pps_60s": "packet-rate growth over 60s",
    "growth_mbps_60s": "bandwidth growth over 60s",
    "growth_pps_10s": "packet-rate growth over 10s",
    "baseline_deviation_pps": "deviation from this hour's normal packet rate",
    "baseline_deviation_mbps": "deviation from this hour's normal bandwidth",
    "baseline_deviation_pps_max_hour": "strongest per-vector deviation on this range",
    "country_hhi": "source-country concentration",
    "country_hhi_delta_60s": "shift in source-country concentration over 60s",
    "account_attack_history": "attack episodes on this account in the past 6 hours",
    "blocked_share": "share of traffic currently blocked",
    "blocked_share_delta_60s": "change in blocked share over 60s",
    "anomaly_persistence": "consecutive abnormal windows",
    "policy_utilisation": "packet rate as a fraction of the configured policy",
    "policy_utilisation_mbps": "bandwidth as a fraction of the configured policy",
}


@dataclass(frozen=True)
class RiskBands:
    """Where each band starts, anchored on the model's decision threshold.

    The threshold is calibrated per model, so a fixed 0.35 / 0.60 / 0.80 scale
    contradicted it: with a threshold of 0.65 a score of 0.64 read "warn" while
    the state stayed `QUIET`, and a model calibrated at 0.40 would have been
    raising warnings from the "watch" band.

    `warn` begins exactly at the threshold, so a window is in the warn band or
    above if and only if it counts as above threshold. `watch` begins at half
    the threshold, and `critical` halfway between the threshold and certainty.
    """

    watch: float
    warn: float
    critical: float

    @classmethod
    def for_threshold(cls, threshold: float) -> RiskBands:
        # `not (0 < x < 1)` also rejects NaN, which every comparison misses.
        if not 0.0 < threshold < 1.0:
            raise ValueError(
                f"decision threshold must lie strictly between 0 and 1, got {threshold}"
            )
        return cls(
            watch=threshold / 2.0,
            warn=threshold,
            critical=threshold + (1.0 - threshold) / 2.0,
        )

    def band(self, score: float) -> str:
        if score >= self.critical:
            return "critical"
        if score >= self.warn:
            return "warn"
        if score >= self.watch:
            return "watch"
        return "calm"


@dataclass(frozen=True)
class Forecast:
    """One risk forecast for one range at one instant."""

    window_start_epoch: int
    range_id: int
    account_id: int
    risk_score: float | None
    risk_band: str
    predicted_state: str
    horizon_seconds: int
    model_version: str
    feature_version: str
    evidence: list[dict] = field(default_factory=list)
    #: For a window at or above threshold: how far the top drivers would have to
    #: come back for the alert to stop (see `counterfactual`). `None` otherwise.
    counterfactual: dict | None = None
    degraded: bool = False
    degraded_reason: str | None = None
    #: The risk score is above the decision threshold but there is no traffic
    #: evidence to back an alert (see `gate`), so none is raised. The score is
    #: still reported, honestly, and `gate_reason` says why nothing fired.
    gated: bool = False
    gate_reason: str | None = None
    #: The decision threshold this forecast's band and state were judged
    #: against: the range's own, if one is set, otherwise the model's.
    decision_threshold: float | None = None

    def to_dict(self) -> dict:
        return {
            "window_start_epoch": self.window_start_epoch,
            "range_id": self.range_id,
            "account_id": self.account_id,
            "risk_score": self.risk_score,
            "risk_band": self.risk_band,
            "predicted_state": self.predicted_state,
            "horizon_seconds": self.horizon_seconds,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "evidence": self.evidence,
            "counterfactual": self.counterfactual,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
            "gated": self.gated,
            "gate_reason": self.gate_reason,
            "decision_threshold": self.decision_threshold,
        }


class ForecasterService:
    """Loads the pickled forecaster and scores windows.

    Deliberately *not* thread-safe for mutation: the model is loaded once at
    construction and only read afterwards.
    """

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or (PATHS.models / "forecaster.pkl")
        self._model: Any = None  # a fitted sklearn estimator once loaded
        self._features: list[str] = list(FEATURE_COLUMNS)
        self._importance: dict[str, float] = {}
        #: Typical quiet-window value of each feature; the yardstick for
        #: per-alert attribution. Empty for models saved before it existed.
        self._reference: dict[str, float] = {}
        self.model_version = "unavailable"
        self.feature_version = FEATURE_VERSION
        self.decision_threshold = 0.55
        self.bands = RiskBands.for_threshold(self.decision_threshold)
        self.min_consecutive = 2
        #: Strongest per-vector deviation an alert needs, or `None` for no gate.
        #: Models saved before the gate existed have none, and behave as before.
        self.evidence_gate_deviation: float | None = None
        self.load_error: str | None = None
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if not self.model_path.exists():
            self.load_error = (
                f"no trained model at {self.model_path}; run `python -m dfsi.forecaster.train`"
            )
            log.warning(self.load_error)
            return
        try:
            with self.model_path.open("rb") as handle:
                payload = pickle.load(handle)
            self._model = payload["model"]
            self._features = list(payload.get("features") or FEATURE_COLUMNS)
            self._importance = payload.get("feature_importance") or {}
            self._reference = {
                name: float(value)
                for name, value in (payload.get("feature_reference") or {}).items()
            }
            self.feature_version = payload.get("feature_version", FEATURE_VERSION)
            self.decision_threshold = float(payload.get("decision_threshold", 0.55))
            # A threshold the bands cannot be built from means a corrupt model:
            # refuse to score (degraded) rather than label risk on a broken scale.
            self.bands = RiskBands.for_threshold(self.decision_threshold)
            self.min_consecutive = int(payload.get("min_consecutive", 2))
            gate = payload.get("evidence_gate_deviation")
            if gate is not None:
                gate = float(gate)
                if not math.isfinite(gate) or gate < 0:
                    raise ValueError(
                        f"evidence gate must be a finite non-negative number, got {gate}"
                    )
            self.evidence_gate_deviation = gate
            version = payload.get("model_version") or ""
            run_id = payload.get("mlflow_run_id") or ""
            self.model_version = (
                f"{self.feature_version}/v{version}"
                if version
                else f"{self.feature_version}/{run_id[:8] or 'local'}"
            )
            if self.feature_version != FEATURE_VERSION:
                self.load_error = (
                    f"model was trained on {self.feature_version} but this build "
                    f"computes {FEATURE_VERSION}; refusing to score"
                )
                self._model = None
                log.error(self.load_error)
        except Exception as exc:
            self.load_error = f"could not load forecaster: {exc}"
            self._model = None
            log.exception("forecaster load failed")

    @property
    def ready(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------ #
    def score_window(self, windows: list[dict], threshold: float | None = None) -> Forecast:
        """Score one instant for one range.

        `windows` holds the per-traffic-type rows for a single
        (range_id, window_start_epoch). The range-level risk is the maximum
        across its traffic types: one vector is enough to ring the bell.

        `threshold` overrides the model's decision threshold for this range.
        The score itself never depends on it, only the band and the state the
        score is read as. `None` uses the model's calibrated threshold.
        """
        if not windows:
            raise ValueError("score_window requires at least one traffic-type row")

        head = windows[0]
        epoch = int(head["window_start_epoch"])
        range_id = int(head["range_id"])
        account_id = int(head.get("account_id", -1))

        if threshold is None:
            threshold, bands = self.decision_threshold, self.bands
        else:
            threshold = float(threshold)
            bands = RiskBands.for_threshold(threshold)

        if not self.ready:
            return self._degraded(
                epoch, range_id, account_id, self.load_error or "model not loaded", threshold
            )

        try:
            frame = pd.DataFrame(windows)
            missing = [c for c in self._features if c not in frame.columns]
            if missing:
                return self._degraded(
                    epoch, range_id, account_id, f"missing features {missing}", threshold
                )
            scores = self._model.predict_proba(frame[self._features])[:, 1]
            best = int(scores.argmax())
            score = float(scores[best])
        except Exception as exc:
            log.exception("scoring failed for range %s", range_id)
            return self._degraded(epoch, range_id, account_id, f"scoring error: {exc}", threshold)

        # A score above threshold raises an alert only with traffic evidence behind
        # it. The score itself is untouched; a gated window reads as `watch`, so
        # "warn or above" still means exactly "an alert is raised". "Above" is
        # against the range's threshold, the same one its bands are built from.
        above = score >= threshold
        deviation = self._deviation(windows)
        gated = above and not has_traffic_evidence(deviation, self.evidence_gate_deviation)
        band = "watch" if gated else bands.band(score)
        evidence = self.build_evidence(windows[best], score)
        return Forecast(
            window_start_epoch=epoch,
            range_id=range_id,
            account_id=account_id,
            risk_score=score,
            risk_band=band,
            predicted_state=STATE_ABOUT_TO_START if above and not gated else "QUIET",
            horizon_seconds=HORIZON_SECONDS,
            model_version=self.model_version,
            feature_version=self.feature_version,
            evidence=evidence,
            # Only a window that actually raises an alert has anything to calm:
            # a gated one (score above threshold, no traffic evidence) has no
            # alert to explain away, and the quiet majority skips the extra
            # model calls. "Calm" means below the range's threshold.
            counterfactual=(
                self._counterfactual(windows[best], score, evidence, threshold)
                if above and not gated
                else None
            ),
            gated=gated,
            gate_reason=self._gate_reason(score, deviation, threshold) if gated else None,
            decision_threshold=threshold,
        )

    @staticmethod
    def _deviation(windows: list[dict]) -> float | None:
        """The strongest per-vector deviation at this instant, or `None` if unknown."""
        values = []
        for row in windows:
            value = row.get(EVIDENCE_FEATURE)
            if value is not None and math.isfinite(float(value)):
                values.append(float(value))
        return max(values) if values else None

    def _gate_reason(self, score: float, deviation: float | None, threshold: float) -> str:
        return (
            f"The risk score ({score:.2f}) is above the {threshold:.2f} "
            f"threshold, but traffic is within its normal range for this hour "
            f"(strongest deviation {deviation:.1f}; an alert needs more than "
            f"{self.evidence_gate_deviation:.1f}). The score is being driven by context "
            "such as recent attack history, so no alert is raised. This is not an all-clear."
        )

    def _degraded(
        self,
        epoch: int,
        range_id: int,
        account_id: int,
        reason: str,
        threshold: float | None = None,
    ) -> Forecast:
        """Fail safe: monitoring-only, no risk number, never 'low risk'."""
        return Forecast(
            window_start_epoch=epoch,
            range_id=range_id,
            account_id=account_id,
            risk_score=None,
            risk_band="unknown",
            predicted_state=STATE_MONITORING_ONLY,
            horizon_seconds=HORIZON_SECONDS,
            model_version=self.model_version,
            feature_version=self.feature_version,
            evidence=[],
            degraded=True,
            degraded_reason=reason,
            decision_threshold=threshold if threshold is not None else self.decision_threshold,
        )

    def _counterfactual(
        self, row: dict, score: float, evidence: list[dict], threshold: float | None = None
    ) -> dict | None:
        """How far the drivers of this alert would have to come back for it to stop.

        "Stop" means falling below `threshold`: the range's own decision
        threshold, or the model's when `None`.

        The drivers are the evidence items pushing the score up, strongest first.
        Needs the quiet reference, so models saved without one return `None`.
        Explanation is not safety: any failure is logged and the forecast is
        returned without it.
        """
        if not self._reference:
            return None
        drivers = [
            item["feature"]
            for item in sorted(
                (i for i in evidence if i.get("contribution", 0.0) > 0.0),
                key=lambda i: i["contribution"],
                reverse=True,
            )
        ]
        try:
            values = [np.nan if row[col] is None else float(row[col]) for col in self._features]
            return find_counterfactual(
                lambda batch: self._model.predict_proba(
                    pd.DataFrame(batch, columns=self._features)
                )[:, 1],
                self._features,
                values,
                self._reference,
                drivers,
                EVIDENCE_LABELS,
                self.decision_threshold if threshold is None else threshold,
                score,
            )
        except Exception:
            log.warning("could not build a counterfactual", exc_info=True)
            return None

    # ------------------------------------------------------------------ #
    def build_evidence(self, row: dict, score: float, top_n: int = 6) -> list[dict]:
        """Rank the features behind *this* score, strongest driver first.

        These are the *only* numbers the agents may cite. The agents receive
        this list through a read-only tool and must quote it verbatim.

        Each item carries `contribution`: how much this score falls when that
        one feature is put back to its typical quiet value. Positive means the
        feature is pushing risk up, negative that it is holding it down (useful
        as contrary evidence). The list is ranked by the size of that, so it
        changes with the alert. Ranking by global importance, as this once did,
        showed every alert the same handful of features whatever had driven it.

        Occlusion measures each feature on its own, so two correlated features
        (growth over 10s and over 60s) split the credit unevenly. Read it as
        "what moves this score", not as an additive decomposition.

        Models saved before the quiet reference existed have no `contribution`
        and keep the previous global-importance ordering.
        """
        items: list[dict] = []
        for name, label in EVIDENCE_LABELS.items():
            if name not in row:
                continue
            value = row[name]
            if value is None or not math.isfinite(float(value)):
                continue
            items.append(
                {
                    "feature": name,
                    "label": label,
                    "value": round(float(value), 4),
                    "importance": round(float(self._importance.get(name, 0.0)), 6),
                    "traffic_type": row.get("traffic_type"),
                }
            )

        contributions = self._contributions(row, score, [item["feature"] for item in items])
        if contributions is None:
            items.sort(key=lambda item: (item["importance"], abs(item["value"])), reverse=True)
        else:
            for item in items:
                item["contribution"] = round(contributions.get(item["feature"], 0.0), 4)
            items.sort(
                key=lambda item: (
                    abs(item["contribution"]),
                    item["importance"],
                    abs(item["value"]),
                ),
                reverse=True,
            )
        return items[:top_n]

    def _contributions(self, row: dict, score: float, names: list[str]) -> dict[str, float] | None:
        """Score drop when each feature alone is reset to its quiet value.

        One batched model call for all features. Explanation is not safety, so
        any failure here is logged and the caller falls back to the global
        ordering instead of losing the forecast.
        """
        names = [name for name in names if name in self._reference and name in self._features]
        if not names:
            return None
        try:
            base = np.array(
                [np.nan if row[col] is None else float(row[col]) for col in self._features],
                dtype=float,
            )
            batch = np.tile(base, (len(names) + 1, 1))
            for offset, name in enumerate(names, start=1):
                batch[offset, self._features.index(name)] = self._reference[name]
            occluded = self._model.predict_proba(pd.DataFrame(batch, columns=self._features))[:, 1]
        except Exception:
            log.warning("could not attribute the score; using global importance", exc_info=True)
            return None
        return {name: float(score - occluded[offset]) for offset, name in enumerate(names, start=1)}
