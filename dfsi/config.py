"""DFSI configuration — thresholds, paths, versions.

DFSI (DDoS Future-State Intelligence) is a read-only, simulation-only predictive
lookout layer that sits *beside* netsec-ddos-shared-state-2. It never mutates
customer configuration, never blocks traffic and never executes a policy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Versions. Bump these when the contract changes; every prediction records
# both so a stored forecast can always be traced back to how it was produced.
# --------------------------------------------------------------------------- #
FEATURE_VERSION = "features-v3"
SCENARIO_SET_VERSION = "synthetic-v2"

# --------------------------------------------------------------------------- #
# Leakage discipline in time. An episode's hour-of-week baseline is built only
# from the same hour in earlier weeks, so the first week is warm-up and never
# reaches the feature table. Evaluation is split by whole weeks counted back
# from the latest one: the model is always tested on a week it has never seen,
# after every week it was fitted or calibrated on.
# --------------------------------------------------------------------------- #
BASELINE_LOOKBACK_WEEKS = 4
TEST_WEEKS = 1
CALIBRATION_WEEKS = 1

# --------------------------------------------------------------------------- #
# Time geometry. ANL summarises blocked traffic in 10-second windows; DFSI
# forecasts whether blocking will *begin* inside the next 120 seconds.
# --------------------------------------------------------------------------- #
WINDOW_SECONDS = 10
HORIZON_SECONDS = 120
HORIZON_WINDOWS = HORIZON_SECONDS // WINDOW_SECONDS  # 12

# Number of consecutive abnormal windows kept for `anomaly_persistence`.
PERSISTENCE_LOOKBACK_WINDOWS = 18  # 180s
# |baseline_deviation| above this counts the window as abnormal.
ANOMALY_DEVIATION_THRESHOLD = 3.0


@dataclass(frozen=True)
class SharedStateThresholds:
    """Mirror of the Shared State ("the bell") lifecycle configuration.

    These values are *read* by DFSI so that the forecast horizon and the label
    definition line up with what the Java service will actually declare. DFSI
    does not own them and must never write them.
    """

    environment: str
    start_duration_seconds: int
    stop_duration_seconds: int
    start_threshold_ratio: float
    stop_threshold_ratio: float
    start_floor_kpps: float
    start_floor_mbps: float

    @property
    def start_duration_windows(self) -> int:
        return max(1, self.start_duration_seconds // WINDOW_SECONDS)


PROD_THRESHOLDS = SharedStateThresholds(
    environment="prod",
    start_duration_seconds=300,
    stop_duration_seconds=10_800,
    start_threshold_ratio=0.30,
    stop_threshold_ratio=0.10,
    start_floor_kpps=15.0,
    start_floor_mbps=50.0,
)

STAGE_THRESHOLDS = SharedStateThresholds(
    environment="stage",
    start_duration_seconds=30,
    stop_duration_seconds=300,
    start_threshold_ratio=0.30,
    stop_threshold_ratio=0.10,
    start_floor_kpps=5.0,
    start_floor_mbps=20.0,
)

THRESHOLDS_BY_ENV = {
    PROD_THRESHOLDS.environment: PROD_THRESHOLDS,
    STAGE_THRESHOLDS.environment: STAGE_THRESHOLDS,
}


def thresholds_for(environment: str | None = None) -> SharedStateThresholds:
    """Return the Shared State thresholds for `environment` (default: stage).

    Stage is the default because the hackathon demo replays minutes, not hours,
    and the 300s prod start duration would not fit inside a demo window.
    """
    env = (environment or os.getenv("DFSI_ENV") or "stage").lower()
    if env not in THRESHOLDS_BY_ENV:
        raise ValueError(
            f"unknown DFSI environment {env!r}; expected one of {sorted(THRESHOLDS_BY_ENV)}"
        )
    return THRESHOLDS_BY_ENV[env]


# --------------------------------------------------------------------------- #
# Range-state vocabulary. The first three mirror NetsecDDOSSharedStateProto's
# RangeState; ABOUT_TO_START is the new state DFSI contributes.
# --------------------------------------------------------------------------- #
STATE_ABOUT_TO_START = "ABOUT_TO_START"
STATE_STARTED = "RANGE_STATE_DDOS_STARTED"
STATE_IN_PROGRESS = "RANGE_STATE_DDOS_IN_PROGRESS"
STATE_ENDED = "RANGE_STATE_DDOS_ENDED"
STATE_UNKNOWN = "RANGE_STATE_UNKNOWN"
# Returned whenever any part of the prediction path fails. Never "low risk".
STATE_MONITORING_ONLY = "MONITORING_ONLY"


# --------------------------------------------------------------------------- #
# Paths. Everything DFSI writes lives under DFSI_HOME (default: this package's
# parent directory), so nothing lands inside the Java service's source tree.
# --------------------------------------------------------------------------- #
def _home() -> Path:
    return Path(os.getenv("DFSI_HOME", Path(__file__).resolve().parent.parent))


@dataclass(frozen=True)
class Paths:
    home: Path

    @property
    def data(self) -> Path:
        return self.home / "data"

    @property
    def raw_windows(self) -> Path:
        return self.data / "raw_windows"

    @property
    def baselines(self) -> Path:
        return self.data / "baselines"

    @property
    def features(self) -> Path:
        return self.data / "features"

    @property
    def replay(self) -> Path:
        """Small per-scenario JSON replays consumed by the API/UI."""
        return self.data / "replay"

    @property
    def models(self) -> Path:
        return self.home / "models"

    @property
    def policy_db(self) -> Path:
        """Seed for the simulated policy store (per range and traffic type)."""
        return Path(os.getenv("DFSI_POLICY_DB", self.home / "raw.json"))

    @property
    def mlruns(self) -> Path:
        return self.home / "mlruns"

    @property
    def decisions(self) -> Path:
        """Append-only audit log of simulated human decisions."""
        return self.data / "decisions"


PATHS = Paths(home=_home())

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", PATHS.mlruns.as_uri())
MLFLOW_EXPERIMENT = os.getenv("DFSI_MLFLOW_EXPERIMENT", "dfsi-forecaster")
REGISTERED_MODEL_NAME = "dfsi-onset-forecaster"

# Risk-score bands (calm / watch / warn / critical) are deliberately *not*
# constants here. The decision threshold is relearned on every retrain, so a
# fixed scale drifts out of step with it and the band shown can contradict the
# state raised. They are derived from each model's own threshold; see
# `forecaster.serve.RiskBands`. The agents are explicitly forbidden from
# producing these numbers — ML owns them.
