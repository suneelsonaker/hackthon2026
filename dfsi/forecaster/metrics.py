"""Forecaster metrics — the three headline numbers, plus the baseline.

    warning lead time   how many seconds before the bell would ring
    false-alarm rate    share of non-attack scenarios that raised a warning
    attacks caught      share of real onsets warned about *early* (recall)

Every number is computed *per scenario*, never per row: a scenario is the unit
a human experiences, and per-row accuracy on 10-second windows is both
inflated and meaningless here.

Two measurement rules, both of which exist because getting them wrong flatters
DFSI or the incumbent by tens of seconds:

**The clock starts at the bell, not at the threshold crossing.** Shared State
crosses the policy-relative threshold, then waits out the start duration before
it declares. `is_onset` marks the declaration. So a detector scored on the
*crossing* is credited with the start duration as though it were foresight — on
this scenario set that is a free 20 seconds. The incumbent's warning is the
bell, which is why `fixed_threshold_baseline` scores `declared` and its lead is
zero by construction.

**Lead time is only defined where there was a runway.** An episode that opens
already over threshold is *left-censored*: there is no pre-attack history to
forecast from, so a short lead there measures the episode boundary rather than
the model. Those scenarios are counted and reported separately (`in_progress`)
instead of being averaged into the head-start figure — folding them in halves
the median and tells the operator something untrue about what warning they can
expect.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..config import HORIZON_SECONDS, WINDOW_SECONDS
from .alerting import first_alert_index

#: Sanity bound for reporting: a lead time longer than the horizon plus one
#: window cannot come from the model's own forecast horizon. Such warnings are
#: genuine — a long ramp is visible in the trailing features well before the
#: label window — but they are counted separately so the headline figure is
#: never quietly inflated by them.
MAX_CREDIBLE_LEAD_SECONDS = HORIZON_SECONDS + WINDOW_SECONDS


@dataclass(frozen=True)
class ScenarioOutcome:
    scenario_id: str
    scenario_type: str
    has_onset: bool
    onset_epoch: int | None
    first_warning_epoch: int | None
    #: The range was already over the start threshold in this scenario's very
    #: first window, so the episode opens mid-attack. There is no pre-attack
    #: runway to forecast from; see the module docstring on left-censoring.
    in_progress_at_start: bool = False

    @property
    def warned(self) -> bool:
        return self.first_warning_epoch is not None

    @property
    def forecastable(self) -> bool:
        """Is a head start even possible on this scenario?

        An attack already underway when the recording begins cannot be
        forecast — only detected. Lead time is undefined there, and averaging
        it in would understate what warning a real onset gets.
        """
        return self.has_onset and not self.in_progress_at_start

    @property
    def lead_seconds(self) -> float | None:
        """Seconds between the first warning and the bell.

        `None` when there was no onset or no warning. Negative values mean the
        warning came *after* the bell — counted as a miss, not a win.
        """
        if self.onset_epoch is None or self.first_warning_epoch is None:
            return None
        return float(self.onset_epoch - self.first_warning_epoch)


@dataclass(frozen=True)
class EvaluationResult:
    n_scenarios: int
    n_attack_scenarios: int
    #: Attacks with a pre-attack runway. `attacks_caught` and every lead-time
    #: figure are computed over these only.
    n_forecastable_scenarios: int
    #: Attacks already underway at the first window — excluded from lead time.
    n_in_progress_scenarios: int
    attacks_caught: float
    false_alarm_rate: float
    median_lead_seconds: float
    mean_lead_seconds: float
    p10_lead_seconds: float
    late_warning_count: int
    #: Share of already-underway attacks the detector flagged at all. Not a
    #: head start — it is the "did we at least notice" number.
    in_progress_detected: float
    #: Warnings earlier than the model's own 120s horizon. Real, but they come
    #: from a trend the features can see further out, so they are reported
    #: rather than folded silently into the headline claim.
    leads_beyond_horizon: int
    threshold: float

    def as_mlflow_metrics(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_{k}": float(v)
            for k, v in asdict(self).items()
            if isinstance(v, (int, float))
        }


def scenario_outcomes(
    frame: pd.DataFrame,
    score_column: str,
    threshold: float,
    min_consecutive: int = 2,
) -> list[ScenarioOutcome]:
    """Reduce scored windows to one outcome per scenario.

    A warning requires `min_consecutive` consecutive windows above threshold
    (see `alerting`): single-window spikes are exactly the noise an operator
    would learn to ignore, so they must not count as warnings. The warning is
    credited at the window that *completes* the run, the first moment anyone
    could have been told, not back to the run's first window.
    """
    has_over_column = "range_over_threshold" in frame.columns
    outcomes: list[ScenarioOutcome] = []
    for scenario_id, group in frame.groupby("scenario_id", sort=True):
        # Collapse traffic types: the range-level view is what rings the bell.
        aggregations = {
            "score": (score_column, "max"),
            "is_onset": ("is_onset", "max"),
            "scenario_type": ("scenario_type", "first"),
        }
        if has_over_column:
            aggregations["over_threshold"] = ("range_over_threshold", "max")
        per_window = group.groupby("window_start_epoch").agg(**aggregations).sort_index()
        above = (per_window["score"] >= threshold).to_numpy()
        epochs = per_window.index.to_numpy()

        alert_index = first_alert_index(above, min_consecutive)
        first_warning = int(epochs[alert_index]) if alert_index is not None else None

        onset_rows = np.flatnonzero(per_window["is_onset"].to_numpy() > 0)
        onset_epoch = int(epochs[onset_rows[0]]) if onset_rows.size else None

        # Already over threshold in the opening window: the episode starts
        # mid-attack, so there is nothing to forecast from.
        in_progress = bool(
            has_over_column and len(per_window) and per_window["over_threshold"].iloc[0] > 0
        )

        outcomes.append(
            ScenarioOutcome(
                scenario_id=str(scenario_id),
                scenario_type=str(per_window["scenario_type"].iloc[0]),
                has_onset=onset_epoch is not None,
                onset_epoch=onset_epoch,
                first_warning_epoch=first_warning,
                in_progress_at_start=in_progress,
            )
        )
    return outcomes


def evaluate(
    frame: pd.DataFrame,
    score_column: str,
    threshold: float,
    min_consecutive: int = 2,
) -> EvaluationResult:
    outcomes = scenario_outcomes(frame, score_column, threshold, min_consecutive)
    attacks = [o for o in outcomes if o.has_onset]
    quiet = [o for o in outcomes if not o.has_onset]

    # Lead time and recall are scored on attacks that could be forecast. The
    # already-underway ones get their own number so they are visible rather
    # than dragging the head-start claim down to the episode boundary.
    forecastable = [o for o in attacks if o.forecastable]
    in_progress = [o for o in attacks if not o.forecastable]

    leads = [o.lead_seconds for o in forecastable if o.lead_seconds is not None]
    useful = [lead for lead in leads if lead > 0]
    late = len(leads) - len(useful)

    caught = len(useful) / len(forecastable) if forecastable else 0.0
    detected = sum(1 for o in in_progress if o.warned) / len(in_progress) if in_progress else 0.0
    false_alarms = sum(1 for o in quiet if o.warned) / len(quiet) if quiet else 0.0

    return EvaluationResult(
        n_scenarios=len(outcomes),
        n_attack_scenarios=len(attacks),
        n_forecastable_scenarios=len(forecastable),
        n_in_progress_scenarios=len(in_progress),
        attacks_caught=caught,
        false_alarm_rate=false_alarms,
        median_lead_seconds=float(np.median(useful)) if useful else 0.0,
        mean_lead_seconds=float(np.mean(useful)) if useful else 0.0,
        p10_lead_seconds=float(np.percentile(useful, 10)) if useful else 0.0,
        late_warning_count=late,
        in_progress_detected=detected,
        leads_beyond_horizon=sum(1 for lead in useful if lead > MAX_CREDIBLE_LEAD_SECONDS),
        threshold=float(threshold),
    )


def fixed_threshold_baseline(frame: pd.DataFrame) -> pd.Series:
    """The incumbent: score 1.0 at the moment Shared State declares.

    This is what the bell does today — it waits for blocking above the start
    threshold to persist for the full start duration, then declares. Its
    warning *is* the declaration, so its lead time is zero by construction,
    which is the point of the comparison.

    Scoring `range_over_threshold` instead would be the subtle mistake: the
    crossing precedes the declaration by the whole start duration, so the
    incumbent would be credited with its own debounce delay (20s on this
    scenario set, 290s in production) as though it were foresight.
    """
    if "declared" not in frame.columns:
        raise KeyError(
            "fixed_threshold_baseline needs the `declared` column: the incumbent's "
            "warning is the declaration, not the threshold crossing"
        )
    return frame["declared"].astype(float)


def lead_time_histogram(
    outcomes: list[ScenarioOutcome], bucket_seconds: int = 30
) -> dict[str, int]:
    """Bucketed lead times for the dashboard.

    Three kinds of "no bar on the chart" are kept apart, because collapsing
    them would turn correct behaviour into apparent failure:

    * `no attack` — a quiet scenario. Silence here is the right answer, and it
      is the large majority of any realistic scenario set;
    * `already under way` — the episode opened mid-attack, so there was no
      runway to forecast from (see the module docstring);
    * `no useful warning` — a real, forecastable onset that DFSI missed or
      warned about too late. This is the only bucket that is a failure.
    """
    histogram: dict[str, int] = {}
    for outcome in outcomes:
        lead = outcome.lead_seconds
        if not outcome.has_onset:
            key = "no attack"
        elif outcome.in_progress_at_start:
            key = "already under way"
        elif lead is None or lead <= 0:
            key = "no useful warning"
        else:
            low = int(lead // bucket_seconds) * bucket_seconds
            key = f"{low}-{low + bucket_seconds}s"
        histogram[key] = histogram.get(key, 0) + 1
    return histogram
