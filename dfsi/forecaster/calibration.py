"""Whether a decision threshold was calibrated, stated rather than assumed.

Kept apart from `train.py` so the API's model card can use it without
importing scikit-learn and MLflow at start-up.
"""

from __future__ import annotations


def calibration_summary(sweep: list[dict], budget: float, threshold: float) -> dict:
    """Whether the decision threshold was calibrated, or is the configured fallback.

    Stated in the report and on the model card, so "calibrated" is never claimed
    for a threshold that no point of the sweep could justify.
    """
    if not sweep:
        return {
            "source": "configured",
            "budget": budget,
            "met_budget": None,
            "threshold": threshold,
        }
    admissible = [row for row in sweep if row["false_alarm_rate"] <= budget]
    best = min(sweep, key=lambda row: row["false_alarm_rate"])
    return {
        "source": "calibrated" if admissible else "fallback",
        "budget": budget,
        "met_budget": bool(admissible),
        "threshold": threshold,
        # The closest the sweep came, so a missed budget is quantified.
        "lowest_false_alarm_rate": best["false_alarm_rate"],
        "lowest_false_alarm_threshold": best["threshold"],
    }
