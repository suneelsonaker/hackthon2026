"""Per-range decision thresholds, configured in raw.json.

The model is calibrated to one decision threshold across every range. A range
in raw.json may set its own `risk_threshold` instead: a noisy range held to a
higher bar, a critical one warned about sooner. A range without one falls back
to the model's calibrated threshold.

The threshold only changes how a score is *read* (its band, and whether it
counts towards a warning), never the score itself. It is read once, when the
API starts; edit raw.json and restart to change it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .serve import RiskBands

log = logging.getLogger("dfsi.forecaster.range_thresholds")


def load_range_thresholds(path: Path) -> dict[int, float]:
    """`range_id -> risk_threshold` for every range in raw.json that sets one.

    A missing file means no overrides. A value the risk bands cannot be built
    from raises: silently scoring a range on the model's threshold when the
    file asked for another would misreport that range's warnings.
    """
    if not path.exists():
        log.warning("no policy db at %s; every range uses the model's threshold", path)
        return {}
    thresholds: dict[int, float] = {}
    for account in json.loads(path.read_text()).get("accounts", []):
        for rng in account.get("ranges", []):
            if rng.get("risk_threshold") is None:
                continue
            range_id = int(rng["range_id"])
            try:
                value = float(rng["risk_threshold"])
                RiskBands.for_threshold(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}: range {range_id} has an invalid risk_threshold: {exc}"
                ) from exc
            thresholds[range_id] = value
    return thresholds
