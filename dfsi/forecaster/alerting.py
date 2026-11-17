"""When does DFSI raise a warning? One rule, used everywhere.

A single window above the decision threshold is exactly the noise an operator
learns to ignore, so a warning needs `min_consecutive` windows in a row at or
above it. Three places must agree on that rule or the numbers stop meaning what
they say:

* the **evaluator** (`metrics.scenario_outcomes`) that produces the published
  head start and false-alarm rate;
* the **live replay stream** that drives the dashboard's gauge and timeline;
* the **default decision epoch** the drill and explain endpoints use.

The live path used to warn on the first window above threshold while the
evaluator required two in a row, so the demo behaved differently from the
numbers quoted about it, and raised alarms the evaluation would have discarded.

The warning is **raised at the window that completes the run**, not credited
back to the run's first window: an operator cannot be told about window *n*
until window *n + min_consecutive - 1* has been seen, and a head start measured
from a moment nobody could have acted on is overstated by that many windows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def _check(min_consecutive: int) -> int:
    if min_consecutive < 1:
        raise ValueError(f"min_consecutive must be at least 1, got {min_consecutive}")
    return int(min_consecutive)


def first_alert_index(above: Sequence[bool], min_consecutive: int) -> int | None:
    """Index of the window at which a warning is first raised, or `None`.

    `above[i]` says whether window *i* scored at or above the decision
    threshold. The warning fires on the `min_consecutive`-th window of an
    unbroken run.
    """
    needed = _check(min_consecutive)
    run = 0
    for index, flag in enumerate(above):
        run = run + 1 if flag else 0
        if run >= needed:
            return index
    return None


@dataclass
class AlertTracker:
    """Streaming form of `first_alert_index`, for one range, one window at a time.

    `update` returns whether the warning is active *after* this window: it
    becomes active when the run reaches `min_consecutive` and stays active
    while windows keep scoring above threshold. A window below threshold, or
    one the forecaster could not score, breaks the run.
    """

    min_consecutive: int = 2
    run: int = 0

    def __post_init__(self) -> None:
        self.min_consecutive = _check(self.min_consecutive)

    def update(self, above: bool) -> bool:
        self.run = self.run + 1 if above else 0
        return self.active

    @property
    def active(self) -> bool:
        return self.run >= self.min_consecutive
