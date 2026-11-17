"""Read-only tools the Explainer agents are allowed to call.

The agents get a deliberately small, read-only surface. Every tool here:

* reads from an evidence bundle that was assembled *before* the agent ran,
* returns plain data the agent must quote rather than recompute,
* cannot mutate anything, reach the network, or touch a customer config.

No tool returns, accepts, or stores an IP address. Source information is
country-concentration and prior-episode counts only.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from ..rehearsal.simulator import CANDIDATES

#: Decimal places an agent may round a quoted value to ("0.4231" -> "0.42").
_VALUE_DECIMALS = range(0, 5)
#: The same value read as a percentage ("0.4231" -> "42.31%").
_PERCENT_DECIMALS = range(0, 3)
#: A rounded rendering counts as quoting the value only while it stays this
#: close to it. "0.4231" may be "0.42" or "42%", but 7.4 may not become "7":
#: without this bound every value donated a nearby integer, and together they
#: covered most small numbers an agent could invent.
MAX_ROUNDING_ERROR = 0.05

_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def _trim(number: float, decimals: int) -> str:
    text = f"{number:.{decimals}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def numeric_forms(value: Any) -> set[str]:
    """Every string an agent may use to quote `value` faithfully.

    The validator extracts numbers from prose without their sign, so a value of
    -0.4231 can only ever be seen as "0.4231". The forms are therefore built
    from the *magnitude*; the direction of a change is left to the agent and
    the reader. Rounded and percentage renderings are accepted because an
    explanation that says "about 42%" is quoting the evidence, not inventing a
    figure. Anything else — a different value, or arithmetic on one — is still
    rejected.

    Non-numbers, booleans and non-finite floats yield no forms.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return set()
    magnitude = abs(float(value))
    if not math.isfinite(magnitude):
        return set()
    forms = _faithful_roundings(magnitude, _VALUE_DECIMALS)
    forms |= _faithful_roundings(magnitude * 100.0, _PERCENT_DECIMALS)
    return forms


def _faithful_roundings(value: float, places: range) -> set[str]:
    """Renderings of `value` at each precision that stay within the rounding bound.

    The most precise rendering is always kept, so an exact quote never fails.
    """
    forms = {_trim(value, places[-1])}
    for decimals in places:
        rounded = round(value, decimals)
        if value == 0 or abs(rounded - value) / value <= MAX_ROUNDING_ERROR:
            forms.add(_trim(value, decimals))
    return forms


def _add_value(values: dict[str, list[float]], name: Any, value: Any) -> None:
    """Record `value` under feature `name`; a name with no usable value still counts."""
    if not isinstance(name, str):
        return
    bucket = values.setdefault(name, [])
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, int | float) and math.isfinite(value):
        bucket.append(float(value))


def _is_timestamp(key: Any) -> bool:
    """Epoch columns locate a window; they are not figures an explanation quotes."""
    return isinstance(key, str) and key.endswith("_epoch")


def _numbers_in(payload: Any) -> list[Any]:
    """Every number in a decoded tool payload, including numbers written in its prose."""
    if isinstance(payload, dict):
        return [
            n
            for key, value in payload.items()
            if not _is_timestamp(key)
            for n in _numbers_in(value)
        ]
    if isinstance(payload, list):
        return [n for value in payload for n in _numbers_in(value)]
    if isinstance(payload, str):
        return [float(match) for match in _NUMBER.findall(payload)]
    return [payload]


@dataclass
class EvidenceBundle:
    """Everything the agents are permitted to see for one forecast.

    Assembled by the API from the forecaster's own output. The agents cannot
    add to it, and the validator later checks that every number an agent cites
    appears somewhere in here.
    """

    range_id: int
    account_id: int
    window_start_epoch: int
    risk_score: float | None
    risk_band: str
    model_version: str
    feature_version: str
    horizon_seconds: int
    evidence: list[dict] = field(default_factory=list)
    recent_windows: list[dict] = field(default_factory=list)
    scenario_type_hint: str | None = None
    simulation: dict | None = None
    #: Tenant the request is scoped to. The validator enforces this.
    tenant_account_id: int | None = None
    #: The instant this evidence is judged against. A replay supplies its own
    #: clock (the decision epoch); live use leaves it unset and the orchestrator
    #: reads the wall clock, which is what makes the freshness check bite.
    as_of_epoch: int | None = None

    def citable_numbers(self) -> set[str]:
        """String forms of every number an agent may legitimately quote."""
        values: set[str] = set()

        def add(value: Any) -> None:
            values.update(numeric_forms(value))

        add(self.risk_score)
        add(self.horizon_seconds)
        add(self.range_id)
        add(self.account_id)
        for item in self.evidence:
            # Every figure the evidence tool shows an agent is one it may quote.
            for key in ("value", "importance", "contribution"):
                add(item.get(key))
        for row in self.recent_windows:
            for key, value in row.items():
                if not _is_timestamp(key):
                    add(value)
        if self.simulation:
            for candidate in self.simulation.get("candidates", []):
                for value in candidate.values():
                    add(value)
            plan = self.simulation.get("policy_plan") or {}
            add(plan.get("history_days"))
            add(plan.get("history_required_days"))
            for suggestion in plan.get("per_type", []) or []:
                for value in suggestion.values():
                    add(value)
        return values

    def citable_values(self) -> dict[str, list[float]]:
        """Feature name -> every value of it an agent was shown.

        Drawn from the evidence items and the recent traffic rows, so a
        citation can be checked against the feature it names, not merely
        against "some number somewhere in the bundle".
        """
        values: dict[str, list[float]] = {}
        for item in self.evidence:
            _add_value(values, item.get("feature"), item.get("value"))
        for row in self.recent_windows:
            for name, value in row.items():
                _add_value(values, name, value)
        return values

    def newest_evidence_epoch(self) -> int:
        """Start of the newest window any agent can see.

        Taken over the forecast *and* the traffic rows, so a bundle that leaks
        windows from after the decision instant shows up as evidence from the
        future rather than hiding behind the forecast's own timestamp.
        """
        epochs = [self.window_start_epoch]
        epochs.extend(
            int(row["window_start_epoch"])
            for row in self.recent_windows
            if row.get("window_start_epoch") is not None
        )
        return max(epochs)

    def scope_violations(self) -> list[str]:
        """Rows that describe a range or account other than the one under review.

        Tenant isolation is a property of how the bundle is assembled, so it is
        checked before any agent runs: once another tenant's rows have been
        sent to a model, rejecting its output is too late.
        """
        account = self.tenant_account_id if self.tenant_account_id is not None else self.account_id
        problems: list[str] = []
        for row in self.recent_windows:
            row_range = row.get("range_id", self.range_id)
            row_account = row.get("account_id", account)
            if int(row_range) != int(self.range_id) or int(row_account) != int(account):
                problems.append(
                    f"window {row.get('window_start_epoch')} is for range {row_range} / "
                    f"account {row_account}, expected range {self.range_id} / account {account}"
                )
        return problems


class ReadOnlyToolbox:
    """Bound tool implementations for a single evidence bundle.

    Each public method is exposed to Claude as a tool. Methods are pure reads
    over `bundle`; there is intentionally no setter, no file write and no
    network call anywhere in this class.
    """

    def __init__(self, bundle: EvidenceBundle) -> None:
        self._bundle = bundle
        self.calls: list[dict] = []
        #: Every payload actually returned to the agent, decoded.
        self._shown: list[Any] = []

    def _record(self, name: str, payload: dict) -> None:
        self.calls.append({"tool": name, "arguments": payload})

    def _reply(self, data: dict) -> str:
        self._shown.append(data)
        return json.dumps(data)

    def citable_numbers(self) -> set[str]:
        """Forms of every number this agent was actually shown.

        The tool replies, plus the three figures its opening message states
        (range, account, horizon). Tighter than the whole bundle: an agent that
        never asked for the drill cannot quote the drill's figures.
        """
        values: set[str] = set()
        stated = (self._bundle.range_id, self._bundle.account_id, self._bundle.horizon_seconds)
        for number in (*stated, *_numbers_in(self._shown)):
            values |= numeric_forms(number)
        return values

    def citable_values(self) -> dict[str, list[float]]:
        """Feature name -> values shown to the agent, for the citation check."""
        values: dict[str, list[float]] = {}
        for reply in self._shown:
            rows = list(reply.get("top_features") or [])
            for item in rows:
                _add_value(values, item.get("feature"), item.get("value"))
            for row in reply.get("windows") or []:
                for name, value in row.items():
                    _add_value(values, name, value)
        return values

    def record_denied(self, name: str, arguments: dict | None = None) -> None:
        """Log an attempt to use a tool outside the agent's role.

        The call is refused, but it must stay on the audit trail: the validator
        fails an agent that reached outside its role, and it cannot see a
        refusal that was never recorded.
        """
        self.calls.append({"tool": name, "arguments": arguments or {}, "denied": True})

    # ------------------------------------------------------------------ #
    def get_forecast_evidence(self) -> str:
        """Return the ranked features behind the current risk score."""
        self._record("get_forecast_evidence", {})
        return self._reply(
            {
                "range_id": self._bundle.range_id,
                "window_start_epoch": self._bundle.window_start_epoch,
                "risk_band": self._bundle.risk_band,
                "model_version": self._bundle.model_version,
                "feature_version": self._bundle.feature_version,
                "horizon_seconds": self._bundle.horizon_seconds,
                "top_features": self._bundle.evidence,
            }
        )

    def get_recent_traffic(self, max_windows: int = 12) -> str:
        """Return the most recent traffic windows for this range."""
        limit = max(1, min(int(max_windows), 36))
        self._record("get_recent_traffic", {"max_windows": limit})
        return self._reply({"windows": self._bundle.recent_windows[-limit:]})

    def get_source_profile(self) -> str:
        """Return the country-concentration profile and attack history.

        Country concentration only — no addresses are available to this tool.
        """
        self._record("get_source_profile", {})
        latest = self._bundle.recent_windows[-1] if self._bundle.recent_windows else {}
        return self._reply(
            {
                "country_concentration_index": latest.get("country_hhi"),
                "country_concentration_change_60s": latest.get("country_hhi_delta_60s"),
                "prior_attack_episodes_on_account": latest.get("account_attack_history"),
                "note": "Country mix only. Raw source addresses are not collected or stored.",
            }
        )

    def get_policy_candidates(self) -> str:
        """Return the four fixed response options and their trade-offs."""
        self._record("get_policy_candidates", {})
        return self._reply(
            {
                "candidates": [
                    {
                        "key": candidate.key,
                        "name": candidate.name,
                        "description": candidate.description,
                        "operational_cost": candidate.operational_cost,
                        "blast_radius": candidate.blast_radius,
                        "engagement_delay_seconds": candidate.engagement_delay_seconds,
                        "reversible_seconds": candidate.reversible_seconds,
                    }
                    for candidate in CANDIDATES
                ],
                "note": "Describing options only. Selecting one requires human approval.",
            }
        )

    def get_simulation_results(self) -> str:
        """Return the rehearsal-room comparison, if a drill has been run."""
        self._record("get_simulation_results", {})
        if not self._bundle.simulation:
            return self._reply({"available": False, "reason": "no drill has been run yet"})
        return self._reply({"available": True, **self._bundle.simulation})


#: Tool names each agent may use. Enforced by the validator, so a model that
#: tries to reach outside its role is caught rather than trusted.
INVESTIGATOR_TOOLS = (
    "get_forecast_evidence",
    "get_recent_traffic",
    "get_source_profile",
)
ADVISOR_TOOLS = (
    "get_forecast_evidence",
    "get_policy_candidates",
    "get_simulation_results",
)
