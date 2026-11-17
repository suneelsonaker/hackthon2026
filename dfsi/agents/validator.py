"""The validator — code, not a model.

Every agent output passes through here before a human sees it. The validator
is deliberately boring, deterministic and written in plain Python: it is the
component that must not be able to hallucinate.

Seven checks, from the safety design:

1. **tenant scope**    — the output may only discuss the requested account/range
2. **output format**   — required fields present, correct types, bounded length
3. **freshness**       — the evidence is recent enough to still be true
4. **evidence**        — every number in the prose appears in the evidence bundle
5. **citations**       — every structured citation names a real feature, at its real value
6. **allowed actions** — only the four fixed candidates, never an executed one
7. **tool use**        — the agent used (or tried) only the tools its role allows

A failure is never silently repaired. The output is replaced with a
monitoring-only notice that states what failed.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import (
    HORIZON_SECONDS,
    STATE_MONITORING_ONLY,
    THRESHOLDS_BY_ENV,
    WINDOW_SECONDS,
)
from ..rehearsal.simulator import CANDIDATES_BY_KEY

#: Oldest evidence the validator will let through, in seconds.
MAX_EVIDENCE_AGE_SECONDS = 90

#: Phrases that would mean the agent is claiming to have *done* something.
_ACTION_CLAIM_PATTERNS = (
    r"\bI (?:have )?(?:applied|enabled|blocked|diverted|deployed|executed|activated)\b",
    r"\bpolicy (?:has been|was) (?:applied|changed|updated|deployed)\b",
    r"\btraffic (?:has been|was) (?:blocked|diverted|dropped)\b",
    r"\bI (?:will|'ll) (?:apply|enable|block|divert|deploy|execute)\b",
    r"\bautomatically (?:applied|mitigated|blocked)\b",
)

#: The agents explain; ML produces the numbers. An agent inventing its own
#: probability is the single most dangerous failure mode in this design.
_RISK_NUMBER_PATTERNS = (
    r"\b(?:risk|probability|likelihood|confidence|chance)\s*(?:score|level|is|of|:|=)?\s*"
    r"(?:is\s*)?(\d{1,3}(?:\.\d+)?)\s*%",
    r"\b(?:risk|probability|likelihood|confidence)\s*(?:score)?\s*(?:is|of|:|=)\s*"
    r"(0?\.\d+|1\.0+)\b",
    r"\bI (?:estimate|calculate|compute|assess) (?:the )?(?:risk|probability)\b",
)

#: Every number in prose must be traceable to the bundle, with two narrow
#: exceptions below: structural durations quoted *with a seconds unit*, and bare
#: small counts. Neither exception covers a rate, a size or a percentage.
#: A trailing unit glued to the digits ("300Mbps", "10x") does not hide a figure.
_NUMBER_IN_TEXT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\d.]\d|\d)")

#: Scientific notation ("1e6", "2.5E+3") is a figure the digit scan would read
#: as a small count; it is never how the bundle states a value, so it is refused.
_SCIENTIFIC = re.compile(r"(?<![\w.])\d+(?:\.\d+)?[eE][+-]?\d+\b")

#: A figure spelled out in words cannot be traced to the bundle. Small counts
#: ("two agents", "four options") stay allowed, like their digit forms.
_NUMBER_WORDS = re.compile(
    r"\b(?:five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|hundreds?|thousands?|millions?|billions?|dozens?)\b",
    re.IGNORECASE,
)

#: Names that contain digits but are not claims about this incident: ticket ids
#: ("PPNG-6760") and CIDR prefix lengths ("/24"). The Advisor's own prompt tells
#: it to cite PPNG-6760 and the /24 blast radius, so counting them as invented
#: figures made the briefing that follows the prompt fail validation. Prefix
#: lengths are limited to /8-/32 so an arbitrary "/87" cannot hide a figure.
_REFERENCE_NAMES = re.compile(r"\b[A-Z]{2,}[A-Z0-9]*(?:-\d+)+\b|/(?:[89]|[12]\d|3[0-2])\b")

#: Durations that describe how DFSI works rather than this incident: the
#: forecast horizon, the window size, the one-minute feature lookback, each
#: environment's start duration and the default waits the drill prices. They
#: are tolerated only as durations ("120 seconds", "a 10s window"), so "120 pps"
#: or "300 Mbps" must still be found in the bundle.
STRUCTURAL_SECONDS: frozenset[int] = frozenset(
    {HORIZON_SECONDS, WINDOW_SECONDS, 60, 90}
    | {t.start_duration_seconds for t in THRESHOLDS_BY_ENV.values()}
)
_STRUCTURAL_DURATION = re.compile(
    r"(?<![\w.])(?:" + "|".join(str(n) for n in sorted(STRUCTURAL_SECONDS, reverse=True)) + r")"
    r"(?:\s*-?\s*)(?:s|secs?|seconds?)\b",
    re.IGNORECASE,
)

#: Bare integers this small are counts ("4 options", "2 windows in a row"), not
#: claims, unless a unit or percent sign makes them a measurement.
_SMALL_COUNTS = frozenset({"0", "1", "2", "3", "4"})
_MEASUREMENT_UNIT = re.compile(
    r"\s*(?:%|percent|[kmg]?pps\b|[kmg]?bps\b|[kmg]b/s|x\b|times\b"
    # A magnitude suffix makes a small digit a large figure: "4M pps", "3k", "4K-fold".
    r"|[kmbg](?=\b|-)|-?fold\b)",
    re.IGNORECASE,
)

#: How far a cited value may sit from the bundle's: agents round for prose, so
#: 0.5512 quoted as 0.55 is the same figure, while 0.6 is not.
CITATION_REL_TOLERANCE = 0.01

#: A range or account named in prose by its numeric id ("range 90110002").
#: Six or more digits, so ordinary figures such as "range 2000 pps" are not
#: read as tenant ids. Only the first id after the keyword is captured.
_TENANT_REFERENCE = re.compile(
    r"\b(range|account)s?\b(?:\s+id)?\s*[#:]?\s*(\d{6,})\b", re.IGNORECASE
)


@dataclass
class ValidationReport:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def fail(self, check: str, reason: str) -> None:
        self.checks[check] = False
        self.failures.append(reason)
        self.passed = False

    def ok(self, check: str) -> None:
        self.checks.setdefault(check, True)


def _iter_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(_iter_strings(item))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_iter_strings(item))
        return out
    return []


def check_tenant_scope(
    output: dict, expected_account_id: int | None, expected_range_id: int, report: ValidationReport
) -> None:
    """The output must reference only the account and range under review.

    Checks the structured ids the output states about itself, then the prose:
    an explanation that discusses "range 90110002" while briefing on another
    range has strayed outside its tenant even if every structured field is right.
    """
    stated_range = output.get("range_id")
    if stated_range is not None and int(stated_range) != int(expected_range_id):
        report.fail(
            "tenant_scope",
            f"output references range {stated_range}, expected {expected_range_id}",
        )
        return
    stated_account = output.get("account_id")
    if (
        expected_account_id is not None
        and stated_account is not None
        and int(stated_account) != int(expected_account_id)
    ):
        report.fail(
            "tenant_scope",
            f"output references account {stated_account}, expected {expected_account_id}",
        )
        return

    for kind, number in _TENANT_REFERENCE.findall(" ".join(_iter_strings(output))):
        expected = expected_range_id if kind.lower() == "range" else expected_account_id
        if expected is not None and int(number) != int(expected):
            report.fail(
                "tenant_scope",
                f"output discusses {kind.lower()} {number}, expected {expected}",
            )
            return
    report.ok("tenant_scope")


def check_output_format(
    output: dict, required_fields: Sequence[str], report: ValidationReport, max_chars: int = 4000
) -> None:
    missing = [field_name for field_name in required_fields if not output.get(field_name)]
    if missing:
        report.fail("output_format", f"missing required fields: {missing}")
        return
    total = sum(len(text) for text in _iter_strings(output))
    if total > max_chars:
        report.fail("output_format", f"output is {total} characters, limit is {max_chars}")
        return
    report.ok("output_format")


def check_freshness(
    evidence_epoch: int,
    now_epoch: int,
    report: ValidationReport,
    forecast_epoch: int | None = None,
    skew_seconds: int = WINDOW_SECONDS,
) -> None:
    """Two halves, each measured on the evidence that can violate it.

    * Look-ahead: the *newest* evidence must not postdate "now" by more than
      `skew_seconds`. Against a live wall clock one window of skew is normal
      (window boundaries vs the clock); a replay's "now" is the exact decision
      instant, so the caller passes 0 and a single leaked window fails.
    * Staleness: the *forecast being explained* must be recent. Measured on the
      forecast rather than the newest row, so a briefing on a ten-minute-old
      forecast fails even when fresher traffic rows sit beside it — which is
      what lets the check bite in a replay, where "now" is the decision instant.
    """
    ahead = evidence_epoch - now_epoch
    if ahead > skew_seconds:
        report.fail("freshness", f"evidence is {ahead}s in the future")
        return
    age = now_epoch - (forecast_epoch if forecast_epoch is not None else evidence_epoch)
    if age > MAX_EVIDENCE_AGE_SECONDS:
        report.fail(
            "freshness",
            f"the forecast is {age}s old, limit is {MAX_EVIDENCE_AGE_SECONDS}s",
        )
        return
    report.ok("freshness")


def check_evidence(output: dict, citable: set[str], report: ValidationReport) -> None:
    """No invented numbers, and no risk score of the agent's own making."""
    texts = _iter_strings(output)
    joined = " ".join(texts)

    for pattern in _RISK_NUMBER_PATTERNS:
        match = re.search(pattern, joined, flags=re.IGNORECASE)
        if match:
            report.fail(
                "evidence",
                f"agent produced its own risk figure: {match.group(0)!r}. "
                "Risk numbers come from the model, never from an agent.",
            )
            return

    # Only the figure scan skips reference names and structural durations; the
    # risk-figure patterns above still read the full text.
    scanned = _STRUCTURAL_DURATION.sub(" ", _REFERENCE_NAMES.sub(" ", joined))
    untraceable: tuple[tuple[re.Pattern[str], str], ...] = (
        (_SCIENTIFIC, "scientific notation"),
        (_NUMBER_WORDS, "a number in words"),
    )
    for spelled, kind in untraceable:
        found = spelled.search(scanned)
        if found:
            report.fail(
                "evidence",
                f"figure written as {kind} cannot be traced to the bundle: "
                f"{found.group(0)!r}. Quote the value in digits, as the tools give it.",
            )
            return
    unsupported: list[str] = []
    for match in _NUMBER_IN_TEXT.finditer(scanned):
        raw = match.group(1)
        normalised = raw.rstrip("0").rstrip(".") if "." in raw else raw
        if normalised in citable or raw in citable:
            continue
        if raw in _SMALL_COUNTS and not _MEASUREMENT_UNIT.match(scanned, match.end()):
            continue
        unsupported.append(raw)

    if unsupported:
        report.fail(
            "evidence",
            f"cited numbers not present in the evidence bundle: {sorted(set(unsupported))[:6]}",
        )
        return

    if not output.get("evidence_cited"):
        report.warnings.append("output cites no specific evidence items")
    report.ok("evidence")


def check_citations(
    output: dict, citable_values: dict[str, list[float]], report: ValidationReport
) -> None:
    """Each structured citation must name a feature the agent was shown, at its value.

    The prose scan proves a number exists somewhere in the bundle; this proves
    it belongs to the feature it is attributed to. "country_hhi 0.82" fails when
    0.82 is the blocked share and the source mix is broad.
    """
    for item in output.get("evidence_cited") or []:
        if not isinstance(item, dict):
            continue  # free-text citations are covered by the prose scan
        feature = item.get("feature")
        if feature is None:
            report.fail("citations", f"citation without a feature name: {item!r}")
            return
        if feature not in citable_values:
            report.fail("citations", f"cited feature {feature!r} is not in the evidence bundle")
            return
        value = item.get("value")
        if value is None:
            continue
        try:
            quoted = float(value)
        except (TypeError, ValueError):
            report.fail("citations", f"cited value for {feature!r} is not a number: {value!r}")
            return
        known = citable_values[feature]
        if known and not any(_same_figure(quoted, actual) for actual in known):
            report.fail(
                "citations",
                f"{feature} cited as {quoted:g}, but the bundle shows "
                f"{sorted({round(v, 4) for v in known})[:4]}",
            )
            return
    report.ok("citations")


def _same_figure(quoted: float, actual: float) -> bool:
    if math.isnan(quoted) or math.isnan(actual):
        return False
    return math.isclose(quoted, actual, rel_tol=CITATION_REL_TOLERANCE, abs_tol=1e-6)


def check_allowed_actions(output: dict, report: ValidationReport) -> None:
    """Only the four fixed candidates may be named, and only as proposals."""
    texts = " ".join(_iter_strings(output))
    for pattern in _ACTION_CLAIM_PATTERNS:
        match = re.search(pattern, texts, flags=re.IGNORECASE)
        if match:
            report.fail(
                "allowed_actions",
                f"output claims an action was or will be taken: {match.group(0)!r}. "
                "DFSI is simulation-only; humans decide and apply.",
            )
            return

    for option in output.get("options", []) or []:
        key = option.get("policy_key") if isinstance(option, dict) else None
        if key is not None and key not in CANDIDATES_BY_KEY:
            report.fail("allowed_actions", f"unknown policy option {key!r}")
            return

    recommended = output.get("recommended_policy_key")
    if recommended is not None and recommended not in CANDIDATES_BY_KEY:
        report.fail("allowed_actions", f"unknown recommended policy {recommended!r}")
        return

    report.ok("allowed_actions")


def check_tool_use(calls: Sequence[dict], allowed: Sequence[str], report: ValidationReport) -> None:
    """The agent may only have used the tools its role permits.

    Refused attempts count: the toolbox records them with `denied: true`, and
    reaching outside its role fails an agent whether or not the call succeeded.
    """
    used = {call.get("tool") for call in calls}
    forbidden = sorted(name for name in used if name and name not in set(allowed))
    if forbidden:
        report.fail("tool_use", f"agent used or attempted tools outside its role: {forbidden}")
        return
    report.ok("tool_use")


def validate(
    output: dict,
    *,
    required_fields: Sequence[str],
    expected_range_id: int,
    expected_account_id: int | None,
    citable_numbers: set[str],
    evidence_epoch: int,
    now_epoch: int,
    tool_calls: Sequence[dict] = (),
    allowed_tools: Sequence[str] = (),
    citable_values: dict[str, list[float]] | None = None,
    forecast_epoch: int | None = None,
    clock_skew_seconds: int = WINDOW_SECONDS,
) -> ValidationReport:
    """Run all checks. Order matters only for which failure is reported first."""
    report = ValidationReport(passed=True)
    check_output_format(output, required_fields, report)
    check_tenant_scope(output, expected_account_id, expected_range_id, report)
    check_freshness(evidence_epoch, now_epoch, report, forecast_epoch, clock_skew_seconds)
    check_evidence(output, citable_numbers, report)
    if citable_values is not None:
        check_citations(output, citable_values, report)
    check_allowed_actions(output, report)
    if allowed_tools:
        check_tool_use(tool_calls, allowed_tools, report)
    return report


def monitoring_only_notice(reason: str, failures: Sequence[str] = ()) -> dict:
    """The fail-safe output. Never says "low risk"; always says "unknown"."""
    return {
        "state": STATE_MONITORING_ONLY,
        "headline": "Monitoring only — explanation withheld",
        "summary": (
            "The explanation could not be verified, so DFSI is reporting "
            "monitoring-only rather than a risk assessment. Treat the "
            "situation as unknown, not as safe."
        ),
        "reason": reason,
        "validator_failures": list(failures),
        "evidence_cited": [],
        "options": [],
        "recommended_policy_key": None,
        "requires_human_decision": True,
    }
