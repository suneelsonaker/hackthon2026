"""The Claude code path, against a scripted client: the request contract and every exit.

No network and no API key. A fake `client.beta.messages.create` replays a
scripted conversation and records each request, so these tests pin down:

* the request shape DFSI sends (model, thinking, effort, structured output,
  strict read-only tools, the refusal fallback, cacheable system prompt);
* the tool loop (results returned in one user message, denied tools recorded);
* that the validator still judges Claude's answer, exactly as it judges rules;
* that every failure (refusal, truncation, garbage, SDK error, runaway loop)
  falls back to the offline rules instead of raising.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_explain_failsafe import make_bundle

from dfsi.agents import orchestrator as orch
from dfsi.agents.orchestrator import ExplainerOrchestrator
from dfsi.agents.tools import INVESTIGATOR_TOOLS


def text(payload) -> SimpleNamespace:
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(type="text", text=body)


def tool_use(name: str, block_id: str = "tu_1", **inputs) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, id=block_id, input=inputs)


def reply(stop_reason: str, *content) -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop_reason, content=list(content))


class ScriptedClient:
    """Stands in for `anthropic.Anthropic`; returns scripted replies in order."""

    def __init__(self, *replies):
        self._replies = list(replies)
        self.requests: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request):
        # Snapshot: the orchestrator keeps appending to one `messages` list.
        self.requests.append({**request, "messages": list(request["messages"])})
        if not self._replies:
            raise AssertionError("client called more often than scripted")
        nxt = self._replies.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def claude(client: ScriptedClient) -> ExplainerOrchestrator:
    agent = ExplainerOrchestrator(api_key=None)
    agent._client = client
    agent.engine = "claude"
    return agent


GOOD_READING = {
    "reading": "attack",
    "headline": "Reading: an attack is beginning",
    "summary": "Source concentration is 0.68, narrow for this range.",
    "evidence_cited": [{"feature": "country_hhi", "value": 0.68, "why": "narrow mix"}],
    "contrary_evidence": [],
}


def read_then_answer(answer: dict) -> ScriptedClient:
    """Claude reads the evidence, then gives `answer`: the honest path, minus the answer."""
    return ScriptedClient(
        reply("tool_use", tool_use("get_forecast_evidence")),
        reply("end_turn"),
        reply("end_turn", text(answer)),
    )


def run_investigator(client: ScriptedClient):
    return claude(client).run_investigator(make_bundle(risk_score=0.9))


# ------------------------------------------------------------- the contract


def test_the_request_contract():
    client = ScriptedClient(
        reply("tool_use", tool_use("get_forecast_evidence")),
        reply("end_turn", text("done reading")),
        reply("end_turn", text(GOOD_READING)),
    )
    result = run_investigator(client)
    assert result.engine == "claude" and result.validated, result.validator_failures

    first, _, final = client.requests
    for request in client.requests:
        assert request["model"] == orch.MODEL == "claude-opus-5-5"
        assert request["max_tokens"] == orch.MAX_TOKENS
        assert request["thinking"] == {"type": "adaptive"}
        assert request["betas"] == ["server-side-fallback-2026-07-01"]
        assert request["fallbacks"] == "default"
        assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert [t["name"] for t in request["tools"]] == list(INVESTIGATOR_TOOLS)
        assert all(t["strict"] is True for t in request["tools"])
    assert "tool_choice" not in first
    assert final["tool_choice"] == {"type": "none"}
    assert final["output_config"]["format"] == {
        "type": "json_schema",
        "schema": orch.INVESTIGATOR_SCHEMA,
    }


def test_tool_results_go_back_in_one_user_message():
    client = ScriptedClient(
        reply(
            "tool_use",
            tool_use("get_forecast_evidence", "a"),
            tool_use("get_recent_traffic", "b", max_windows=3),
        ),
        reply("end_turn", text("ok")),
        reply("end_turn", text(GOOD_READING)),
    )
    run_investigator(client)
    results = client.requests[1]["messages"][-1]
    assert results["role"] == "user"
    assert [r["tool_use_id"] for r in results["content"]] == ["a", "b"]
    assert not any(r.get("is_error") for r in results["content"])


def test_a_tool_outside_the_role_is_refused_and_fails_validation():
    client = ScriptedClient(
        reply("tool_use", tool_use("get_simulation_results")),  # an Advisor tool
        reply("end_turn", text("ok")),
        reply("end_turn", text(GOOD_READING)),
    )
    result = run_investigator(client)
    refused = client.requests[1]["messages"][-1]["content"][0]
    assert refused["is_error"] is True and "not available" in refused["content"]
    assert not result.validated
    assert result.validator_checks["tool_use"] is False


def test_a_failing_tool_is_reported_to_the_model_not_raised():
    client = ScriptedClient(
        reply("tool_use", tool_use("get_recent_traffic", max_windows="many")),
        reply("end_turn", text("ok")),
        reply("end_turn", text(GOOD_READING)),
    )
    run_investigator(client)
    errored = client.requests[1]["messages"][-1]["content"][0]
    assert errored["is_error"] is True and errored["content"].startswith("Tool failed")


def test_claude_is_held_to_the_same_validator_as_the_rules():
    invented = {**GOOD_READING, "summary": "Expect 900 Mbps within the minute."}
    result = run_investigator(read_then_answer(invented))
    assert result.engine == "claude" and not result.validated
    assert result.validator_failures == [
        "cited numbers not present in the evidence bundle: ['900']"
    ]


def test_a_misattributed_citation_from_claude_fails():
    wrong = {
        **GOOD_READING,
        "evidence_cited": [{"feature": "country_hhi", "value": 0.12, "why": "x"}],
    }
    result = run_investigator(read_then_answer(wrong))
    assert not result.validated and result.validator_checks["citations"] is False
    assert result.validator_checks["evidence"] is True  # the number itself was shown


def test_claude_may_only_quote_what_its_tools_returned():
    """0.68 is in the bundle, but a Claude that never read the evidence cannot know it."""
    blind = run_investigator(
        ScriptedClient(reply("end_turn"), reply("end_turn", text(GOOD_READING)))
    )
    assert not blind.validated
    assert blind.validator_checks["evidence"] is False

    read = run_investigator(
        ScriptedClient(
            reply("tool_use", tool_use("get_forecast_evidence")),
            reply("end_turn"),
            reply("end_turn", text(GOOD_READING)),
        )
    )
    assert read.validated, read.validator_failures


# ---------------------------------------------------- every failure falls back


@pytest.mark.parametrize(
    "replies",
    [
        pytest.param([reply("refusal")], id="refusal-mid-loop"),
        pytest.param([reply("end_turn"), reply("refusal")], id="refusal-final"),
        pytest.param([reply("end_turn"), reply("max_tokens", text('{"read'))], id="truncated"),
        pytest.param([reply("end_turn"), reply("end_turn")], id="no-text"),
        pytest.param([reply("end_turn"), reply("end_turn", text("not json"))], id="garbage"),
        pytest.param([RuntimeError("connection reset")], id="sdk-error"),
        pytest.param([reply("tool_use", tool_use("get_forecast_evidence"))] * 6, id="runaway"),
    ],
)
def test_every_failure_falls_back_to_the_offline_rules(replies):
    result = run_investigator(ScriptedClient(*replies))
    assert result.engine == "offline-rules"
    assert result.output["agent"] == "investigator"


def test_the_full_briefing_runs_both_agents_on_claude():
    advice = {
        "headline": "Hold and watch",
        "summary": "Holding costs nothing while the reading firms up.",
        "options": [{"policy_key": "hold", "rationale": "reversible"}],
        "recommended_policy_key": "hold",
    }
    client = ScriptedClient(
        reply("tool_use", tool_use("get_forecast_evidence")),
        reply("end_turn"),
        reply("end_turn", text(GOOD_READING)),
        reply("end_turn"),
        reply("end_turn", text(advice)),
    )
    briefing = claude(client).explain(make_bundle(risk_score=0.9))
    assert briefing["investigator"]["engine"] == "claude"
    assert briefing["advisor"]["engine"] in {"claude", "offline-rules"}
    assert "reading is: attack" in client.requests[3]["messages"][0]["content"]


# ------------------------------------------------------------ client set-up


def test_without_a_key_the_engine_is_offline(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert ExplainerOrchestrator().engine == "offline-rules"


def test_a_client_that_cannot_be_built_means_offline(monkeypatch):
    import anthropic

    def broken(**_):
        raise ValueError("bad key format")

    monkeypatch.setattr(anthropic, "Anthropic", broken)
    assert ExplainerOrchestrator(api_key="sk-test").engine == "offline-rules"


def test_with_a_key_the_engine_is_claude():
    assert ExplainerOrchestrator(api_key="sk-test").engine == "claude"


def test_calling_claude_without_a_client_is_an_error():
    with pytest.raises(RuntimeError, match="not configured"):
        ExplainerOrchestrator(api_key=None)._require_client()


def test_an_unknown_tool_schema_is_a_programming_error():
    with pytest.raises(ValueError, match="no tool schema"):
        orch._tool_definition("delete_customer")
