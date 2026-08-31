"""Where the tracing attaches, and why it is nowhere near the loop (D10, step 2).

`run_tool_loop` has been byte-identical since D2 — proved rather than claimed:
D10 step 1 hashed its source on `main` and on this branch and got
`161bdc1c…9d00` both times. D5 changed where tools come from and D9 changed
what the model is told, what it remembers and what it may say, and neither
opened it. Instrumentation is the easiest thing in this project to justify
putting inside that `while`, and it is the one that would prove the claim was
only ever true because nothing had wanted it enough.

So it goes where D9's did: around. Both wrappers below are the shape
`GuardedClient` already is — forward everything by `__getattr__`, intercept the
one method that matters — which means the loop still receives a client with
`chat_with_tools` and a registry with `dispatch`, and still knows nothing about
either.

**Nothing is measured twice.** `llm/usage.py` already computes tokens and cost
per call and `AssistantMessage` already carries the result; `ToolResult`
already carries a tool's outcome. These wrappers read what exists and add one
thing neither has: how long it took. Recomputing a cost here would be a second
pricing table, and the first symptom would be a trace and a CLI disagreeing
about the same conversation — which is exactly the comparison D10 uses to check
this file.
"""

from __future__ import annotations

import time
from typing import Any

from shopagent.obs.tracing import Observation, Tracer
from shopagent.tools.registry import ToolResult


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


class TracedClient:
    """An LLM client that records every call as a generation.

    **Innermost, under `GuardedClient`**, and the order is the point. The amount
    guardrail can send a second, corrected request, and that request is really
    billed — a wrapper placed outside it would see one call where two happened
    and report half the cost. Under it, both appear, and the trace shows the
    retry as what it is.
    """

    def __init__(self, client: Any, tracer: Tracer) -> None:
        self._client = client
        self._tracer = tracer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def chat_with_tools(self, messages: Any, tools: Any = None) -> Any:
        span = self._tracer.generation(
            name="chat", messages=messages, model=getattr(self._client, "model", "unknown")
        )
        started = time.perf_counter()
        try:
            reply = self._client.chat_with_tools(messages, tools)
        except Exception as exc:
            # Ended rather than dropped: a generation that never closes is the
            # one a reader most needs to see, because it is where the
            # conversation stopped.
            span.end(
                level="ERROR",
                status_message=f"{type(exc).__name__}: {exc}",
                metadata={"latency_ms": _elapsed_ms(started)},
            )
            raise

        span.end(**_generation_fields(reply, _elapsed_ms(started)))
        return reply


def _generation_fields(reply: Any, latency_ms: float) -> dict[str, Any]:
    """What one finished call contributes, read from what already measured it.

    The output is deliberately *not* the model's prose. `redact_messages` would
    have to be applied to it and there is nothing left after that a reader can
    use, so what is sent instead is the shape of the turn: whether it answered
    or asked for tools, and which tools. That is the half of an assistant turn
    the redaction rule permits and the half a trace is read for.
    """
    fields: dict[str, Any] = {
        "output": {
            "answered": bool(getattr(reply, "content", None)),
            "tool_calls": [call.name for call in getattr(reply, "tool_calls", []) or []],
        },
        "metadata": {"latency_ms": latency_ms},
    }

    usage = getattr(reply, "usage", None)
    if usage is not None:
        fields["usage_details"] = {
            "input": usage.prompt_tokens,
            "output": usage.completion_tokens,
            "cache_read_input_tokens": usage.cached_tokens,
            "total": usage.total_tokens,
        }
        # Our own number, from our own table. Langfuse prices what it
        # recognises and recognises none of the `gpt-5.6-*` family, so leaving
        # this out would put a confident $0.00 next to a CLI reporting cents.
        fields["cost_details"] = {"total": usage.cost_usd}
        fields["model"] = usage.model
    return fields


class TracedRegistry:
    """A registry that records every dispatch as a tool observation.

    A forwarding wrapper rather than a fourth subclass of `ToolRegistry`.
    `RememberingRegistry` and `GuardedRegistry` are subclasses because each
    *changes what dispatch does* — one records into a conversation's memory,
    the other refuses calls. This one changes nothing and only watches, and
    extending the chain would tie tracing to guarding: a caller wanting one
    would get the other.

    Everything but `dispatch` forwards, which is what lets `_run_session` keep
    calling `openai_schemas()`, `names()` and `specs()` on it, and lets the gate
    keep reaching `memory` through it.
    """

    def __init__(self, registry: Any, tracer: Tracer) -> None:
        self._registry = registry
        self._tracer = tracer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._registry, name)

    def dispatch(self, name: str, raw_args: Any = None) -> ToolResult:
        span = self._tracer.tool(name=name, arguments=_arguments(raw_args))
        started = time.perf_counter()
        result = self._registry.dispatch(name, raw_args)
        _end_tool_span(span, result, _elapsed_ms(started))
        return result


def _arguments(raw_args: Any) -> Any:
    """A tool call's arguments as a mapping, so `query` can be found in them.

    The model sends a JSON *string* and `ToolRegistry.dispatch` is what decodes
    it. Decoding again here is not a second owner of that contract — nothing is
    validated and nothing is passed on; a payload this cannot read is traced as
    the string it was, which is what a reader debugging a malformed call needs
    to see anyway.
    """
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        import json

        try:
            parsed = json.loads(raw_args or "{}")
        except ValueError:
            return raw_args
        return parsed if isinstance(parsed, dict) else raw_args
    return raw_args


def _end_tool_span(span: Observation, result: ToolResult, latency_ms: float) -> None:
    """What one finished dispatch contributes.

    A refusal is a `WARNING` carrying `result.error`, which is the short
    machine-facing half of a `ToolResult` — the long half is written for the
    model and would put this shop's own sentences in a vendor's database for no
    gain. That is also what makes two of D9's three guardrails visible here for
    free: the confirmation gate and the unknown-variant refusal both come back
    as a failed `ToolResult` with a reason, so neither needs instrumenting
    separately. The third lives in `GuardedClient` and reports itself.

    `ok` results carry no output. A tool result is this shop's own data and the
    redaction rule permits it, but it is also the largest thing in the
    conversation — one catalogue search is 4,728 characters — and a trace is
    read for which tools ran and what they cost, not for replaying their
    payloads. The CLI already prints them and the MCP server already logs them.
    """
    fields: dict[str, Any] = {
        "metadata": {"latency_ms": latency_ms, "ok": result.ok},
        "output": {"ok": result.ok, "chars": len(result.content)},
    }
    if not result.ok:
        fields["level"] = "WARNING"
        fields["status_message"] = result.error or "the tool refused"
    span.end(**fields)
