"""What the tools did on one turn, kept where somebody can read it (D11, step 1).

The browser's activity strip needs four facts per tool call — the name, the
arguments, how long it took, and whether it worked — and before adding anything
this project asked whether it already had them. It did not, and the two places
that came closest are worth naming because each is missing a different half:

- `RememberingRegistry` (D9) sees every dispatch and keeps `last_search` and the
  sets of ids and amounts the model has been shown. Those are answers to
  questions about *meaning*: which list is the customer looking at, was this
  variant ever put in front of the model. It keeps no call log at all, and no
  timings.
- `TracedRegistry` (D10) measures exactly the missing half — it wraps `dispatch`
  with a `perf_counter` and records the name, the arguments and the outcome —
  and then sends all of it to Langfuse. There is no return path: a `Tracer` is
  a one-way door by design, it is inert when no keys are configured, and it may
  not raise. A UI that read its panel out of a vendor SDK would go blank the
  moment somebody ran the shop without Langfuse keys, which is a state D10
  declared ordinary.

So this is the same measurement with the other destination. It is a
**forwarding wrapper**, not a fourth `ToolRegistry` subclass, for the reason
`TracedRegistry` is one: `RememberingRegistry` and `GuardedRegistry` subclass
because each *changes what dispatch does*, and this one only watches. Extending
that chain would tie a UI's activity panel to the confirmation gate.

**It goes outermost.** The gate's refusals and the unknown-variant refusal are
`ToolResult`s returned by `GuardedRegistry.dispatch` rather than exceptions, so
only a wrapper above it sees them — and a refused checkout is the single most
important row the panel can show. The cost is that the duration includes the
tracing wrapper's own overhead, which is a `perf_counter` and a dict.

**Nothing here is redacted, and that is a boundary rather than an oversight.**
`obs/redaction.py` exists because a trace leaves this machine for a third
party. This log does not leave the process it was made in: it is rendered back
to the person who typed the query, in their own browser, which is the same
audience the CLI already prints every tool call and every argument to. The day
something ships this structure anywhere else, it is `obs/redaction.py` that has
to be reached in the same commit — the rule CLAUDE.md already states for tool
results in a trace.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from shopagent.tools.registry import ToolResult

# How much of one call's arguments is kept for display. The panel is a strip in
# a conversation, not a log file, and the model's own arguments are short — the
# longest this project produces is a search with seven filters. A result that
# needs more than this is one the CLI's `--` output or the MCP server's log is
# the right place to read.
MAX_ARGUMENT_CHARS = 400


@dataclass(frozen=True)
class ToolCallRecord:
    """One dispatch, as somebody watching the conversation would describe it."""

    name: str
    arguments: str
    ok: bool
    duration_ms: float
    # The short machine-facing half of a failed `ToolResult`. `None` on success.
    # The long half is written for the model and reads oddly to a person.
    error: str | None = None
    # How much the model was handed back. The number is what a panel shows:
    # one catalogue search is 4,728 characters, and a strip that inlined them
    # would bury the conversation it sits inside — the same call
    # `obs/instrumentation.py` makes about a trace.
    result_chars: int = 0
    # The payload itself, kept for one reader and not for display. `repr=False`
    # so it cannot arrive in a log line or a debugger view by accident, and
    # empty on a failure — an error message is written for the model and is
    # already summarised in `error`.
    #
    # It exists because a turn's search results have to be captured *at the
    # moment they are produced*, onto the message that produced them. Reading
    # them back off `ConversationMemory.last_search` at render time is the
    # obvious alternative and it is wrong: D9 made every new search replace the
    # previous one deliberately, so a history rendered that way would show the
    # newest cards under every older message.
    content: str = field(default="", repr=False)


@dataclass
class ActivityLog:
    """Every tool call of one turn, in the order the model asked for them.

    Owned by whatever is presenting the conversation, and cleared by it at the
    start of each turn through `begin_turn`. Deliberately not owned by
    `ConversationMemory`: that object answers questions about what the model was
    *shown* and lives for the whole conversation, and this one answers what
    happened *just now*. D9 split `last_search` from `seen_variant_ids` over
    exactly this distinction, and merging a per-turn record into a
    per-conversation one would be the same mistake a third time.
    """

    calls: list[ToolCallRecord] = field(default_factory=list)

    def begin_turn(self) -> None:
        self.calls = []

    def record(self, record: ToolCallRecord) -> None:
        self.calls.append(record)

    @property
    def failures(self) -> list[ToolCallRecord]:
        return [call for call in self.calls if not call.ok]

    def results_of(self, tool: str) -> list[Any]:
        """Every successful JSON payload `tool` returned on this turn.

        Decoded here rather than kept decoded, because most calls are never
        asked about and a tool is allowed to answer prose — `get_time` does.
        Anything that is not JSON contributes nothing and is not an error,
        which is the rule `ConversationMemory.observe` already follows.
        """
        payloads = []
        for call in self.calls:
            if call.name != tool or not call.ok:
                continue
            try:
                payloads.append(json.loads(call.content))
            except (ValueError, TypeError):
                continue
        return payloads


def summarise_arguments(raw_args: Any) -> str:
    """A tool call's arguments as one short readable line.

    The model sends a JSON string and `ToolRegistry.dispatch` is what decodes
    it; this decodes again only to re-encode it compactly, and hands back the
    original string when it cannot — a payload that will not parse is exactly
    what somebody debugging a malformed call needs to see verbatim.
    """
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args or "{}")
        except ValueError:
            return _clip(raw_args)
    if isinstance(raw_args, dict) and not raw_args:
        return ""
    try:
        return _clip(json.dumps(raw_args, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError):
        return _clip(repr(raw_args))


def _clip(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) <= MAX_ARGUMENT_CHARS:
        return flat
    return f"{flat[:MAX_ARGUMENT_CHARS]}… ({len(text):,} chars)"


class RecordingRegistry:
    """A registry that files every dispatch into an `ActivityLog`.

    Everything but `dispatch` forwards, which is what lets the caller keep
    asking it for `openai_schemas()`, `names()` and `specs()`, and lets the gate
    keep reaching `memory` through it — the same shape and the same reason as
    `TracedRegistry`.
    """

    def __init__(self, registry: Any, log: ActivityLog) -> None:
        self._registry = registry
        self._log = log

    def __getattr__(self, name: str) -> Any:
        return getattr(self._registry, name)

    @property
    def activity(self) -> ActivityLog:
        return self._log

    def dispatch(self, name: str, raw_args: Any = None) -> ToolResult:
        started = time.perf_counter()
        result = self._registry.dispatch(name, raw_args)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        record = ToolCallRecord(
            name=name,
            arguments=summarise_arguments(raw_args),
            ok=result.ok,
            duration_ms=elapsed_ms,
            error=None if result.ok else (result.error or "the tool refused"),
            result_chars=len(result.content),
            content=result.content if result.ok else "",
        )
        self._log.record(record)
        return result
