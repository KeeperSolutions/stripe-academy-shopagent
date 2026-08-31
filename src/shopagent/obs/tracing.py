"""Langfuse, behind a facade that cannot break a conversation (D10, step 2).

Three properties, and each is a decision rather than a convenience.

**Unconfigured is a normal state.** No keys means no tracing and nothing else
changes — the same shape `payments/stripe_svc.py` gives a missing Stripe key,
and for the same reason: a cart that cannot be browsed because an observability
vendor was not configured would be the wrong failure. There is no startup check
and no warning at import, only a note in the CLI banner so nobody demos an
untraced session believing otherwise.

**Nothing here may raise.** Every call into the SDK is caught, because the
alternative is a diagnostic tool causing the outage it is meant to describe.
`warn_on_account_mismatch` in `payments/` already carries that argument; this
is the same one with more surface, since a trace exporter talks to a host over
the network on a background thread and can fail in ways nothing in this process
chose. The first failure is logged once, at WARNING, and then the tracer stops
trying for the rest of the session — a conversation that logs a stack trace per
turn is one nobody can read.

**No structured-log fallback.** The plan offers one and it is declined, for a
reason worth stating rather than skipping. The CLI already prints every tool
call, every result, the tokens, the cost per turn and the session total, and
the MCP server already logs every call with its arguments and duration. A
fallback would be a third rendering of the same facts, in a third format, kept
in step with two others by hand — which is the thing this repository refuses
everywhere else. What a fallback would genuinely add over those two is one
thing only: a *trace*, the nesting that says this generation happened inside
that turn, and a flat log line cannot carry it. So the honest answer to "no
keys" is that there is no trace, not that there is a worse one.

**The trace is the conversation.** One root observation per session, opened
when the REPL starts and closed when it ends; a `generation` per LLM call
nested inside it, a `tool` per dispatch, a `guardrail` when one refuses. That
is the plan's third requirement taken literally. Langfuse ingests observations
as they arrive rather than waiting for the root to close, and `flush()` runs
after every turn, so a conversation in progress is visible in the UI while it
is still going.

**Cost is sent, not inferred.** `cost_details` carries the number
`llm/usage.py` computed from its own table. Langfuse prices what it recognises,
and it recognises none of `gpt-5.6-luna`, `gpt-5.6-sol` or `gpt-5.6-terra` — so
a trace that let the vendor guess would report a confident zero next to a CLI
saying $0.0024. Sending our own number also means the two can be compared,
which is the check that catches this file double-counting a retry.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack, contextmanager
from typing import Any, Iterator

from shopagent.config import get_settings
from shopagent.obs import redaction

logger = logging.getLogger(__name__)

# What a conversation's root observation is called in the UI.
CONVERSATION = "conversation"

# The note the CLI prints when nothing is configured. Here rather than in
# `llm/loop.py` because the condition it describes belongs to this module.
UNCONFIGURED_NOTE = "tracing off (no Langfuse keys)"


class Observation:
    """A handle on one span, which is allowed to be nothing at all.

    Returned by every method below, including when the tracer is off or the SDK
    raised. That is what lets callers write straight-line code — `span.update(...)`,
    `span.end()` — with no branch on whether tracing is happening, which is the
    only way an instrumentation layer stays out of the way of what it measures.
    """

    __slots__ = ("_span", "_tracer")

    def __init__(self, tracer: "Tracer", span: Any = None) -> None:
        self._tracer = tracer
        self._span = span

    def update(self, **fields: Any) -> None:
        if self._span is None:
            return
        self._tracer._guard(lambda: self._span.update(**fields), "update an observation")

    def end(self, **fields: Any) -> None:
        if self._span is None:
            return
        if fields:
            self.update(**fields)
        self._tracer._guard(self._span.end, "end an observation")
        self._span = None


class Tracer:
    """The whole of this project's contact with Langfuse.

    Built through `build_tracer()`, which returns one with no client when the
    keys are absent. Every method is safe to call on either.
    """

    def __init__(self, client: Any = None) -> None:
        self._client = client
        # Set once, the first time the SDK misbehaves. From then on this tracer
        # is inert: a broken exporter fails on every span, and a conversation
        # carrying one warning per turn is one nobody reads to the end.
        self._broken = False

    @property
    def enabled(self) -> bool:
        return self._client is not None and not self._broken

    # --- the three shapes of observation ---------------------------------

    @contextmanager
    def conversation(self, *, shopper_id: str | None, model: str) -> Iterator[Observation]:
        """The root observation. Everything else nests inside it.

        `start_as_current_observation` rather than `start_observation`, and the
        difference is the whole feature: only the "current" form puts the span
        in the OTEL context, and without it every tool and generation below
        starts a *new trace* of its own. The plan's third requirement — the
        whole conversation as one trace — is that one word.

        `user_id` is the shopper's identifier through `redact_identifier`, so
        Langfuse still groups a person's conversations together while nothing
        in the UI says who they are. It is set through `propagate_attributes`,
        which puts it on every child span, so a span added later cannot forget
        it.
        """
        if not self.enabled:
            yield Observation(self)
            return

        opened = ExitStack()
        span = None
        try:
            span = opened.enter_context(
                self._client.start_as_current_observation(
                    name=CONVERSATION,
                    as_type="agent",
                    metadata={"model": model, "redacted": redaction.redacting()},
                )
            )
            opened.enter_context(
                _propagate(
                    user_id=redaction.redact_identifier(shopper_id),
                    trace_name=CONVERSATION,
                )
            )
        except Exception as exc:  # noqa: BLE001 - see the module docstring
            self._give_up("start the conversation observation", exc)
            opened.close()
            opened = ExitStack()
            span = None

        try:
            yield Observation(self, span)
        finally:
            # Closed here rather than by the caller, so a conversation that
            # ended on an exception still closes its root — an unfinished trace
            # is the one a reader most needs, because it is where the
            # conversation stopped.
            self._guard(opened.close, "end the conversation observation")
            self.flush()

    def generation(self, *, name: str, messages: Any, model: str) -> Observation:
        """One LLM call. Closed by the caller with the reply and the usage."""
        if not self.enabled:
            return Observation(self)
        span = self._guard(
            lambda: self._client.start_observation(
                name=name,
                as_type="generation",
                model=model,
                input=redaction.redact_messages(messages),
            ),
            "start a generation",
        )
        return Observation(self, span)

    def tool(self, *, name: str, arguments: Any) -> Observation:
        """One tool dispatch. Closed by the caller with the result."""
        if not self.enabled:
            return Observation(self)
        span = self._guard(
            lambda: self._client.start_observation(
                name=name,
                as_type="tool",
                input=redaction.redact_arguments(arguments),
            ),
            "start a tool observation",
        )
        return Observation(self, span)

    def guardrail(self, *, name: str, outcome: str, detail: Any = None) -> None:
        """A rule that refused something, recorded as a point in the trace.

        Langfuse has an observation type for exactly this, which is why it is
        one rather than a metadata field on whatever it interrupted: the four
        questions a trace has to answer include "did a guardrail fire, and
        which", and an answer buried in another span's metadata is one nobody
        finds.

        `detail` is expected to be structured — the amounts that could not be
        traced, the variant id nobody was shown — and never the model's prose.
        Amounts and ids are this shop's own data and pass the redaction rule on
        purpose; the sentence they appeared in does not.
        """
        if not self.enabled:
            return
        span = self._guard(
            lambda: self._client.start_observation(
                name=name,
                as_type="guardrail",
                input=detail,
                output=outcome,
                level="WARNING",
            ),
            "start a guardrail observation",
        )
        if span is not None:
            self._guard(span.end, "end a guardrail observation")

    # --- housekeeping ----------------------------------------------------

    def flush(self) -> None:
        """Send whatever is queued. Called after every turn.

        Langfuse batches, so without this a conversation would appear in the UI
        only when the process exits — which is the wrong time for something
        whose purpose is watching a conversation happen.
        """
        if not self.enabled:
            return
        self._guard(self._client.flush, "flush")

    def shutdown(self) -> None:
        if self._client is None:
            return
        self._guard(self._client.shutdown, "shut down")

    def _guard(self, call, what: str) -> Any:
        """Run one SDK call, or give up on tracing for the rest of the session."""
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - see the module docstring
            self._give_up(what, exc)
            return None

    def _give_up(self, what: str, exc: BaseException) -> None:
        """Log the first failure and stop trying.

        Broad on purpose and for the reason `build_tool_setup` is broad: this
        reaches a vendor SDK, a background exporter thread and a network, and
        every failure from a bad key to a DNS timeout to a version mismatch
        arrives here. None of them is worth ending a conversation over, and
        none is worth a second warning — a broken exporter fails on every span,
        and a transcript carrying one stack trace per turn is one nobody reads
        to the end.
        """
        if not self._broken:
            logger.warning(
                "tracing is off for the rest of this session: could not %s (%s: %s)",
                what,
                type(exc).__name__,
                exc,
            )
        self._broken = True


def _propagate(**attributes: Any):
    """Imported lazily so this module loads with no Langfuse installed."""
    from langfuse import propagate_attributes

    return propagate_attributes(**attributes)


def build_tracer(**overrides: Any) -> Tracer:
    """The tracer this process will use, or an inert one.

    Inert when either key is absent, which is a state this project treats as
    ordinary. Inert *also* when building the client raises — an unusable
    observability vendor is not a reason to refuse to run a shop.

    `overrides` go straight to the SDK constructor and exist for one caller:
    a test handing in its own `span_exporter` so the real client can be
    exercised without a network.
    """
    settings = get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return Tracer()

    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            **overrides,
        )
    except Exception as exc:  # noqa: BLE001 - a missing or broken SDK is not fatal
        logger.warning(
            "tracing is off: could not build the Langfuse client (%s: %s)",
            type(exc).__name__,
            exc,
        )
        return Tracer()

    return Tracer(client)
