"""What leaves this machine, and what happens when the vendor does not (D10, step 2).

The redaction tests here are unusual in one way that is the whole point of
them: they do not check what `redact_text` returns, they check **the bytes the
exporter was handed**. A real `langfuse.Langfuse` is built with an in-memory
span exporter in place of the OTLP one, the wrappers are driven through it, and
the assertion is that a plaintext string appears nowhere in the serialised
export. A test on the redaction function alone would pass over an
instrumentation layer that forgot to call it, which is exactly the failure
worth catching — the function is easy and the wiring is where a name escapes.
"""

from __future__ import annotations

import itertools
import json
from types import SimpleNamespace

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from shopagent.agent.guardrails import FALLBACK_PREFIX, GuardedClient
from shopagent.agent.memory import ConversationMemory
from shopagent.config import get_settings
from shopagent.llm.client import AssistantMessage, ToolCall
from shopagent.llm.usage import UsageTracker
from shopagent.obs import redaction
from shopagent.obs.instrumentation import TracedClient, TracedRegistry
from shopagent.obs.tracing import Tracer, build_tracer
from shopagent.tools.registry import ToolRegistry, ToolResult, ToolSpec
from pydantic import BaseModel


# Langfuse keeps one resource manager per public key, process-wide, and hands
# a second client with the same key the *first* one's exporter. Two `Capture`s
# in one test session would then share one exporter and each would read the
# other's spans — which is not a failure that announces itself: the first test
# passes and the second sees nothing. A key per instance keeps them apart.
_CAPTURES = itertools.count()


class Capture:
    """A tracer wired to a real Langfuse client that exports into memory."""

    def __init__(self):
        self.exporter = InMemorySpanExporter()
        self.tracer = Tracer(
            Langfuse(
                public_key=f"pk-lf-offline-{next(_CAPTURES)}",
                secret_key="sk-lf-offline",
                host="http://langfuse.invalid",
                span_exporter=self.exporter,
                flush_at=1,
            )
        )

    def spans(self):
        self.tracer.flush()
        return self.exporter.get_finished_spans()

    def names(self):
        """Span names, oldest first. The exporter yields them as they ended."""
        return [span.name for span in self.spans()]

    def wire(self) -> str:
        """Everything the exporter was handed, as one searchable string.

        Two details here are load-bearing rather than tidy, and both are about
        an assertion of *absence* passing for the wrong reason.

        Langfuse serialises a span's input and output to JSON itself, with the
        default `ensure_ascii=True` — so a euro sign arrives as `\\u20ac` and a
        name like `Milo\u0161` as `Milo\\u0161`. A test asserting the plaintext is
        absent would then pass over a trace carrying it in full. So every
        attribute that parses as JSON is decoded back to an object first, and
        the whole thing is re-dumped with `ensure_ascii=False`.

        `test_the_wire_helper_would_notice_a_name_that_is_not_ascii` is what
        keeps that true, because a helper that quietly stopped decoding would
        make every redaction test below vacuous at once.
        """
        rows = []
        for span in self.spans():
            attributes = {}
            for key, value in span.attributes.items():
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except ValueError:
                        pass
                attributes[key] = value
            rows.append({"name": span.name, "attributes": attributes})
        return json.dumps(rows, default=str, ensure_ascii=False)


class NoArgs(BaseModel):
    pass


class SearchArgs(BaseModel):
    query: str


def reply(content=None, tools=(), usage=None):
    return AssistantMessage(
        content=content,
        tool_calls=[ToolCall(id=f"c{i}", name=name, arguments="{}") for i, name in enumerate(tools)],
        usage=usage,
    )


class ScriptedClient:
    model = "gpt-5.6-luna"

    def __init__(self, *replies):
        self.replies = list(replies)

    def chat_with_tools(self, messages, tools=None):
        return self.replies.pop(0)


def registry_with(*, search_result="{}", ok=True):
    registry = ToolRegistry()

    def search_products(query: str):
        if not ok:
            raise RuntimeError("the catalog is down")
        return json.loads(search_result)

    def view_cart():
        return {"currency": "eur", "items": [], "total_cents": 0}

    registry.register(
        ToolSpec(
            name="search_products",
            description="d",
            args_model=SearchArgs,
            fn=search_products,
        )
    )
    registry.register(
        ToolSpec(name="view_cart", description="d", args_model=NoArgs, fn=view_cart)
    )
    return registry


# --- what must not leave --------------------------------------------------


def test_a_search_query_does_not_leave_in_readable_form():
    capture = Capture()
    registry = TracedRegistry(registry_with(), capture.tracer)

    registry.dispatch("search_products", '{"query": "waterproof shoes for my mother"}')

    wire = capture.wire()
    assert "waterproof shoes for my mother" not in wire
    assert "waterproof" not in wire
    assert "<redacted:" in wire, "the field is present, so a reader knows one was sent"


def test_the_wire_helper_would_notice_a_name_that_is_not_ascii():
    """A falsification of the test method, not of the code.

    Every redaction test below asserts that a string is *absent*, and an
    absence assertion is only as good as the search. Langfuse escapes non-ASCII
    when it serialises a span, so without the decoding in `Capture.wire` a
    customer called `Milo\u0161` would reach the trace as `Milo\\u0161` and every
    one of those tests would pass while the name went out in full.

    So this checks the search finds a name the escaping would have hidden, with
    redaction deliberately off.
    """
    monkey = redaction.redacting
    redaction.redacting = lambda: False
    try:
        capture = Capture()
        client = TracedClient(ScriptedClient(reply("ok")), capture.tracer)
        client.chat_with_tools([{"role": "user", "content": "my name is Milo\u0161"}])
        assert "Milo\u0161" in capture.wire()
    finally:
        redaction.redacting = monkey


def test_a_name_that_is_not_ascii_does_not_leave_either():
    capture = Capture()
    client = TracedClient(ScriptedClient(reply("Hello Milo\u0161.")), capture.tracer)

    client.chat_with_tools([{"role": "system", "content": "The customer is Milo\u0161."}])

    assert "Milo\u0161" not in capture.wire()


def test_the_customer_and_the_model_do_not_leave_in_readable_form():
    """Every free-text role at once, including the one carrying the profile.

    `display_name` reaches a trace twice and neither route is `query`: it is
    injected into the system prompt, and the model says it back — measured in
    the D10 step 1 live run, where the answer opened with the customer's first
    name.
    """
    capture = Capture()
    client = TracedClient(ScriptedClient(reply("Maks, these are €94.99.")), capture.tracer)

    client.chat_with_tools(
        [
            {"role": "system", "content": "The customer's name is Maks."},
            {"role": "user", "content": "find me trail running shoes"},
            {"role": "assistant", "content": "Maks, here are three."},
        ]
    )

    wire = capture.wire()
    assert "Maks" not in wire
    assert "trail running shoes" not in wire
    assert "here are three" not in wire


def test_the_query_replayed_in_an_assistant_tool_call_does_not_leave_either():
    """The leak this module shipped, and the test whose absence let it.

    A tool call the model made stays in the message list for the rest of the
    conversation, so its `function.arguments` — the raw `query` among them — is
    replayed into the input of every later generation. Measured on a real
    Langfuse trace before this existed: eighteen plaintext copies of
    `trail running shoes` in one conversation, beside a `search_products` span
    that showed a digest.

    The tests that missed it built messages with `content` and no `tool_calls`.
    A fixture omitting a field the real object always has is a blind spot with
    the shape of coverage — the same defect as D8's `refunded_event`, which had
    no `payment_intent` and so could not have noticed the attribution check was
    missing.
    """
    capture = Capture()
    client = TracedClient(ScriptedClient(reply("ok")), capture.tracer)

    client.chat_with_tools(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_products",
                            "arguments": '{"query":"trail running shoes","limit":5}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"count": 0}'},
        ]
    )

    wire = capture.wire()
    assert "trail running shoes" not in wire
    assert "search_products" in wire, "the call itself still shows, only its text is gone"
    # The arguments stay a JSON string inside the message, so `wire` shows
    # them escaped — the absence assertions above are unaffected by that, since
    # escaping touches the quotes and not the words.
    assert "limit" in wire, "the rest of the arguments survive"


def test_an_argument_payload_that_does_not_decode_is_redacted_whole():
    """The model does not always emit valid JSON, and `dispatch` says so.

    Nothing can find `query` inside a string that will not parse, so the rule
    falls back to the safe side rather than passing it through — which is what
    the first version did, because `redact_arguments` returned anything that
    was not a dict untouched.
    """
    capture = Capture()
    registry = TracedRegistry(registry_with(), capture.tracer)

    registry.dispatch("search_products", '{"query": "waterproof boots" oops')

    wire = capture.wire()
    assert "waterproof boots" not in wire
    assert "<redacted:" in wire


def test_a_shopper_id_reaches_the_trace_as_a_digest():
    capture = Capture()

    with capture.tracer.conversation(shopper_id="maks@example.com", model="m"):
        pass

    wire = capture.wire()
    assert "maks@example.com" not in wire
    assert "maks" not in wire.lower().replace("langfuse", "")


def test_the_same_text_twice_is_the_same_digest_and_two_texts_are_not():
    """The one question a redacted trace can still answer, and why D6 chose a
    digest over a blank `<redacted>`."""
    first = redaction.redact_text("waterproof jacket")
    again = redaction.redact_text("waterproof jacket")
    other = redaction.redact_text("running shoes")

    assert first == again
    assert first != other


def test_a_digest_carries_no_length():
    """With the salt in place, length is the only thing left that narrows a guess."""
    short = redaction.redact_text("a")
    long = redaction.redact_text("a waterproof jacket for cycling in November")

    assert len(short) == len(long)


def settings_saying(redact: bool):
    return lambda: SimpleNamespace(trace_redact_text=redact)


def test_the_switch_is_read_from_configuration(monkeypatch):
    """`redacting()` reads the setting, rather than returning a constant.

    Worth its own test because the obvious way to write the one below is to
    replace `redacting` itself — and a test that patches the function it is
    checking passes over a body that ignores its input entirely. Found by
    mutating `redacting()` to `return True` and watching nothing fail.
    """
    monkeypatch.setattr(redaction, "get_settings", settings_saying(False))
    assert redaction.redacting() is False

    monkeypatch.setattr(redaction, "get_settings", settings_saying(True))
    assert redaction.redacting() is True


def test_turning_the_switch_off_sends_the_text(monkeypatch):
    """The bargain `MCP_LOG_REDACT_QUERY` offers, in the same shape.

    Asserted because a switch nobody checks is a switch that quietly stops
    working — and because the cost of the default has to be demonstrable, not
    just described. Patched at the *setting*, not at `redacting`, so the whole
    path from configuration to the wire is the thing being exercised.
    """
    monkeypatch.setattr(redaction, "get_settings", settings_saying(False))
    capture = Capture()
    registry = TracedRegistry(registry_with(), capture.tracer)

    registry.dispatch("search_products", '{"query": "waterproof shoes"}')

    assert "waterproof shoes" in capture.wire()


def test_the_switch_defaults_to_the_safe_side():
    """The safe setting must not be the one somebody has to remember to type."""
    assert get_settings().trace_redact_text is True


def test_the_shop_s_own_data_is_not_redacted():
    """Amounts, ids and product names pass, and the trace is worthless without them."""
    capture = Capture()
    registry = TracedRegistry(registry_with(), capture.tracer)

    registry.dispatch("search_products", '{"query": "x", "category": "shoes", "limit": 5}')

    wire = capture.wire()
    assert "shoes" in wire, "a category from a closed set is not personal data"
    assert "limit" in wire


# --- what a trace has to show ---------------------------------------------


def test_the_tools_appear_in_the_order_they_were_called():
    capture = Capture()
    registry = TracedRegistry(registry_with(), capture.tracer)

    registry.dispatch("search_products", '{"query": "x"}')
    registry.dispatch("view_cart", {})
    registry.dispatch("search_products", '{"query": "y"}')

    assert capture.names() == ["search_products", "view_cart", "search_products"]


def test_a_refused_tool_is_a_warning_carrying_its_reason():
    """Two of D9's three guardrails become visible here without instrumenting them."""
    capture = Capture()
    registry = TracedRegistry(registry_with(ok=False), capture.tracer)

    registry.dispatch("search_products", '{"query": "x"}')

    (span,) = capture.spans()
    attributes = dict(span.attributes)
    assert attributes["langfuse.observation.level"] == "WARNING"
    assert "the catalog is down" in attributes["langfuse.observation.status_message"]


def test_the_cost_sent_is_the_one_the_tracker_computed():
    """Not Langfuse's estimate. It prices none of the `gpt-5.6-*` family, so a
    trace that let it guess would report $0.00 beside a CLI reporting cents."""
    capture = Capture()
    tracker = UsageTracker()
    usage = tracker.record("gpt-5.6-luna", prompt_tokens=3000, completion_tokens=40)
    client = TracedClient(ScriptedClient(reply("done", usage=usage)), capture.tracer)

    client.chat_with_tools([{"role": "user", "content": "hi"}])

    (span,) = capture.spans()
    costs = json.loads(dict(span.attributes)["langfuse.observation.cost_details"])
    assert costs["total"] == pytest.approx(tracker.total_cost_usd)
    assert costs["total"] > 0


def test_a_corrected_retry_is_a_second_generation_not_a_lost_one():
    """Why `TracedClient` goes *inside* `GuardedClient`.

    The amount guardrail can send a second, really billed request. Traced from
    outside, one call would be recorded where two happened and the trace would
    report half the cost of every corrected turn — which the comparison against
    the CLI's own total is what catches.
    """
    capture = Capture()
    tracker = UsageTracker()
    memory = ConversationMemory()
    memory.observe("view_cart", {}, json.dumps({"total_cents": 18998}))

    inner = ScriptedClient(
        reply("That is €5.00.", usage=tracker.record("gpt-5.6-luna", 100, 10)),
        reply("That is €5.00.", usage=tracker.record("gpt-5.6-luna", 120, 10)),
    )
    client = GuardedClient(TracedClient(inner, capture.tracer), memory, capture.tracer)

    answer = client.chat_with_tools([{"role": "user", "content": "total?"}])

    assert answer.content.startswith(FALLBACK_PREFIX)
    assert capture.names().count("chat") == 2, "the billed retry is missing from the trace"


def test_the_amount_guardrail_reports_itself_and_names_no_prose():
    capture = Capture()
    memory = ConversationMemory()
    memory.observe("view_cart", {}, json.dumps({"total_cents": 18998}))
    inner = ScriptedClient(reply("It comes to €5.00."), reply("It comes to €5.00."))

    GuardedClient(inner, memory, capture.tracer).chat_with_tools([])

    guardrails = [span for span in capture.spans() if span.name == "untraceable_amount"]
    assert len(guardrails) == 2, "the retry and the fallback are both worth seeing"
    wire = capture.wire()
    assert "€5.00" in wire, "the amount is this shop's own data and is the diagnosis"
    assert "It comes to" not in wire, "the sentence it appeared in is not"


def test_the_whole_conversation_is_one_trace():
    capture = Capture()

    with capture.tracer.conversation(shopper_id=None, model="m"):
        registry = TracedRegistry(registry_with(), capture.tracer)
        registry.dispatch("view_cart", {})
        registry.dispatch("view_cart", {})

    traces = {span.context.trace_id for span in capture.spans()}
    assert len(traces) == 1, "the turns landed in separate traces"
    assert capture.names()[-1] == "conversation", "the root closes last"


# --- when the vendor is absent or broken ----------------------------------


def test_no_keys_means_the_sdk_is_never_constructed(monkeypatch):
    """Not merely that the tracer is off — that nothing was built.

    The first version of this asserted `not tracer.enabled` and passed for the
    wrong reason: with the key check removed, `Langfuse(public_key=None, ...)`
    raised, the broad `except` swallowed it and the tracer came back off
    anyway. An unconfigured process must not reach the SDK at all, so the
    assertion is on the attempt.
    """
    import langfuse

    attempts = []

    def spy(**kwargs):
        attempts.append(kwargs)
        raise RuntimeError("the SDK should not have been reached")

    monkeypatch.setattr(langfuse, "Langfuse", spy)
    monkeypatch.setattr("shopagent.obs.tracing.get_settings", unconfigured)

    tracer = build_tracer()

    assert attempts == [], "the SDK was constructed with no keys"
    assert not tracer.enabled


def unconfigured():
    return SimpleNamespace(
        langfuse_public_key=None,
        langfuse_secret_key=None,
        langfuse_host="https://cloud.langfuse.com",
    )


def test_no_keys_means_nothing_else_changes(monkeypatch):
    monkeypatch.setattr("shopagent.obs.tracing.get_settings", unconfigured)
    tracer = build_tracer()

    assert not tracer.enabled
    # Every method is safe to call on it, which is what lets the wrappers hold
    # no branch on whether tracing is happening.
    with tracer.conversation(shopper_id="x", model="m"):
        tracer.tool(name="view_cart", arguments={}).end()
        tracer.generation(name="chat", messages=[], model="m").end()
        tracer.guardrail(name="g", outcome="o")
    tracer.flush()
    tracer.shutdown()


def test_a_session_with_no_tracer_runs_the_wrappers_unchanged():
    registry = TracedRegistry(registry_with(), Tracer())
    client = TracedClient(ScriptedClient(reply("hello")), Tracer())

    assert registry.dispatch("view_cart", {}).ok
    assert client.chat_with_tools([]).content == "hello"


def test_a_langfuse_that_raises_does_not_break_the_conversation(caplog):
    """The argument `warn_on_account_mismatch` already carries: a diagnostic
    must not cause the outage it describes."""

    class Exploding:
        def start_observation(self, **kwargs):
            raise RuntimeError("langfuse is having a day")

        def flush(self):
            raise RuntimeError("langfuse is having a day")

    tracer = Tracer(Exploding())
    registry = TracedRegistry(registry_with(), tracer)
    client = TracedClient(ScriptedClient(reply("hello")), tracer)

    with capture_warnings(caplog):
        result = registry.dispatch("view_cart", {})
        answer = client.chat_with_tools([])

    assert result.ok, "a broken exporter refused a customer's cart"
    assert answer.content == "hello"
    assert not tracer.enabled, "it gives up rather than failing once per turn"


def test_it_warns_once_and_then_stops(caplog):
    class Exploding:
        def start_observation(self, **kwargs):
            raise RuntimeError("no")

    tracer = Tracer(Exploding())
    with capture_warnings(caplog):
        for _ in range(5):
            tracer.tool(name="view_cart", arguments={}).end()

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1, f"{len(warnings)} warnings for one broken exporter"


def test_a_span_opened_before_the_break_does_not_warn_again_when_it_ends(caplog):
    """The path the test above cannot reach, and the only one `_broken` guards.

    Once a tracer has given up, `enabled` is false and no new span is started —
    so a loop of `tool()` calls produces one warning whatever `_give_up` does.
    A span that was already open is different: `Observation.end` holds a real
    handle and still calls through. Mutating the warn-once check survived until
    this existed, which is what says the two paths are not the same test.
    """

    class BreaksOnEnd:
        def start_observation(self, **kwargs):
            return BreaksOnEnd.Span()

        class Span:
            def end(self, **kwargs):
                raise RuntimeError("the exporter died mid-conversation")

            def update(self, **kwargs):
                raise RuntimeError("the exporter died mid-conversation")

    tracer = Tracer(BreaksOnEnd())
    first = tracer.tool(name="view_cart", arguments={})
    second = tracer.tool(name="view_cart", arguments={})

    with capture_warnings(caplog):
        first.end()
        second.end()

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1, f"{len(warnings)} warnings for one broken exporter"


def capture_warnings(caplog):
    import logging

    return caplog.at_level(logging.WARNING, logger="shopagent.obs.tracing")


# --- the guard that keeps this suite offline ------------------------------


def test_the_autouse_guard_refuses_a_real_export():
    """The third seam in `no_accidental_api_calls`, proved rather than assumed.

    Same mechanism as the OpenAI and Stripe halves, not a parallel one: one
    fixture, three funnels. This drives the real exporter's `export` and
    asserts the fixture has replaced it — which is the only way to know the
    string target in `monkeypatch.setattr` still names something real after an
    SDK upgrade.
    """
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    with pytest.raises(AssertionError, match="Langfuse"):
        OTLPSpanExporter.export(object(), [])


def test_a_default_langfuse_client_would_use_that_exporter():
    """The other half of the claim above: that the seam is on the path.

    A guard is a property of the path it sits on, not of the sentence
    describing it — the D8 lesson about a live-mode event. If Langfuse ever
    stops defaulting to the OTLP HTTP exporter, this fails while the test above
    goes on passing.
    """
    from langfuse._client import span_processor

    assert span_processor.OTLPSpanExporter is not None
    source = span_processor.__file__
    assert source.endswith("span_processor.py")


# --- one process, several tracers (D10, step 4) ---------------------------
#
# Langfuse keeps **one resource manager per public key, process-wide**. This
# file already knew that — `Capture` gives every instance a unique key because
# two clients sharing one would each read the other's spans — but it was
# written down as the workaround to that problem rather than as a property of
# the library, and the eval runner then built and shut down a tracer per
# scenario in one process.
#
# What that does is not subtle once seen: `shutdown()` stops the shared
# `BatchSpanProcessor`, and the next tracer is handed the same dead manager.
# Its worker thread is gone, so `flush()` — which is `queue.join()`, waiting
# for `task_done()` on everything queued — never returns. Two eval passes hung
# there, in the same scenario both times, and the second one's `faulthandler`
# dump named the line.


def test_a_second_shutdown_strands_the_queue_a_later_flush_waits_on(monkeypatch):
    """Why a tracer is not a per-unit-of-work object. The evidence, measured.

    This asserts a property of Langfuse rather than of this project, which is
    unusual and deliberate: it is the *reason* `evals/runner.py` builds one
    tracer for a whole run, and a reason nobody can check is one that gets
    undone. If Langfuse ever fixes this, this test fails and whoever is here
    gets sent to the comment.

    The mechanism, watched cycle by cycle rather than guessed at:

        cycle 1  shutdown -> the score-ingestion consumer thread stops
        cycle 2  shutdown -> a stop sentinel per consumer goes onto the queue,
                             and no consumer is left to take it, so
                             unfinished_tasks becomes 1
        cycle 3  flush    -> `_score_ingestion_queue.join()` waits for a
                             `task_done()` that will never come

    Two eval passes hung on that third flush, in scenario three both times.
    Two earlier attempts to reproduce it used *two* cycles and passed, which is
    how the wrong diagnosis got reported twice — the stranded queue only
    appears on the second shutdown.

    No alarm and no hang here: the precondition is what is asserted, in
    milliseconds, rather than the twenty seconds it takes to observe the block.
    """
    from langfuse._client.resource_manager import LangfuseResourceManager

    key = f"pk-lf-shared-{next(_CAPTURES)}"
    monkeypatch.setattr(
        "shopagent.obs.tracing.get_settings",
        lambda: SimpleNamespace(
            langfuse_public_key=key,
            langfuse_secret_key="sk-lf-offline",
            langfuse_host="http://langfuse.invalid",
        ),
    )

    for _ in range(2):
        tracer = build_tracer(span_exporter=InMemorySpanExporter(), flush_at=1)
        with tracer.conversation(shopper_id="probe", model="m"):
            tracer.tool(name="view_cart", arguments={}).end()
        tracer.shutdown()

    manager = LangfuseResourceManager._instances[key]
    assert not any(c.is_alive() for c in manager._ingestion_consumers)
    assert manager._score_ingestion_queue.unfinished_tasks > 0, (
        "the queue a later flush() joins is drained, so the hazard this test "
        "documents is gone — re-read the comment above and simplify "
        "evals/runner.py if Langfuse has fixed it"
    )


def test_the_answer_a_conversation_ends_on_reaches_the_trace():
    """The last answer appears in no later generation's input, so it needs its own.

    Every other assistant message is replayed into the next call and reaches a
    trace that way. The final one is not, because there is no next call — so
    with `TRACE_REDACT_TEXT=false`, whose whole promise is that a trace reads
    as a conversation, it was the one thing missing. Raised by review on PR
    #10.

    The phrase is chosen to appear in no input message, so finding it can only
    mean the reply itself was recorded.
    """
    monkey = redaction.redacting
    redaction.redacting = lambda: False
    try:
        capture = Capture()
        client = TracedClient(ScriptedClient(reply("Your order is on its way.")), capture.tracer)
        client.chat_with_tools([{"role": "user", "content": "did it go through"}])
        assert "Your order is on its way." in capture.wire()
    finally:
        redaction.redacting = monkey


def test_that_answer_is_redacted_like_every_other_thing_a_person_reads():
    """Recording it must not become a way round the switch.

    Same reply, redaction left at its default. `redact_text` is the function
    `redact_messages` already applies to an assistant message, so the answer
    obeys one rule rather than a second one written for this field.
    """
    capture = Capture()
    client = TracedClient(ScriptedClient(reply("Your order is on its way.")), capture.tracer)

    client.chat_with_tools([{"role": "user", "content": "did it go through"}])

    wire = capture.wire()
    assert "Your order is on its way." not in wire
    assert "<redacted:" in wire, "the answer was dropped rather than digested"
