"""Tests for shopagent.llm.client.

No network calls — the OpenAI client is replaced by a fake stream that mirrors
the shape of real chunks, including the final chunk carrying usage alongside an
empty `choices` list (confirmed against the live API on 2026-08-14).
"""

from types import SimpleNamespace

import pytest

from shopagent.llm import usage as usage_mod
from shopagent.llm.client import AssistantMessage, LLMClient, ToolCall
from shopagent.llm.usage import UsageTracker

MODEL = "fake-model"
MSGS = [{"role": "user", "content": "x"}]


def _chunk(text=None, usage=None):
    choices = []
    if text is not None:
        choices = [SimpleNamespace(delta=SimpleNamespace(content=text))]
    return SimpleNamespace(choices=choices, usage=usage)


def _usage(prompt=1_000, completion=500, cached=800):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )


FULL_STREAM = [
    _chunk("Hel"),
    _chunk("lo"),
    _chunk(" from "),
    _chunk("ShopAgent."),
    _chunk(None, usage=_usage()),  # final chunk: choices == [], carries usage
]


@pytest.fixture
def client(monkeypatch):
    """An LLMClient with a fake SDK and known prices."""
    monkeypatch.setitem(usage_mod.PRICING, MODEL, (1.0, 2.0, 0.10))
    monkeypatch.setattr(
        "shopagent.llm.client.OpenAI", lambda **kw: SimpleNamespace()
    )
    c = LLMClient(tracker=UsageTracker())
    c.model = MODEL
    return c


def _install_stream(client, chunks):
    client._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: iter(chunks))
        )
    )


def test_stream_yields_deltas_and_records_usage(client):
    _install_stream(client, FULL_STREAM)

    deltas = list(client.stream_chat(MSGS))

    assert deltas == ["Hel", "lo", " from ", "ShopAgent."]
    assert "".join(deltas) == "Hello from ShopAgent."
    assert len(client.tracker.calls) == 1
    call = client.tracker.calls[0]
    assert (call.prompt_tokens, call.completion_tokens, call.cached_tokens) == (
        1_000,
        500,
        800,
    )


def test_empty_choices_in_final_chunk_does_not_raise(client):
    """The final chunk has choices == []; chunk.choices[0] would IndexError."""
    _install_stream(client, FULL_STREAM)

    list(client.stream_chat(MSGS))  # must not raise

    assert len(client.tracker.calls) == 1


def test_interrupt_after_second_delta_still_records_the_call(client):
    """Ctrl+C / break mid-stream must not leave the tracker empty."""
    _install_stream(client, FULL_STREAM)

    gen = client.stream_chat(MSGS)
    received = []
    for i, delta in enumerate(gen):
        received.append(delta)
        if i == 1:  # interrupt after the second delta
            break
    gen.close()  # deterministically triggers the finally branch

    assert received == ["Hel", "lo"]
    assert client.tracker.calls, "tracker is empty — the call was lost"
    assert len(client.tracker.calls) == 1

    # Usage never arrived, so the record is incomplete — zeros, not made-up
    # numbers.
    call = client.tracker.calls[0]
    assert call.model == MODEL
    assert call.total_tokens == 0


def test_exception_mid_stream_still_records_the_call(client):
    def failing(**kw):
        yield _chunk("Hel")
        yield _chunk("lo")
        raise RuntimeError("connection dropped")

    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=failing))
    )

    with pytest.raises(RuntimeError):
        list(client.stream_chat(MSGS))

    assert len(client.tracker.calls) == 1


def test_usage_is_not_recorded_twice(client):
    """A cleanly finished stream must write exactly one record."""
    _install_stream(client, FULL_STREAM)

    list(client.stream_chat(MSGS))

    assert len(client.tracker.calls) == 1


def test_temperature_is_omitted_when_none(client):
    sent = {}

    def capture(**kw):
        sent.update(kw)
        return iter(FULL_STREAM)

    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=capture))
    )

    list(client.stream_chat(MSGS))
    assert "temperature" not in sent

    list(client.stream_chat(MSGS, temperature=0))
    assert sent["temperature"] == 0, "zero is falsy — it must not be swallowed"


# --- chat_with_tools (D2) ----------------------------------------------
#
# Non-streaming on purpose: assembling tool calls from streamed deltas means
# accumulating name and fragmented arguments per index, which D2 does not do.


def _sdk_tool_call(call_id, name, arguments, type_="function"):
    return SimpleNamespace(
        id=call_id,
        type=type_,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _response(content=None, tool_calls=None, usage=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=content, tool_calls=tool_calls
        ))],
        usage=usage if usage is not None else _usage(),
    )


def _install_response(client, response):
    sent = {}

    def capture(**kw):
        sent.update(kw)
        return response

    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=capture))
    )
    return sent


def test_chat_with_tools_returns_plain_text_when_no_tool_is_called(client):
    _install_response(client, _response(content="Hello."))

    reply = client.chat_with_tools(MSGS, tools=[])

    assert reply.content == "Hello."
    assert reply.tool_calls == []


def test_chat_with_tools_unpacks_id_name_and_raw_arguments(client):
    _install_response(client, _response(
        tool_calls=[_sdk_tool_call("call_1", "get_time", '{"timezone": "UTC"}')]
    ))

    reply = client.chat_with_tools(MSGS, tools=[{"type": "function"}])

    assert reply.tool_calls == [ToolCall("call_1", "get_time", '{"timezone": "UTC"}')]


def test_arguments_stay_a_raw_string(client):
    """The registry validates them; parsing here would duplicate that, badly."""
    _install_response(client, _response(
        tool_calls=[_sdk_tool_call("call_1", "echo", "{not json")]
    ))

    reply = client.chat_with_tools(MSGS, tools=[{"type": "function"}])

    assert reply.tool_calls[0].arguments == "{not json"


def test_content_may_be_none_alongside_tool_calls(client):
    _install_response(client, _response(
        content=None, tool_calls=[_sdk_tool_call("call_1", "get_time", "{}")]
    ))

    reply = client.chat_with_tools(MSGS, tools=[{"type": "function"}])

    assert reply.content is None
    assert len(reply.tool_calls) == 1


def test_chat_with_tools_records_usage_including_cached_tokens(client):
    _install_response(client, _response(content="Hi"))

    reply = client.chat_with_tools(MSGS, tools=[])

    assert len(client.tracker.calls) == 1
    assert reply.usage is client.tracker.calls[0]
    assert (reply.usage.prompt_tokens, reply.usage.cached_tokens) == (1_000, 800)


def test_tools_parameter_is_omitted_when_there_are_none(client):
    """An empty `tools` list is a 400, not "no tools"."""
    sent = _install_response(client, _response(content="Hi"))

    client.chat_with_tools(MSGS, tools=[])
    assert "tools" not in sent

    client.chat_with_tools(MSGS, tools=None)
    assert "tools" not in sent

    client.chat_with_tools(MSGS, tools=[{"type": "function"}])
    assert sent["tools"] == [{"type": "function"}]


def test_reasoning_effort_is_sent_with_tools_when_configured(client):
    """gpt-5.6-luna rejects function tools unless reasoning_effort is 'none'."""
    client.reasoning_effort = "none"
    sent = _install_response(client, _response(content="Hi"))

    client.chat_with_tools(MSGS, tools=[{"type": "function"}])

    assert sent["reasoning_effort"] == "none"


def test_reasoning_effort_is_omitted_when_unset(client):
    """Models that do not know the parameter return 400 if it is sent."""
    client.reasoning_effort = None
    sent = _install_response(client, _response(content="Hi"))

    client.chat_with_tools(MSGS, tools=[{"type": "function"}])

    assert "reasoning_effort" not in sent


def test_non_function_tool_calls_are_ignored(client):
    """The response type is a union; only function calls are dispatchable."""
    _install_response(client, _response(tool_calls=[
        _sdk_tool_call("call_1", "get_time", "{}"),
        SimpleNamespace(id="call_2", type="custom", custom=SimpleNamespace(input="x")),
    ]))

    reply = client.chat_with_tools(MSGS, tools=[{"type": "function"}])

    assert [c.id for c in reply.tool_calls] == ["call_1"]


def test_to_message_carries_tool_calls_in_the_api_shape(client):
    """Replay the call ids verbatim, or the matching tool messages are orphans."""
    reply = AssistantMessage(
        content=None,
        tool_calls=[ToolCall("call_1", "get_time", '{"timezone": "UTC"}')],
        usage=client.tracker.record(MODEL, 1, 1),
    )

    assert reply.to_message() == {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_time", "arguments": '{"timezone": "UTC"}'},
        }],
    }


def test_to_message_omits_the_key_entirely_when_there_are_no_tool_calls(client):
    reply = AssistantMessage(
        content="Hello.", tool_calls=[], usage=client.tracker.record(MODEL, 1, 1)
    )

    assert reply.to_message() == {"role": "assistant", "content": "Hello."}


# --- embeddings --------------------------------------------------------

EMBEDDING_MODEL = "fake-embedding-model"


def _install_embeddings(client, vectors, *, prompt_tokens=42, shuffle=False):
    """Fake the embeddings endpoint, recording the request it was given.

    `shuffle` returns the items out of order with their `index` fields intact,
    which is the case `embed` sorts for. The real API returns them in order;
    that it is documented to be keyed by `index` is the reason not to rely on
    it.
    """
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        data = [
            SimpleNamespace(index=position, embedding=vector)
            for position, vector in enumerate(vectors)
        ]
        if shuffle:
            data = list(reversed(data))
        return SimpleNamespace(
            data=data,
            usage=SimpleNamespace(prompt_tokens=prompt_tokens),
        )

    client._client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    return calls


@pytest.fixture
def embedding_client(client):
    """The same fake client, priced as an embedding model."""
    usage_mod.PRICING[EMBEDDING_MODEL] = (0.02, 0.00, 0.02)
    client.embedding_model = EMBEDDING_MODEL
    yield client
    usage_mod.PRICING.pop(EMBEDDING_MODEL, None)


def test_embed_sends_the_whole_batch_as_one_request(embedding_client):
    calls = _install_embeddings(embedding_client, [[0.1], [0.2], [0.3]])

    vectors, _ = embedding_client.embed(["a", "b", "c"])

    assert len(calls) == 1
    assert calls[0]["input"] == ["a", "b", "c"]
    assert vectors == [[0.1], [0.2], [0.3]]


def test_embed_uses_the_embedding_model_not_the_chat_model(embedding_client):
    calls = _install_embeddings(embedding_client, [[0.1]])

    embedding_client.embed(["a"])

    assert calls[0]["model"] == EMBEDDING_MODEL
    assert calls[0]["model"] != embedding_client.model


def test_embed_returns_vectors_in_index_order_whatever_order_they_arrive_in(
    embedding_client,
):
    _install_embeddings(embedding_client, [[0.1], [0.2], [0.3]], shuffle=True)

    vectors, _ = embedding_client.embed(["a", "b", "c"])

    assert vectors == [[0.1], [0.2], [0.3]]


def test_embed_records_usage_against_the_embedding_model(embedding_client):
    _install_embeddings(embedding_client, [[0.1], [0.2]], prompt_tokens=1_336)

    _, call = embedding_client.embed(["a", "b"])

    assert call.model == EMBEDDING_MODEL
    assert call.prompt_tokens == 1_336
    # An embedding call has nothing to complete, so nothing is billed as output.
    assert call.completion_tokens == 0
    assert embedding_client.tracker.total_tokens == 1_336
    assert call.cost_usd == pytest.approx(1_336 * 0.02 / 1_000_000)


def test_embed_does_not_land_in_unknown_models(embedding_client):
    _install_embeddings(embedding_client, [[0.1]])

    embedding_client.embed(["a"])

    assert embedding_client.tracker.unknown_models == set()


def test_embed_refuses_an_empty_batch_without_calling_the_api(embedding_client):
    """A request with no input is a 400 from the API and a bug here."""
    calls = _install_embeddings(embedding_client, [])

    with pytest.raises(ValueError, match="empty batch"):
        embedding_client.embed([])

    assert calls == []
    assert embedding_client.tracker.calls == []


# --- the request timeout (D10, step 4) ------------------------------------
#
# `OpenAI(api_key=...)` defaults to `read=600s` with two retries, which is not
# a default anybody chose: a connection the peer has dropped stalls a turn for
# up to thirty minutes with nothing printed. That is not hypothetical — a D10
# eval pass sat there for ten minutes before it was killed, with `lsof` showing
# no open socket to OpenAI and four in CLOSE_WAIT.


def test_the_client_carries_the_configured_timeout_and_retries():
    """Read from the client, not from a constant this file wrote.

    The same shape as D7's `STRIPE_API_VERSION` pin: asserting that a module
    constant equals itself proves nothing about the object that was built.
    """
    from shopagent.config import get_settings

    settings = get_settings()
    built = LLMClient()._client

    assert built.timeout.connect == settings.openai_connect_timeout_seconds
    assert built.timeout.read == settings.openai_read_timeout_seconds
    assert built.timeout.write == settings.openai_read_timeout_seconds
    assert built.timeout.pool == settings.openai_read_timeout_seconds
    assert built.max_retries == settings.openai_max_retries


def test_every_phase_is_bounded_by_something_this_project_chose():
    """`write` and `pool` too, and `pool` for the reason the outage happened.

    The stalled sockets were in CLOSE_WAIT, and waiting for one of those to
    free up is its own way to hang — a bound on `read` alone would have left
    that path open.
    """
    timeout = LLMClient()._client.timeout

    assert None not in (timeout.connect, timeout.read, timeout.write, timeout.pool)


def test_the_timeout_is_not_the_sdk_default():
    """The half of the claim the test above cannot make.

    D7 recorded why: comparing a client's effective value against the constant
    proves nothing on its own, because a client that was never configured
    resolves to *some* value too. What says this project chose it is that it
    differs from what the SDK would have used.
    """
    from openai._constants import DEFAULT_TIMEOUT

    from shopagent.config import get_settings

    assert get_settings().openai_read_timeout_seconds < DEFAULT_TIMEOUT.read
    assert LLMClient()._client.timeout.read < DEFAULT_TIMEOUT.read


def test_a_dead_connection_is_given_up_on_inside_five_minutes():
    """The number nobody derives while reading three separate fields.

    `(connect + read) x (1 + retries)` is what a person waits when the
    connection has *stopped* — nothing arriving, which is the outage this
    setting exists for. It is the criterion the values were chosen against, so
    it is asserted rather than left in a comment: raising `read` to the SDK's
    600 fails here with the reason attached.

    **It is not a request deadline, and the name says stall rather than wait
    for that reason.** `httpx.Timeout` is per phase and measures inactivity, so
    a response arriving slowly can outlast `read` legitimately, and the SDK's
    backoff between retries adds more. Asserting a total bound would mean
    building one — a deadline around the call — which is a change this project
    has not made and should not pretend to have. Raised by review on PR #10.

    The dominant term is the retries: lowering `OPENAI_MAX_RETRIES` shortens
    this at the cost of recovering from a transient 5xx, where lowering the
    read timeout starts cutting off answers that were going to arrive.
    """
    from shopagent.config import get_settings

    settings = get_settings()
    worst_case = (
        settings.openai_connect_timeout_seconds + settings.openai_read_timeout_seconds
    ) * (1 + settings.openai_max_retries)

    assert worst_case == 300.0
    assert worst_case <= 300.0, f"a dead connection would stall a turn for {worst_case}s"


def test_a_timed_out_turn_reaches_the_customer_as_a_sentence(monkeypatch, capsys):
    """A timeout that prints a traceback is half a fix.

    `_run_session` already catches, rolls the broken turn back and prints one
    line; this pins that the timeout arrives there rather than escaping. The
    message is terse — "APITimeoutError: Request timed out." — but it is a
    message, the prompt comes back, and the conversation is not corrupted.
    """
    import builtins
    from contextlib import ExitStack

    import httpx
    import openai

    from shopagent.agent import profile as profiles
    from shopagent.llm.loop import _run_session, build_tool_setup

    class TimesOut:
        model = "gpt-5.6-luna"

        def chat_with_tools(self, messages, tools=None):
            raise openai.APITimeoutError(
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
            )

    typed = iter(["what is in my cart?", "/exit"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(typed))
    monkeypatch.setattr(profiles, "load_for_session", lambda shopper_id: (None, None))

    with ExitStack() as stack:
        setup = build_tool_setup(stack, catalog_enabled=False)
        _run_session(TimesOut(), UsageTracker(), setup)

    printed = capsys.readouterr().out
    assert "[error] APITimeoutError" in printed
    assert "Traceback" not in printed
