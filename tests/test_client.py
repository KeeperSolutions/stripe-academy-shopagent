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
