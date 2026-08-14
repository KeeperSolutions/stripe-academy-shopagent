"""Tests for shopagent.llm.client.

No network calls — the OpenAI client is replaced by a fake stream that mirrors
the shape of real chunks, including the final chunk carrying usage alongside an
empty `choices` list (confirmed against the live API on 2026-08-14).
"""

from types import SimpleNamespace

import pytest

from shopagent.llm import usage as usage_mod
from shopagent.llm.client import LLMClient
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
