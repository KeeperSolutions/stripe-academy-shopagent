"""Tests for shopagent.mcp_client (D5, step 1).

Split the way the two halves differ. The adapter is pure translation, so it is
tested offline against tools invented here — which is also the proof that it is
not keying off our own server's tool names. The client spawns the real D4 server
as a subprocess, and that server reads Postgres, so those carry the `db` marker.

The last test in this file is the one worth keeping honest about: it asks the
operating system whether the server process is gone, rather than trusting the
client to say so. A leaked subprocess is invisible for an hour and then is not.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

import pytest

from shopagent.mcp_client.adapter import (
    EMPTY_RESULT,
    MISSING_DESCRIPTION,
    is_error,
    result_to_content,
    to_openai_tool,
    to_openai_tools,
)
from shopagent.mcp_client.client import MCPToolClient

EXPECTED_TOOLS = {"ping", "search_products", "get_product_details", "check_stock"}


# --- stand-ins, so the adapter is tested against tools it has never seen ---


@dataclass
class FakeTool:
    """An MCP tool as the adapter reads one: a name, a description, a schema."""

    name: str
    description: str | None
    input_schema: dict[str, Any]


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeImageBlock:
    data: str = "<base64>"
    type: str = "image"


@dataclass
class FakeResult:
    content: list[Any] = field(default_factory=list)
    structured_content: dict[str, Any] | None = None
    is_error: bool = False


# --- the adapter ----------------------------------------------------------


def test_a_tool_becomes_the_nested_chat_completions_shape():
    """The exact shape `ToolSpec.to_openai_schema` produces, so the loop cannot tell."""
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}

    converted = to_openai_tool(FakeTool("do_thing", "Does the thing.", schema))

    assert converted == {
        "type": "function",
        "function": {
            "name": "do_thing",
            "description": "Does the thing.",
            "parameters": schema,
        },
    }


def test_the_input_schema_is_passed_through_untouched():
    """Not rewritten, not copied — a doctored schema is one the server disagrees with."""
    schema = {"type": "object", "properties": {}, "title": "Args"}

    converted = to_openai_tool(FakeTool("t", "d", schema))

    assert converted["function"]["parameters"] is schema


def test_a_tool_without_a_description_gets_a_visible_placeholder():
    """MCP allows no description; an empty string would hide that from us."""
    converted = to_openai_tool(FakeTool("t", None, {"type": "object"}))

    assert converted["function"]["description"] == MISSING_DESCRIPTION


def test_the_adapter_converts_tools_it_has_never_heard_of():
    """The point of the whole step: nothing here knows our server's tool names.

    These four are invented. If the adapter ever grows a special case for a
    real tool name, this keeps passing — but the same fabricated input is what
    makes such a case obvious in review, and the server-side test below is what
    proves the real tools also come through.
    """
    invented = [
        FakeTool("book_flight", "Books a flight.", {"type": "object", "properties": {}}),
        FakeTool("feed_cat", "Feeds the cat.", {"type": "object", "properties": {}}),
        FakeTool("launch_probe", None, {"type": "object", "properties": {}}),
        FakeTool("zzz_last", "Sorts last.", {"type": "object", "properties": {}}),
    ]

    converted = to_openai_tools(invented)

    assert [entry["function"]["name"] for entry in converted] == [
        "book_flight",
        "feed_cat",
        "launch_probe",
        "zzz_last",
    ]
    assert all(entry["type"] == "function" for entry in converted)


def test_an_empty_tool_list_converts_to_an_empty_list():
    """A server offering nothing is not an error, and must not crash the adapter."""
    assert to_openai_tools([]) == []


# --- results, on the way back --------------------------------------------


def test_a_text_result_becomes_the_tool_message_content():
    assert result_to_content(FakeResult(content=[FakeTextBlock('{"count": 1}')])) == '{"count": 1}'


def test_several_text_blocks_are_joined():
    """A list return arrives as one block per item; the model needs all of them."""
    result = FakeResult(content=[FakeTextBlock("first"), FakeTextBlock("second")])

    assert result_to_content(result) == "first\nsecond"


def test_non_text_blocks_are_skipped():
    """An image has no place in a `tool` message, and stringifying it is noise."""
    result = FakeResult(content=[FakeImageBlock(), FakeTextBlock("the answer")])

    assert result_to_content(result) == "the answer"


def test_structured_content_is_the_fallback_when_there_is_no_text():
    """Content first, structured second — see the adapter for why that order."""
    result = FakeResult(content=[], structured_content={"count": 0, "results": []})

    assert result_to_content(result) == '{"count": 0, "results": []}'


def test_a_result_carrying_nothing_still_produces_text():
    """A `tool` message with empty content is a turn the model cannot read."""
    assert result_to_content(FakeResult()) == EMPTY_RESULT


def test_is_error_reads_the_flag_and_defaults_to_success():
    assert is_error(FakeResult(is_error=True)) is True
    assert is_error(FakeResult(is_error=False)) is False
    assert is_error(object()) is False


# --- against the real server ---------------------------------------------


def _child_pids() -> set[int]:
    """The PIDs this process has spawned, straight from the OS."""
    listed = subprocess.run(
        ["pgrep", "-P", str(os.getpid())], capture_output=True, text=True
    ).stdout
    return {int(pid) for pid in listed.split()}


def _is_alive(pid: int) -> bool:
    """Whether a PID still exists. Signal 0 checks without delivering anything."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


@pytest.fixture
def client(engine):
    """A started client, torn down whatever the test does.

    Depends on `engine` only so the test skips with a useful message when
    Postgres is down: the server would start fine and then fail every catalog
    call, which reads as a client bug.
    """
    with MCPToolClient() as started:
        yield started


@pytest.mark.db
def test_the_client_lists_every_tool_the_server_offers(client):
    """Dynamically — the client asks, it does not assume."""
    names = {tool.name for tool in client.list_tools()}

    assert names == EXPECTED_TOOLS


@pytest.mark.db
def test_every_listed_tool_carries_a_description_and_a_schema(client):
    """What the adapter needs from each one, checked at the source."""
    for tool in client.list_tools():
        assert tool.description, f"{tool.name} has no description"
        assert tool.input_schema["type"] == "object"


@pytest.mark.db
def test_the_real_tools_convert_without_special_cases(client):
    """The fabricated-tool test proves generality; this proves it on real input."""
    converted = to_openai_tools(client.list_tools())

    assert {entry["function"]["name"] for entry in converted} == EXPECTED_TOOLS
    for entry in converted:
        assert entry["type"] == "function"
        assert entry["function"]["description"] != MISSING_DESCRIPTION
        assert entry["function"]["parameters"]["type"] == "object"


@pytest.mark.db
def test_search_products_through_the_client_returns_real_products(client):
    """The whole path: sync call, portal, subprocess, Postgres, and back.

    `query` is left out on purpose — the MCP tool always embeds a query, so
    passing one here would spend money on every run.
    """
    result = client.call_tool("search_products", {"category": "shoes", "limit": 3})

    assert is_error(result) is False
    payload = result.structured_content
    assert 1 <= payload["count"] <= 3
    assert payload["count"] == len(payload["results"])
    for product in payload["results"]:
        assert product["category"] == "shoes"
        assert product["variants"]
        assert isinstance(product["variants"][0]["price_cents"], int)


@pytest.mark.db
def test_a_result_from_the_real_server_renders_as_tool_message_text(client):
    """What the loop will actually put in front of the model."""
    result = client.call_tool("search_products", {"category": "shoes", "limit": 1})

    content = result_to_content(result)

    assert content.strip()
    assert '"count"' in content


@pytest.mark.db
def test_an_unknown_product_id_comes_back_as_an_error(client):
    """The D4 contract, seen from the other side of the pipe."""
    result = client.call_tool("get_product_details", {"product_id": 10_000_000})

    assert is_error(result) is True
    assert "search_products" in result_to_content(result)


@pytest.mark.db
def test_a_successful_call_is_not_reported_as_an_error(client):
    """The other half of the flag, so `is_error` is not trivially always true."""
    assert is_error(client.call_tool("ping", {})) is False


@pytest.mark.db
def test_the_server_process_does_not_outlive_the_client(engine):
    """Asked of the operating system, not of the client.

    The client claiming a clean shutdown is exactly what a leak would also
    claim. `pgrep` names the process the client spawned, and `kill -0` after
    the block says whether it is still there.
    """
    before = _child_pids()

    with MCPToolClient() as started:
        started.list_tools()
        spawned = _child_pids() - before

    assert spawned, "expected the client to have spawned a server process"
    for pid in spawned:
        assert not _is_alive(pid), f"server process {pid} outlived the client"


@pytest.mark.db
def test_the_server_process_dies_even_when_the_block_raises(engine):
    """The path that actually leaks in production: an exception mid-conversation."""
    before = _child_pids()
    spawned: set[int] = set()

    with pytest.raises(RuntimeError, match="something went wrong"):
        with MCPToolClient() as started:
            started.list_tools()
            spawned = _child_pids() - before
            raise RuntimeError("something went wrong")

    assert spawned, "expected the client to have spawned a server process"
    for pid in spawned:
        assert not _is_alive(pid), f"server process {pid} survived an exception"


# --- using the client wrongly --------------------------------------------


def test_calling_before_starting_says_so():
    """A clear message beats an AttributeError on a None session."""
    client = MCPToolClient()

    with pytest.raises(RuntimeError, match="not started"):
        client.list_tools()


def test_closing_a_client_that_never_started_is_harmless():
    """`__exit__` runs whatever happened inside the block, including a failure."""
    MCPToolClient().close()


@pytest.mark.db
def test_starting_twice_is_refused(client):
    """Two sessions behind one handle would leak the first server silently."""
    with pytest.raises(RuntimeError, match="already started"):
        client.start()


