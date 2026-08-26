"""Tests for shopagent.mcp_server.server (D4, step 1).

Unmarked: no database, no network, no subprocess. The server object is built at
import time, and the SDK can list and call its tools in-process, so the whole
protocol surface is testable without a transport.

Two ways in, and the difference matters. `Client(server)` runs the SDK's
in-memory transport: requests go through the same handlers a stdio client
reaches, so a failure comes back as `is_error=True` exactly as it would on the
wire. `server.call_tool(...)` is the bare Python API underneath that, and it
*raises* `ToolError` instead. Step 3 depends on that distinction, so it is
pinned here rather than discovered later.

These tests are about the transport contract only. There is one tool and it
returns a constant; the assertions worth writing about schema quality arrive
with the real tools in step 2.
"""

from __future__ import annotations

import anyio
import pytest
from mcp.client.client import Client
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from shopagent.mcp_server.server import SERVER_NAME, ping, server


def call(fn):
    """Run one async SDK call from a sync test.

    The project has no async test plugin and this step does not justify adding
    one: `anyio` is already a transitive dependency of `mcp`, and `anyio.run`
    turns each coroutine into an ordinary function call.
    """
    return anyio.run(fn)


def test_server_is_an_mcp_server():
    assert isinstance(server, MCPServer)


def test_server_name_is_the_recognisable_one():
    """The name is what a client shows in its server list, so it is pinned."""
    assert server.name == SERVER_NAME
    assert SERVER_NAME == "shopagent-catalog"


def test_ping_is_callable_without_the_server():
    """The tool is a plain function; the decorator only registers it.

    Same rule as `tools/basic.py` in CLAUDE.md — a tool stays testable without
    the machinery that wraps it.
    """
    assert ping() == "pong"


def test_ping_is_registered_under_its_own_name():
    tools = call(server.list_tools)
    assert [tool.name for tool in tools] == ["ping"]


def test_ping_advertises_a_schema_with_no_arguments():
    """An argument-free tool still needs a schema, and it must not invent one."""
    (tool,) = call(server.list_tools)

    assert tool.input_schema["type"] == "object"
    assert tool.input_schema.get("properties", {}) == {}
    assert tool.input_schema.get("required", []) == []


def test_ping_description_comes_from_the_docstring():
    """The docstring is the contract the model reads; MCP derives it from here.

    Asserting the first line rather than the whole string keeps this from
    breaking every time the prose is edited, while still failing loudly if the
    docstring stops reaching the schema at all.
    """
    (tool,) = call(server.list_tools)

    assert tool.description is not None
    assert tool.description.startswith("Check that the catalog server is reachable.")


def test_ping_over_the_in_memory_transport_returns_pong():
    """The end-to-end path a stdio client takes, minus the pipe."""

    async def scenario():
        async with Client(server) as client:
            return await client.call_tool("ping", {})

    result = call(scenario)

    assert result.is_error is False
    assert [content.text for content in result.content] == ["pong"]


def test_unknown_tool_is_an_error_result_not_a_crash():
    """A client asking for a tool that is not there must get `is_error`.

    This is the shape step 3 has to reproduce deliberately: an error the client
    can see as an error, rather than a successful result whose text happens to
    describe a failure.
    """

    async def scenario():
        async with Client(server) as client:
            return await client.call_tool("no_such_tool", {})

    result = call(scenario)

    assert result.is_error is True
    assert "no_such_tool" in result.content[0].text


def test_listing_tools_over_the_transport_matches_the_direct_listing():
    """What the wire advertises is what the server holds — no adapter in between."""

    async def scenario():
        async with Client(server) as client:
            return await client.list_tools()

    assert [tool.name for tool in call(scenario).tools] == ["ping"]


def test_direct_call_tool_raises_where_the_transport_reports():
    """The bare API raises; the transport converts. Step 3 turns on this.

    `server.call_tool` is the layer below the protocol handler, and it lets a
    `ToolError` out. The handler above it is what catches that and produces
    `is_error=True`. A tool that wants the client to see a failure therefore
    raises — it does not return a string describing the problem.
    """
    with pytest.raises(ToolError, match="no_such_tool"):
        call(lambda: server.call_tool("no_such_tool", {}))
