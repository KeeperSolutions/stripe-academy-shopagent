"""Tests for shopagent.mcp_client.client (D5, step 1).

The client spawns the real D4 server as a subprocess and that server reads
Postgres, so most of this carries the `db` marker.

The last test in this file is the one worth keeping honest about: it asks the
operating system whether the server process is gone, rather than trusting the
client to say so. A leaked subprocess is invisible for an hour and then is not.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from shopagent.mcp_client.client import MCPToolClient

EXPECTED_TOOLS = {"ping", "search_products", "get_product_details", "check_stock"}


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
def test_search_products_through_the_client_returns_real_products(client):
    """The whole path: sync call, portal, subprocess, Postgres, and back.

    `query` is left out on purpose — the MCP tool always embeds a query, so
    passing one here would spend money on every run.
    """
    result = client.call_tool("search_products", {"category": "shoes", "limit": 3})

    assert result.is_error is False
    payload = result.structured_content
    assert 1 <= payload["count"] <= 3
    assert payload["count"] == len(payload["results"])
    for product in payload["results"]:
        assert product["category"] == "shoes"
        assert product["variants"]
        assert isinstance(product["variants"][0]["price_cents"], int)


@pytest.mark.db
def test_an_unknown_product_id_comes_back_as_an_error(client):
    """The D4 contract, seen from the other side of the pipe."""
    result = client.call_tool("get_product_details", {"product_id": 10_000_000})

    assert result.is_error is True
    assert "search_products" in result.content[0].text


@pytest.mark.db
def test_a_successful_call_is_not_reported_as_an_error(client):
    """The other half of the flag, so `is_error` is not trivially always true."""
    assert client.call_tool("ping", {}).is_error is False


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


