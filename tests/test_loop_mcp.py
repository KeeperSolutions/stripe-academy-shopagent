"""How the agent loop gets its tools (D5, step 3).

The claim this file exists to hold is narrow and worth stating: `run_tool_loop`
did not change. Every test here is about `build_tool_setup` — which tools end up
in the registry, what happens when the catalog will not open, and that a result
fetched over MCP reaches the model as a `tool` message. The loop is exercised by
the last test the same way `tests/test_tool_loop.py` already exercises it, with
a fake client, and it is the unmodified function in both.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any

import pytest

from shopagent.llm.client import LLMClient
from shopagent.agent.prompt import CATALOG_PROMPT, NO_CATALOG_PROMPT, initial_messages
from shopagent.llm.loop import build_tool_setup, run_tool_loop
from shopagent.mcp_client.client import MCPToolClient

LOCAL_TOOLS = ["get_time", "calculator"]
# D9 put five more in every session, from a third source. They are listed here
# rather than folded into LOCAL_TOOLS because the distinction these tests are
# about is where a tool comes from: `MCP_CATALOG_ENABLED` moves the middle
# group and nothing else.
COMMERCE_TOOLS = [
    "add_to_cart",
    "view_cart",
    "remove_from_cart",
    "create_checkout",
    "check_order_status",
]
# Three since D9 — `ping` is no longer advertised by the server. The list is
# the interface, so it is pinned in order and by name: this fails when one
# disappears and equally when one appears that nobody meant to publish.
CATALOG_TOOLS = ["search_products", "get_product_details", "check_stock"]


# --- a catalog that refuses to open --------------------------------------


class ExplodingClient:
    """Stands in for a server that cannot start: bad path, no database, no pipe."""

    def __init__(self) -> None:
        raise OSError("no such file or directory: python")


class FailsOnHandshake:
    """Starts, then fails once the session is being opened."""

    def __enter__(self):
        raise RuntimeError("handshake timed out")

    def __exit__(self, *exc_info: object) -> None:  # pragma: no cover - never reached
        return None


def test_the_catalog_switch_off_leaves_only_the_local_tools():
    """The proof the D5 demo rests on: same binary, the catalog tools gone.

    Not "two tools" any more — D9 adds five that reach a different service, and
    the claim was never about the count. What the switch has to do is remove
    the four the catalog server publishes and touch nothing else.
    """
    with ExitStack() as stack:
        setup = build_tool_setup(stack, catalog_enabled=False)

    assert setup.registry.names() == LOCAL_TOOLS + COMMERCE_TOOLS
    assert setup.catalog_available is False
    assert "disabled" in setup.note


def test_a_client_that_cannot_start_does_not_stop_the_session():
    """A missing catalog is smaller than a CLI that refuses to run."""
    with ExitStack() as stack:
        setup = build_tool_setup(stack, client_factory=ExplodingClient)

    assert setup.registry.names() == LOCAL_TOOLS + COMMERCE_TOOLS
    assert setup.catalog_available is False
    assert "OSError" in setup.note


def test_a_client_that_fails_while_connecting_is_handled_the_same_way():
    """The failure can land at construction or at handshake; both are survivable."""
    with ExitStack() as stack:
        setup = build_tool_setup(stack, client_factory=FailsOnHandshake)

    assert setup.registry.names() == LOCAL_TOOLS + COMMERCE_TOOLS
    assert setup.catalog_available is False
    assert "handshake timed out" in setup.note


def test_the_local_tools_still_work_when_the_catalog_is_missing():
    """Degraded, not broken — the calculator is unaffected by a dead server."""
    with ExitStack() as stack:
        setup = build_tool_setup(stack, client_factory=ExplodingClient)
        result = setup.registry.dispatch("calculator", {"expression": "6 * 7"})

    assert result.ok is True
    assert result.content == "42"


# --- what the model is told ----------------------------------------------


def test_the_system_prompt_gains_the_catalog_rules_when_it_is_available():
    (message,) = initial_messages(catalog_available=True)

    assert CATALOG_PROMPT in message["content"]
    assert NO_CATALOG_PROMPT not in message["content"]


def test_the_system_prompt_says_so_when_the_catalog_is_missing():
    """Otherwise the model apologises for its memory instead of naming the cause."""
    (message,) = initial_messages(catalog_available=False)

    assert NO_CATALOG_PROMPT in message["content"]
    assert CATALOG_PROMPT not in message["content"]


def test_the_catalog_rules_forbid_answering_from_memory():
    """The one instruction D9's guardrails will later have to enforce in code."""
    assert "never from memory" in CATALOG_PROMPT
    assert "count of 0" in CATALOG_PROMPT


# --- with the real server -------------------------------------------------


@pytest.mark.db
def test_the_catalog_switch_on_adds_the_server_tools(engine):
    """Ten tools: the local two, D9's five, then the three the server lists."""
    with ExitStack() as stack:
        setup = build_tool_setup(stack, catalog_enabled=True)
        names = setup.registry.names()
        available = setup.catalog_available

    assert names == LOCAL_TOOLS + COMMERCE_TOOLS + CATALOG_TOOLS
    assert len(names) == 10
    assert "ping" not in names
    assert available is True


@pytest.mark.db
def test_the_server_process_is_released_with_the_stack(engine):
    """The catalog's lifetime is the session's, and `main` relies on it.

    The probe used to be `ping`, which D9 stopped advertising. A category
    browse replaces it for the same reason it does in `test_mcp_client.py`: it
    always succeeds against the seed and, carrying no `query`, embeds nothing
    and costs nothing.
    """
    browse = {"category": "shoes", "limit": 1}
    with ExitStack() as stack:
        setup = build_tool_setup(stack, catalog_enabled=True)
        assert setup.registry.dispatch("search_products", browse).ok is True

    # Outside the stack the client is closed, so the tool can no longer run.
    assert setup.registry.dispatch("search_products", browse).ok is False


# --- the loop, unmodified, driving an MCP tool ---------------------------


@dataclass
class FakeToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class FakeReply:
    content: str | None = None
    tool_calls: list[FakeToolCall] = field(default_factory=list)

    def to_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        return message


class ScriptedClient:
    """An LLM that asks for one catalog tool, then answers. No network."""

    def __init__(self, replies: list[FakeReply]) -> None:
        self._replies = list(replies)
        self.seen_tools: list[list[dict]] = []

    def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> FakeReply:
        self.seen_tools.append(tools)
        return self._replies.pop(0)


@pytest.mark.db
def test_a_full_round_trip_through_the_unmodified_loop(engine, capsys):
    """Model asks for search_products, MCP answers, the result becomes a tool message.

    `run_tool_loop` is the D2 function unchanged — that is the point of the
    test. The only thing D5 altered is which registry it is handed.
    """
    with ExitStack() as stack:
        setup = build_tool_setup(stack, catalog_enabled=True)
        client = ScriptedClient([
            FakeReply(tool_calls=[FakeToolCall(
                id="call_1",
                name="search_products",
                arguments=json.dumps({"category": "shoes", "limit": 2}),
            )]),
            FakeReply(content="Here are two pairs of shoes."),
        ])
        messages: list[Any] = initial_messages(setup.catalog_available)
        messages.append({"role": "user", "content": "show me shoes"})

        run_tool_loop(client, setup.registry, messages, setup.registry.openai_schemas())

    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"

    payload = json.loads(tool_messages[0]["content"])
    assert payload["count"] >= 1
    assert all(product["category"] == "shoes" for product in payload["results"])

    # The model was offered all ten, and the terminal showed the call.
    assert len(client.seen_tools[0]) == 10
    assert "search_products" in capsys.readouterr().out


def test_the_loop_signature_still_takes_a_registry_and_schemas():
    """A guard on the abstraction itself, not on behaviour.

    If `run_tool_loop` ever grows an MCP-shaped parameter, the claim that D5
    left it alone stops being true, and this is where that shows up.
    """
    import inspect

    parameters = list(inspect.signature(run_tool_loop).parameters)

    assert parameters == ["client", "registry", "messages", "tools"]
    assert LLMClient is not None  # imported for the type it documents
    assert MCPToolClient is not None


# --- the commerce tools (D9, step 1) -------------------------------------
#
# `build_tool_setup` is where a session's tools are decided, so it is where
# "the agent can actually buy something" is asserted. These stay next to the
# MCP tests rather than in `test_commerce_tools.py` because what they check is
# the assembly, not the tools: the same function, one more source.


COMMERCE_TOOLS = [
    "add_to_cart",
    "view_cart",
    "remove_from_cart",
    "create_checkout",
    "check_order_status",
]


def test_the_commerce_tools_are_in_every_session():
    """Even with the catalog off — they reach a different service.

    `MCP_CATALOG_ENABLED=false` is the catalog's off switch and nothing else.
    A session without product search can still show a cart and a payment link,
    and folding the two sources into one switch would make an unrelated
    failure look like this one.
    """
    with ExitStack() as stack:
        setup = build_tool_setup(stack, catalog_enabled=False)

    assert [name for name in setup.registry.names() if name in COMMERCE_TOOLS] == COMMERCE_TOOLS


def test_a_session_holds_one_cart_across_its_commerce_tools():
    """Two tools, one basket — the state that is deliberately not global."""
    with ExitStack() as stack:
        first = build_tool_setup(stack, catalog_enabled=False)
        second = build_tool_setup(stack, catalog_enabled=False)

    assert first.commerce is not None
    assert first.commerce is not second.commerce


# --- the whole list, offline (D9, step 2) --------------------------------


@dataclass
class FakeCatalogTool:
    name: str
    description: str = "a catalog tool"
    input_schema: dict = field(default_factory=lambda: {"type": "object", "properties": {}})


class FakeCatalogClient:
    """A catalog server that lists what the real one lists, without starting one.

    The composition of the tool list is the agent's whole interface, so it is
    worth an assertion that runs on every `pytest tests/` rather than only when
    Postgres happens to be up. What this cannot check is that the real server
    publishes these three names — `test_the_catalog_switch_on_adds_the_server_tools`
    does that, against the real process, and would fail if the two drifted.
    """

    def __enter__(self):
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def list_tools(self):
        return [FakeCatalogTool(name) for name in CATALOG_TOOLS]

    def call_tool(self, name: str, arguments=None):  # pragma: no cover - never called
        raise AssertionError("this fake is for listing only")


def test_the_model_is_offered_exactly_these_ten_tools():
    """Named and ordered, so it fails on a disappearance and on an arrival.

    The same shape as D6's hand-written table of foreign-key expectations: a
    check derived from whatever the code happens to register would pass on the
    day something is published by accident. `ping` is the reason this exists —
    it sat in the list from D5 to D9 and no test objected, because no test said
    what the list was supposed to be.
    """
    with ExitStack() as stack:
        setup = build_tool_setup(
            stack, catalog_enabled=True, client_factory=FakeCatalogClient
        )

    assert setup.registry.names() == [
        "get_time",
        "calculator",
        "add_to_cart",
        "view_cart",
        "remove_from_cart",
        "create_checkout",
        "check_order_status",
        "search_products",
        "get_product_details",
        "check_stock",
    ]
    assert len(setup.registry.names()) == 10
    assert "ping" not in setup.registry.names()
