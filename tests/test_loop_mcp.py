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

from shopagent.llm.client import AssistantMessage, LLMClient, ToolCall
from shopagent.agent.prompt import CATALOG_PROMPT, NO_CATALOG_PROMPT, initial_messages
from shopagent.agent.memory import ConversationMemory
from shopagent.agent.confirmation import CONFIRMED_NOTE, DECLINED_NOTE
from shopagent.agent.guardrails import FALLBACK_PREFIX, GuardedClient
from shopagent.llm.loop import (
    _print_payment_link,
    _settle_confirmation,
    build_tool_setup,
    run_tool_loop,
)
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


def test_a_session_holds_one_memory_across_its_commerce_tools():
    """Two tools, one basket — the state that is deliberately not global."""
    with ExitStack() as stack:
        first = build_tool_setup(stack, catalog_enabled=False)
        second = build_tool_setup(stack, catalog_enabled=False)

    assert first.memory is not None
    assert first.memory is not second.memory


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


# --- the payment link the CLI prints (PR #9) ------------------------------
#
# A Checkout Session URL is 475 opaque characters, and the end-to-end run for
# PR #9 measured what happens when the model relays one: asked twice for the
# same session, it reproduced the URL correctly once and changed a single
# character the second time (`TlZQ` to `TlVQ`, position 329). Stripe answers
# 401 for that, so the customer gets a payment page that does not work.
#
# The link therefore never enters the conversation. `tools/commerce.py` puts it
# on the memory and the CLI prints it. These tests are about that last step.


def test_the_cli_prints_the_link_exactly_as_the_shop_issued_it(capsys):
    url = (
        "https://checkout.stripe.com/c/pay/cs_test_a1lat7VO5sgjzwy5bFZM4N4mqoDv7c6"
        "#fidnandhYHdWcXxpYCc%2FJ2FgY2RwaXEnKSdicGRmZGhqaWBTZHdsZGtxJz8nZmprcXdqaSc"
    )
    memory = ConversationMemory(checkout_url=url)

    _print_payment_link(memory)

    printed = capsys.readouterr().out
    assert url in printed, "the link must appear byte-for-byte"
    assert "Pay here" in printed


def test_a_turn_that_produced_no_link_prints_nothing(capsys):
    """Otherwise every answer in the conversation carries a payment page."""
    _print_payment_link(ConversationMemory())

    assert capsys.readouterr().out == ""


def test_the_link_is_printed_once_and_not_under_every_later_answer(capsys):
    """A payment page shown again beneath "your order is paid" is one somebody clicks."""
    memory = ConversationMemory(checkout_url="https://checkout.stripe.com/c/pay/cs_test_1")

    _print_payment_link(memory)
    first = capsys.readouterr().out
    _print_payment_link(memory)
    second = capsys.readouterr().out

    assert "cs_test_1" in first
    assert second == ""


def test_a_session_with_no_memory_at_all_does_not_crash(capsys):
    """`ToolSetup.memory` is optional, and a missing one is not a reason to fail a turn."""
    _print_payment_link(None)

    assert capsys.readouterr().out == ""


# --- the CLI's half of the confirmation protocol (D10, step 1) ------------
#
# The gate parks a question and returns; `_settle_confirmation` is what puts it
# to a person and carries their answer back into a second turn. These tests are
# about that second half only — what the gate decides is asserted in
# `tests/test_guardrails.py`, and the protocol under both in
# `tests/test_confirmation.py`.
#
# The point of driving it here is the one thing neither of those can show: the
# follow-up is a normal `run_tool_loop` call on the same unmodified function,
# not a suspended one being resumed.


class ConfirmationClient:
    """An LLM answering from a list, and recording the messages it was sent."""

    # `_run_session` prints the model name in its banner, and `GuardedClient`
    # forwards every attribute but one to whatever it wraps.
    model = "fake-model"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def chat_with_tools(self, messages, tools=None):
        self.calls.append(list(messages))
        return self.replies.pop(0)


def _reply(content=None, tool=None):
    calls = [ToolCall(id="c1", name=tool, arguments="{}")] if tool else []
    return AssistantMessage(content=content, tool_calls=calls)


def _gated_setup(confirm):
    """A real session's gate and memory, over a commerce API answering in-process.

    Built directly rather than through `build_tool_setup`, which would reach a
    live MCP subprocess and a live HTTP client for nothing: what these tests
    are about is the two halves of the protocol, and the pieces that carry it
    are the registry, the memory and the confirmer.
    """
    import httpx

    from shopagent.llm.loop import ToolSetup
    from shopagent.agent.guardrails import GuardedRegistry
    from shopagent.tools.commerce import register_commerce_tools
    from shopagent.tools.http import CommerceAPI

    def handler(request):
        path = request.url.path
        if path.startswith("/cart"):
            return httpx.Response(
                200,
                json={
                    "cart_id": "c-1",
                    "status": "open",
                    "currency": "eur",
                    "items": [
                        {
                            "item_id": "i-1",
                            "variant_id": 86263,
                            "sku": "FF-TRLGTX-42-BLK",
                            "product_name": "Trail Runner GTX",
                            "variant_label": "42 / black",
                            "quantity": 2,
                            "unit_price_cents": 9499,
                            "line_total_cents": 18998,
                        }
                    ],
                    "total_cents": 18998,
                },
            )
        if path.endswith("/checkout"):
            return httpx.Response(
                200, json={"checkout_url": "https://pay.example/cs_test_1", "status": "pending"}
            )
        if path == "/orders":
            return httpx.Response(
                201,
                json={
                    "order_id": "o-1",
                    "status": "pending",
                    "currency": "eur",
                    "items": [],
                    "total_cents": 18998,
                },
            )
        return httpx.Response(404, json={"detail": "no such path"})

    memory = ConversationMemory(cart_id="c-1")
    registry = GuardedRegistry(memory, can_confirm=confirm is not None)
    api = CommerceAPI(
        base_url="http://commerce.test", api_key="k", transport=httpx.MockTransport(handler)
    )
    register_commerce_tools(registry, api, memory)
    return ToolSetup(
        registry=registry, catalog_available=False, memory=memory, confirm=confirm
    )


def test_the_cli_asks_after_the_turn_and_then_drives_one_more(capsys):
    """The whole protocol, end to end, through the function the CLI calls.

    Two turns, two `run_tool_loop` entries: the first parks the question, the
    second spends the answer. The customer is asked in between, which is the
    only moment at which a browser could have been asked instead.
    """
    shown = []
    setup = _gated_setup(confirm=lambda summary: shown.append(summary) or True)
    client = ConfirmationClient(
        _reply(tool="create_checkout"),
        _reply("Waiting for your confirmation."),
        _reply(tool="create_checkout"),
        _reply("Your order is placed."),
    )
    messages = [{"role": "user", "content": "check me out"}]
    schemas = setup.registry.openai_schemas()

    setup.memory.begin_turn(from_customer=True)
    run_tool_loop(client, setup.registry, messages, schemas)
    assert setup.memory.pending_confirmation is not None, "nothing was parked"

    _settle_confirmation(client, setup.registry, messages, schemas, setup)

    assert len(shown) == 1
    assert "Total: €189.98" in shown[0]
    assert setup.memory.order_id == "o-1"
    assert setup.memory.checkout_url == "https://pay.example/cs_test_1"
    assert client.replies == [], "both turns ran"
    assert [message["role"] for message in messages].count("system") == 1


def test_the_answer_reaches_the_model_as_a_system_note_not_as_the_customer(capsys):
    """The customer pressed a key in the shop's interface; they did not speak."""
    setup = _gated_setup(confirm=lambda summary: True)
    client = ConfirmationClient(
        _reply(tool="create_checkout"),
        _reply("Waiting."),
        _reply("Placed."),
    )
    messages = [{"role": "user", "content": "check me out"}]
    schemas = setup.registry.openai_schemas()

    setup.memory.begin_turn(from_customer=True)
    run_tool_loop(client, setup.registry, messages, schemas)
    _settle_confirmation(client, setup.registry, messages, schemas, setup)

    note = [message for message in messages if message.get("role") == "system"]
    assert note == [{"role": "system", "content": CONFIRMED_NOTE}]
    assert not any(
        message.get("role") == "user" and message["content"] == CONFIRMED_NOTE
        for message in messages
    )


def test_declining_drives_the_turn_that_tells_the_customer_and_buys_nothing():
    setup = _gated_setup(confirm=lambda summary: False)
    client = ConfirmationClient(
        _reply(tool="create_checkout"),
        _reply("Waiting."),
        _reply("Understood, I have not placed it."),
    )
    messages = [{"role": "user", "content": "check me out"}]
    schemas = setup.registry.openai_schemas()

    setup.memory.begin_turn(from_customer=True)
    run_tool_loop(client, setup.registry, messages, schemas)
    _settle_confirmation(client, setup.registry, messages, schemas, setup)

    assert setup.memory.order_id is None, "nothing was ordered"
    assert setup.memory.checkout_url is None
    assert any(message.get("content") == DECLINED_NOTE for message in messages)
    assert client.replies == [], "the customer still gets told, which costs a turn"


def test_a_turn_that_asked_nobody_anything_drives_no_second_turn():
    """The ordinary case, which is every turn that did not reach a checkout."""
    setup = _gated_setup(confirm=lambda summary: True)
    client = ConfirmationClient(_reply("Here is your cart."))
    messages = [{"role": "user", "content": "hello"}]
    schemas = setup.registry.openai_schemas()

    setup.memory.begin_turn(from_customer=True)
    run_tool_loop(client, setup.registry, messages, schemas)
    _settle_confirmation(client, setup.registry, messages, schemas, setup)

    assert client.replies == []
    assert len(client.calls) == 1


def test_a_session_with_nobody_to_ask_never_reaches_the_confirmer():
    """`confirm=None` refuses at the gate, so nothing is parked to be answered."""
    setup = _gated_setup(confirm=None)
    client = ConfirmationClient(
        _reply(tool="create_checkout"),
        _reply("I cannot complete that here."),
    )
    messages = [{"role": "user", "content": "check me out"}]
    schemas = setup.registry.openai_schemas()

    setup.memory.begin_turn(from_customer=True)
    run_tool_loop(client, setup.registry, messages, schemas)
    assert setup.memory.pending_confirmation is None

    _settle_confirmation(client, setup.registry, messages, schemas, setup)

    assert setup.memory.order_id is None
    assert client.replies == []


# --- the amount guard over the turn the gate created (D10, step 1) --------
#
# The confirmation protocol added a second `run_tool_loop` entry per checkout,
# and a new path is a new place a rule can fail to reach. Two answers now exist
# that did not before: the sentence the model writes while a person is being
# asked, and the one it writes after they have answered. Both state figures to
# a customer, so both have to be checked against `seen_amount_cents` — a gate
# that opened a hole in the amount guardrail would have traded one rule for
# another.
#
# `_settle_confirmation` takes whatever client it is given, and `_run_session`
# gives it the `GuardedClient` it wraps once at the top. These tests assert the
# consequence rather than the wiring: a bad figure in either sentence comes back
# as the fallback.


def _guarded(setup, *replies):
    inner = ConfirmationClient(*replies)
    return GuardedClient(inner, setup.memory), inner


def test_an_invented_amount_after_the_confirmation_is_caught_like_any_other():
    """The follow-up turn is not a privileged one."""
    setup = _gated_setup(confirm=lambda summary: True)
    client, inner = _guarded(
        setup,
        _reply(tool="create_checkout"),
        _reply("Waiting for your confirmation."),
        _reply(tool="create_checkout"),
        _reply("Done — that came to €5.00."),
        # The retry the correction buys, wrong in the same way.
        _reply("Done — that came to €5.00."),
    )
    messages = [{"role": "user", "content": "check me out"}]
    schemas = setup.registry.openai_schemas()

    setup.memory.begin_turn(from_customer=True)
    run_tool_loop(client, setup.registry, messages, schemas)
    _settle_confirmation(client, setup.registry, messages, schemas, setup)

    assert inner.replies == [], "the retry was never asked for"
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"].startswith(FALLBACK_PREFIX)
    assert "€5.00" not in messages[-1]["content"].removeprefix(FALLBACK_PREFIX).split(")")[1]


def test_an_invented_amount_while_the_customer_is_being_asked_is_caught_too():
    """The other new sentence: the one written before anybody has answered.

    It is the turn the gate ends, and the model is told to say briefly that it
    is waiting — which is an invitation to restate the total from memory.
    """
    setup = _gated_setup(confirm=lambda summary: True)
    client, inner = _guarded(
        setup,
        _reply(tool="create_checkout"),
        _reply("Confirm the €5.00 and I will place it."),
        _reply("Confirm the €5.00 and I will place it."),
        _reply(tool="create_checkout"),
        _reply("Placed."),
    )
    messages = [{"role": "user", "content": "check me out"}]
    schemas = setup.registry.openai_schemas()

    setup.memory.begin_turn(from_customer=True)
    run_tool_loop(client, setup.registry, messages, schemas)

    waiting = [m for m in messages if m.get("role") == "assistant" and m.get("content")]
    assert waiting[-1]["content"].startswith(FALLBACK_PREFIX)

    # And the purchase is unaffected: a blocked sentence is not a blocked gate.
    _settle_confirmation(client, setup.registry, messages, schemas, setup)
    assert setup.memory.order_id == "o-1"


def test_the_cart_total_quoted_after_a_confirmation_passes_untouched():
    """The control. A guard that rejected the right answer too would pass the
    two tests above and be worthless."""
    setup = _gated_setup(confirm=lambda summary: True)
    client, inner = _guarded(
        setup,
        _reply(tool="create_checkout"),
        _reply("Waiting for your confirmation."),
        _reply(tool="create_checkout"),
        _reply("Placed — €189.98 in total."),
    )
    messages = [{"role": "user", "content": "check me out"}]
    schemas = setup.registry.openai_schemas()

    setup.memory.begin_turn(from_customer=True)
    run_tool_loop(client, setup.registry, messages, schemas)
    _settle_confirmation(client, setup.registry, messages, schemas, setup)

    assert messages[-1]["content"] == "Placed — €189.98 in total."
    assert inner.replies == [], "a correct answer must not cost a retry"


def test_the_repl_itself_guards_the_turn_the_gate_created():
    """The claim the three tests above cannot make on their own.

    They hand `_settle_confirmation` a client they wrapped themselves, so they
    prove the guard works when it is there — not that the CLI puts it there. An
    edit that built a second, unwrapped client for the follow-up would leave
    all three green and let an invented figure reach a customer on the one turn
    a purchase happens.

    So this drives `_run_session`, which is the function that does the wrapping,
    with `input` standing in for a person at the keyboard: a customer message, a
    `y` at the confirmation prompt, and `/exit`. What reaches the terminal is
    what a customer would have read.
    """
    import builtins

    from shopagent.agent import profile as profiles
    from shopagent.llm.loop import _ask_to_confirm, _run_session
    from shopagent.llm.usage import UsageTracker

    setup = _gated_setup(confirm=_ask_to_confirm)
    client = ConfirmationClient(
        _reply(tool="create_checkout"),
        _reply("Waiting for your confirmation."),
        _reply(tool="create_checkout"),
        _reply("Done — that came to €5.00."),
        _reply("Done — that came to €5.00."),
    )
    typed = iter(["check me out", "y", "/exit"])
    printed = []

    original_input = builtins.input
    original_print = builtins.print
    original_load = profiles.load_for_session
    builtins.input = lambda prompt="": next(typed)
    builtins.print = lambda *args, **kwargs: printed.append(" ".join(str(a) for a in args))
    profiles.load_for_session = lambda shopper_id: (None, None)
    try:
        _run_session(client, UsageTracker(), setup)
    finally:
        builtins.input = original_input
        builtins.print = original_print
        profiles.load_for_session = original_load

    transcript = "\n".join(printed)
    assert "About to place this order:" in transcript, "the person was never asked"
    assert "Total: €189.98" in transcript
    assert FALLBACK_PREFIX in transcript, (
        "an amount no tool produced reached the customer on the turn the "
        "confirmation created"
    )
    assert client.replies == [], "the correction was never sent"


# --- where the tracing attaches (D10, step 2) -----------------------------
#
# `_run_session` builds two wrappers around each of the two things the loop is
# handed, and the *order* of the client's is load-bearing: `TracedClient` goes
# inside `GuardedClient` so the corrected retry the amount guardrail can send —
# a second, really billed call — appears as a second generation.
#
# Asserted here rather than in `tests/test_tracing.py` because that file builds
# the order itself and would go on passing over a session that wired it the
# other way. Mutating the line in `llm/loop.py` survived the whole suite until
# this existed.


def _capture_tracer():
    """A tracer on a real Langfuse client exporting into memory."""
    import itertools

    from langfuse import Langfuse
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from shopagent.obs.tracing import Tracer

    exporter = InMemorySpanExporter()
    key = f"pk-lf-loop-{next(_capture_tracer.keys)}"
    tracer = Tracer(
        Langfuse(
            public_key=key,
            secret_key="sk-lf-offline",
            host="http://langfuse.invalid",
            span_exporter=exporter,
            flush_at=1,
        )
    )
    return tracer, exporter


_capture_tracer.keys = __import__("itertools").count()


def _drive_session(setup, client, tracer, typed):
    """Run the REPL with `input` standing in for a person at the keyboard."""
    import builtins

    from shopagent.agent import profile as profiles
    from shopagent.llm.loop import _run_session
    from shopagent.llm.usage import UsageTracker

    lines = iter(typed)
    original_input, original_print = builtins.input, builtins.print
    original_load = profiles.load_for_session
    builtins.input = lambda prompt="": next(lines)
    builtins.print = lambda *args, **kwargs: None
    profiles.load_for_session = lambda shopper_id: (None, None)
    try:
        _run_session(client, UsageTracker(), setup, tracer)
    finally:
        builtins.input, builtins.print = original_input, original_print
        profiles.load_for_session = original_load


def test_the_session_traces_the_billed_retry_the_amount_guardrail_sends():
    """`TracedClient` inside `GuardedClient`, proved through the real session.

    Wired the other way round, the trace records one call where two were paid
    for — and reports half the cost of every corrected turn, which is exactly
    what the CLI-versus-Langfuse comparison is meant to catch.
    """
    tracer, exporter = _capture_tracer()
    setup = _gated_setup(confirm=lambda summary: True)
    # An amount from nowhere, twice: the guardrail corrects once and then falls
    # back, so two requests are really sent.
    client = ConfirmationClient(
        _reply("Your total is €5.00."),
        _reply("Your total is €5.00."),
    )

    _drive_session(setup, client, tracer, ["what is my total?", "/exit"])

    tracer.flush()
    names = [span.name for span in exporter.get_finished_spans()]
    assert names.count("chat") == 2, f"the billed retry is missing from the trace: {names}"
    assert "untraceable_amount" in names, "the guardrail did not report itself"
    assert names[-1] == "conversation", "the whole conversation is one trace"


def test_the_session_puts_every_span_in_one_trace():
    tracer, exporter = _capture_tracer()
    setup = _gated_setup(confirm=lambda summary: True)
    client = ConfirmationClient(
        _reply(tool="view_cart"),
        _reply("Your cart is here."),
        _reply(tool="view_cart"),
        _reply("Still there."),
    )

    _drive_session(setup, client, tracer, ["what is in my cart?", "and now?", "/exit"])

    tracer.flush()
    spans = exporter.get_finished_spans()
    assert len({span.context.trace_id for span in spans}) == 1, "two turns, two traces"
    assert [s.name for s in spans].count("view_cart") == 2
