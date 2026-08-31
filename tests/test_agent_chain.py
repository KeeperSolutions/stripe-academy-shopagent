"""Does the model hold a five-tool chain together? (D9, step 1.)

This file exists to settle one open question rather than to guard a behaviour.
JOURNAL has carried it since D2: `reasoning_effort='none'` is the price of
using function tools on Chat Completions with `gpt-5.6-luna`, and D2 could not
tell whether that mattered, because two independent tools chain trivially. D9
is the case the entry named — search, check stock, add to cart, view cart,
check out, where *choosing the next call* is the reasoning, and each step needs
an argument that only the previous step's result can supply.

So the assertion is not that the answers read well. It is **which tools were
called and in what order**, recorded as they happen — the same shape of test as
D6 recording the SQL `render_order` issues and D7 recording the checkout
payload, and for the same reason: an implementation that produced a plausible
final sentence while calling nothing at all would satisfy every assertion about
the text.

**Two scripts, one variable.** The first run of this test called nothing on
`"add it to my cart"`, and that single observation cannot tell "the model does
not hold a chain" apart from "the model reasonably asked which of four checked
variants *it* meant". So the script comes in two versions, identical except for
that one turn: `ambiguous` says "add it", `named` names the product. A test
that only ran the first would keep producing a result with two explanations.

**Turn 5 needs a confirmer, since step 5.** Run B stopped there: the
`create_checkout` description asked for "an explicit yes first" and never said
the customer's previous message could be that yes, so the model showed the cart
and asked again. Step 5 removed the sentence and replaced it with a gate in
code — `agent/guardrails.py` intercepts the call, shows a person what they are
buying at a total read from `view_cart`, and asks. A registry built without a
`confirm` callable refuses the purchase, which is the safe default and is what
this test gets unless it passes one. Step 6 is the run that means something.

**Marking.** `network` and `db`, and the running API is a skip rather than a
third marker. A marker decides whether a test is *selected*, and selection in
this project turns on what a run costs — `network` spends tokens, `stripe`
needs an account. Whether uvicorn happens to be up is not a property of the
run, it is a property of the minute, and it changes between two invocations of
the same command; that is what `db`'s skip already models, and a fourth marker
would mean the test that decides D9's plan is one nobody selects by accident.

**It writes real rows, and it takes them back out by id.** The fifth turn
places an order and opens a Stripe Checkout Session in test mode. Teardown
cancels the order — which is what releases the reservation and closes the
payment page — and then deletes exactly the four rows this run created, matched
by primary key. Never `DELETE FROM orders`: the lesson `manual_test_state.py`
was written for is that a cleanup which restores to a remembered constant
destroys whatever it did not know about, and a truncating teardown here would
take a developer's own manual run with it. When any step of that cannot be
completed, the test says so by name — "left cart X and order Y behind" — rather
than leaving a quietly dirty database behind a passing run.

The full transcript of each run, including everything the model said, is
written to `notes/`, which is not tracked. The assertions below are about the
tool calls; the prose is what a person reads afterwards to find out why.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from sqlalchemy import text

from shopagent.config import REPO_ROOT, get_settings
from shopagent.llm.client import LLMClient
from shopagent.agent.prompt import initial_messages
from shopagent.llm.loop import build_tool_setup, run_tool_loop
from shopagent.llm.usage import UsageTracker

# One turn, and the tool that turn cannot be answered honestly without. The
# English is the repo's rule, not a preference — every string committed here is
# English, including the ones a person would type.
#
# Turn 3 is the variable, and it is the only one. Everything else is held
# identical between the two scripts so that a difference in the result has one
# candidate explanation rather than five.
AMBIGUOUS_TURN_3 = "add it to my cart"
NAMED_TURN_3 = "add the Trail Runner GTX in size 42 to my cart"
# The reference step 3 is about. Not a third reading of the same question: the
# `ambiguous` script asks whether the model calls a write tool at all, and this
# one asks whether it can count rows in a list it was shown several turns ago
# without being handed a mechanism for doing so.
ORDINAL_TURN_3 = "add the second one to my cart"


def script(turn_3: str) -> list[tuple[str, str]]:
    return [
        ("find me some trail running shoes", "search_products"),
        ("do you have those in size 42?", "check_stock"),
        (turn_3, "add_to_cart"),
        ("what is in my cart?", "view_cart"),
        ("yes, order it", "create_checkout"),
    ]


SCRIPTS = {
    "ambiguous": script(AMBIGUOUS_TURN_3),
    "named": script(NAMED_TURN_3),
    "ordinal": script(ORDINAL_TURN_3),
}


# --- recording -----------------------------------------------------------


@dataclass
class ModelTurn:
    """One call to the model, as it came back."""

    turn: int
    content: str | None
    tool_calls: list[tuple[str, str]]
    finish_reason: str | None
    tools_offered: int


@dataclass
class Trace:
    """Everything one run did, in the order it happened."""

    replies: list[ModelTurn] = field(default_factory=list)
    dispatched: list[tuple[int, str, Any, bool]] = field(default_factory=list)
    turn: int = 0

    def names_in(self, turn: int) -> list[str]:
        return [name for index, name, _, _ in self.dispatched if index == turn]

    def said_in(self, turn: int) -> str:
        return "\n".join(
            reply.content for reply in self.replies if reply.turn == turn and reply.content
        )


class RecordingRegistry:
    """Everything `run_tool_loop` asks of a registry, plus a transcript.

    A wrapper rather than a subclass or a monkeypatch, because the loop takes
    the registry as a parameter — the same seam D5 used to swap the tool source
    without touching the loop. The loop under test is the unmodified one.
    """

    def __init__(self, registry, trace: Trace) -> None:
        self._registry = registry
        self._trace = trace

    def dispatch(self, name: str, raw_args):
        result = self._registry.dispatch(name, raw_args)
        self._trace.dispatched.append((self._trace.turn, name, raw_args, result.ok))
        return result


def record_replies(client: LLMClient, trace: Trace) -> None:
    """Capture each raw completion on its way back through the SDK.

    At the SDK boundary rather than around `chat_with_tools`, because
    `finish_reason` is the one field that does not survive the trip:
    `AssistantMessage` deliberately drops it, and adding it there to satisfy a
    test would put a diagnostic into production code — this measurement is not
    allowed to change the thing it measures. Wrapping here also gets the number
    of tools that were actually offered, taken from the outgoing request rather
    than from what the test believes it registered.
    """
    original = client._client.chat.completions.create

    def create(**kwargs):
        response = original(**kwargs)
        choice = response.choices[0]
        trace.replies.append(
            ModelTurn(
                turn=trace.turn,
                content=choice.message.content,
                tool_calls=[
                    (call.function.name, call.function.arguments or "")
                    for call in (choice.message.tool_calls or [])
                    if getattr(call, "type", "function") == "function"
                ],
                finish_reason=choice.finish_reason,
                tools_offered=len(kwargs.get("tools") or []),
            )
        )
        return response

    client._client.chat.completions.create = create


def write_transcript(name: str, turns, trace: Trace, tracker: UsageTracker, leftovers) -> str:
    """The whole run, on disk, because stdout is where the last one was lost."""
    path = REPO_ROOT / "notes" / f"d9-chain-{name}.md"
    lines = [f"# D9 chain run: {name}", ""]
    for index, (prompt, expected) in enumerate(turns, start=1):
        lines += [f"## turn {index} — {prompt!r}  (expected: {expected})", ""]
        for reply in (r for r in trace.replies if r.turn == index):
            lines += [
                f"- finish_reason: `{reply.finish_reason}` · tools offered: {reply.tools_offered}",
                "",
                "  model said:",
                "",
            ]
            lines += [f"  > {line}" for line in (reply.content or "(no text)").splitlines()]
            lines.append("")
            for call_name, arguments in reply.tool_calls:
                lines.append(f"  - requested `{call_name}({arguments})`")
            lines.append("")
        called = trace.names_in(index)
        lines += [f"- tools actually run: {called or 'none'}", ""]
    lines += ["## cost", "", "```", tracker.summary(), "```", ""]
    if leftovers:
        lines += ["## left behind", "", *[f"- {item}" for item in leftovers], ""]
    path.write_text("\n".join(lines))
    return str(path)


# --- fixtures ------------------------------------------------------------


@pytest.fixture
def commerce_api():
    """Skip, with the command to fix it, when the API is not running."""
    base_url = get_settings().commerce_api_base_url
    try:
        response = httpx.get(f"{base_url}/health", timeout=2.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(
            f"the commerce API is not answering at {base_url} ({type(exc).__name__}). "
            "Run: uvicorn shopagent.api.main:app --reload"
        )
    return base_url


# --- the measurement -----------------------------------------------------


@pytest.mark.network
@pytest.mark.db
@pytest.mark.parametrize("name", list(SCRIPTS), ids=list(SCRIPTS))
def test_the_model_holds_the_five_tool_chain(name, commerce_api, engine, capsys):
    turns = SCRIPTS[name]
    tracker = UsageTracker()
    client = LLMClient(tracker=tracker)
    trace = Trace()
    record_replies(client, trace)

    confirmations = []

    def confirm(summary: str) -> bool:
        """A test saying yes on a person's behalf, and saying so out loud.

        The gate refuses when there is nobody to ask, which is what a chain run
        without this argument would hit — correctly, and uninformatively. What
        this measurement is about is whether the model reaches the gate, so the
        harness answers it and records the summary it was shown, and the
        assertion below is that a person would have been shown a real cart
        rather than a total the model made up.
        """
        confirmations.append(summary)
        return True

    with ExitStack() as stack:
        setup = build_tool_setup(stack, catalog_enabled=True, confirm=confirm)
        if not setup.catalog_available:
            pytest.skip(f"the catalog server did not start: {setup.note}")

        registry = RecordingRegistry(setup.registry, trace)
        messages = initial_messages(catalog_available=True)
        schemas = setup.registry.openai_schemas()

        try:
            for index, (prompt, _) in enumerate(turns, start=1):
                trace.turn = index
                messages.append({"role": "user", "content": prompt})
                run_tool_loop(client, registry, messages, schemas)
        finally:
            leftovers = clean_up(commerce_api, setup.memory, engine)
            path = write_transcript(name, turns, trace, tracker, leftovers)
            print(f"\n--- {name}: {path}")
            for index, (prompt, _) in enumerate(turns, start=1):
                print(f"  {index}. {prompt!r} -> {trace.names_in(index) or 'no tool called'}")
            print(tracker.summary())
            for item in leftovers:
                print(f"[LEFT BEHIND] {item}")

    assert not leftovers, "teardown could not undo this run:\n" + "\n".join(leftovers)

    missed = [
        f"turn {index} ({prompt!r}) expected {expected}, called {trace.names_in(index) or 'nothing'}"
        for index, (prompt, expected) in enumerate(turns, start=1)
        if expected not in trace.names_in(index)
    ]
    assert not missed, f"the chain broke ({name}), see {path}:\n" + "\n".join(missed)

    # The gate was reached and a person was shown the cart, not a figure the
    # model produced. Without this the test would pass on a chain that called
    # `create_checkout` through a gate that had been quietly disabled.
    assert confirmations, "create_checkout ran without anybody being asked"
    assert "Total:" in confirmations[-1]


# --- taking the rows back out --------------------------------------------


def clean_up(base_url: str, commerce, engine) -> list[str]:
    """Undo exactly this run, by id, and say plainly what could not be undone.

    Cancelling comes before deleting and is not optional: the cancel is what
    releases `inventory.reserved`, and an order row deleted without it takes
    the release with it — the units would be unsellable with nothing left in
    the database to explain why. So a cancel that fails stops the deletion of
    that order rather than proceeding without it.

    Returns the list of things still in the database. Empty means the run left
    no trace, and the test asserts that: a cleanup nobody checks is a cleanup
    that stops working silently.
    """
    leftovers: list[str] = []
    if commerce is None:
        return leftovers

    order_id = commerce.order_id
    cart_id = commerce.cart_id
    if order_id and cart_id is None:
        # `create_checkout` releases the cart id once the order exists, which
        # is correct for the tools and leaves teardown without it. The order
        # remembers which cart it came from, so ask the database.
        with engine.begin() as connection:
            row = connection.execute(
                text("SELECT cart_id FROM orders WHERE id = :id"), {"id": order_id}
            ).first()
        if row is not None:
            cart_id = str(row[0])

    if order_id and not _cancelled(base_url, order_id, leftovers):
        # Deliberately keeps the cart too: the order still points at it.
        return leftovers + [f"cart {cart_id} (kept, its order could not be cancelled)"]

    try:
        with engine.begin() as connection:
            if order_id:
                connection.execute(
                    text("DELETE FROM order_items WHERE order_id = :id"), {"id": order_id}
                )
                connection.execute(text("DELETE FROM orders WHERE id = :id"), {"id": order_id})
            if cart_id:
                connection.execute(
                    text("DELETE FROM cart_items WHERE cart_id = :id"), {"id": cart_id}
                )
                connection.execute(text("DELETE FROM carts WHERE id = :id"), {"id": cart_id})
    except Exception as exc:  # noqa: BLE001 - the message is the point
        leftovers.append(
            f"order {order_id} and cart {cart_id} could not be deleted "
            f"({type(exc).__name__}: {exc}). Run: python scripts/manual_test_state.py restore"
        )

    return leftovers


def _cancelled(base_url: str, order_id: str, leftovers: list[str]) -> bool:
    """Cancel the order, treating an already-cancelled one as done."""
    headers = {"X-API-Key": get_settings().shopagent_api_key}
    try:
        response = httpx.post(
            f"{base_url}/orders/{order_id}/cancel", headers=headers, timeout=10.0
        )
        if response.is_success:
            return True
        status = httpx.get(
            f"{base_url}/orders/{order_id}", headers=headers, timeout=10.0
        ).json().get("status")
        if status in ("cancelled", "refunded"):
            return True
        leftovers.append(
            f"order {order_id} is {status} and could not be cancelled "
            f"({response.status_code}: {_detail(response)}). Its stock is still reserved. "
            "Run: python scripts/manual_test_state.py restore"
        )
    except httpx.HTTPError as exc:
        leftovers.append(
            f"order {order_id} could not be cancelled ({type(exc).__name__}). Its stock is "
            "still reserved. Run: python scripts/manual_test_state.py restore"
        )
    return False


def _detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail", ""))[:200]
    except (ValueError, AttributeError):
        return ""
