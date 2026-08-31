"""Rules enforced in code, because the prompt already asked (D9, step 5).

Two debts are paid here and both say the same thing. D2 asked the model never
to do arithmetic in its head and measured it answering `5 factorial` from
memory with zero tool calls — *"an amount that appears in an answer without
appearing in the context has to be blocked, not discouraged"*. Run B of the
chain test measured the other half: `create_checkout`'s description asked for
"an explicit yes first" and the model, having just been told "yes, order it",
showed the cart and asked for the yes again. An instruction describing a
precondition without describing how it is satisfied is unsatisfiable.

So there are three rules here and none of them is a sentence in a prompt:

*Nothing buys anything until a person says so.* The gate intercepts the call
before it runs, prints what is being bought at a total read from `view_cart`,
and asks. The model cannot set the flag, because there is no flag.

*No amount reaches the customer that did not come from a tool.* One retry with
a correction, then a fallback that says plainly which figure could not be
traced — never silence, and never a third attempt.

*No variant is added that the model has not been shown.* The surface for that
was built in step 3 and left unused; this is what reads it.

Deliberately not here: counting. "All three are available" over four rows is
the D5 debt, it is the same shape, and it is a different rule — step 6 records
it as still open rather than half-building it next to one that works.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from shopagent.agent.guardrails import (
    CONFIRM_BEFORE,
    FALLBACK_PREFIX,
    GuardedClient,
    GuardedRegistry,
    unsupported_amounts,
)
from shopagent.agent.memory import ConversationMemory
from shopagent.llm.client import AssistantMessage, ToolCall
from shopagent.tools.registry import ToolResult, ToolSpec

CART = {
    "currency": "eur",
    "line_count": 1,
    "unit_count": 2,
    "items": [
        {
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
}


class NoArgs(BaseModel):
    pass


class AddArgs(BaseModel):
    variant_id: int
    quantity: int = 1


def build(confirm=None, cart=CART, memory=None):
    """A registry holding a fake cart and a fake checkout, with the gate on."""
    memory = memory or ConversationMemory()
    registry = GuardedRegistry(memory, confirm=confirm)
    ran = []

    def view_cart():
        if isinstance(cart, Exception):
            raise cart
        return cart

    def create_checkout():
        ran.append("create_checkout")
        return {"order_id": "o-1", "checkout_url": "https://pay.example/1", "total_cents": 18998}

    def add_to_cart(variant_id: int, quantity: int = 1):
        ran.append(f"add_to_cart:{variant_id}")
        return cart

    registry.register(ToolSpec(name="view_cart", description="d", args_model=NoArgs, fn=view_cart))
    registry.register(
        ToolSpec(name="create_checkout", description="d", args_model=NoArgs, fn=create_checkout)
    )
    registry.register(
        ToolSpec(name="add_to_cart", description="d", args_model=AddArgs, fn=add_to_cart)
    )
    return registry, memory, ran


# --- the gate ------------------------------------------------------------


def test_checkout_is_the_call_that_needs_a_person():
    assert CONFIRM_BEFORE == frozenset({"create_checkout"})


def test_nothing_is_bought_before_the_answer_comes_back():
    """The order matters: the tool must not have run when the human is asked."""
    order = []

    def confirm(summary):
        order.append("asked")
        return True

    registry, _, ran = build(confirm=confirm)
    registry.dispatch("create_checkout", {})

    assert order == ["asked"]
    assert ran == ["create_checkout"]


def test_declining_leaves_the_tool_unrun_and_tells_the_model_why():
    registry, _, ran = build(confirm=lambda summary: False)

    result = registry.dispatch("create_checkout", {})

    assert ran == []
    assert not result.ok
    assert "did not confirm" in result.content
    assert "do not call create_checkout again" in result.content.lower()


def test_the_total_a_person_confirms_comes_from_the_cart_not_the_model():
    """A person approving a number the model invented is worse than no gate."""
    seen = []
    registry, _, _ = build(confirm=lambda summary: seen.append(summary) or True)

    registry.dispatch("create_checkout", {})

    (summary,) = seen
    assert "€189.98" in summary
    assert "Trail Runner GTX" in summary
    assert "42 / black" in summary


def test_a_registry_with_nobody_to_ask_refuses_to_buy():
    """The safe default. A gate that cannot reach a person is not a gate."""
    registry, _, ran = build(confirm=None)

    result = registry.dispatch("create_checkout", {})

    assert ran == []
    assert not result.ok


def test_a_cart_that_cannot_be_read_stops_the_purchase():
    """Nothing is confirmed blind: no cart, no summary, no sale."""
    registry, _, ran = build(confirm=lambda summary: True, cart=RuntimeError("API down"))

    result = registry.dispatch("create_checkout", {})

    assert ran == []
    assert not result.ok


def test_an_ordinary_tool_is_never_gated():
    registry, _, _ = build(confirm=lambda summary: False)

    assert registry.dispatch("view_cart", {}).ok


# --- amounts -------------------------------------------------------------


def memory_holding_the_cart():
    memory = ConversationMemory()
    memory.observe("view_cart", {}, json.dumps(CART))
    return memory


@pytest.mark.parametrize(
    "answer",
    [
        "That comes to €189.98.",
        "The total is 189.98 EUR.",
        "Each pair is €94.99.",
        "The line total is 18998 EUR in minor units.",
        "Your cart holds 2 pairs in size 42, 3 left in stock.",
        "I could not find anything matching that.",
    ],
)
def test_an_amount_the_shop_actually_quoted_passes(answer):
    """Both sides of the one conversion the prompt allows, and no false alarms
    on the integers this domain is full of — sizes, quantities, stock counts."""
    assert unsupported_amounts(answer, memory_holding_the_cart()) == []


@pytest.mark.parametrize(
    "answer",
    [
        "With your discount that comes to €94.00.",
        "Two pairs would be €200.00.",
        "That is 150.00 EUR after the reduction.",
        "I can do €1.00 for you.",
    ],
)
def test_an_amount_that_came_from_nowhere_is_caught(answer):
    assert unsupported_amounts(answer, memory_holding_the_cart())


def test_an_empty_memory_supports_no_amount_at_all():
    assert unsupported_amounts("That is €94.99.", ConversationMemory())


# --- retry, then a fallback that says something --------------------------


class FakeClient:
    """An LLM that answers from a script and records what it was asked."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def chat_with_tools(self, messages, tools=None):
        self.calls.append(list(messages))
        return self.answers.pop(0)


def answer(text, tool_calls=()):
    return AssistantMessage(content=text, tool_calls=list(tool_calls))


def test_a_good_answer_is_returned_untouched_and_costs_one_call():
    client = FakeClient(answer("That comes to €189.98."))
    guarded = GuardedClient(client, memory_holding_the_cart())

    reply = guarded.chat_with_tools([{"role": "user", "content": "total?"}])

    assert reply.content == "That comes to €189.98."
    assert len(client.calls) == 1


def test_a_bad_answer_is_retried_once_with_a_correction():
    client = FakeClient(answer("That is €94.00."), answer("That comes to €189.98."))
    guarded = GuardedClient(client, memory_holding_the_cart())

    reply = guarded.chat_with_tools([{"role": "user", "content": "total?"}])

    assert reply.content == "That comes to €189.98."
    assert len(client.calls) == 2
    correction = client.calls[1][-1]
    assert correction["role"] == "system"
    assert "€94.00" in correction["content"]


def test_a_second_bad_answer_becomes_a_fallback_rather_than_a_third_try():
    client = FakeClient(answer("That is €94.00."), answer("Sorry, €94.00."))
    guarded = GuardedClient(client, memory_holding_the_cart())

    reply = guarded.chat_with_tools([{"role": "user", "content": "total?"}])

    assert len(client.calls) == 2
    assert reply.content.startswith(FALLBACK_PREFIX)
    assert "€94.00" in reply.content


def test_the_fallback_is_something_a_customer_can_act_on():
    """Not an empty answer and not an apology with no next step."""
    client = FakeClient(answer("€94.00"), answer("€94.00"))
    guarded = GuardedClient(client, memory_holding_the_cart())

    reply = guarded.chat_with_tools([{"role": "user", "content": "total?"}])

    assert len(reply.content) > 60
    assert "cart" in reply.content.lower()


def test_a_turn_that_is_still_calling_tools_is_not_validated():
    """Mid-chain narration is not the answer, and the numbers may not be in yet."""
    reply_with_calls = answer("Let me check €94.00.", [ToolCall(id="1", name="view_cart", arguments="{}")])
    client = FakeClient(reply_with_calls)
    guarded = GuardedClient(client, memory_holding_the_cart())

    reply = guarded.chat_with_tools([{"role": "user", "content": "total?"}])

    assert reply is reply_with_calls
    assert len(client.calls) == 1


# --- a variant the model has not been shown ------------------------------


def test_adding_a_variant_that_never_appeared_is_refused():
    registry, memory, ran = build()

    result = registry.dispatch("add_to_cart", {"variant_id": 999999})

    assert ran == []
    assert not result.ok
    assert "999999" in result.content
    assert "search" in result.content.lower()


def test_adding_a_variant_the_model_was_shown_goes_through():
    registry, memory, ran = build()
    memory.observe("search_products", {}, json.dumps({"results": [{"variants": [{"variant_id": 86263}]}]}))

    result = registry.dispatch("add_to_cart", {"variant_id": 86263})

    assert result.ok
    assert ran == ["add_to_cart:86263"]
