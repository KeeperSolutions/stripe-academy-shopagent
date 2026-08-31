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

from shopagent.agent import confirmation
from shopagent.agent.guardrails import (
    AWAITING_ANSWER,
    CONFIRM_BEFORE,
    FALLBACK_PREFIX,
    PLACING,
    RESUMING,
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


# What `view_cart` answers once `create_checkout` has closed the cart, and what
# `check_order_status` answers for the order that closed it. Same shape, which
# is what lets the gate render either.
EMPTY_CART = {"currency": "eur", "line_count": 0, "unit_count": 0, "items": [], "total_cents": 0}

ORDER = {
    "order_id": "o-1",
    "status": "pending",
    "currency": "eur",
    "line_count": 1,
    "items": [
        {
            "variant_id": 86287,
            "product_name": "Storm Guard Shell",
            "variant_label": "M / red",
            "quantity": 1,
            "unit_price_cents": 19999,
            "line_total_cents": 19999,
        }
    ],
    "total_cents": 19999,
}

EMPTY_ORDER = {"error": "no order has been placed in this conversation"}


def build(can_confirm=True, cart=CART, memory=None, order=ORDER):
    """A registry holding a fake cart and a fake checkout, with the gate on.

    `can_confirm` replaced D9's `confirm` callable when the gate stopped
    blocking (D10, step 1). The registry no longer asks anybody; it needs to
    know only whether anybody is reachable, and `False` is still a refusal
    rather than a pass.
    """
    memory = memory or ConversationMemory()
    registry = GuardedRegistry(memory, can_confirm=can_confirm)
    ran = []

    def view_cart():
        if isinstance(cart, Exception):
            raise cart
        return cart

    def check_order_status():
        return order

    def create_checkout():
        ran.append("create_checkout")
        return {"order_id": "o-1", "checkout_url": "https://pay.example/1", "total_cents": 18998}

    def add_to_cart(variant_id: int, quantity: int = 1):
        ran.append(f"add_to_cart:{variant_id}")
        return cart

    registry.register(ToolSpec(name="view_cart", description="d", args_model=NoArgs, fn=view_cart))
    registry.register(
        ToolSpec(
            name="check_order_status", description="d", args_model=NoArgs, fn=check_order_status
        )
    )
    registry.register(
        ToolSpec(name="create_checkout", description="d", args_model=NoArgs, fn=create_checkout)
    )
    registry.register(
        ToolSpec(name="add_to_cart", description="d", args_model=AddArgs, fn=add_to_cart)
    )
    return registry, memory, ran


# --- the gate ------------------------------------------------------------


def settle(memory, said_yes):
    """What a caller does between turns, and nothing a caller does on its own.

    The CLI's `_settle_confirmation` and the eval runner's driver both reduce
    to these two calls: put whatever is parked to somebody, then begin the turn
    that carries their answer back. Driving the protocol here rather than
    either caller is the point — if a test needed the CLI, the protocol would
    not be one.

    Returns the summaries the person was actually shown.
    """
    shown = []
    confirmation.resolve_pending(memory, lambda summary: shown.append(summary) or said_yes)
    memory.begin_turn(from_customer=False)
    return shown


def test_checkout_is_the_call_that_needs_a_person():
    assert CONFIRM_BEFORE == frozenset({"create_checkout"})


def test_the_first_checkout_asks_and_buys_nothing():
    """Phase one: the question is parked, the tool has not run, nothing blocks."""
    registry, memory, ran = build()

    result = registry.dispatch("create_checkout", {})

    assert ran == [], "the tool ran before anybody was asked"
    assert not result.ok
    assert result.content == AWAITING_ANSWER

    pending = memory.pending_confirmation
    assert pending is not None
    assert pending.tool == "create_checkout"
    assert pending.answer is None
    assert "Total: €189.98" in pending.summary


def test_confirming_lets_the_next_call_through():
    """Phase two, the yes. One dispatch asks, the next one buys."""
    registry, memory, ran = build()
    registry.dispatch("create_checkout", {})

    shown = settle(memory, True)
    result = registry.dispatch("create_checkout", {})

    assert len(shown) == 1
    assert ran == ["create_checkout"]
    assert result.ok


def test_declining_leaves_the_tool_unrun_and_tells_the_model_why():
    """Phase two, the no. The refusal is the same sentence D9 wrote."""
    registry, memory, ran = build()
    registry.dispatch("create_checkout", {})

    settle(memory, False)
    result = registry.dispatch("create_checkout", {})

    assert ran == []
    assert not result.ok
    assert "did not confirm" in result.content
    assert "do not call create_checkout again" in result.content.lower()


def test_a_second_call_in_the_same_turn_asks_nothing_new():
    """One question in front of a person at a time.

    A second `create_checkout` before anybody has answered must not read the
    cart again and park a second summary: the person is looking at the first
    one, and approving it would then approve something they were never shown.
    Identity is the assertion, because a rebuilt summary would compare equal.
    """
    registry, memory, ran = build()
    registry.dispatch("create_checkout", {})
    first = memory.pending_confirmation

    result = registry.dispatch("create_checkout", {})

    assert memory.pending_confirmation is first
    assert result.content == AWAITING_ANSWER
    assert ran == []


def test_the_total_a_person_confirms_comes_from_the_cart_not_the_model():
    """A person approving a number the model invented is worse than no gate."""
    registry, memory, _ = build()

    registry.dispatch("create_checkout", {})
    (summary,) = settle(memory, True)

    assert "€189.98" in summary
    assert "Trail Runner GTX" in summary
    assert "42 / black" in summary


def test_what_the_model_said_the_total_was_reaches_the_summary_nowhere():
    """The falsification of the rule above, from the model's side.

    The gate reads `view_cart`; it does not read the conversation, and the
    difference is invisible while the model is telling the truth. So the model
    is made to say something else — a final answer claiming a total of €5.00,
    passed through `GuardedClient` exactly as a real one would be — and the
    summary a person is shown is checked against the cart rather than against
    it.

    The claim is also *caught*, which is the other guard doing its own job:
    €5.00 came from no tool result, so the answer never reaches the customer
    either. Both facts are asserted, because a summary that stayed correct
    only because the prose was suppressed would prove nothing about the gate.
    """
    registry, memory, _ = build()
    lie = "Your total is €5.00. Shall I place the order?"
    client = GuardedClient(FakeClient(answer(lie), answer(lie)), memory)

    reply = client.chat_with_tools([{"role": "user", "content": "check me out"}])
    registry.dispatch("create_checkout", {})
    (summary,) = settle(memory, True)

    assert "€5.00" not in summary
    assert "€189.98" in summary
    assert reply.content.startswith(FALLBACK_PREFIX), "the lie was passed on as well"


def test_a_resume_confirms_the_order_and_never_a_total_of_zero():
    """`create_checkout` clears the cart, so a resume reads an empty one.

    The gate summarised that cart anyway and asked a person to approve "About
    to place this order: Total: €0.00" for a purchase of €199.99 that was
    already made — a person approving a figure that is not the real one, which
    is the exact failure the gate exists to prevent. Found in the end-to-end
    run for PR #9.
    """
    registry, memory, ran = build(cart=EMPTY_CART)
    registry.dispatch("create_checkout", {})

    (summary,) = settle(memory, True)
    result = registry.dispatch("create_checkout", {})

    assert result.ok
    assert ran == ["create_checkout"]
    assert "0.00" not in summary
    assert "€199.99" in summary
    assert RESUMING in summary
    assert PLACING not in summary, "this is not a new order and must not say it is"


def test_an_order_being_placed_still_says_so():
    """The other heading, so the two cannot collapse into one."""
    registry, memory, _ = build()

    registry.dispatch("create_checkout", {})
    (summary,) = settle(memory, True)

    assert PLACING in summary
    assert RESUMING not in summary


def test_an_empty_cart_with_no_order_is_left_to_the_tool_to_refuse():
    """Nothing to confirm is not a question to ask a person.

    "The cart is empty, add something" is the tool's own sentence and a better
    answer than a gate asking whether to buy nothing.
    """
    registry, memory, ran = build(cart=EMPTY_CART, order=EMPTY_ORDER)

    registry.dispatch("create_checkout", {})

    assert memory.pending_confirmation is None, "nobody should be asked about nothing"
    assert ran == ["create_checkout"], "the tool answers instead"


def test_a_registry_with_nobody_to_ask_refuses_to_buy():
    """The safe default. A gate that cannot reach a person is not a gate."""
    registry, memory, ran = build(can_confirm=False)

    result = registry.dispatch("create_checkout", {})

    assert ran == []
    assert not result.ok
    assert memory.pending_confirmation is None, (
        "a question nobody can answer is worse than a refusal: it parks state "
        "that will be found later by something looking for an approval"
    )


def test_a_cart_that_cannot_be_read_stops_the_purchase():
    """Nothing is confirmed blind: no cart, no summary, no sale."""
    registry, memory, ran = build(cart=RuntimeError("API down"))

    result = registry.dispatch("create_checkout", {})

    assert ran == []
    assert not result.ok
    assert memory.pending_confirmation is None


def test_an_ordinary_tool_is_never_gated():
    registry, _, _ = build()

    assert registry.dispatch("view_cart", {}).ok


# --- how long an approval lasts ------------------------------------------
#
# The expiry is the part of this protocol that is a decision rather than a
# mechanism, so it gets its own tests. An approval is spendable on exactly one
# turn, and any customer message drops it.
#
# The failure it is set against is one a colleague's repository showed: a
# classifier deciding "did they say yes" from the shape of the word will read a
# "yes" aimed at some other question as authorisation to spend. This gate never
# had that defect, because it asks the question itself — but a pending approval
# that outlives its turn hands the immunity back, since it is then once again
# an answer sitting apart from what it answered.


def test_an_approval_does_not_survive_another_customer_message():
    registry, memory, ran = build()
    registry.dispatch("create_checkout", {})
    settle(memory, True)

    memory.begin_turn(from_customer=True)
    result = registry.dispatch("create_checkout", {})

    assert ran == [], "an approval given before another message was spent after it"
    assert result.content == AWAITING_ANSWER, "and the person is asked again"


def test_an_unanswered_question_does_not_survive_another_customer_message():
    registry, memory, _ = build()
    registry.dispatch("create_checkout", {})

    memory.begin_turn(from_customer=True)

    assert memory.pending_confirmation is None


def test_an_approval_is_good_for_one_turn_and_not_two():
    registry, memory, ran = build()
    registry.dispatch("create_checkout", {})
    settle(memory, True)
    memory.begin_turn(from_customer=False)

    result = registry.dispatch("create_checkout", {})

    assert ran == []
    assert result.content == AWAITING_ANSWER


def test_an_approval_given_but_never_carried_back_dies_on_the_next_message():
    """The ordering the expiry actually exists for.

    A person approves, and then — instead of the caller driving the turn that
    carries the answer to the model — the customer types something else. The
    approval is now sitting one turn away from a question that has scrolled
    past, which is precisely the state a yes-detector is in when it authorises
    a "yes" meant for something else. It has to die here, and the turn counter
    alone does not kill it: the customer's message advances to exactly the turn
    the approval was good for.
    """
    registry, memory, ran = build()
    registry.dispatch("create_checkout", {})
    confirmation.resolve_pending(memory, lambda summary: True)

    memory.begin_turn(from_customer=True)
    result = registry.dispatch("create_checkout", {})

    assert ran == [], "an approval was spent on a turn the customer started"
    assert result.content == AWAITING_ANSWER


def test_an_approval_cannot_be_spent_in_the_turn_it_was_given_in():
    """The answer is carried back in a *new* turn, which is the whole protocol.

    A caller that answers and re-dispatches without advancing the turn has
    written the blocking gate again with extra steps. It gets nothing.
    """
    registry, memory, ran = build()
    registry.dispatch("create_checkout", {})
    confirmation.resolve_pending(memory, lambda summary: True)

    result = registry.dispatch("create_checkout", {})

    assert ran == []
    assert result.content == AWAITING_ANSWER


def test_an_approval_is_spent_once():
    """Two checkouts on one yes would be a second purchase nobody was asked about."""
    registry, memory, ran = build()
    registry.dispatch("create_checkout", {})
    settle(memory, True)

    registry.dispatch("create_checkout", {})
    second = registry.dispatch("create_checkout", {})

    assert ran == ["create_checkout"]
    assert second.content == AWAITING_ANSWER


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


def test_a_correction_answered_with_a_tool_call_dispatches_it():
    """The correction says "call a tool"; doing so must not reach the fallback.

    A tool-call reply normally carries no text, and the retry was accepted only
    when `retry.content` was truthy — so the one behaviour `CORRECTION`
    explicitly asks for was the one that could never satisfy it. The model
    obeyed, the call was thrown away, and the customer got the fallback instead
    of the looked-up figure. Found by review on PR #9.
    """
    looks_it_up = answer(None, [ToolCall(id="1", name="view_cart", arguments="{}")])
    client = FakeClient(answer("That is €94.00."), looks_it_up)
    guarded = GuardedClient(client, memory_holding_the_cart())

    reply = guarded.chat_with_tools([{"role": "user", "content": "total?"}])

    assert reply is looks_it_up
    assert reply.tool_calls, "the lookup has to survive to be dispatched"
    assert not reply.content or not reply.content.startswith(FALLBACK_PREFIX)


def test_a_correction_answered_with_narration_and_a_tool_call_also_passes():
    """Text alongside a tool call is still not a final answer."""
    both = answer("Let me check: €94.00.", [ToolCall(id="1", name="view_cart", arguments="{}")])
    client = FakeClient(answer("That is €94.00."), both)
    guarded = GuardedClient(client, memory_holding_the_cart())

    assert guarded.chat_with_tools([{"role": "user", "content": "total?"}]) is both


@pytest.mark.parametrize(
    "answer_text",
    ["That will be 94 euros.", "That will be 94 euro.", "It costs euros 94."],
)
def test_an_amount_written_as_a_word_is_caught_like_one_written_with_a_symbol(answer_text):
    """"94 euros" is a money claim with no symbol and no decimals.

    Documented as a known miss for one review round, on the argument that the
    prompt teaches the symbol form. That is an argument about what the model
    usually does, and this guard exists for when it does something else — a
    bypass reachable by writing a word is still a bypass. Raised on PR #9.
    """
    assert unsupported_amounts(answer_text, memory_holding_the_cart())


def test_the_word_form_quotes_the_whole_claim_back_not_just_the_number():
    """A correction naming `94` is about a string the model cannot find."""
    (written,) = unsupported_amounts("That will be 94 euros.", memory_holding_the_cart())

    assert written == "94 euros"


def test_a_real_amount_written_as_a_word_is_not_flagged():
    """The guard must not start refusing figures the shop actually quoted."""
    assert unsupported_amounts("That is 189.98 euros.", memory_holding_the_cart()) == []


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


def test_an_approval_is_not_spent_on_a_cart_that_changed_after_it_was_given():
    """The person approved a total, not a tool name.

    The follow-up turn that carries an answer back is a full `run_tool_loop`
    with every tool available, so the model can call `add_to_cart` and then
    `create_checkout` inside it. Binding the approval to the tool name alone
    would spend a yes given for €189.98 on a basket that is now €569.94 — the
    exact laundering the summary exists to prevent, one step later. Raised by
    review on PR #10.
    """
    basket = json.loads(json.dumps(CART))
    registry, memory, ran = build(cart=basket)

    registry.dispatch("create_checkout", {})
    (shown,) = settle(memory, said_yes=True)
    assert "€189.98" in shown

    # What the model can do in the follow-up turn, before checking out.
    basket["items"][0]["quantity"] = 6
    basket["items"][0]["line_total_cents"] = 56994
    basket["unit_count"] = 6
    basket["total_cents"] = 56994

    result = registry.dispatch("create_checkout", {})

    assert ran == [], "an approval for one basket bought another"
    assert not result.ok
