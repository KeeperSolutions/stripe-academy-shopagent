"""Rules the code enforces, because asking the model did not work (D9, step 5).

Two measurements put this file here rather than in the prompt.

D2 told the model never to do arithmetic in its head, asked it for `5
factorial`, and got `120` with zero tool calls. The answer was right, which is
the uncomfortable part: nothing marked it as unverified. The entry that
recorded it named the fix — *"an amount that appears in an answer without
appearing in the context has to be blocked, not discouraged"*.

Run B of the chain test measured the other half. `create_checkout`'s
description said to "get an explicit yes first"; the customer had just said
"yes, order it"; the model showed the cart and asked for the yes again, because
the sentence never said the previous message could be that yes. An instruction
that states a precondition without stating how it is met cannot be met.

Three rules, and the shape of all three is the same: something the model
produces is checked against something the tools produced.

**The gate asks a person.** `create_checkout` is intercepted before it runs,
the cart is read through `view_cart`, and a human is shown what they are buying
at a total that came from the shop rather than from the model. There is no
`confirmed` argument, deliberately: an argument the model sets is a suggestion
with a type annotation.

D10 made that question non-blocking and changed nothing else about it. The gate
parks the summary and returns; somebody answers between turns; the next call
spends the answer. The reason is in `agent/confirmation.py` — an eval runner
and a browser both answer in a different turn from the one that asked, and a
protocol built on blocking cannot be adapted to either. The total a person
approves still comes from `view_cart` and never from the model's prose, which
is the property the whole gate exists for.

**No amount reaches a customer that no tool produced.** One retry with a
correction, then a fallback that names the figure it could not trace. Not a
third attempt, because regeneration is billed; not silence, because a blocked
answer with nothing in its place is worse than a wrong one the customer can
see.

**No variant is added that the model has not been shown.** `seen_variant_ids`
was built in step 3 and left unused; this reads it.

**Where the boundary is.** These guards live between the agent and the tools,
which means they bind the model and nothing else. `tools/commerce.py` holds
plain functions anybody can import, and the commerce API answers any client
holding the key — a person with `curl` can place an order without passing
through here, and that is correct: the gate exists because a model can be
talked into spending money, not because HTTP is dangerous. The authoritative
protections for the *shop* are elsewhere and unchanged — `place_order` locks
inventory, the lifecycle table refuses illegal transitions, and only a signed
webhook may mark an order paid.
"""

from __future__ import annotations

import json
import re
from typing import Any

from shopagent.agent.confirmation import PendingConfirmation
from shopagent.agent.memory import ConversationMemory, RememberingRegistry
from shopagent.config import get_settings
from shopagent.money import WORDS, ZERO_DECIMAL_CURRENCIES, SYMBOLS, format_amount
from shopagent.obs.tracing import Tracer
from shopagent.tools.registry import ToolResult

# The calls that spend money. A set rather than a check inside the tool,
# because the tool is a plain function reachable without this layer and the
# rule belongs to the agent: it is about who is allowed to decide, not about
# what a checkout is.
# The calls that move money, in either direction. `request_refund` is here for
# a reason the name "spend" does not cover: a refund gives money *back*, so it
# cannot be the theft the gate was built against — but it is **terminal**.
# `refunded` has no outgoing transition and `paid -> paid` is refused, so a
# refund the customer did not ask for cannot be undone by this system at all.
# "Irreversible" is the property that earns a confirmation, and spending is only
# the most obvious way to be irreversible.
CONFIRM_BEFORE = frozenset({"create_checkout", "request_refund"})

# The tool the gate reads the confirmation total from, and the tool whose
# argument is checked against what the model has been shown. Named here for the
# reason `agent/memory.py` names `search_products`: this is the layer whose job
# is to know what the tools mean.
CART_TOOL = "view_cart"
ORDER_TOOL = "check_order_status"
ADD_TOOL = "add_to_cart"
CHECKOUT_TOOL = "create_checkout"
REFUND_TOOL = "request_refund"

# The two things `create_checkout` can be about, and the person confirming has
# to be told which. It places an order from the cart; it also hands back the
# payment link of an order already placed, for a customer who lost it. Those
# are not the same commitment, and the second one reads an empty cart — which
# summarised as "About to place this order: Total: €0.00", a figure that is
# neither the order's nor anything else's. Found in the end-to-end run for
# PR #9, in the fix that made the resume reachable.
# Indented to sit with the summary lines under it, which is where the CLI used
# to add the indentation itself.
PLACING = "  About to place this order:"
RESUMING = "  That order is already placed. This only fetches its payment link again:"

# What a refund is about. It says "in full" because that is the only refund
# this system can issue — `create_refund` takes no amount, and a partial one
# has nowhere to live — and a person approving "a refund" without that word
# could reasonably think they were returning one item.
REFUNDING = "  About to refund this whole order:"

# What the model is told while a person is being asked (D10, step 1). The gate
# no longer blocks for an answer, so this is the result of the *first*
# `create_checkout` in a conversation: the question has been put, and the tool
# has not run.
#
# It says three things and each is there because of a way the model could
# otherwise fill the gap. That nothing was ordered, because a result the model
# cannot read as success or failure is one it guesses about. That it must not
# ask for the confirmation itself, because the shop is already asking and two
# questions about one purchase is how a customer ends up answering the wrong
# one. And that the answer arrives later, because a model told only "not yet"
# retries.
AWAITING_ANSWER = (
    "The customer is being asked to confirm this purchase right now: the shop "
    "is showing them the order and its total and waiting for their answer. "
    "Nothing has been ordered and nothing has been charged. Do not call "
    "create_checkout again and do not ask them to confirm yourself — the shop "
    "is asking them directly. Say briefly that you are waiting for their "
    "confirmation and stop there. Their answer will reach you next."
)

BASKET_CHANGED = (
    "The basket is not the one the customer approved — it changed after they "
    "said yes, so their approval does not cover it and nothing has been "
    "ordered or charged. The shop is now showing them the new total and "
    "waiting for a fresh answer. Do not call create_checkout again and do not "
    "ask them to confirm yourself. Say briefly that the order changed and you "
    "are waiting for their confirmation, and stop there."
)

# The same two, for a refund. Written out rather than templated off the tool
# name, because the facts differ and not only the verb: a parked *purchase* has
# charged nothing, while a parked *refund* leaves an order that is still paid,
# and telling the model "nothing has been charged" about it would be false.
AWAITING_REFUND_ANSWER = (
    "The customer is being asked to confirm this refund right now: the shop is "
    "showing them the order and its total and waiting for their answer. No "
    "refund has been requested and their order is unchanged. Do not call "
    "request_refund again and do not ask them to confirm yourself — the shop "
    "is asking them directly. Say briefly that you are waiting for their "
    "confirmation and stop there. Their answer will reach you next."
)

REFUND_ORDER_CHANGED = (
    "The order is not the one the customer approved a refund for — it changed "
    "after they said yes, so their approval does not cover it and no refund "
    "has been requested. The shop is now showing them the order as it stands "
    "and waiting for a fresh answer. Do not call request_refund again and do "
    "not ask them to confirm yourself. Say briefly that the order changed and "
    "you are waiting for their confirmation, and stop there."
)

# Which pair a gated tool speaks with. The mapping is the reason
# `_unconfirmed` and `_spend` take the tool name through rather than reaching
# for a module constant: two gated tools now, and the checkout's wording is
# wrong for the other one in both branches.
_AWAITING = {CHECKOUT_TOOL: AWAITING_ANSWER, REFUND_TOOL: AWAITING_REFUND_ANSWER}
_CHANGED = {CHECKOUT_TOOL: BASKET_CHANGED, REFUND_TOOL: REFUND_ORDER_CHANGED}


# --- amounts -------------------------------------------------------------
#
# What counts as an amount claim, and what deliberately does not.
#
# Caught: a number carrying the currency symbol or its ISO code on either side,
# and any bare number written with exactly two decimal places. In this shop a
# two-decimal number is money by construction — nothing else here is fractional.
#
# Not caught: a bare integer. This domain is full of them — size 42, 3 in
# stock, 2 units, variant 86263 — and flagging them would make the guardrail
# noisy in exactly the situation where it has to be trusted. That is the same
# trade `find_column_gaps` makes when it declines to report extra columns.
#
# Also caught: the currency written as a word, singular or plural — "190 euros".
# This was a documented miss for one review round, on the argument that the
# prompt teaches the symbol form and every measured run had used it. That is an
# argument about what the model usually does, and a guardrail is for the times
# it does something else: an unambiguous claim about money that the guard
# cannot see is a bypass, and one the model reaches by writing a word. The
# spellings live in `money.WORDS`, one entry, under the same policy as
# `money.SYMBOLS` — the shop sells in one currency at a time. Raised in review
# on PR #9.

_DECIMAL = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?"


def _patterns(currency: str) -> list[re.Pattern[str]]:
    """The ways an amount can be written, most specific first.

    Order is not cosmetic. `amount_claims` keeps the first match on a span and
    drops anything overlapping it, so whichever pattern runs first decides what
    the model gets quoted back. The bare-two-decimal pattern would find `190.00`
    inside `190.00 euros`, and a correction naming `190.00` is about a
    different string than the one the model wrote — so every form that carries
    a currency marker is tried before the one that carries none.
    """
    symbol = re.escape(SYMBOLS[currency]) if currency in SYMBOLS else None
    code = re.escape(currency)
    # Longest first, so "euros" is matched whole rather than as "euro" with a
    # stray "s" left behind.
    words = sorted(WORDS.get(currency, ()), key=len, reverse=True)
    spelled = "|".join(re.escape(word) for word in words) if words else None

    patterns: list[re.Pattern[str]] = []

    if symbol:
        # €94.99 and 94.99€
        patterns.append(re.compile(rf"{symbol}\s*({_DECIMAL})"))
        patterns.append(re.compile(rf"({_DECIMAL})\s*{symbol}"))

    if spelled:
        # 190 euros, and the rarer "euros 190"
        patterns.append(re.compile(rf"({_DECIMAL})\s*(?:{spelled})\b", re.IGNORECASE))
        patterns.append(re.compile(rf"\b(?:{spelled})\s*({_DECIMAL})", re.IGNORECASE))

    # 94.99 EUR and EUR 94.99
    patterns.append(re.compile(rf"({_DECIMAL})\s*{code}\b", re.IGNORECASE))
    patterns.append(re.compile(rf"\b{code}\s*({_DECIMAL})", re.IGNORECASE))

    # A bare number with exactly two decimals. Last, because it carries no
    # marker and would otherwise claim the number out of every form above.
    patterns.append(
        re.compile(rf"(?<![\d.,]) ?(\d{{1,3}}(?:,\d{{3}})*\.\d{{2}}|\d+\.\d{{2}})(?![\d])")
    )
    return patterns


def amount_claims(text: str, currency: str | None = None) -> list[tuple[str, str]]:
    """Every amount in `text`, as `(as written, the number in it)`.

    Both halves are needed and they are not the same string. The number is what
    gets converted to minor units and looked up; the written form is what a
    correction quotes back, and a message naming `94.00` when the model wrote
    `€94.00` is a message about a different string than the one it has to fix.

    Overlapping matches are dropped: `€94.99` is found by the symbol pattern
    first, and the bare-two-decimal pattern would find `94.99` inside it a
    moment later. Reporting both would tell the model it stated two amounts
    when it stated one.
    """
    currency = (currency or get_settings().currency).lower()
    found: list[tuple[str, str]] = []
    taken: list[tuple[int, int]] = []
    for pattern in _patterns(currency):
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < other_end and other_start < end for other_start, other_end in taken):
                continue
            taken.append((start, end))
            found.append((match.group(0).strip(), match.group(1).strip()))
    return found


def _candidates(written: str, currency: str) -> set[int]:
    """The minor-unit values a written amount could legitimately mean.

    Two, when the number has no decimal part, and that is the conversion
    `MONEY_PROMPT` explicitly permits: `9499` in a tool result may be quoted as
    `€94.99`, so `€9499` and `€94.99` both have to be checked against the set
    rather than one of them being refused for being the other side of a rule
    the prompt allows.
    """
    plain = written.replace(",", "")
    try:
        value = float(plain)
    except ValueError:  # pragma: no cover - the patterns cannot produce this
        return set()

    if "." in plain:
        return {round(value * 100)}
    if currency in ZERO_DECIMAL_CURRENCIES:
        return {int(value)}
    return {int(value) * 100, int(value)}


def unsupported_amounts(text: str, memory: ConversationMemory, currency: str | None = None) -> list[str]:
    """Amounts in `text` that no tool result in this conversation produced.

    Returned as they were written, so a correction can quote the model back to
    itself — a message naming `18998` when the model wrote `€189.98` is a
    message about a different string than the one it has to fix.
    """
    currency = (currency or get_settings().currency).lower()
    seen = memory.seen_amount_cents
    return [
        written
        for written, number in amount_claims(text, currency)
        if not (_candidates(number, currency) & seen)
    ]


CORRECTION = (
    "Your last answer stated {amounts}, and no tool result in this "
    "conversation contains that amount. Every figure you give the customer has "
    "to come from a tool result — a price from the catalogue, a line total or a "
    "cart total — and you may not add, subtract or compare amounts to produce a "
    "new one. Answer again using only amounts that appeared in a tool result, "
    "or call a tool to get the right figure. If you cannot, say plainly that "
    "you do not have that number."
)

FALLBACK_PREFIX = "I nearly quoted a figure I cannot trace to this shop's own data"

FALLBACK = (
    FALLBACK_PREFIX + " ({amounts}), so I have not stated it. Ask me to show "
    "your cart or to look the product up again, and I will quote only the "
    "prices that come back from the shop."
)


class GuardedClient:
    """An LLM client that will not pass on an amount no tool produced.

    A wrapper around the client rather than a change to `run_tool_loop`, which
    is what keeps that function what D2 left and D5 relied on — it still takes
    a registry and a list of schemas and knows nothing else. The retry happens
    entirely inside this call, on a copy of the message list, so the
    conversation the loop is holding never contains the rejected answer or the
    correction: a transcript that carried both would show the customer being
    corrected for something they never saw.

    Only a *final* answer is checked. A turn that is still asking for tools is
    narration on the way to an answer, and the numbers it mentions may be about
    to arrive.

    `tracer` is optional and defaults to one that records nothing. It is here
    rather than outside because this is the only place that knows the amount
    rule fired: the retry and the fallback both leave the loop looking like an
    ordinary answer, and the alternative — a wrapper further out sniffing for
    `FALLBACK_PREFIX` in the text — would make a guard's visibility depend on
    matching a sentence somebody may rewrite. D9's other two guardrails need
    nothing like this: the gate and the unknown-variant refusal both come back
    as a failed `ToolResult`, which `obs/instrumentation.py` already sees.
    """

    def __init__(
        self,
        client: Any,
        memory: ConversationMemory,
        tracer: Tracer | None = None,
    ) -> None:
        self._client = client
        self._memory = memory
        self._tracer = tracer or Tracer()

    def __getattr__(self, name: str) -> Any:
        # Everything else — `model`, `stream_chat`, the tracker — is the real
        # client's. Only the one method with a rule attached is intercepted.
        return getattr(self._client, name)

    def chat_with_tools(self, messages: Any, tools: Any = None) -> Any:
        reply = self._client.chat_with_tools(messages, tools)
        if reply.tool_calls or not reply.content:
            return reply

        bad = unsupported_amounts(reply.content, self._memory)
        if not bad:
            return reply

        self._tracer.guardrail(
            name="untraceable_amount", outcome="retried with a correction", detail=bad
        )
        retry = self._client.chat_with_tools(
            [
                *messages,
                reply.to_message(),
                {"role": "system", "content": CORRECTION.format(amounts=", ".join(bad))},
            ],
            tools,
        )
        # A retry that asks for a tool is not a final answer and is not
        # checked, exactly like a first attempt that asks for one. `CORRECTION`
        # ends by telling the model to "call a tool to get the right figure",
        # and a tool-call reply normally carries no text at all — so treating
        # an empty `content` as a failed retry replaced the very behaviour the
        # correction asked for with the fallback, and the lookup was never
        # dispatched. Raised in review on PR #9.
        if retry.tool_calls:
            return retry

        if retry.content and not unsupported_amounts(retry.content, self._memory):
            return retry

        # Two attempts is where this stops. A third is billed, and a model that
        # has produced the same untraceable figure twice is not one round away
        # from tracing it.
        still_bad = unsupported_amounts(retry.content or "", self._memory) or bad
        self._tracer.guardrail(
            name="untraceable_amount", outcome="answered with the fallback", detail=still_bad
        )
        return type(reply)(
            content=FALLBACK.format(amounts=", ".join(still_bad)),
            tool_calls=[],
            usage=retry.usage,
        )


class GuardedRegistry(RememberingRegistry):
    """The registry, plus the two rules that have to run before a tool does.

    On top of `RememberingRegistry` rather than beside it, because both are
    about `dispatch` and a conversation needs one registry: the memory is what
    the guards read, so splitting them would mean wiring the same object into
    two wrappers and hoping they stayed in the same order.

    `can_confirm` says whether anybody can be reached at all. False refuses a
    purchase rather than allowing it — a gate that cannot reach a person is not
    a gate, and defaulting the other way would make every non-interactive
    caller an exception to the rule. That was D9's rule and it has not moved;
    what moved is that this is a fact rather than a callable.

    D9 held the confirmer here and called it from inside `dispatch`. It does
    not any more, and the boolean is the honest shape of what is left: this
    class needs to know that somebody exists, and holding a callable it never
    invokes would read as a bug to everyone after the first reader. The
    confirmer belongs to whatever is presenting the conversation, which is the
    only thing that knows how to reach a person — see `agent/confirmation.py`.
    """

    def __init__(self, memory: ConversationMemory, can_confirm: bool = False) -> None:
        super().__init__(memory)
        self._can_confirm = can_confirm

    def dispatch(self, name: str, raw_args: Any = None) -> ToolResult:
        if name == ADD_TOOL:
            refusal = self._unknown_variant(raw_args)
            if refusal is not None:
                return refusal

        if name in CONFIRM_BEFORE:
            refusal = self._unconfirmed(name)
            if refusal is not None:
                return refusal

        return super().dispatch(name, raw_args)

    # --- a variant nobody was shown --------------------------------------

    def _unknown_variant(self, raw_args: Any) -> ToolResult | None:
        """Refuse an id that has not appeared in a tool result here.

        The model can only have got a `variant_id` from a tool result or from
        nowhere, and every successful result is recorded — so this refuses
        exactly the ids it invented. A customer who knows a sku says the sku,
        and the model still has to look it up, which puts the id in the set
        before it can be used.
        """
        arguments = _as_dict(raw_args)
        if arguments is None:
            # Not this guard's problem: `dispatch` explains malformed
            # arguments to the model far better than a guess here would.
            return None

        variant_id = arguments.get("variant_id")
        if not isinstance(variant_id, int) or isinstance(variant_id, bool):
            return None
        if variant_id in self.memory.seen_variant_ids:
            return None

        return ToolResult(
            ok=False,
            content=(
                f"Error: variant_id {variant_id} has not appeared in any result "
                f"in this conversation, so it cannot be added. Do not use an id "
                f"from memory or from an earlier conversation. Search for the "
                f"product first and add a variant from the results that come "
                f"back."
            ),
            error=f"variant_id {variant_id} was never shown in this conversation",
        )

    # --- the gate --------------------------------------------------------

    def _unconfirmed(self, name: str) -> ToolResult | None:
        """Ask a person, with a total that came from the cart.

        The summary is built from a real `view_cart` call rather than from
        anything the model said. A person approving a figure the model invented
        is worse than no gate at all: it launders the invention through a human
        and leaves a record saying they agreed to it. That property is the one
        thing D10 was not allowed to lose while making the gate non-blocking,
        and it is unchanged: the same `view_cart` dispatch, the same
        `_summarise`, the same `money.format_amount`.

        What changed is when the answer arrives. The question is parked and the
        call returns; a person answers it between turns; the *next* call
        through here spends that answer. Three outcomes, and each is a branch
        below: answered yes, answered no, and nothing answered yet.
        """
        answered = self.memory.take_confirmation(name)
        if answered is not None:
            if answered.answer:
                return self._spend(name, answered)
            if name == REFUND_TOOL:
                return ToolResult(
                    ok=False,
                    content=(
                        "Error: the customer did not confirm the refund, so no "
                        "refund was requested and their order is unchanged. "
                        "Acknowledge that, and do not call request_refund again "
                        "unless they ask for it in a later message."
                    ),
                    error="the customer declined the refund",
                )
            return ToolResult(
                ok=False,
                content=(
                    "Error: the customer did not confirm the purchase, so no "
                    "order was placed and nothing was charged. Acknowledge "
                    "that, and do not call create_checkout again unless they "
                    "ask for it in a later message."
                ),
                error="the customer declined the purchase",
            )

        parked = self.memory.pending_confirmation
        if parked is not None and parked.tool == name:
            # Asked already, in this same turn, and nobody has answered yet.
            # Repeat the sentence rather than reading the cart and asking a
            # second time: a person is looking at one question, and a second
            # summary would be a second thing to approve.
            return ToolResult(
                ok=False,
                content=_AWAITING[name],
                error="waiting for the customer to confirm",
            )

        if not self._can_confirm:
            return ToolResult(
                ok=False,
                content=(
                    "Error: this checkout could not be confirmed with the "
                    "customer, so nothing was ordered and nothing was charged. "
                    "Tell them the order was not placed."
                ),
                error="no confirmation is possible in this session",
            )

        summary = self._describe(name)
        if isinstance(summary, ToolResult) or summary is None:
            return summary

        self.memory.park_confirmation(name, summary)
        return ToolResult(
            ok=False, content=_AWAITING[name], error="waiting for the customer to confirm"
        )

    def _spend(self, name: str, approved: PendingConfirmation) -> ToolResult | None:
        """Let the checkout run, but only against what the person actually saw.

        A person approves a basket, not a tool name. The turn that carries
        their answer back is a whole `run_tool_loop` with every tool available,
        so the model can call `add_to_cart` between the yes and the checkout —
        and an approval bound to the name alone would spend a yes given for
        €189.98 on a basket that is now €569.94. That is the same laundering
        `_summarise` exists to prevent, one step later in the protocol: a
        record saying somebody agreed to a figure they were never shown.

        The comparison is against the summary itself rather than a separate
        fingerprint, because the summary already *is* the thing they approved —
        every line, every quantity, every price and the total, rendered through
        `money.format_amount`. A second representation would be a second record
        of one fact, and the first symptom would be the two disagreeing about
        what "the same basket" means.

        A change is not an error and is not a refusal: it is a new question.
        The new summary is parked, the customer is asked about what the basket
        is *now*, and — because `_settle_confirmation` drives exactly one
        follow-up turn — that question lapses at their next message rather than
        driving a loop the model controls. Raised by review on PR #10.
        """
        current = self._describe(name)
        if isinstance(current, ToolResult) or current is None:
            # Unreadable, or nothing left to buy. Both are already answered
            # above: the first is a sentence about the cart, and the second
            # lets the tool say "the cart is empty" in its own words.
            return current
        if current == approved.summary:
            # The one path on which a checkout runs.
            return None

        self.memory.park_confirmation(name, current)
        return ToolResult(
            ok=False,
            content=_CHANGED[name],
            error="the basket changed after it was approved",
        )

    def _describe(self, tool: str) -> ToolResult | str | None:
        """What a person would be shown right now, read from the shop.

        Three answers, and each is a different thing for the caller to do: a
        string is the summary, `None` means there is nothing to confirm and the
        tool should answer in its own words, and a `ToolResult` is the shop
        being unreadable. Still **one function**, which is the property that
        matters: parking a question and spending its answer have to describe the
        same thing the same way, and two renderers would be two opinions about
        whether it changed.

        `tool` selects which question is being asked, not how it is rendered.
        Both branches end in `_summarise`, which already reads a cart and an
        order identically because `view_cart` and `check_order_status` return
        the same shape. So adding the refund added a *question* and no second
        renderer — the thing the one-function rule was protecting is untouched.
        """
        if tool == REFUND_TOOL:
            return self._describe_order_for_refund()
        cart = super().dispatch(CART_TOOL, {})
        if not cart.ok:
            return ToolResult(
                ok=False,
                content=(
                    "Error: the cart could not be read, so the customer could "
                    "not be shown what they were about to buy and nothing was "
                    "ordered. Call view_cart and tell them what you find."
                ),
                error="the cart could not be read before confirming",
            )

        if _has_lines(cart.content):
            return f"{PLACING}\n{_summarise(cart.content)}"

        # An empty cart here is not a mistake: `create_checkout` clears the
        # cart when it places the order, so this is what a resume looks like.
        # Summarise the order instead — the alternative is asking a person to
        # approve a total of zero for a purchase that is real.
        order = super().dispatch(ORDER_TOOL, {})
        if not order.ok or not _has_lines(order.content):
            # Nothing to confirm at all. Let the tool answer: "the cart is
            # empty, add something" is its sentence and a better one than any
            # question this gate could ask about nothing.
            return None
        return f"{RESUMING}\n{_summarise(order.content)}"

    def _describe_order_for_refund(self) -> ToolResult | str | None:
        """The order a full refund would be issued against.

        The cart is not read at all, and that is the whole difference between
        the two questions. By the time a refund is possible the cart is empty —
        `create_checkout` clears it when it places the order — so summarising a
        basket here would put "Total: €0.00" in front of somebody about to give
        up a real order. That exact figure is the defect PR #9 found on the
        resume path, arriving again from the other direction.

        `None` when there is no order: `request_refund` then says "nothing has
        been placed in this conversation" in its own words, which is a better
        sentence than any question this gate could ask about nothing. That is
        the same handover the empty-cart branch above makes.

        **"There is no order" is read from the memory and not from the tool's
        refusal, and that distinction is load-bearing.** Both arrive as a
        failed `ToolResult` — `check_order_status` refuses an absent order, and
        a commerce API that is down refuses everything — so a single `not
        order.ok` check cannot tell them apart. Treating both as "nothing to
        confirm" would let the gate stand aside on a *transport* failure, and
        `request_refund` would then run with nobody asked. The checkout branch
        above gets away with the same shape only because an empty cart makes
        `create_checkout` refuse anyway; a refund has no such second lock, so
        this asks the memory, which is the same field the tool itself checks.
        """
        if self.memory.order_id is None:
            return None
        order = super().dispatch(ORDER_TOOL, {})
        if not order.ok or not _has_lines(order.content):
            # An order this conversation placed, that cannot be read right now.
            # Never `None`: that would run the refund unconfirmed.
            return ToolResult(
                ok=False,
                content=(
                    "Error: the order could not be read, so the customer could "
                    "not be shown what they were about to have refunded and no "
                    "refund was requested. Call check_order_status and tell "
                    "them what you find."
                ),
                error="the order could not be read before confirming",
            )
        return f"{REFUNDING}\n{_summarise(order.content)}"


def _as_dict(raw_args: Any) -> dict[str, Any] | None:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args or "{}")
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _has_lines(content: str) -> bool:
    """Whether a cart or an order actually holds anything.

    One reader for both, because `view_cart` and `check_order_status` return
    the same shape — items with a name, a label, a quantity and a line total.
    That is what lets the gate summarise either without a second renderer.
    """
    try:
        payload = json.loads(content)
    except ValueError:
        return False
    return bool(isinstance(payload, dict) and payload.get("items"))


def _summarise(content: str) -> str:
    """The cart, as the person about to spend the money reads it.

    Rendered through `shopagent.money.format_amount`, which is the same
    function the checkout page uses and the same one the system prompt teaches
    the model to imitate. Three surfaces, one format, so the figure somebody
    approves here is the figure they see on the payment page.
    """
    try:
        cart = json.loads(content)
    except ValueError:
        return content

    currency = cart.get("currency") or get_settings().currency
    lines = []
    for item in cart.get("items", []):
        label = " / ".join(
            part for part in (item.get("product_name"), item.get("variant_label")) if part
        )
        lines.append(
            f"  {item.get('quantity')} x {label} — "
            f"{format_amount(item.get('line_total_cents'), currency)}"
        )
    lines.append(f"  Total: {format_amount(cart.get('total_cents'), currency)}")
    return "\n".join(lines)
