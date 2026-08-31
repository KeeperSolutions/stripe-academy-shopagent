"""What one conversation holds outside its message list (D9, step 3).

This replaces the `CommerceSession` seam step 1 left behind, which said in its
own docstring that it was temporary and what would take it over.

**The message list is not the thing this improves.** Every tool result is
already in the history, so the model can read the second row of an earlier
search and pull a `variant_id` out of it without any help from here. What a
memory adds is three things the history cannot do:

1. **State the model must not see.** The cart id and the order id belong to the
   tool layer, appear in no schema and in no result, and have to live
   somewhere that is not a message.
2. **Survival of a trimmed context.** When the history grows, tool results are
   the first thing to go, and they are exactly what an ordinal reference needs.
3. **A surface something else can check against.** `seen_variant_ids` is the
   set of ids that actually appeared in a tool result in this conversation.
   Step 5 is what refuses an id that never did — the same rule as an amount
   that appears in an answer without appearing in the context. **Nothing here
   refuses anything.** This module records; the rule that reads the record is
   written separately so the two can be reviewed apart.

**Two lifetimes, on purpose.**

`last_search` is *replaced* by every new search. "The second one" can only mean
the second row of the list the customer is currently looking at, and keeping
older lists would let a reference resolve against a list that has scrolled
away — putting the wrong shoe in a basket, with nothing on screen to show it
happened. So an older list is not available to be resolved against at all, and
a reference into one comes back as a sentence telling the model to search
again.

`seen_variant_ids` *accumulates* for the whole conversation. It answers a
different question — "was this id ever put in front of the model here?" — and
that question does not stop being about a variant because another search has
since happened. A shoe found, stock-checked, discussed and then added six
messages later was seen the entire time.

Collapsing the two into one field would mean choosing one of those answers and
being wrong about the other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from shopagent.tools.registry import ToolRegistry, ToolResult

# The tool whose result *is* the list a customer is looking at.
#
# This is a match on a tool's name, which D5 forbade — in `mcp_client/`, where
# it would make this project's client know about this project's server and
# break the one property that module exists to demonstrate. The prohibition is
# about that layer, not about the word. `agent/` is the layer whose job is to
# know what the tools mean: it decides what to tell the model they are for, and
# on step 5 what a bad answer looks like. A layer that may not name a tool
# could not do either. Step 2 leaned on the same distinction from the other
# side, fixing `ping` on the server rather than filtering it in the client.
#
# If the catalog ever publishes a second search, this becomes a set, and it
# still belongs here.
SEARCH_TOOL = "search_products"

# The key a variant id travels under, everywhere it appears: a search result's
# nested `variants`, `check_stock`'s answer, a cart line, an order line. One
# name across four shapes is what makes a recursive sweep the right tool
# rather than four bespoke readers that would each need updating.
VARIANT_ID_KEY = "variant_id"

# Every amount in this project ends in `_cents`, in every result the model can
# see: `price_cents` from the catalog, `unit_price_cents`, `line_total_cents`
# and `total_cents` from carts and orders. That is not a coincidence to lean on
# lightly — it is the naming rule CLAUDE.md draws between a column
# (`amount_cents`) and what a reader gets (`price_cents`), and it means the set
# of amounts the model has been shown can be collected without knowing which
# tool produced them.
#
# Collected rather than every integer, which would be the easy version and the
# wrong one: variant ids, quantities, sizes and stock counts are integers too,
# and a set holding 86263 would quietly support a claim of "€862.63". Step 5
# reads this set to decide whether an amount in an answer came from anywhere.
AMOUNT_SUFFIX = "_cents"


@dataclass(frozen=True)
class LastSearch:
    """One search, in the order the tool returned it."""

    tool: str
    arguments: Any
    results: list[dict]


@dataclass(frozen=True)
class Reference:
    """The outcome of resolving an ordinal, which may be a refusal in words.

    A message rather than an exception, and prose rather than a code, because
    the only reader is the model: the caller's job is to pass it on, not to
    interpret it. `result is None` and `message is not None` always travel
    together — a refusal with nothing to say would be a silence, which is the
    one answer this whole module exists to avoid.
    """

    result: dict | None = None
    message: str | None = None

    @property
    def resolved(self) -> bool:
        return self.result is not None


@dataclass
class ConversationMemory:
    """One conversation's state. Never shared, never global.

    A plain object passed to whatever needs it, for the reason the step 1 seam
    was a closure: two conversations in one process must not be able to reach
    each other's basket, and the way to guarantee that is to have no place
    where a second conversation could look.
    """

    cart_id: str | None = None
    # Set as soon as an order exists, before its payment link is created, so a
    # checkout that fails partway still leaves the order findable.
    order_id: str | None = None

    _last_search: LastSearch | None = None
    _seen_variant_ids: set[int] = field(default_factory=set)
    _seen_amount_cents: set[int] = field(default_factory=set)

    # --- reading ---------------------------------------------------------

    @property
    def last_search(self) -> LastSearch | None:
        return self._last_search

    @property
    def seen_amount_cents(self) -> frozenset[int]:
        """Every amount, in minor units, that a tool has shown the model here.

        Cumulative for the whole conversation, for the same reason
        `seen_variant_ids` is: a price quoted four messages ago is still a
        price this shop gave, and a customer asking "what was that one again"
        is asking about it.
        """
        return frozenset(self._seen_amount_cents)

    @property
    def seen_variant_ids(self) -> frozenset[int]:
        """Every variant id that has appeared in a tool result here.

        Frozen on the way out so a caller cannot add to it: an id becomes
        "seen" by being shown to the model, and any other way of getting into
        this set would make it mean something weaker than it says.
        """
        return frozenset(self._seen_variant_ids)

    def nth_from_last_search(self, position: int) -> Reference:
        """The nth row of the most recent search, counting from one.

        From one because the customer says "the second one" and means the
        second, and an off-by-one here is not a crash — it is the wrong
        product, bought.

        Only the most recent list can be counted in, and there is no way to ask
        for an older one. A `search_id` argument existed for one step so a
        caller could name the list it meant and be refused if that list had
        been replaced — it came out again because the measurement said the
        function it was serving does not exist: the model resolves "the second
        one" from the message history by itself, twice measured, and no tool
        takes an ordinal. Surface for a caller nobody has is surface to get
        wrong. It comes back the day step 5 asks for it.

        Nothing calls this yet.
        """
        if self._last_search is None:
            return Reference(
                message=(
                    f"No search has been run in this conversation yet, so there is no "
                    f"list to count in. Call {SEARCH_TOOL} first, then refer to a "
                    f"result from it."
                )
            )

        count = len(self._last_search.results)
        if position < 1 or position > count:
            return Reference(
                message=(
                    f"The last search returned {count} result(s), so there is no "
                    f"result number {position}. Refer to one between 1 and {count}, "
                    f"or search again."
                )
            )

        return Reference(result=self._last_search.results[position - 1])

    # --- writing ---------------------------------------------------------

    def observe(self, tool: str, arguments: Any, content: str) -> None:
        """Take from one successful tool result whatever is worth keeping.

        Called for every tool, not only the ones this module knows about,
        because `seen_variant_ids` is about what the model was shown rather
        than about which tool showed it. A result that is not JSON — a time, a
        sentence — contributes nothing and is not an error: tools are allowed
        to answer prose.
        """
        try:
            payload = json.loads(content)
        except (ValueError, TypeError):
            return

        found_ids, found_amounts = _numbers_in(payload)
        self._seen_variant_ids.update(found_ids)
        self._seen_amount_cents.update(found_amounts)

        if tool == SEARCH_TOOL:
            self._last_search = LastSearch(
                tool=tool,
                arguments=arguments,
                results=list(_results_in(payload)),
            )


def _results_in(payload: Any) -> list[dict]:
    """The ordered rows of a search result envelope.

    `{count, results}` is the shape the MCP wrapper adds so that "nothing
    matched" can be told apart from "nothing happened"; a bare list is what a
    caller reaching `catalog.search_products` directly gets. Both are read,
    because which one arrives depends on how the tool was reached rather than
    on anything about the search.
    """
    if isinstance(payload, dict):
        rows = payload.get("results", [])
    else:
        rows = payload
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _numbers_in(payload: Any) -> tuple[set[int], set[int]]:
    """Every variant id and every amount anywhere in a decoded result.

    One walk for both, because they are found the same way and at the same
    depths: the results that carry them nest them differently — a search puts
    variants two levels down, `check_stock` puts one at the top, a cart line
    carries both — and a reader written per shape is a place to update the day
    a fifth shape arrives.

    `bool` is excluded explicitly. It is a subclass of `int` in Python, so
    `in_stock: true` would otherwise be recorded as the amount 1 and quietly
    support a claim of "€0.01".
    """
    ids: set[int] = set()
    amounts: set[int] = set()
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if not isinstance(value, int) or isinstance(value, bool):
                    continue
                if key == VARIANT_ID_KEY:
                    ids.add(value)
                elif key.endswith(AMOUNT_SUFFIX):
                    amounts.add(value)
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return ids, amounts


class RememberingRegistry(ToolRegistry):
    """A registry that files every successful result into a conversation's memory.

    A subclass rather than a call inside `run_tool_loop`, so the loop stays what
    D2 left and D5 relied on: it takes a registry and calls `dispatch`, and has
    never needed to know what a tool means. Every tool goes through `dispatch`,
    which makes this the one place that cannot be forgotten — the same argument
    `api/services/orders.py::apply_transition` makes about acting on a
    transition's effects.

    Only successful calls are recorded. A refusal is not a list the customer is
    looking at, and an error message with a number in it is not a variant the
    model was shown.
    """

    def __init__(self, memory: ConversationMemory) -> None:
        super().__init__()
        self._memory = memory

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    def dispatch(self, name: str, raw_args: Any = None) -> ToolResult:
        result = super().dispatch(name, raw_args)
        if result.ok:
            self._memory.observe(name, raw_args, result.content)
        return result
