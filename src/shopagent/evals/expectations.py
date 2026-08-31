"""What a run did, and whether that satisfies what the scenario claimed.

**The criterion is tool calls and database state, not text.** That is the shape
`tests/test_agent_chain.py` settled on in D9 and the only one that separates
"the model answered convincingly" from "the model did the right thing" — the
same class as D6 recording the SQL `render_order` issues and D7 recording the
outgoing checkout payload. An implementation that produced a plausible final
sentence while calling nothing at all satisfies every assertion about the text.

Text is measured in exactly one place, `answer_matches`, and only where the
text *is* the thing: scenario 8 asks whether an ambiguous request produces a
question, and there is nothing else about that a tool log could show.

**Invariant versus variance.** D9 measured two identical runs producing eight
and nine model calls, and an eval that fails one run in five is one people
learn to ignore. So the vocabulary here can only express the invariants:

  * *which* tools were called — `tools_called`, `tools_not_called`
  * the order of the ones where the order is causal — `tools_in_order`,
    a subsequence rather than an exact list
  * what the tools answered and what the database holds afterwards
  * whether a guardrail fired

and it deliberately cannot express the variance: how many times `check_stock`
ran, how many model calls a turn took, or the exact words. There is no
"n failures out of m runs" threshold anywhere, because a threshold is a number
somebody tunes until the suite is green. If a claim is not deterministic, the
claim is wrong — not the threshold.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from shopagent.agent.guardrails import unsupported_amounts
from shopagent.agent.memory import ConversationMemory
from shopagent.evals.spec import Expectation

SEARCH_TOOL = "search_products"
ADD_TOOL = "add_to_cart"
CART_TOOLS = ("view_cart", "add_to_cart", "remove_from_cart")

# What `order_status` may be asked for beyond a real status. `none` is the
# claim that no order was placed at all, which is what scenario 5 is about and
# is not expressible as a status.
NO_ORDER = "none"


@dataclass
class Dispatch:
    """One tool call, as it happened."""

    turn: int
    name: str
    arguments: Any
    ok: bool
    content: str

    def payload(self) -> Any:
        try:
            return json.loads(self.content)
        except (ValueError, TypeError):
            return None


@dataclass
class Observed:
    """Everything one scenario run produced, and nothing derived from it.

    Filled by the runner as the conversation happens; read by the checks below.
    Kept apart from both so a failing expectation can be re-evaluated against a
    recorded run without paying for another one.
    """

    dispatches: list[Dispatch] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    confirmations: list[str] = field(default_factory=list)
    memory: ConversationMemory | None = None
    # Read from the database after the conversation, by id. `None` when no
    # order was placed.
    order_status: str | None = None

    def names(self) -> list[str]:
        return [dispatch.name for dispatch in self.dispatches]

    def successful(self, name: str) -> list[Dispatch]:
        return [d for d in self.dispatches if d.name == name and d.ok]

    def last_answer(self) -> str:
        return self.answers[-1] if self.answers else ""


@dataclass(frozen=True)
class Verdict:
    """One expectation, checked."""

    expectation: Expectation
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"{'PASS' if self.passed else 'FAIL'}  {self.expectation}  — {self.detail}"


def check(expectation: Expectation, observed: Observed) -> Verdict:
    """Evaluate one expectation. Never raises: a broken check is a FAIL."""
    checker = CHECKS[expectation.key]
    try:
        passed, detail = checker(expectation.argument, observed)
    except Exception as exc:  # noqa: BLE001 - a check that throws is a check that failed
        passed, detail = False, f"the check itself raised {type(exc).__name__}: {exc}"
    return Verdict(expectation=expectation, passed=passed, detail=detail)


# --- what the model did ---------------------------------------------------


def _tools_called(wanted: list, observed: Observed) -> tuple[bool, str]:
    called = observed.names()
    missing = [name for name in wanted if name not in called]
    return not missing, (
        f"called {called}" if not missing else f"never called {missing}; called {called}"
    )


def _tools_not_called(forbidden: list, observed: Observed) -> tuple[bool, str]:
    called = observed.names()
    seen = [name for name in forbidden if name in called]
    return not seen, ("none of them ran" if not seen else f"but {seen} ran")


def _tools_in_order(wanted: list, observed: Observed) -> tuple[bool, str]:
    """A *subsequence*, not an exact list.

    Anything stricter would be a claim about variance: a model may check stock
    three times between a search and an add, and D9 measured exactly that. What
    is invariant is that the search happened before the add, because the add
    needs an id only the search could supply.
    """
    remaining = list(wanted)
    for name in observed.names():
        if remaining and name == remaining[0]:
            remaining.pop(0)
    return not remaining, (
        f"in order within {observed.names()}"
        if not remaining
        else f"never reached {remaining[0]}; called {observed.names()}"
    )


# --- what the tools answered ----------------------------------------------


def _search_rows(observed: Observed) -> list[dict]:
    rows: list[dict] = []
    for dispatch in observed.successful(SEARCH_TOOL):
        payload = dispatch.payload()
        found = payload.get("results") if isinstance(payload, dict) else payload
        if isinstance(found, list):
            rows.extend(row for row in found if isinstance(row, dict))
    return rows


def _search_returned_results(wanted: bool, observed: Observed) -> tuple[bool, str]:
    rows = _search_rows(observed)
    return bool(rows) is wanted, f"{len(rows)} result(s)"


def _search_results_cost_at_most_cents(limit: int, observed: Observed) -> tuple[bool, str]:
    """Every price in every search result is within the bound the customer said.

    Reads `price_cents`, which is the flattened field the model actually sees —
    the rename `api/schemas.py` performs, not the `amount_cents` column. Reading
    the column name here would be checking a different number from the one that
    reached the answer.
    """
    rows = _search_rows(observed)
    if not rows:
        return False, "no search results to check"
    over = [
        (row.get("name"), price)
        for row in rows
        for price in _prices_in(row)
        if price > limit
    ]
    return not over, (
        f"{len(rows)} result(s), all at or under {limit}" if not over else f"over the bound: {over}"
    )


def _prices_in(row: dict) -> list[int]:
    prices = []
    top = row.get("price_cents")
    if isinstance(top, int) and not isinstance(top, bool):
        prices.append(top)
    for variant in row.get("variants", []) or []:
        value = variant.get("price_cents") if isinstance(variant, dict) else None
        if isinstance(value, int) and not isinstance(value, bool):
            prices.append(value)
    return prices


def _added_variant_is_search_row(position: int, observed: Observed) -> tuple[bool, str]:
    """The ordinal claim, checked against the list the model was shown.

    This is the one expectation that can tell "the model resolved *the second
    one*" from "the model added something plausible": the answer is a specific
    id, and the only place it could legitimately come from is row `position` of
    the last search this conversation ran. D9 measured the model doing this
    with no mechanism at all, and this is what would notice it stopping.
    """
    added = [d for d in observed.dispatches if d.name == ADD_TOOL]
    if not added:
        return False, f"{ADD_TOOL} was never called"
    arguments = added[-1].arguments
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except ValueError:
            return False, f"{ADD_TOOL} arguments were not JSON: {added[-1].arguments!r}"
    sent = arguments.get("variant_id") if isinstance(arguments, dict) else None

    memory = observed.memory
    last = memory.last_search if memory is not None else None
    if last is None:
        return False, "no search is recorded in this conversation's memory"
    rows = last.results
    if not 1 <= position <= len(rows):
        return False, f"the last search returned {len(rows)} row(s); no row {position}"

    expected = _variant_ids(rows[position - 1])
    return sent in expected, (
        f"added variant {sent}, row {position} offers {sorted(expected)}"
    )


def _variant_ids(row: dict) -> set[int]:
    """Every variant id row `n` of a search result offers.

    A search row is a *product* with variants under it, so "the second one"
    names a product and the model picks a variant within it. Accepting any of
    that product's variants is the honest reading — insisting on one would make
    the check about colour, which the customer never said.
    """
    found = set()
    top = row.get("variant_id")
    if isinstance(top, int) and not isinstance(top, bool):
        found.add(top)
    for variant in row.get("variants", []) or []:
        value = variant.get("variant_id") if isinstance(variant, dict) else None
        if isinstance(value, int) and not isinstance(value, bool):
            found.add(value)
    return found


def _cart_total_cents(wanted: int, observed: Observed) -> tuple[bool, str]:
    """The total the last cart-shaped tool result carried.

    Read from the tool result rather than recomputed, because a cart total is
    computed from the database on every read — recomputing it here would make
    this check a second implementation of the rule it is testing.
    """
    totals = [
        payload.get("total_cents")
        for dispatch in observed.dispatches
        if dispatch.name in CART_TOOLS and dispatch.ok
        for payload in [dispatch.payload()]
        if isinstance(payload, dict) and "total_cents" in payload
    ]
    if not totals:
        return False, "no cart result to read a total from"
    return totals[-1] == wanted, f"last cart total was {totals[-1]}"


# --- what the shop's state became -----------------------------------------


def _order_status(wanted: str, observed: Observed) -> tuple[bool, str]:
    """Read from the database by id, after the conversation.

    Not from `check_order_status`, deliberately. A tool result is what the
    model was told; the row is what is true, and the two coming apart is
    precisely the failure worth catching.
    """
    actual = observed.order_status or NO_ORDER
    return actual == wanted, f"the database says {actual}"


# --- what the guardrails did ----------------------------------------------


def _confirmation_requested(wanted: bool, observed: Observed) -> tuple[bool, str]:
    """Whether the gate put a purchase in front of a person.

    The claim is about the *gate*, not about the model's prose. Measured in
    D10 step 2: the model sometimes asks for confirmation itself before calling
    the tool, which is neither required nor forbidden — the gate is what
    decides, and it decides the same either way.
    """
    asked = bool(observed.confirmations)
    return asked is wanted, f"{len(observed.confirmations)} confirmation(s) requested"


def _confirmation_summary_matches(pattern: str, observed: Observed) -> tuple[bool, str]:
    """What a person was actually shown, as text — and it is not the model's.

    The exception to "no text assertions" that proves the rule: this string was
    built by `agent/guardrails.py` from a real `view_cart` result and rendered
    through `money.format_amount`, so asserting on it is asserting on the
    shop's own number. That is the D9 property the whole gate exists for.
    """
    if not observed.confirmations:
        return False, "nobody was asked to confirm anything"
    last = observed.confirmations[-1]
    return bool(re.search(pattern, last)), f"summary was {last.strip()!r}"


def _every_amount_traceable(wanted: bool, observed: Observed) -> tuple[bool, str]:
    """No figure reached the customer that no tool produced.

    This is scenario 6, and it is a claim about the *property* rather than
    about the mechanism. Making a real model invent a price on demand is not
    something that can be ordered up — D9 recorded the fallback never firing in
    a live run and D10 step 2 measured the same — so an eval asserting "the
    guardrail fired" would be asserting something no run reliably produces, and
    faking it with a scripted client would be a unit test of the guardrail
    wearing an eval's clothes.

    What holds in both branches is this: whatever the model did, no untraceable
    amount is in the answer. If it invented one, `GuardedClient` corrected or
    replaced it and this passes; if it did not, this passes for the plainer
    reason. Which branch happened is reported rather than asserted.
    """
    memory = observed.memory
    if memory is None:
        return False, "no conversation memory to check against"
    bad = [
        amount
        for answer in observed.answers
        for amount in unsupported_amounts(answer, memory)
    ]
    return (not bad) is wanted, (
        "every amount came from a tool result" if not bad else f"untraceable: {bad}"
    )


# --- what the customer read -----------------------------------------------


def _answer_matches(pattern: str, observed: Observed) -> tuple[bool, str]:
    answer = observed.last_answer()
    return bool(re.search(pattern, answer)), f"final answer was {answer.strip()[:160]!r}"


CHECKS = {
    "tools_called": _tools_called,
    "tools_not_called": _tools_not_called,
    "tools_in_order": _tools_in_order,
    "search_returned_results": _search_returned_results,
    "search_results_cost_at_most_cents": _search_results_cost_at_most_cents,
    "added_variant_is_search_row": _added_variant_is_search_row,
    "cart_total_cents": _cart_total_cents,
    "order_status": _order_status,
    "confirmation_requested": _confirmation_requested,
    "confirmation_summary_matches": _confirmation_summary_matches,
    "every_amount_traceable": _every_amount_traceable,
    "answer_matches": _answer_matches,
}
