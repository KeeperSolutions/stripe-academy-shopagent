"""What the model is told before the conversation starts (D9, step 2).

Here rather than in `llm/loop.py` for one reason: the prompt is the part of
this system most likely to change, and it should be changeable without opening
the file that holds the agent loop. `loop.py` is a mechanism — a `while`, a
message list, a dispatch — and it has been byte-stable since D2 on purpose,
which is a claim D5 and D9 both leaned on. Editing a sentence about how to
quote a price should not touch it. `agent/` is also where steps 3 and 4 put
memory and guardrails, and all three are the same kind of thing: policy about
how this assistant behaves, as opposed to how the loop runs.

**The prompt is assembled, not written once.** Four blocks: who the assistant
is, whether the catalog is reachable, what it may do with a cart, and how it
handles money. The catalog block has two versions because the catalog has an
off switch and the model has to be told which world it is in; the commerce and
money blocks have one, because those tools are always present.

**What is deliberately not in here.** Nothing about confirming a checkout —
that is step 5, and it will be a gate in code, which can see who said yes where
an instruction cannot. Nothing about memory, which is steps 3 and 4. And no
restatement of what a tool says when it fails: `tools/commerce.py` writes those
sentences against measured failures, and a paraphrase here would be a second
copy of a contract, aging quietly.
"""

from __future__ import annotations

from shopagent.config import get_settings
from shopagent.llm.client import Message
from shopagent.money import format_amount

# The currency the shop sells in, and one amount rendered the way every other
# surface renders it. The example is *generated* rather than typed: this
# sentence used to say `$94.99` while the checkout page said `42.00 USD`, which
# is one amount in two formats seen by one shopper a click apart. Calling the
# formatter here means the assistant cannot drift from the page even in
# principle — `test_the_prompt_and_the_checkout_page_share_one_formatter`
# asserts it is that function and not a copy of it.
_CURRENCY = get_settings().currency
_EXAMPLE_MINOR_UNITS = 9499
_EXAMPLE = format_amount(_EXAMPLE_MINOR_UNITS, _CURRENCY)

# Who it is, and the two rules D1 and D2 measured the need for. `never do
# arithmetic in your head` is from D2, where the model answered "5 factorial"
# from memory rather than reaching for the calculator; MONEY_PROMPT below
# carves out the one exception it has to have, in the place where the exception
# is made.
SYSTEM_PROMPT = (
    "You are ShopAgent, an online shopping assistant. "
    "Always reply in English, regardless of the language of the question. "
    "Keep answers short and concrete, with no preamble and no restating of "
    "the question. If you do not know something or lack the data, say so "
    "plainly — never guess. "
    "Use the tools for anything they cover: you have no clock, so never state "
    "a time from memory, and never do arithmetic in your head."
)

# What the catalog tools are for, not how they work. Each one already carries a
# description written for a model to read, and repeating that here would give
# the same contract two authors and let them drift. This says only when to
# reach for them, and what is never allowed to come from memory.
CATALOG_PROMPT = (
    " The product catalogue is available through tools. Every product name, "
    "price, size, colour and stock level you state must have come from a tool "
    "result in this conversation — never from memory, and never inferred from "
    "what a product sounds like. When the user asks about products, search "
    "first and answer from what comes back. If a search returns a count of 0, "
    "say plainly that nothing matched and suggest a broader search; do not "
    "offer a product that was not in a result."
)

# Said when the catalog server could not be reached, so the model does not
# apologise for its own memory when the real answer is that a tool is missing.
NO_CATALOG_PROMPT = (
    " The product catalogue is NOT available in this session: the tools that "
    "search it could not be loaded. If the user asks about products, prices or "
    "stock, say the catalogue is unavailable right now. Do not answer from "
    "memory and do not invent products."
)

# The cart. Two things are load-bearing and neither is about being polite.
#
# The first is that no identifier is the model's to carry: the tools hold the
# cart, so a model that mentions one is either reading back something it should
# not have or inventing one. Saying it here as well as withholding it from
# every schema costs one sentence and covers the case where the model decides a
# customer would like a reference number.
#
# The second is what to do with a failure. Those messages are written for this
# reader and each ends by saying what to do next; the failure mode they guard
# against is a model that reads "the ordering system is not answering" and
# tells the customer their card was declined. So the instruction is to follow
# the message, not to explain it.
COMMERCE_PROMPT = (
    " You can also hold a shopping cart for this customer and start a payment. "
    "The cart belongs to this conversation and is handled for you: never ask "
    "for a cart or basket number, never mention one, and never invent an order "
    "reference — the tools that need one already have it. "
    "Before telling a customer that something can be bought, check its stock: "
    "a search result says a variant exists, not that it can be sold today. "
    "When a tool comes back with an error, that message is written for you and "
    "says what to do next — follow it, and tell the customer what it says "
    "rather than a cause you worked out. Do not repeat a call the message told "
    "you not to repeat, and never suggest the customer did something wrong "
    "unless the message says so."
)

# Money. The exception to `never do arithmetic in your head` is stated in the
# same breath as the ban, because every amount in this system arrives as an
# integer number of minor units and somebody has to turn 9499 into a
# readable amount. A
# base rule the model must break in order to answer at all is a rule it stops
# reading, so the one permitted conversion is named and everything else is
# refused — totals especially, which the tools compute from the database on
# every read for exactly this reason.
MONEY_PROMPT = (
    " Amounts are whole numbers of minor units — cents — in the currency named "
    f"beside them, so {_EXAMPLE_MINOR_UNITS} in {_CURRENCY} is {_EXAMPLE}. "
    "Show amounts to the customer in that form, and treat that conversion as "
    "the only arithmetic you may do. "
    "Never add, subtract or compare amounts to produce a new one: a line "
    "total, a cart total and an order total each arrive from a tool, and a "
    "figure you worked out yourself is one the shop cannot honour. Every "
    "amount you state must come from a tool result in this conversation."
)


def initial_messages(catalog_available: bool = True) -> list[Message]:
    """The conversation's opening state: one system message, nothing else.

    A list rather than a string, because that is what the caller appends to,
    and returning the assembled history is what lets `/reset` in the CLI be one
    line rather than a re-derivation of what a fresh conversation looks like.
    """
    catalog = CATALOG_PROMPT if catalog_available else NO_CATALOG_PROMPT
    return [
        {"role": "system", "content": SYSTEM_PROMPT + catalog + COMMERCE_PROMPT + MONEY_PROMPT}
    ]
