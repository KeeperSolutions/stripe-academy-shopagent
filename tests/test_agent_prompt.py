"""What the model is told before it is asked anything (D9, step 2).

The prompt is the one part of this system with no type checker and no schema,
so what holds it in place is a test that reads it. Two kinds of assertion live
here and the second is the one with teeth:

*It says what it must.* Prices come from tools, stock is checked before
availability is claimed, amounts are minor units. These are cheap to assert and
would survive a rewrite of the prose, because each names a rule rather than a
sentence.

*It does not say what belongs somewhere else.* No rule about confirming a
checkout — that is step 5, and it is going to be a gate in code that can see
who said yes, not an instruction to a model that cannot. A prompt that got
there first would decide the question by squatting on it: the gate would arrive
to find the behaviour already half-implemented in prose, and nobody would be
able to say which of the two was actually stopping a purchase. The same goes
for memory, which is steps 3 and 4.

And it does not restate the tool failure messages. Those were written for the
model in `tools/commerce.py`, sentence by sentence, against measured failures.
A prompt that paraphrased them would give one contract two authors, and the
paraphrase is the copy that would go stale.
"""

from __future__ import annotations

import pytest

from shopagent.config import get_settings
from shopagent.agent.prompt import (
    CATALOG_PROMPT,
    COMMERCE_PROMPT,
    MONEY_PROMPT,
    NO_CATALOG_PROMPT,
    SYSTEM_PROMPT,
    initial_messages,
)
from shopagent.tools import commerce


def assembled(catalog_available: bool = True) -> str:
    (message,) = initial_messages(catalog_available=catalog_available)
    return message["content"]


def test_the_prompt_is_one_system_message():
    messages = initial_messages()

    assert [message["role"] for message in messages] == ["system"]


def test_every_block_reaches_the_model():
    content = assembled()

    for block in (SYSTEM_PROMPT, COMMERCE_PROMPT, MONEY_PROMPT, CATALOG_PROMPT):
        assert block in content


def test_the_commerce_rules_are_there_with_or_without_the_catalog():
    """The catalog switch is the catalog's, and the cart is a different service."""
    assert COMMERCE_PROMPT in assembled(catalog_available=False)
    assert MONEY_PROMPT in assembled(catalog_available=False)
    assert NO_CATALOG_PROMPT in assembled(catalog_available=False)
    assert CATALOG_PROMPT not in assembled(catalog_available=False)


# --- what it must say ----------------------------------------------------


def test_prices_may_never_come_from_the_model_s_memory():
    assert "never from memory" in CATALOG_PROMPT
    assert "must come from a tool result" in MONEY_PROMPT


def test_stock_is_checked_before_availability_is_claimed():
    assert "check its stock" in COMMERCE_PROMPT
    assert "can be sold" in COMMERCE_PROMPT


def test_amounts_are_minor_units_and_the_conversion_is_the_only_arithmetic():
    """The one exception to the D1 rule, stated where it is made.

    `SYSTEM_PROMPT` says never do arithmetic in your head, and every price in
    this system arrives as an integer number of cents that somebody has to
    turn into $94.99. Leaving that unsaid would make the base rule one the
    model has to break to answer at all, and a rule that is broken on every
    turn stops being read on the turn that matters.
    """
    assert "never do arithmetic in your head" in SYSTEM_PROMPT
    assert "cents" in MONEY_PROMPT
    assert "the only arithmetic" in MONEY_PROMPT
    assert "Never add, subtract or compare amounts" in MONEY_PROMPT


def test_a_tool_error_is_followed_rather_than_reinterpreted():
    assert "written for you" in COMMERCE_PROMPT
    assert "do not repeat a call" in COMMERCE_PROMPT.lower()


# --- what it must not say ------------------------------------------------


CONFIRMATION_WORDS = ["confirm", "confirmation", "explicit yes", "say yes", "permission"]


@pytest.mark.parametrize("word", CONFIRMATION_WORDS)
def test_the_prompt_decides_nothing_about_confirming_a_checkout(word):
    """The boundary this file exists to hold. Step 5 owns confirmation.

    Note what is *not* asserted: the `create_checkout` tool description still
    carries "get an explicit yes first", and run B of the chain test measured
    it stopping the chain at turn 4 — it never tells the model that the
    customer's last message can be that yes. Removing a weak instruction before
    the strong gate exists is the wrong order, so it stays and this assertion
    is about the prompt only.
    """
    assert word not in assembled().lower()


@pytest.mark.parametrize("word", ["remember", "memory of", "earlier search", "previous results"])
def test_the_prompt_decides_nothing_about_memory(word):
    """Steps 3 and 4. The same reasoning as confirmation."""
    assert word not in assembled().lower()


def test_the_prompt_does_not_restate_the_tool_failure_messages():
    """One contract, one author.

    Every sentence a tool sends back on failure is in `tools/commerce.py`,
    written against a measured failure and naming what to do next. The prompt
    tells the model to follow those messages; it must not carry a second,
    aging copy of what they say.
    """
    content = assembled().lower()

    for message in (commerce._UNREACHABLE, commerce._UNAUTHORIZED, commerce._BROKEN):
        for sentence in message.split(". "):
            words = sentence.strip().lower()
            if len(words) > 30:
                assert words not in content
    assert "temporarily unavailable" not in content


# --- one amount, one format (D9) -----------------------------------------


def test_the_prompt_and_the_checkout_page_share_one_formatter():
    """Not two tests asserting the same string — one asserting the same object.

    The prompt used to teach `$94.99` while `checkout_pages.py` rendered
    `42.00 USD`: one amount in two formats, seen by the same shopper in the
    chat and then on the payment page a click later. Two tests pinning two
    literals would have passed happily through that, because each was right
    about its own half.

    So the assertion is identity. `shopagent.money.format_amount` is the only
    implementation, the page imports the name, and the prompt's worked example
    is generated by calling it — which is what makes a change to the format
    reach both surfaces or neither.
    """
    from shopagent.api.routers import checkout_pages
    from shopagent import money

    assert checkout_pages.format_amount is money.format_amount
    assert money.format_amount(9499, get_settings().currency) in MONEY_PROMPT


def test_the_worked_example_is_in_the_shop_s_own_currency():
    """A prompt teaching dollars in a euro shop is a wrong contract.

    Generated from `CURRENCY` rather than typed, so this cannot go stale the
    way the tool descriptions did between the switch to EUR and the pass that
    fixed them.
    """
    currency = get_settings().currency

    assert f"9499 in {currency} is" in MONEY_PROMPT
