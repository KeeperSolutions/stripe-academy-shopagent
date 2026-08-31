"""How an amount is rendered for a person (D7, shared on D9).

`shopagent/money.py` is the one place minor units become something readable,
and it has two callers that a shopper meets a click apart: the checkout page
and the system prompt's worked example. This file pins what it produces; the
test that they both call *this* function rather than a copy lives in
`test_agent_prompt.py`, next to the prompt that would be the one to drift.
"""

from __future__ import annotations

import pytest

from shopagent.money import format_amount


@pytest.mark.parametrize(
    ("minor_units", "currency", "expected"),
    [
        # The shop's own currency, which is the case that matters: symbol
        # first, no space, two decimals.
        (9499, "eur", "€94.99"),
        (18998, "eur", "€189.98"),
        (100, "eur", "€1.00"),
        (0, "eur", "€0.00"),
        (123456789, "eur", "€1,234,567.89"),
        # A currency with no symbol here falls back to the ISO code rather than
        # guessing one. Correct and unambiguous, which is the point of not
        # keeping a table of every symbol.
        (28497, "usd", "284.97 USD"),
        (4200, "gbp", "42.00 GBP"),
        # Zero-decimal: the smallest unit *is* the unit, so dividing invents a
        # decimal part the currency does not have.
        (5000, "jpy", "5,000 JPY"),
        # Not zero. An amount Stripe did not report is unknown, and rendering
        # it as 0.00 would say the shopper paid nothing.
        (None, "eur", "—"),
    ],
)
def test_amounts_are_rendered_for_a_person(minor_units, currency, expected):
    assert format_amount(minor_units, currency) == expected


def test_an_unknown_currency_is_never_shown_a_borrowed_symbol():
    """The failure this guards is silent: €42.00 charged in dollars."""
    assert "€" not in format_amount(4200, "usd")
    assert "$" not in format_amount(4200, "eur")


# --- exactness ------------------------------------------------------------
#
# Raised in review on PR #9: the formatter divided by 100, which turns an exact
# integer of minor units into a binary float at the very last step — the one
# conversion this project's "money is an int" rule exists to prevent.


@pytest.mark.parametrize(
    "minor_units, expected",
    [
        # The reviewer's example. Above 2**53 a float cannot hold the integer,
        # and `/ 100` rendered this one cent high.
        (9007199254740993, "€90,071,992,547,409.93"),
        (9007199254740994, "€90,071,992,547,409.94"),
        # Ordinary amounts, which have to keep working.
        (9499, "€94.99"),
        (1, "€0.01"),
        (0, "€0.00"),
        (100, "€1.00"),
        (123456789, "€1,234,567.89"),
    ],
)
def test_every_amount_renders_from_the_integer_it_was_given(minor_units, expected):
    assert format_amount(minor_units, "eur") == expected


def test_no_amount_survives_a_round_trip_changed():
    """A property, over a range no float argument can wave away.

    Each amount is rendered and read back as minor units; a formatter that
    rounds anywhere fails here without anybody having to guess which value it
    rounds at.
    """
    for minor_units in list(range(0, 1000)) + [
        2**53 - 1,
        2**53,
        2**53 + 1,
        99999999999999,
    ]:
        rendered = format_amount(minor_units, "eur").lstrip("€").replace(",", "")
        whole, _, fraction = rendered.partition(".")
        assert int(whole) * 100 + int(fraction) == minor_units


@pytest.mark.parametrize(
    "minor_units, currency, expected",
    [
        (-1, "eur", "-€0.01"),
        (-9499, "eur", "-€94.99"),
        (-4200, "usd", "-42.00 USD"),
        (-4200, "jpy", "-4,200 JPY"),
    ],
)
def test_a_negative_amount_is_written_the_way_people_write_one(
    minor_units, currency, expected
):
    """Nothing in this shop produces one, and the formatter still owes it.

    `divmod(-1, 100)` is `(-1, 99)`, so the obvious integer version renders one
    cent below zero as `-1.99` — a wrong number, not a cosmetic one.
    """
    assert format_amount(minor_units, currency) == expected
