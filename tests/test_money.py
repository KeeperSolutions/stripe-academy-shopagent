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
