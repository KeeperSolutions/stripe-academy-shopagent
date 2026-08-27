"""Tests for shopagent.api.lifecycle (D6).

Offline, and pointedly so. `transition()` reads and writes one attribute, so a
two-line stub stands in for an order row and the whole transition table can be
swept without Postgres. That is the argument for keeping the lifecycle free of
the ORM in the first place, and this file is where it pays.

The sweep below is the important one. It walks every ordered pair of statuses
and asserts the pair is either in the table and permitted or absent and
refused — there is no third answer. A status added on D8 and left out of
`ALLOWED_TRANSITIONS` therefore fails here, rather than being discovered when a
webhook cannot move an order out of it.
"""

from __future__ import annotations

import itertools

import pytest

from shopagent.api.lifecycle import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATUSES,
    IllegalTransition,
    OrderStatus,
    transition,
)


class StubOrder:
    """The smallest thing `transition()` accepts: something with a status."""

    def __init__(self, status: OrderStatus) -> None:
        self.status = status


ALL_PAIRS = list(itertools.product(OrderStatus, OrderStatus))


# --- the whole table, pair by pair --------------------------------------


@pytest.mark.parametrize(("current", "requested"), ALL_PAIRS)
def test_every_status_pair_is_either_allowed_or_refused(current, requested):
    order = StubOrder(current)
    permitted = requested in ALLOWED_TRANSITIONS[current]

    if permitted:
        assert transition(order, requested) is requested
        assert order.status is requested
    else:
        with pytest.raises(IllegalTransition):
            transition(order, requested)
        # A refused transition leaves the order where it was. Half-applying it
        # would be worse than refusing, because the caller catching the
        # exception would have no reason to suspect the object had moved.
        assert order.status is current


def test_the_transition_table_covers_every_status():
    # Not a tautology: a status added to the enum without a row here would
    # raise KeyError inside `transition()` rather than refusing cleanly.
    assert set(ALLOWED_TRANSITIONS) == set(OrderStatus)


# --- the specific rules D8 leans on -------------------------------------


@pytest.mark.parametrize("status", sorted(TERMINAL_STATUSES))
def test_a_terminal_status_has_no_way_out(status):
    assert ALLOWED_TRANSITIONS[status] == frozenset()

    for requested in OrderStatus:
        order = StubOrder(status)
        with pytest.raises(IllegalTransition):
            transition(order, requested)


def test_cancelled_and_refunded_are_the_terminal_statuses():
    assert TERMINAL_STATUSES == frozenset(
        {OrderStatus.CANCELLED, OrderStatus.REFUNDED}
    )


@pytest.mark.parametrize("status", list(OrderStatus))
def test_a_status_cannot_transition_to_itself(status):
    # The case that matters is `paid -> paid`: Stripe delivers the same event
    # more than once, and the second delivery has to be visibly refused rather
    # than absorbed as a no-op that reads like success.
    order = StubOrder(status)
    with pytest.raises(IllegalTransition):
        transition(order, status)


def test_paid_cannot_go_back_to_pending_or_cancelled():
    for requested in (OrderStatus.PENDING, OrderStatus.CANCELLED):
        order = StubOrder(OrderStatus.PAID)
        with pytest.raises(IllegalTransition):
            transition(order, requested)


def test_a_refund_is_reachable_from_paid_and_fulfilled_only():
    reaching_refunded = {
        status
        for status, allowed in ALLOWED_TRANSITIONS.items()
        if OrderStatus.REFUNDED in allowed
    }
    assert reaching_refunded == {OrderStatus.PAID, OrderStatus.FULFILLED}


# --- the exception itself ------------------------------------------------


def test_the_error_names_both_the_current_and_the_requested_status():
    order = StubOrder(OrderStatus.FULFILLED)
    with pytest.raises(IllegalTransition) as excinfo:
        transition(order, OrderStatus.PENDING)

    message = str(excinfo.value)
    assert "fulfilled" in message
    assert "pending" in message
    # Carried as attributes too, because the D6 router builds a 409 body from
    # them and parsing its own message would be an odd way to get there.
    assert excinfo.value.current is OrderStatus.FULFILLED
    assert excinfo.value.requested is OrderStatus.PENDING


def test_the_error_from_a_terminal_status_says_it_is_terminal():
    order = StubOrder(OrderStatus.CANCELLED)
    with pytest.raises(IllegalTransition) as excinfo:
        transition(order, OrderStatus.PAID)

    assert "terminal" in str(excinfo.value)


def test_a_plain_string_status_is_accepted_and_normalised():
    # `StrEnum` means the column, the JSON and the enum all compare equal, so
    # a caller handing over `"pending"` is not a mistake worth refusing.
    order = StubOrder("pending")
    assert transition(order, "paid") is OrderStatus.PAID
    assert order.status is OrderStatus.PAID


def test_an_unknown_status_is_a_value_error():
    order = StubOrder(OrderStatus.PENDING)
    with pytest.raises(ValueError):
        transition(order, "shipped")
