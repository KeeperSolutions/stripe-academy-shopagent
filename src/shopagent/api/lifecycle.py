"""Order status and the transitions between them (D6).

Deliberately free of FastAPI. D8 calls `transition()` from a Stripe webhook
handler and, later, from a reconciliation pass that runs outside any HTTP
request at all; raising `HTTPException` there would produce an exception with
nobody to catch it and a 500 where the log should have said what was refused.
The mapping from `IllegalTransition` to HTTP 409 belongs in the router, which
is the one layer that knows it is speaking HTTP.

It is also free of the ORM. `transition()` needs an object with a `status`
attribute and nothing more, which is what keeps the whole transition table
testable without a database.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol


class OrderStatus(StrEnum):
    """Where an order is in its life.

    `StrEnum` so the value is its own name: the column stores `"paid"`, the
    JSON a client reads says `"paid"`, and `order.status == "paid"` is true.
    Without it every boundary would need an explicit `.value`, and the one
    that got forgotten would compare an enum against a string and be silently
    false.

    `refunded` is a status of its own rather than a spelling of `cancelled`.
    Money moved and then moved back, which is not the same event as an order
    that never charged, and D8 has to tell them apart to reconcile against
    Stripe.
    """

    PENDING = "pending"
    PAID = "paid"
    FULFILLED = "fulfilled"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


# The whole lifecycle, in one place, as data. A status missing from this map is
# not reachable, and a status present with an empty set is terminal.
#
# Two absences are deliberate and worth naming, because they look like
# oversights:
#
#   paid -> pending    Stripe delivers the same event more than once, which is
#                      its documented behaviour, not a fault. A second
#                      `checkout.session.completed` for an order already paid
#                      must be refused rather than quietly reopened; D8 leans
#                      on that refusal for idempotency.
#   paid -> cancelled  Once a charge has settled the only way back is a
#                      refund, which is a movement of money and has its own
#                      status. Letting `cancelled` absorb it would lose the
#                      distinction exactly where an accountant needs it.
ALLOWED_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.PENDING: frozenset({OrderStatus.PAID, OrderStatus.CANCELLED}),
    OrderStatus.PAID: frozenset({OrderStatus.FULFILLED, OrderStatus.REFUNDED}),
    OrderStatus.FULFILLED: frozenset({OrderStatus.REFUNDED}),
    OrderStatus.CANCELLED: frozenset(),
    OrderStatus.REFUNDED: frozenset(),
}

# Statuses nothing leaves. Derived rather than listed, so it cannot fall out of
# step with the table above.
TERMINAL_STATUSES: frozenset[OrderStatus] = frozenset(
    status for status, allowed in ALLOWED_TRANSITIONS.items() if not allowed
)


class IllegalTransition(Exception):
    """Raised when a status change is not one the lifecycle permits.

    Carries both statuses as attributes rather than only in the message,
    because the D6 router turns this into a 409 whose body names them and
    string-parsing its own exception would be an odd way to get there.
    """

    def __init__(self, current: OrderStatus, requested: OrderStatus) -> None:
        self.current = current
        self.requested = requested
        allowed = ALLOWED_TRANSITIONS[current]
        if allowed:
            tail = f"allowed from {current.value}: {', '.join(sorted(allowed))}"
        else:
            tail = f"{current.value} is terminal, no transition leaves it"
        super().__init__(
            f"cannot move an order from {current.value} to {requested.value}: {tail}"
        )


class HasStatus(Protocol):
    """Anything with an order status — in practice `api.models.Order`.

    Structural rather than a concrete import, which is what lets the
    transition table be tested against a two-line stub instead of a database
    row. `transition()` reads and writes this attribute and touches nothing
    else.
    """

    status: OrderStatus


def transition(order: HasStatus, new_status: OrderStatus) -> OrderStatus:
    """Move `order` to `new_status`, or refuse.

    Returns the new status so a caller can chain or log it. Mutates in place
    and does not commit: the session that owns the object owns the write, and
    a function that committed on its own would break the webhook handler's
    ability to do this alongside other work in one transaction.

    A status transitioning to itself is refused like any other pair not in the
    table. That is the point rather than an edge case: a repeated Stripe event
    arriving as `paid -> paid` should be visibly rejected, not absorbed as a
    no-op that reads like success in the log.
    """
    current = OrderStatus(order.status)
    requested = OrderStatus(new_status)

    if requested not in ALLOWED_TRANSITIONS[current]:
        raise IllegalTransition(current, requested)

    order.status = requested
    return requested
