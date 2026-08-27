"""Turning an order into a Stripe Checkout Session (D7, step 3).

No FastAPI, by the rule `api/services/` follows: D8's webhook and D9's agent
tools reach this outside any HTTP request.

**Checkout Session, not Payment Link.** A Payment Link is a durable URL bound
to a Price and reusable by anyone who has it; it carries no per-order
`metadata`, so a webhook firing against one cannot say *which* order was paid.
That single fact decides it — D8 exists to flip one order to `paid` on one
event, and a Payment Link would leave it guessing. A Checkout Session is
created per order, carries `metadata.order_id`, and expires. Payment Links are
the right tool for selling one product from a tweet; they are the wrong tool
for a cart.

**`line_items` are built from the `order_items` snapshot, never from a Stripe
Price.** The sync in `catalog_sync.py` wrote `stripe_price_id` onto every
variant and nothing here reads it, on purpose. `order_items` froze the price at
order time — D6 asserts that by recording the SQL and failing if the catalog is
touched — while a Stripe Price is a separate object that a re-sync, a dashboard
edit, or a local price change can move. Referencing one would mean the shopper
is charged Stripe's number while `orders.total_amount_cents` claims another,
and the two would diverge silently. `price_data` carries the snapshot straight
into the session instead, so there is exactly one number.

That is checked rather than trusted: `_build_line_items` refuses to hand back
lines whose total is not the order's total. See `CheckoutTotalMismatch`.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from shopagent.api.lifecycle import OrderStatus
from shopagent.api.models import Order, OrderItem
from shopagent.config import get_settings
from shopagent.payments import stripe_svc

# What Stripe calls the three states of a Checkout Session. Read off
# `stripe.checkout.Session` in the installed SDK rather than assumed — D7 has
# already been caught twice by guessing at SDK shapes (`Account.livemode` does
# not exist, `metadata.get` is not a method).
SESSION_OPEN = "open"
SESSION_COMPLETE = "complete"
SESSION_EXPIRED = "expired"

# `status` and `payment_status` are different axes and are easy to conflate.
# `status` is the lifecycle of the *session*; `payment_status` is one of
# "no_payment_required" | "paid" | "unpaid" and describes the money. A session
# can be open and unpaid, which is simply a shopper who has not finished yet.


class CheckoutError(Exception):
    """Base for everything this module raises."""


class OrderNotFound(CheckoutError):
    """No order with that id."""


class OrderNotPayable(CheckoutError):
    """The order is not in a state where a payment can be started."""


class PaymentAlreadyInProgress(CheckoutError):
    """A session for this order has completed, but the order is still pending.

    Exactly the window D8 exists to close: Stripe has taken the payment and the
    webhook that flips the order to `paid` has not arrived or has not been
    processed. Handing out a second session here is how the same basket gets
    charged twice, so this is refused rather than retried.
    """


class CheckoutTotalMismatch(CheckoutError):
    """The line items do not add up to what the order says it costs.

    Cheap to check and impossible to ignore: a session built from lines that
    disagree with `orders.total_amount_cents` would charge the shopper an
    amount the order does not recognise, and D8 would then mark an order paid
    for money that never matched it. Refusing to create the session is the only
    safe answer, because the alternative is a charge nobody can reconcile.
    """

    def __init__(self, *, order_id: uuid.UUID, lines_total: int, order_total: int) -> None:
        self.order_id = order_id
        self.lines_total = lines_total
        self.order_total = order_total
        super().__init__(
            f"refusing to start checkout for order {order_id}: the line items "
            f"add up to {lines_total} but the order total is {order_total}. "
            "These must agree exactly; the order was not charged."
        )


def _load_order(session: Session, order_id: uuid.UUID) -> Order:
    """Load the order with a row lock, because what follows decides a write.

    Two things race without it. Two concurrent checkout requests both see no
    stored session, both create one, and the order ends up with two payable
    URLs while remembering only the second — the first stays open and
    chargeable with nothing pointing at it. And a concurrent cancellation can
    commit `cancelled` between this read and the write below, leaving a fresh
    payable session attached to an order that no longer exists.

    The stored `stripe_checkout_session_id` only makes a *later* call
    idempotent; it cannot order two calls that overlap. `cancel_order` takes
    the same lock, so the two serialise against each other.
    """
    order = session.scalar(
        select(Order).where(Order.id == order_id).with_for_update()
    )
    if order is None:
        raise OrderNotFound(f"no order with id {order_id}")
    return order


def build_line_items(session: Session, order: Order) -> list[dict[str, Any]]:
    """The `line_items` payload, straight from the snapshot.

    `price_data` rather than `price`. Every field here comes from
    `order_items`, which is a copy taken at order time and never refreshed —
    that is what makes an order render correctly years later, and it is what
    makes the amount charged the amount the order recorded.

    `unit_amount` takes `unit_amount_cents` unchanged. Stripe wants minor
    units, D3 stored minor units, and there is no conversion here to get wrong.
    """
    rows = session.scalars(
        select(OrderItem)
        .where(OrderItem.order_id == order.id)
        .order_by(OrderItem.sku.asc())
    ).all()

    lines: list[dict[str, Any]] = []
    for row in rows:
        name = row.product_name
        if row.variant_label:
            name = f"{name} ({row.variant_label})"

        lines.append(
            {
                "quantity": row.quantity,
                "price_data": {
                    # From the order, not from settings: an order carries the
                    # currency it was placed in, and a later change to the shop
                    # currency must not reprice history.
                    "currency": row.currency,
                    "unit_amount": row.unit_amount_cents,
                    "product_data": {
                        "name": name,
                        # The sku is what ties a Stripe line back to a variant
                        # without needing the catalog.
                        "metadata": {"sku": row.sku},
                    },
                },
            }
        )

    lines_total = sum(
        line["quantity"] * line["price_data"]["unit_amount"] for line in lines
    )
    if lines_total != order.total_amount_cents:
        raise CheckoutTotalMismatch(
            order_id=order.id,
            lines_total=lines_total,
            order_total=order.total_amount_cents,
        )

    return lines


def _reusable_session(order: Order) -> Any | None:
    """The order's existing Stripe session, if it can still be paid.

    Returns None when there is nothing to reuse and a new session should be
    created. Raises when the existing session says a payment already went
    through, because that is not a case for a new session at all.
    """
    if not order.stripe_checkout_session_id:
        return None

    existing = stripe_svc.retrieve_checkout_session(order.stripe_checkout_session_id)

    if existing.status == SESSION_OPEN:
        return existing

    if existing.status == SESSION_COMPLETE:
        raise PaymentAlreadyInProgress(
            f"order {order.id} already has a completed Checkout Session "
            f"({existing.id}) and is still pending. The payment is being "
            "confirmed; it is not safe to start another. If this does not "
            "resolve, the webhook that marks the order paid has not arrived."
        )

    # Expired, or a status this SDK version does not know about. Either way the
    # shopper cannot pay through it, so a fresh session is the right answer.
    return None


def create_checkout_session(session: Session, order_id: uuid.UUID) -> Any:
    """Start (or resume) a Stripe Checkout for one order.

    Idempotent by lookup rather than by idempotency key: the key would only
    cover 24 hours, while `orders.stripe_checkout_session_id` is durable and
    lets a shopper who closed the tab come back to the same session. A repeat
    call therefore returns the session that already exists whenever it is still
    open, and only creates one when there is nothing usable.
    """
    order = _load_order(session, order_id)

    if OrderStatus(order.status) is not OrderStatus.PENDING:
        raise OrderNotPayable(
            f"order {order_id} is {OrderStatus(order.status).value} and cannot "
            "be paid. Only a pending order can start a checkout."
        )

    reusable = _reusable_session(order)
    if reusable is not None:
        return reusable

    line_items = build_line_items(session, order)
    settings = get_settings()

    # Stripe refuses a session carrying both: "You may only specify one of
    # these parameters: customer, customer_email." Verified against the API in
    # test mode rather than assumed. The Customer wins when there is one — it
    # is the richer object, and it is what puts this payment on that customer's
    # timeline in the dashboard instead of leaving it unattached.
    buyer: dict[str, str] = {}
    if order.stripe_customer_id:
        buyer["customer"] = order.stripe_customer_id
    elif order.customer_email:
        buyer["customer_email"] = order.customer_email

    created = stripe_svc.create_checkout_session(
        line_items=line_items,
        buyer=buyer,
        # The field D8 is built on. Without it a webhook says a payment
        # succeeded and cannot say for what.
        metadata={"order_id": str(order.id)},
        # The same identifier, copied onto the PaymentIntent. Stripe does not
        # propagate a session's metadata down the chain — the PaymentIntent and
        # Charge come back with `metadata: {}` — so `payment_intent.succeeded`
        # would otherwise be unattributable. Sending it in both places lets D8
        # subscribe to whichever event suits it.
        payment_intent_metadata={"order_id": str(order.id)},
        # Duplicated on purpose, and safe to duplicate because both are written
        # once from the same value and never updated. `metadata` is what D8
        # reads; this one is for a human in the dashboard, where it is a
        # first-class searchable column rather than a key in a bag.
        client_reference_id=str(order.id),
        success_url=settings.success_url,
        cancel_url=settings.cancel_url,
    )

    order.stripe_checkout_session_id = created.id
    session.commit()

    return created
