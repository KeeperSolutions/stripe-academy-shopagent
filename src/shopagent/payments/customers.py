"""Attaching a buyer to an order (D7, step 4).

No FastAPI, the same rule the rest of `payments/` follows.

**There is no `customers` table, and that is a decision rather than an
omission.** D6 has no concept of a user, and D9 introduces long-term memory —
a name, preferences, the orders somebody placed before — which is the
requirement that would actually shape such a table. Creating it now would mean
guessing that shape a week early and then living with the guess. An order
carries the little it knows, in two nullable columns, and nothing in this
project claims to be a customer record.

**Deduplication is ours to do.** Stripe stores as many Customers with the same
email as it is asked to; it treats the field as data, not identity. An
idempotency key would only cover 24 hours and says nothing about the same
shopper returning next week, so the mechanism here is to look before creating —
first locally, which is free and immediately consistent, then at Stripe.
"""

from __future__ import annotations

import hashlib
import logging

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from shopagent.api.models import Order
from shopagent.payments import stripe_svc

logger = logging.getLogger(__name__)


def normalise_email(email: str) -> str:
    """Lowercase and strip, so "A@B.com " and "a@b.com" are one shopper.

    Only the domain is formally case-insensitive, but treating the local part
    as case-sensitive would mean two Customers for one person at every provider
    people actually use. The cost of being wrong in this direction is merging
    two accounts that a mail server would have kept apart; the cost of the
    other is a duplicate for every capital letter.
    """
    return email.strip().lower()


def find_local_customer_id(session: Session, email: str) -> str | None:
    """A Stripe customer id this application already stored for that email.

    The cheap half of deduplication, and the one that matters most: it costs no
    API call, it cannot lag, and it covers every customer this application has
    ever created — which, since nothing else writes to this Stripe account, is
    all of them.
    """
    return session.scalar(
        select(Order.stripe_customer_id)
        .where(
            Order.customer_email == normalise_email(email),
            Order.stripe_customer_id.is_not(None),
        )
        .order_by(Order.created_at.desc())
        .limit(1)
    )


def get_or_create_customer(session: Session, email: str, name: str | None = None) -> str:
    """Return a Stripe customer id for this email, creating one only if needed.

    Three steps, cheapest first. A local hit answers without touching the
    network. A Stripe `list` filtered by email catches the case where this
    database was rebuilt but the Stripe account was not — which is exactly what
    a disposable catalog and a long-lived test account produce. Only then is a
    Customer created.
    """
    email = normalise_email(email)

    local = find_local_customer_id(session, email)
    if local:
        return local

    existing = stripe_svc.find_customers_by_email(email)
    if existing:
        return existing[0].id

    # Derived from the address, never random: two concurrent creations for the
    # same shopper must collide rather than both succeed.
    digest = hashlib.sha256(email.encode("utf-8")).hexdigest()[:32]
    created = stripe_svc.create_customer(
        email=email, name=name, idempotency_key=f"shopagent-customer-v1-{digest}"
    )
    return created.id


def attach_customer(session: Session, order: Order, email: str | None) -> Order:
    """Record the buyer on the order, creating a Stripe Customer if there is one.

    Called from `place_order`'s caller rather than from `place_order` itself:
    creating a Stripe object inside the transaction that reserves stock would
    put a network round trip inside a `SELECT ... FOR UPDATE`, holding inventory
    row locks for as long as Stripe takes to answer. The order is written
    first, then the buyer is attached.
    """
    if not email:
        return order

    # The email is ours and is written first, on its own. `place_order` has
    # already committed the order, the reservation and the terminal cart
    # status, so an exception from here would surface as a 500 for an order
    # that exists — and the retry would be refused with 409 because the cart is
    # already ordered, leaving the caller without the order id at all.
    order.customer_email = normalise_email(email)
    session.commit()

    # The Stripe Customer is a convenience on top of that: it puts the payment
    # on a timeline in the dashboard. It is not required to place an order, to
    # read one, or to pay for one — a session falls back to `customer_email`.
    # So a Stripe outage degrades the result rather than failing the request.
    try:
        order.stripe_customer_id = get_or_create_customer(session, email)
        session.commit()
    except Exception:
        session.rollback()
        logger.warning(
            "could not attach a Stripe Customer to order %s; the order stands "
            "and checkout will fall back to customer_email",
            order.id,
            exc_info=True,
        )

    return order
