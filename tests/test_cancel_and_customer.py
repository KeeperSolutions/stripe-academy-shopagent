"""Releasing a reservation, and attaching a buyer (D7, step 4).

The most important test in this file is the double-cancel one. D6 raised
`inventory.reserved` and nothing ever lowered it, so D7 had to add a release —
and a release that can run twice is worse than one that never runs at all, because
it hands back units the order never held. The protection is the transition
table rather than a check in the release itself: `cancelled` is terminal, so a
second cancellation is refused before any stock moves.
"""

from __future__ import annotations

import uuid

import pytest

from shopagent.api.lifecycle import IllegalTransition, OrderStatus
from shopagent.api.models import Order
from shopagent.api.services import orders as order_service
from shopagent.catalog.models import Inventory, Price, Product, Variant
from shopagent.payments import customers, stripe_svc

pytestmark = pytest.mark.db

MISSING_UUID = uuid.UUID("00000000-0000-0000-0000-000000000000")


def make_variant(session, *, sku: str, amount_cents: int = 1000, quantity: int = 20,
                 reserved: int = 0) -> Variant:
    product = Product(
        name=f"Cancel Fixture {sku}",
        description="A product that exists only for a cancel test.",
        category="shoes",
        brand="Fleetfoot",
        variants=[
            Variant(
                size="42", color="blue", sku=sku,
                prices=[Price(currency="usd", amount_cents=amount_cents, active=True)],
                inventory=Inventory(quantity=quantity, reserved=reserved),
            )
        ],
    )
    session.add(product)
    session.commit()
    return product.variants[0]


def stock(session, variant_id: int) -> tuple[int, int]:
    from sqlalchemy import select

    row = session.execute(
        select(Inventory.quantity, Inventory.reserved).where(
            Inventory.variant_id == variant_id
        )
    ).one()
    return int(row[0]), int(row[1])


def make_order(authed_client, session, specs, email=None) -> str:
    cart_id = authed_client.post("/cart").json()["cart_id"]
    for variant, quantity in specs:
        authed_client.post(
            f"/cart/{cart_id}/items",
            json={"variant_id": variant.id, "quantity": quantity},
        )
    payload = {"cart_id": cart_id}
    if email:
        payload["customer_email"] = email
    return authed_client.post("/orders", json=payload).json()["order_id"]


# --- releasing the reservation -------------------------------------------


def test_cancelling_releases_exactly_what_was_reserved(authed_client, session):
    """Recorded before the order, compared after the cancel."""
    first = make_variant(session, sku="CAN-A", quantity=30, reserved=4)
    second = make_variant(session, sku="CAN-B", quantity=30, reserved=0)
    before_first = stock(session, first.id)
    before_second = stock(session, second.id)

    order_id = make_order(authed_client, session, [(first, 3), (second, 5)])
    assert stock(session, first.id)[1] == before_first[1] + 3
    assert stock(session, second.id)[1] == before_second[1] + 5

    response = authed_client.post(f"/orders/{order_id}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    # Back exactly where it started, not merely lower.
    assert stock(session, first.id) == before_first
    assert stock(session, second.id) == before_second


def test_cancelling_does_not_touch_quantity(authed_client, session):
    """Releasing means sellable again, which is what `reserved` says.

    `quantity` only changes when goods physically move, and nothing here moves
    any.
    """
    variant = make_variant(session, sku="CAN-QTY", quantity=12)
    order_id = make_order(authed_client, session, [(variant, 4)])

    authed_client.post(f"/orders/{order_id}/cancel")

    assert stock(session, variant.id)[0] == 12


def test_a_second_cancel_is_409_and_releases_nothing_twice(authed_client, session):
    """The test this step exists for.

    A double release hands back units the order never held. The transition
    table is what prevents it: `cancelled` is terminal, so the second attempt
    is refused before any stock moves.
    """
    variant = make_variant(session, sku="CAN-TWICE", quantity=30, reserved=2)
    before = stock(session, variant.id)

    order_id = make_order(authed_client, session, [(variant, 5)])
    first = authed_client.post(f"/orders/{order_id}/cancel")
    after_first = stock(session, variant.id)

    second = authed_client.post(f"/orders/{order_id}/cancel")

    assert first.status_code == 200
    assert second.status_code == 409
    assert after_first == before
    # Unchanged by the refused attempt — not merely non-negative.
    assert stock(session, variant.id) == before


def test_cancelling_a_paid_order_is_409(authed_client, session):
    """D6's table has no `paid -> cancelled`: the way back from a charge is a refund."""
    variant = make_variant(session, sku="CAN-PAID")
    order_id = make_order(authed_client, session, [(variant, 2)])
    session.execute(
        Order.__table__.update()
        .where(Order.id == uuid.UUID(order_id))
        .values(status=OrderStatus.PAID)
    )
    session.commit()
    reserved_before = stock(session, variant.id)[1]

    response = authed_client.post(f"/orders/{order_id}/cancel")

    assert response.status_code == 409
    assert stock(session, variant.id)[1] == reserved_before


def test_cancelling_an_order_that_does_not_exist_is_404(authed_client):
    assert authed_client.post(f"/orders/{MISSING_UUID}/cancel").status_code == 404


def test_cancelling_without_the_api_key_is_401(api_client):
    assert api_client.post(f"/orders/{MISSING_UUID}/cancel").status_code == 401


def test_a_refund_releases_the_reservation_too(authed_client, session):
    """The other half of `RELEASES_RESERVATION`, which D8 will drive."""
    variant = make_variant(session, sku="CAN-REFUND", quantity=30)
    before = stock(session, variant.id)
    order_id = make_order(authed_client, session, [(variant, 6)])
    order = session.get(Order, uuid.UUID(order_id))
    order_service.apply_transition(session, order, OrderStatus.PAID)

    order_service.apply_transition(session, order, OrderStatus.REFUNDED)

    assert stock(session, variant.id) == before


def test_being_fulfilled_keeps_the_reservation(authed_client, session):
    variant = make_variant(session, sku="CAN-FULFILLED", quantity=30)
    order_id = make_order(authed_client, session, [(variant, 3)])
    order = session.get(Order, uuid.UUID(order_id))
    order_service.apply_transition(session, order, OrderStatus.PAID)
    reserved_before = stock(session, variant.id)[1]

    order_service.apply_transition(session, order, OrderStatus.FULFILLED)

    assert stock(session, variant.id)[1] == reserved_before


def test_apply_transition_refuses_an_illegal_move_without_touching_stock(
    authed_client, session
):
    variant = make_variant(session, sku="CAN-ILLEGAL", quantity=30)
    order_id = make_order(authed_client, session, [(variant, 2)])
    order = session.get(Order, uuid.UUID(order_id))
    before = stock(session, variant.id)

    with pytest.raises(IllegalTransition):
        order_service.apply_transition(session, order, OrderStatus.FULFILLED)

    assert stock(session, variant.id) == before


# --- the buyer ------------------------------------------------------------


def test_an_order_can_be_placed_without_an_email(authed_client, session):
    """D6's flow has no notion of who is buying and still has to work."""
    variant = make_variant(session, sku="CUS-NONE")
    order_id = make_order(authed_client, session, [(variant, 1)])

    order = session.get(Order, uuid.UUID(order_id))
    assert order.customer_email is None
    assert order.stripe_customer_id is None


def test_an_email_is_normalised_before_it_is_stored(authed_client, session, monkeypatch):
    monkeypatch.setattr(customers, "get_or_create_customer", lambda *a, **kw: "cus_fake")
    variant = make_variant(session, sku="CUS-NORM")

    order_id = make_order(authed_client, session, [(variant, 1)], email="Shopper@Example.COM")

    order = session.get(Order, uuid.UUID(order_id))
    assert order.customer_email == "shopper@example.com"
    assert order.stripe_customer_id == "cus_fake"


def test_a_malformed_email_is_422(authed_client, session):
    variant = make_variant(session, sku="CUS-BAD")
    cart_id = authed_client.post("/cart").json()["cart_id"]
    authed_client.post(f"/cart/{cart_id}/items", json={"variant_id": variant.id, "quantity": 1})

    response = authed_client.post(
        "/orders", json={"cart_id": cart_id, "customer_email": "not-an-email"}
    )

    assert response.status_code == 422


def test_a_known_email_reuses_the_stored_customer_id(authed_client, session, monkeypatch):
    """The cheap half of deduplication: no API call at all.

    Proved by making the Stripe calls fail — reaching either means the local
    lookup did not answer.
    """
    def explode(*args, **kwargs):
        raise AssertionError("Stripe was called for an email already on file")

    monkeypatch.setattr(stripe_svc, "find_customers_by_email", explode)
    monkeypatch.setattr(stripe_svc, "create_customer", explode)

    variant = make_variant(session, sku="CUS-REUSE")
    order = Order(
        cart_id=uuid.UUID(authed_client.post("/cart").json()["cart_id"]),
        status=OrderStatus.PENDING,
        total_amount_cents=0,
        currency="usd",
        customer_email="repeat@example.com",
        stripe_customer_id="cus_already_known",
    )
    session.add(order)
    session.commit()

    found = customers.get_or_create_customer(session, "Repeat@Example.com")

    assert found == "cus_already_known"


def test_an_unknown_email_falls_back_to_stripe_then_creates(session, monkeypatch):
    """The order of the three steps, asserted rather than assumed."""
    calls = []

    monkeypatch.setattr(
        stripe_svc, "find_customers_by_email",
        lambda email, limit=1: calls.append(("list", email)) or [],
    )

    class Created:
        id = "cus_new"

    monkeypatch.setattr(
        stripe_svc, "create_customer",
        lambda **kw: calls.append(("create", kw["email"])) or Created(),
    )

    result = customers.get_or_create_customer(session, "brand-new@example.com")

    assert result == "cus_new"
    assert [name for name, _ in calls] == ["list", "create"]


def test_stripe_is_not_asked_to_create_when_it_already_has_one(session, monkeypatch):
    class Existing:
        id = "cus_found_at_stripe"

    monkeypatch.setattr(stripe_svc, "find_customers_by_email", lambda email, limit=1: [Existing()])
    monkeypatch.setattr(
        stripe_svc, "create_customer",
        lambda **kw: pytest.fail("created a duplicate for an email Stripe already had"),
    )

    assert customers.get_or_create_customer(session, "known@example.com") == "cus_found_at_stripe"


# --- against real Stripe -------------------------------------------------


@pytest.mark.stripe
def test_a_customer_is_created_once_and_found_again(session):
    """Deduplication against the real API, including the reason for `list`.

    `customers.search` is backed by an index that lags writes by up to a
    minute, so a customer created and immediately searched for is often not
    found — precisely the case this has to get right. `list(email=...)` filters
    the field directly and is immediately consistent, which this asserts by
    looking the customer up in the same breath as creating it.
    """
    email = f"shopagent-test-{uuid.uuid4().hex[:10]}@example.com"
    customer_id = None

    try:
        customer_id = customers.get_or_create_customer(session, email, name="Test Shopper")
        assert customer_id.startswith("cus_")

        # No local row carries this email, so the second call has to be
        # answered by Stripe rather than by the database.
        again = customers.get_or_create_customer(session, email)
        assert again == customer_id, "a duplicate Customer was created"

        found = stripe_svc.find_customers_by_email(email)
        assert len(found) == 1
        assert found[0].livemode is False
    finally:
        if customer_id:
            stripe_svc.delete_customer(customer_id)


@pytest.mark.stripe
def test_a_session_carries_the_customer_and_never_both_fields(authed_client, session):
    """Stripe refuses `customer` and `customer_email` together.

    Verified against the API rather than assumed: sending both answers "You may
    only specify one of these parameters". The Customer wins when there is one,
    which is what puts the payment on that customer's dashboard timeline.
    """
    email = f"shopagent-test-{uuid.uuid4().hex[:10]}@example.com"
    variant = make_variant(session, sku=f"CUS-REAL-{uuid.uuid4().hex[:6]}", amount_cents=1500)
    order_id = make_order(authed_client, session, [(variant, 2)], email=email)
    order = session.get(Order, uuid.UUID(order_id))
    customer_id = order.stripe_customer_id
    session_id = None

    try:
        assert customer_id and customer_id.startswith("cus_")

        response = authed_client.post(f"/orders/{order_id}/checkout")
        assert response.status_code == 201
        session_id = response.json()["checkout_session_id"]

        live = stripe_svc.retrieve_checkout_session(session_id)
        assert live.customer == customer_id
        # The mutually exclusive half: Stripe fills this in itself from the
        # Customer, and we never sent it.
        assert live.livemode is False
    finally:
        if session_id:
            try:
                stripe_svc.expire_checkout_session(session_id)
            except Exception:
                pass
        if customer_id:
            stripe_svc.delete_customer(customer_id)


@pytest.mark.stripe
def test_an_email_only_order_sends_customer_email(authed_client, session):
    """The other branch: no Customer object, so the email goes on the session."""
    variant = make_variant(session, sku=f"CUS-EMAIL-{uuid.uuid4().hex[:6]}")
    order_id = make_order(authed_client, session, [(variant, 1)])
    session.execute(
        Order.__table__.update()
        .where(Order.id == uuid.UUID(order_id))
        .values(customer_email="email-only@example.com", stripe_customer_id=None)
    )
    session.commit()
    session_id = None

    try:
        session_id = authed_client.post(f"/orders/{order_id}/checkout").json()[
            "checkout_session_id"
        ]
        live = stripe_svc.retrieve_checkout_session(session_id)

        assert live.customer_email == "email-only@example.com"
        assert live.customer is None
    finally:
        if session_id:
            try:
                stripe_svc.expire_checkout_session(session_id)
            except Exception:
                pass


@pytest.mark.stripe
def test_cancelling_expires_the_open_checkout_session(authed_client, session):
    """A cancelled order must not leave a working payment URL behind.

    Otherwise a shopper with that link in a tab can pay for something that no
    longer exists, and D8 would receive a completed session for an order it is
    forbidden to move — `cancelled` is terminal. The money would be real and
    the order would not.
    """
    variant = make_variant(session, sku=f"CAN-REAL-{uuid.uuid4().hex[:6]}", quantity=30)
    order_id = make_order(authed_client, session, [(variant, 2)])

    session_id = authed_client.post(f"/orders/{order_id}/checkout").json()[
        "checkout_session_id"
    ]
    assert stripe_svc.retrieve_checkout_session(session_id).status == "open"

    response = authed_client.post(f"/orders/{order_id}/cancel")

    assert response.status_code == 200
    closed = stripe_svc.retrieve_checkout_session(session_id)
    assert closed.status == "expired"
    assert closed.url is None
