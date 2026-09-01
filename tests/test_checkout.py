"""Tests for the Checkout Session built from the order snapshot (D7, step 3).

The offline half never reaches Stripe: `build_line_items` reads `order_items`
and returns a dict, so the whole mapping — amounts, currency, quantity, names,
and the refusal when the totals disagree — is answerable without an account.

The `stripe`-marked half creates a real session in test mode and expires it
afterwards. Stripe keeps Checkout Sessions permanently and offers no delete, so
`expire` is the whole of what cleanup can mean.
"""

from __future__ import annotations

import contextlib
import uuid

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine

from shopagent.config import get_settings
from shopagent.money import format_amount
from shopagent.api.lifecycle import OrderStatus
from shopagent.api.models import Order, OrderItem
from shopagent.catalog.models import Inventory, Price, Product, Variant
from shopagent.payments import checkout, stripe_svc
from shopagent.payments.checkout import (
    CheckoutTotalMismatch,
    OrderNotPayable,
    PaymentAlreadyInProgress,
    build_line_items,
    create_checkout_session,
)


# The shop's currency, read rather than written. A test that creates a price
# row in a literal currency and then asks a service to find it is testing its
# own literal: the service filters on `settings.currency`, so the two have to
# agree by construction, and the day they stopped agreeing is the day 125 of
# them failed at once. What a specific currency *is* still gets pinned, in the
# tests that are actually about that — `format_amount`, and the two that prove
# a foreign currency is treated as foreign.
CURRENCY = get_settings().currency

pytestmark = pytest.mark.db

MISSING_UUID = uuid.UUID("00000000-0000-0000-0000-000000000000")


@contextlib.contextmanager
def recorded_sql():
    """Every SQL statement executed inside the block.

    Row locks are invisible in single-threaded behaviour, so the statements are
    read directly — the same technique `test_api_orders.py` uses on
    `place_order`, and for the same reason.
    """
    statements: list[str] = []

    def listener(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(Engine, "before_cursor_execute", listener)
    try:
        yield statements
    finally:
        event.remove(Engine, "before_cursor_execute", listener)


def make_variant(session, *, sku: str, amount_cents: int = 2500) -> Variant:
    product = Product(
        name=f"Checkout Fixture {sku}",
        description="A product that exists only for a checkout test.",
        category="shoes",
        brand="Fleetfoot",
        variants=[
            Variant(
                size="42",
                color="blue",
                sku=sku,
                prices=[Price(currency=CURRENCY, amount_cents=amount_cents, active=True)],
                inventory=Inventory(quantity=20, reserved=0),
            )
        ],
    )
    session.add(product)
    session.commit()
    return product.variants[0]


def make_order(authed_client, session, specs: list[tuple[str, int, int]]) -> str:
    """Build a real cart and place a real order. Returns the order id."""
    cart_id = authed_client.post("/cart").json()["cart_id"]
    for sku, amount, quantity in specs:
        variant = make_variant(session, sku=sku, amount_cents=amount)
        authed_client.post(
            f"/cart/{cart_id}/items",
            json={"variant_id": variant.id, "quantity": quantity},
        )
    return authed_client.post("/orders", json={"cart_id": cart_id}).json()["order_id"]


# --- the payload, offline ------------------------------------------------


def test_line_items_come_from_the_snapshot(authed_client, session):
    order_id = make_order(authed_client, session, [("CHK-A", 2500, 2), ("CHK-B", 999, 3)])
    order = session.get(Order, uuid.UUID(order_id))

    lines = build_line_items(session, order)

    assert len(lines) == 2
    first, second = lines
    assert first["quantity"] == 2
    assert first["price_data"]["unit_amount"] == 2500
    assert first["price_data"]["currency"] == CURRENCY
    assert "Checkout Fixture CHK-A" in first["price_data"]["product_data"]["name"]
    assert first["price_data"]["product_data"]["metadata"]["sku"] == "CHK-A"
    assert second["price_data"]["unit_amount"] == 999
    assert second["quantity"] == 3


def test_the_variant_label_reaches_the_line_name(authed_client, session):
    """A shopper paying for two sizes of one shoe has to tell them apart."""
    order_id = make_order(authed_client, session, [("CHK-LABEL", 1500, 1)])
    order = session.get(Order, uuid.UUID(order_id))

    (line,) = build_line_items(session, order)

    assert "42 / blue" in line["price_data"]["product_data"]["name"]


def test_no_stripe_price_id_appears_anywhere_in_the_payload(authed_client, session):
    """The decision this whole module rests on, asserted rather than assumed.

    `catalog_sync.py` wrote a `stripe_price_id` onto every variant. If one ever
    reached a line item, the shopper would be charged Stripe's number while
    `orders.total_amount_cents` claimed another.
    """
    order_id = make_order(authed_client, session, [("CHK-NOPRICE", 4200, 1)])
    order = session.get(Order, uuid.UUID(order_id))
    # Give the variant a Stripe price id, so its absence below is meaningful.
    session.execute(
        Variant.__table__.update()
        .where(Variant.sku == "CHK-NOPRICE")
        .values(stripe_price_id="price_must_not_be_used")
    )
    session.commit()

    lines = build_line_items(session, order)

    rendered = repr(lines)
    assert "price_must_not_be_used" not in rendered
    assert "stripe_price_id" not in rendered
    for line in lines:
        assert "price" not in line, "a line referenced a Stripe Price object"
        assert "price_data" in line


def test_the_currency_comes_from_the_order_not_the_settings(authed_client, session):
    """An order carries the currency it was placed in.

    A later change to the shop currency must not reprice history.
    """
    order_id = make_order(authed_client, session, [("CHK-CUR", 1000, 1)])
    session.execute(
        OrderItem.__table__.update()
        .where(OrderItem.sku == "CHK-CUR")
        .values(currency="gbp")
    )
    session.commit()
    order = session.get(Order, uuid.UUID(order_id))

    (line,) = build_line_items(session, order)

    assert line["price_data"]["currency"] == "gbp"


# --- the totals must agree -----------------------------------------------


def test_lines_that_do_not_add_up_refuse_to_build(authed_client, session):
    """The check that makes every future line-building bug loud.

    A session built from lines that disagree with the order total would charge
    an amount the order does not recognise, and D8 would then mark it paid.
    """
    order_id = make_order(authed_client, session, [("CHK-SUM", 1000, 2)])
    session.execute(
        Order.__table__.update()
        .where(Order.id == uuid.UUID(order_id))
        .values(total_amount_cents=9999)
    )
    session.commit()
    order = session.get(Order, uuid.UUID(order_id))

    with pytest.raises(CheckoutTotalMismatch) as excinfo:
        build_line_items(session, order)

    message = str(excinfo.value)
    assert "2000" in message and "9999" in message
    assert "was not charged" in message


def test_a_mismatch_stops_the_session_before_stripe_is_called(
    authed_client, session, monkeypatch
):
    """Proved by breaking the create call, not by trusting the order of lines."""
    order_id = make_order(authed_client, session, [("CHK-SUM-2", 500, 1)])
    session.execute(
        Order.__table__.update()
        .where(Order.id == uuid.UUID(order_id))
        .values(total_amount_cents=1)
    )
    session.commit()

    def explode(**kwargs):
        raise AssertionError("Stripe was called despite a total mismatch")

    monkeypatch.setattr(stripe_svc, "create_checkout_session", explode)

    with pytest.raises(CheckoutTotalMismatch):
        create_checkout_session(session, uuid.UUID(order_id))


# --- metadata is what D8 reads -------------------------------------------


def test_the_session_carries_the_order_id_in_metadata(
    authed_client, session, monkeypatch
):
    """Without this a webhook says a payment succeeded and cannot say for what."""
    order_id = make_order(authed_client, session, [("CHK-META", 1200, 1)])
    captured = {}

    class FakeSession:
        id = "cs_test_fake"
        url = "https://checkout.stripe.com/c/pay/cs_test_fake"

    def capture(**kwargs):
        captured.update(kwargs)
        return FakeSession()

    monkeypatch.setattr(stripe_svc, "create_checkout_session", capture)

    create_checkout_session(session, uuid.UUID(order_id))

    assert captured["metadata"] == {"order_id": order_id}
    assert captured["client_reference_id"] == order_id
    assert "{CHECKOUT_SESSION_ID}" in captured["success_url"]
    assert captured["cancel_url"]


def test_the_session_id_is_written_back_to_the_order(
    authed_client, session, monkeypatch
):
    order_id = make_order(authed_client, session, [("CHK-WRITEBACK", 700, 1)])

    class FakeSession:
        id = "cs_test_written_back"
        url = "https://example.test/pay"

    monkeypatch.setattr(stripe_svc, "create_checkout_session", lambda **kw: FakeSession())

    create_checkout_session(session, uuid.UUID(order_id))

    order = session.get(Order, uuid.UUID(order_id))
    session.refresh(order)
    assert order.stripe_checkout_session_id == "cs_test_written_back"


# --- reuse, and the states Stripe actually has ---------------------------


def _fake_existing(monkeypatch, status_value: str, session_id="cs_test_existing"):
    class Existing:
        id = session_id
        status = status_value
        url = "https://example.test/existing" if status_value == "open" else None

    monkeypatch.setattr(stripe_svc, "retrieve_checkout_session", lambda sid: Existing())
    return Existing


def test_an_open_session_is_reused_rather_than_recreated(
    authed_client, session, monkeypatch
):
    order_id = make_order(authed_client, session, [("CHK-REUSE", 800, 1)])
    session.execute(
        Order.__table__.update()
        .where(Order.id == uuid.UUID(order_id))
        .values(stripe_checkout_session_id="cs_test_existing")
    )
    session.commit()
    _fake_existing(monkeypatch, "open")
    monkeypatch.setattr(
        stripe_svc,
        "create_checkout_session",
        lambda **kw: pytest.fail("a second session was created for an open one"),
    )

    result = create_checkout_session(session, uuid.UUID(order_id))

    assert result.id == "cs_test_existing"


def test_an_expired_session_is_replaced(authed_client, session, monkeypatch):
    order_id = make_order(authed_client, session, [("CHK-EXPIRED", 800, 1)])
    session.execute(
        Order.__table__.update()
        .where(Order.id == uuid.UUID(order_id))
        .values(stripe_checkout_session_id="cs_test_old")
    )
    session.commit()
    _fake_existing(monkeypatch, "expired", session_id="cs_test_old")

    class Fresh:
        id = "cs_test_fresh"
        url = "https://example.test/fresh"

    monkeypatch.setattr(stripe_svc, "create_checkout_session", lambda **kw: Fresh())

    result = create_checkout_session(session, uuid.UUID(order_id))

    assert result.id == "cs_test_fresh"
    order = session.get(Order, uuid.UUID(order_id))
    session.refresh(order)
    assert order.stripe_checkout_session_id == "cs_test_fresh"


def test_a_completed_session_on_a_pending_order_is_refused(
    authed_client, session, monkeypatch
):
    """The window D8 exists to close.

    Stripe has taken the money and the webhook has not landed. Handing out a
    second session here is how one basket gets charged twice.
    """
    order_id = make_order(authed_client, session, [("CHK-COMPLETE", 800, 1)])
    session.execute(
        Order.__table__.update()
        .where(Order.id == uuid.UUID(order_id))
        .values(stripe_checkout_session_id="cs_test_done")
    )
    session.commit()
    _fake_existing(monkeypatch, "complete", session_id="cs_test_done")
    monkeypatch.setattr(
        stripe_svc,
        "create_checkout_session",
        lambda **kw: pytest.fail("a second session was created after a completed one"),
    )

    with pytest.raises(PaymentAlreadyInProgress) as excinfo:
        create_checkout_session(session, uuid.UUID(order_id))

    assert "being confirmed" in str(excinfo.value)


def test_the_state_names_match_the_installed_sdk():
    """Read off the SDK rather than trusted, twice bitten in this project.

    `Account.livemode` did not exist in step 1 and `metadata.get` was not a
    method in step 2. If a future SDK renames a session state, this fails here
    rather than silently making `_reusable_session` fall through to "create a
    new one" for a session that is actually complete.
    """
    import re
    import inspect as pyinspect

    import stripe

    source = pyinspect.getsource(stripe.checkout.Session)
    match = re.search(r'status: Optional\[\s*Literal\[([^\]]*)\]', source)
    assert match, "could not find the status literals in the SDK"

    literals = {value.strip().strip('"') for value in match.group(1).split(",") if value.strip()}
    assert {checkout.SESSION_OPEN, checkout.SESSION_COMPLETE, checkout.SESSION_EXPIRED} == literals


# --- order state ---------------------------------------------------------


def test_an_order_that_is_not_pending_cannot_start_a_checkout(
    authed_client, session, monkeypatch
):
    order_id = make_order(authed_client, session, [("CHK-PAID", 800, 1)])
    session.execute(
        Order.__table__.update()
        .where(Order.id == uuid.UUID(order_id))
        .values(status=OrderStatus.PAID)
    )
    session.commit()

    with pytest.raises(OrderNotPayable) as excinfo:
        create_checkout_session(session, uuid.UUID(order_id))

    assert "paid" in str(excinfo.value)


# --- the endpoint, one row of the status table each ----------------------


def test_checkout_returns_201_and_a_url(authed_client, session, monkeypatch):
    order_id = make_order(authed_client, session, [("CHK-EP-OK", 3300, 2)])

    class FakeSession:
        id = "cs_test_endpoint"
        url = "https://checkout.stripe.com/c/pay/cs_test_endpoint"

    monkeypatch.setattr(stripe_svc, "create_checkout_session", lambda **kw: FakeSession())

    response = authed_client.post(f"/orders/{order_id}/checkout")

    assert response.status_code == 201
    body = response.json()
    assert body["order_id"] == order_id
    assert body["checkout_session_id"] == "cs_test_endpoint"
    assert body["checkout_url"].startswith("https://")


def test_checkout_on_a_missing_order_is_404(authed_client):
    assert authed_client.post(f"/orders/{MISSING_UUID}/checkout").status_code == 404


def test_checkout_on_a_non_pending_order_is_409(authed_client, session):
    order_id = make_order(authed_client, session, [("CHK-EP-409", 800, 1)])
    session.execute(
        Order.__table__.update()
        .where(Order.id == uuid.UUID(order_id))
        .values(status=OrderStatus.CANCELLED)
    )
    session.commit()

    response = authed_client.post(f"/orders/{order_id}/checkout")

    assert response.status_code == 409
    assert "cancelled" in response.json()["detail"]


def test_checkout_with_a_completed_session_is_409(authed_client, session, monkeypatch):
    order_id = make_order(authed_client, session, [("CHK-EP-DONE", 800, 1)])
    session.execute(
        Order.__table__.update()
        .where(Order.id == uuid.UUID(order_id))
        .values(stripe_checkout_session_id="cs_test_done")
    )
    session.commit()
    _fake_existing(monkeypatch, "complete", session_id="cs_test_done")

    response = authed_client.post(f"/orders/{order_id}/checkout")

    assert response.status_code == 409


def test_checkout_without_a_stripe_key_is_503(authed_client, session, monkeypatch):
    """A server that cannot take payments is unavailable, not broken.

    503 says this capability is not there right now, which an operator can act
    on. The cart and order API is unaffected, and 500 would suggest otherwise.
    """
    order_id = make_order(authed_client, session, [("CHK-EP-503", 800, 1)])

    def no_key(**kwargs):
        raise stripe_svc.MissingStripeKey("STRIPE_SECRET_KEY is not set")

    monkeypatch.setattr(stripe_svc, "create_checkout_session", no_key)

    response = authed_client.post(f"/orders/{order_id}/checkout")

    assert response.status_code == 503
    assert "STRIPE_SECRET_KEY" in response.json()["detail"]


def test_checkout_with_a_broken_total_is_500_and_charges_nothing(
    authed_client, session, monkeypatch
):
    order_id = make_order(authed_client, session, [("CHK-EP-500", 800, 1)])
    session.execute(
        Order.__table__.update()
        .where(Order.id == uuid.UUID(order_id))
        .values(total_amount_cents=12345)
    )
    session.commit()
    monkeypatch.setattr(
        stripe_svc,
        "create_checkout_session",
        lambda **kw: pytest.fail("Stripe was called with a broken total"),
    )

    response = authed_client.post(f"/orders/{order_id}/checkout")

    assert response.status_code == 500
    assert "was not charged" in response.json()["detail"]


def test_checkout_without_the_api_key_is_401(api_client):
    assert api_client.post(f"/orders/{MISSING_UUID}/checkout").status_code == 401


def test_a_session_with_no_url_is_409_rather_than_a_null_url(
    authed_client, session, monkeypatch
):
    """`Session.url` is Optional in the SDK; a null checkout URL must not ship."""
    order_id = make_order(authed_client, session, [("CHK-EP-NOURL", 800, 1)])

    class UrllessSession:
        id = "cs_test_nourl"
        url = None

    monkeypatch.setattr(
        stripe_svc, "create_checkout_session", lambda **kw: UrllessSession()
    )

    response = authed_client.post(f"/orders/{order_id}/checkout")

    assert response.status_code == 409
    assert "no longer open" in response.json()["detail"]


# --- against real Stripe -------------------------------------------------


@pytest.mark.stripe
def test_a_real_session_matches_the_order_and_is_reused(authed_client, session):
    """One real session in test mode, expired afterwards.

    Stripe keeps Checkout Sessions permanently and offers no delete, so
    `expire` is the whole of what cleanup can mean here — it stops the session
    being payable, which is the part that matters.
    """
    order_id = make_order(authed_client, session, [("CHK-REAL-A", 2500, 2), ("CHK-REAL-B", 1000, 1)])
    order = session.get(Order, uuid.UUID(order_id))
    expected_total = order.total_amount_cents
    session_id = None

    try:
        response = authed_client.post(f"/orders/{order_id}/checkout")
        assert response.status_code == 201
        body = response.json()
        session_id = body["checkout_session_id"]

        assert session_id.startswith("cs_test_")
        assert body["checkout_url"].startswith("https://")

        live = stripe_svc.retrieve_checkout_session(session_id)
        assert live.livemode is False
        assert live.status == checkout.SESSION_OPEN
        # The two claims this whole step rests on.
        assert live.metadata._data["order_id"] == order_id
        assert live.amount_total == expected_total == 6000

        # A repeat call must hand back the same session, not a second one.
        again = authed_client.post(f"/orders/{order_id}/checkout")
        assert again.status_code == 201
        assert again.json()["checkout_session_id"] == session_id

        session.refresh(order)
        assert order.stripe_checkout_session_id == session_id
    finally:
        if session_id:
            stripe_svc.expire_checkout_session(session_id)


@pytest.mark.stripe
def test_expiring_a_session_makes_it_unpayable_but_still_readable(
    authed_client, session
):
    """What cleanup can and cannot mean, asserted rather than assumed."""
    order_id = make_order(authed_client, session, [("CHK-REAL-EXP", 1500, 1)])

    created = authed_client.post(f"/orders/{order_id}/checkout").json()
    session_id = created["checkout_session_id"]

    expired = stripe_svc.expire_checkout_session(session_id)

    assert expired.status == checkout.SESSION_EXPIRED
    assert expired.url is None
    # Still resolvable: Stripe keeps sessions permanently.
    assert stripe_svc.retrieve_checkout_session(session_id).id == session_id


# --- the identifier is copied onto the PaymentIntent (D7 step 5) ---------


def test_the_order_id_is_sent_for_the_payment_intent_too(
    authed_client, session, monkeypatch
):
    """Stripe does not propagate a session's metadata down the object chain.

    Measured against a real payment: the PaymentIntent and Charge produced by a
    completed session both came back with `metadata: {}`. A webhook on
    `payment_intent.succeeded` would therefore receive a successful charge it
    could not attribute to an order, so the identifier is sent twice.
    """
    order_id = make_order(authed_client, session, [("CHK-PI-META", 1100, 1)])
    captured = {}

    class FakeSession:
        id = "cs_test_pi_meta"
        url = "https://example.test/pay"

    monkeypatch.setattr(
        stripe_svc,
        "create_checkout_session",
        lambda **kw: captured.update(kw) or FakeSession(),
    )

    create_checkout_session(session, uuid.UUID(order_id))

    assert captured["payment_intent_metadata"] == {"order_id": order_id}
    # Both, not one instead of the other: D8 may subscribe to either event.
    assert captured["metadata"] == {"order_id": order_id}


# --- the pages Stripe redirects back to ----------------------------------


def test_the_success_page_needs_no_api_key(api_client):
    """Stripe redirects a browser here, and a browser carries no key."""
    response = api_client.get("/checkout/success")

    assert response.status_code == 200
    assert "No payment to look up" in response.text


def test_the_cancel_page_needs_no_api_key_and_says_nothing_was_charged(api_client):
    response = api_client.get("/checkout/cancel")

    assert response.status_code == 200
    assert "Nothing was charged" in response.text
    # The distinction that actually matters to a shopper: closing the payment
    # page is not cancelling the order. Said in words a customer can act on
    # rather than as the HTTP route that does it, which was the old text.
    assert "not the same as cancelling the order" in response.text


def test_the_success_page_says_it_cannot_find_an_unknown_order(api_client, monkeypatch):
    """An order id from the session that names no row here.

    The page used to answer this case by describing the *session* and asserting
    the order was still pending, which it had not looked at. Now it says it
    could not find the order — and, because a charge may still have happened,
    it refuses to say anything about the money either way.
    """
    class PaidSession:
        id = "cs_test_paid"
        status = "complete"
        payment_status = "paid"
        amount_total = 4200
        currency = CURRENCY

        class metadata:
            _data = {"order_id": "abc-123"}

    monkeypatch.setattr(stripe_svc, "retrieve_checkout_session", lambda sid: PaidSession())

    response = api_client.get("/checkout/success?session_id=cs_test_paid")

    assert response.status_code == 200
    assert "could not find that order" in response.text
    assert "does not say whether a payment went through" in response.text
    assert "abc-123" in response.text
    # Formatted for a person, not echoed in minor units: `4200` reads as four
    # thousand to the shopper who just paid forty-two. Rendered through
    # `shopagent.money.format_amount`, which is the same function the system
    # prompt teaches the model to imitate — `tests/test_money.py` pins what it
    # produces, and `test_the_prompt_and_the_checkout_page_share_one_formatter`
    # pins that these two surfaces cannot drift apart.
    assert format_amount(4200, CURRENCY) in response.text
    assert "4200" not in response.text


def test_the_success_page_does_not_change_the_order(authed_client, session, monkeypatch):
    """Asserted against a real order rather than described in a comment."""
    order_id = make_order(authed_client, session, [("CHK-PAGE-NOOP", 900, 1)])

    class PaidSession:
        id = "cs_test_noop"
        status = "complete"
        payment_status = "paid"
        amount_total = 900
        currency = CURRENCY

        class metadata:
            _data = {"order_id": order_id}

    monkeypatch.setattr(stripe_svc, "retrieve_checkout_session", lambda sid: PaidSession())

    authed_client.get("/checkout/success?session_id=cs_test_noop")

    order = session.get(Order, uuid.UUID(order_id))
    session.refresh(order)
    assert order.status is OrderStatus.PENDING


# --- what the success page says, once it reads the order (D11 follow-up) ---


def _returning_from(order_id: str, amount: int = 900, payment_status: str = "paid"):
    """A Stripe session as the success page receives one, with an order on it.

    `payment_status` is a parameter rather than a constant because the page now
    reads it: an order is `pending` from the moment it is placed, and this URL
    is one anybody can open, so "your payment is being confirmed" is only true
    when the session says the money actually arrived.
    """

    class Returned:
        id = "cs_test_back"
        status = "complete"
        amount_total = amount
        currency = CURRENCY

        class metadata:
            _data = {"order_id": order_id}

    Returned.payment_status = payment_status
    return Returned


@pytest.mark.parametrize(
    "status, expected",
    [
        (OrderStatus.PENDING, "being confirmed"),
        (OrderStatus.PAID, "Payment received"),
        (OrderStatus.FULFILLED, "Order complete"),
        (OrderStatus.CANCELLED, "was cancelled"),
        (OrderStatus.REFUNDED, "was refunded"),
    ],
)
def test_the_success_page_reports_the_status_the_order_actually_has(
    authed_client, session, monkeypatch, status, expected
):
    """The defect this replaced: a fixed sentence that went stale in a second.

    The page used to assert `pending` without looking, and the signed delivery
    that makes an order `paid` routinely lands before a person finishes
    reading — so the page contradicted the shop about the shopper's own money.
    Every status is covered, including the two nobody expects to arrive on: a
    refund issued while somebody sat on the payment page lands here, and a
    branch that did not exist would have fallen through to a sentence
    describing none of them.

    The status is written directly rather than through `apply_transition`
    because this is a test of what a page *reads*, and driving each of five
    statuses through the lifecycle would test the lifecycle instead.
    """
    order_id = make_order(authed_client, session, [(f"CHK-ST-{status}", 900, 1)])
    order = session.get(Order, uuid.UUID(order_id))
    order.status = status
    session.flush()

    monkeypatch.setattr(
        stripe_svc, "retrieve_checkout_session", lambda sid: _returning_from(order_id)
    )

    response = authed_client.get("/checkout/success?session_id=cs_test_back")

    assert response.status_code == 200
    assert expected in response.text
    # The claim the old page made about every order, now made about none of
    # them without looking.
    assert "has not been marked paid yet" not in response.text


# Every word that would tell a customer which repository they are shopping in.
# A sweep rather than one assertion per phrase, because the phrases that leaked
# were not the ones anybody would have listed: "which is what Day 8 builds" sat
# on this page for four days beside an HTTP route a shopper cannot call.
PROJECT_WORDS = (
    "Day ",
    "webhook",
    "shopagent",
    "ShopAgent",
    "/orders/",
    "POST ",
    "GET ",
    ".py",
    "lifecycle",
    "metadata",
    "X-API-Key",
)

# Every rendering either page can produce, named by the branch that makes it.
# A sweep over one branch is what let a first draft of this test pass while the
# unconfigured-Stripe page still said "the Day 8 webhook endpoint": the branches
# that leak are the rare ones, because they are the ones nobody rereads.
PAGE_BRANCHES = (
    "no-session-id",
    "no-stripe-key",
    "unreadable-session",
    "unknown-order",
    "order-pending",
    "order-paid",
    "order-fulfilled",
    "order-cancelled",
    "order-refunded",
    "cancel-page",
)


def _render_branch(branch, authed_client, session, monkeypatch):
    """Drive one branch of one page and return what a browser would receive."""
    if branch == "cancel-page":
        return authed_client.get("/checkout/cancel")
    if branch == "no-session-id":
        return authed_client.get("/checkout/success")

    if branch == "no-stripe-key":
        def raise_missing(session_id):
            raise stripe_svc.MissingStripeKey("STRIPE_SECRET_KEY is not set")

        monkeypatch.setattr(stripe_svc, "retrieve_checkout_session", raise_missing)
        return authed_client.get("/checkout/success?session_id=cs_test_back")

    if branch == "unreadable-session":
        def boom(session_id):
            raise RuntimeError("No such checkout.session")

        monkeypatch.setattr(stripe_svc, "retrieve_checkout_session", boom)
        return authed_client.get("/checkout/success?session_id=cs_test_back")

    if branch == "unknown-order":
        monkeypatch.setattr(
            stripe_svc,
            "retrieve_checkout_session",
            lambda sid: _returning_from("abc-123"),
        )
        return authed_client.get("/checkout/success?session_id=cs_test_back")

    status = OrderStatus(branch.removeprefix("order-"))
    order_id = make_order(authed_client, session, [(f"CHK-LK-{status}", 900, 1)])
    order = session.get(Order, uuid.UUID(order_id))
    order.status = status
    session.flush()
    monkeypatch.setattr(
        stripe_svc, "retrieve_checkout_session", lambda sid: _returning_from(order_id)
    )
    return authed_client.get("/checkout/success?session_id=cs_test_back")


@pytest.mark.parametrize(
    "payment_status, expected, forbidden",
    [
        ("paid", "being confirmed", "has not been paid"),
        ("no_payment_required", "being confirmed", "has not been paid"),
        ("unpaid", "has not been paid", "do not need to pay again"),
        (None, "has not been paid", "do not need to pay again"),
    ],
)
def test_a_pending_order_is_only_called_a_payment_in_flight_when_it_is_one(
    authed_client, session, monkeypatch, payment_status, expected, forbidden
):
    """`pending` is two situations and only one of them is a payment arriving.

    An order is `pending` from the moment it is placed, and this URL is one
    anybody can open — a shopper who reached the payment page and backed out
    lands here with an `unpaid` session, and "you do not need to pay again"
    would tell somebody who has not paid that they have. The allow-list is
    `SETTLED_PAYMENT_STATUSES`, the same one the webhook uses before moving an
    order to `paid`, which is why `no_payment_required` is in it and this is
    not `== "paid"`. Raised by review on PR #11.
    """
    order_id = make_order(authed_client, session, [(f"CHK-PS-{payment_status}", 900, 1)])
    monkeypatch.setattr(
        stripe_svc,
        "retrieve_checkout_session",
        lambda sid: _returning_from(order_id, payment_status=payment_status),
    )

    response = authed_client.get("/checkout/success?session_id=cs_test_back")

    assert response.status_code == 200
    assert expected in response.text
    assert forbidden not in response.text


@pytest.mark.parametrize("branch", PAGE_BRANCHES)
def test_no_page_a_customer_sees_names_the_project_it_was_built_in(
    authed_client, session, monkeypatch, branch
):
    """These two pages are the only thing a non-developer reads.

    Every branch, not one of them. The first version of this test drove the
    happy path only, and a deliberate mutation putting "the Day 8 webhook
    endpoint" into the unconfigured-Stripe branch survived it — which is the
    same shape as the leak being fixed here, since the sentence that sat on
    this page for four days was in the branch nobody was rereading either.
    """
    response = _render_branch(branch, authed_client, session, monkeypatch)

    assert response.status_code == 200
    leaked = [word for word in PROJECT_WORDS if word in response.text]
    assert leaked == [], f"{branch} names the project: {leaked}"


# Promises this system does not control. Every one of these was on the page and
# every one was removed for a reason recorded in `checkout_pages.py`; the email
# receipt was the one measured against Stripe rather than argued about, and
# `charge.receipt_number` came back null for the live payment of 2026-09-01.
#
# A word list rather than a sentence list, because the sentences will be
# rewritten and the promises are what must not come back — "we will email you"
# is the same claim as "Stripe has emailed you a receipt".
BROKEN_PROMISES = (
    "email",
    "e-mail",
    "inbox",
    "receipt from",
    "emailed",
    "few seconds",
    "few days",
    "whenever you like",
    "will be told",
    "will be returned",
    "sent out",
    "shipped",
)


@pytest.mark.parametrize("branch", PAGE_BRANCHES)
def test_no_page_promises_something_this_shop_does_not_control(
    authed_client, session, monkeypatch, branch
):
    """The page may report; it may not undertake.

    Measured, not assumed: no receipt was emailed for the live payment this
    project made. `charge.receipt_number` is null — Stripe sets it only once a
    receipt has actually been sent — and `receipt_email` is null on both the
    PaymentIntent and the Charge, because nothing here sets it. In test mode
    Stripe does not email receipts unless the dashboard says so, which is a
    setting this repository neither reads nor owns.

    The other removals are of the same kind and needed no API call: a settlement
    time this code does not control, a notification nothing sends, a refund
    nobody issues, a payment window the expiry handler closes, and an order
    cancellation the assistant has no tool for.
    """
    response = _render_branch(branch, authed_client, session, monkeypatch)

    assert response.status_code == 200
    promised = [word for word in BROKEN_PROMISES if word in response.text.lower()]
    assert promised == [], f"{branch} promises: {promised}"


@pytest.mark.parametrize("path", ["/checkout/success", "/checkout/cancel"])
def test_both_pages_send_a_shopper_back_to_the_tab_they_came_from(api_client, path):
    """The tab first, the link second, and the link says what it costs.

    The payment button opens Stripe in a new tab, so the conversation is
    usually still open behind this page — and following the link starts a fresh
    browser session with an empty transcript, which is the wrong thing to do to
    somebody who still has theirs.
    """
    response = api_client.get(path)

    assert response.status_code == 200
    assert get_settings().ui_base_url in response.text
    assert "should still be open in the tab you came from" in response.text
    assert "starts a fresh conversation" in response.text


@pytest.mark.parametrize(
    "status", [OrderStatus.PENDING, OrderStatus.PAID, OrderStatus.CANCELLED]
)
def test_reading_the_success_page_never_moves_an_order(
    authed_client, session, monkeypatch, status
):
    """Reporting is not deciding, asserted from every status it can report.

    A redirect is a URL anybody can open, so a page that wrote anything would
    be trusting a browser with the shop's money. The page now reads the status
    it prints, which is exactly the change that makes this worth asserting more
    than once: the read is right beside the place a write would go.
    """
    order_id = make_order(authed_client, session, [(f"CHK-RO-{status}", 900, 1)])
    order = session.get(Order, uuid.UUID(order_id))
    order.status = status
    session.flush()

    monkeypatch.setattr(
        stripe_svc, "retrieve_checkout_session", lambda sid: _returning_from(order_id)
    )

    authed_client.get("/checkout/success?session_id=cs_test_back")

    session.refresh(order)
    assert order.status is status


def test_an_unknown_session_id_does_not_raise(api_client, monkeypatch):
    def boom(session_id):
        raise RuntimeError("No such checkout.session")

    monkeypatch.setattr(stripe_svc, "retrieve_checkout_session", boom)

    response = api_client.get("/checkout/success?session_id=cs_test_nope")

    assert response.status_code == 200
    assert "could not be read" in response.text


@pytest.mark.stripe
def test_stripe_accepts_payment_intent_data_on_a_real_session(authed_client, session):
    """The half of the claim a test can reach, and a note on the half it cannot.

    **Confirmed by a manual payment on 2026-08-27**: with
    `payment_intent_data={"metadata": {...}}` on the session, the resulting
    PaymentIntent *and* its Charge both carry `metadata.order_id`. Before the
    parameter was added, both came back `{}`. So D8 may subscribe to
    `checkout.session.completed`, `payment_intent.succeeded` or
    `charge.succeeded` and attribute any of them.

    This test cannot reproduce that, and the reason is worth stating rather
    than working around: **no PaymentIntent exists until a shopper starts
    paying** — `session.payment_intent` is null on a fresh session, the session
    does not echo `payment_intent_data` back, and a hosted Checkout page cannot
    be completed through the API. Pinning the id of the manually paid session
    here would make the suite depend on one object in one Stripe account and
    would still not catch the regression that matters.

    That regression — this project quietly ceasing to send the parameter — is
    caught offline by `test_the_order_id_is_sent_for_the_payment_intent_too`,
    which reads the payload. What is left for this test is that Stripe accepts
    the parameter alongside everything else the session carries.
    """
    order_id = make_order(authed_client, session, [("CHK-PI-REAL", 1300, 2)])
    session_id = None

    try:
        response = authed_client.post(f"/orders/{order_id}/checkout")
        assert response.status_code == 201
        session_id = response.json()["checkout_session_id"]

        live = stripe_svc.retrieve_checkout_session(session_id)
        assert live.status == "open"
        assert live.metadata._data["order_id"] == order_id
        # The documented shape of the gap, so nobody later mistakes this test
        # for proof that the PaymentIntent carries the id.
        assert live.payment_intent is None
        assert "payment_intent_data" not in live._data
    finally:
        if session_id:
            stripe_svc.expire_checkout_session(session_id)


# --- what review on PR #7 turned up --------------------------------------


def test_an_unreadable_session_does_not_claim_nothing_was_charged(
    api_client, monkeypatch
):
    """A server that cannot check is not a server that knows.

    A key can be rotated or removed after a session was created and paid.
    Telling a shopper who was charged that they were not is the worst of the
    three possible answers, so neither failure branch says it.
    """
    def no_key(session_id):
        raise stripe_svc.MissingStripeKey("STRIPE_SECRET_KEY is not set")

    monkeypatch.setattr(stripe_svc, "retrieve_checkout_session", no_key)

    response = api_client.get("/checkout/success?session_id=cs_test_x")

    assert response.status_code == 200
    assert "Nothing was charged" not in response.text
    assert "does not mean the payment did not go through" in response.text


def test_creating_a_checkout_locks_the_order_row(authed_client, session, monkeypatch):
    """Two concurrent checkouts would otherwise leave two payable sessions.

    Both read "no stored session", both create one, and the order remembers
    only the second — the first stays open and chargeable with nothing pointing
    at it. Invisible single-threaded, so the SQL is read instead.
    """
    order_id = make_order(authed_client, session, [("CHK-LOCK-SQL", 700, 1)])

    class FakeSession:
        id = "cs_test_lock"
        url = "https://example.test/pay"

    monkeypatch.setattr(stripe_svc, "create_checkout_session", lambda **kw: FakeSession())

    with recorded_sql() as statements:
        create_checkout_session(session, uuid.UUID(order_id))

    normalised = [" ".join(s.lower().split()) for s in statements]
    locked = [s for s in normalised if "from orders" in s and "for update" in s]
    assert locked, "the order row was read without FOR UPDATE"


def test_applying_a_transition_locks_the_order_row(authed_client, session):
    """The sanctioned boundary carries the lock, so no caller has to remember.

    Two concurrent refunds would otherwise both read `paid`, both pass the
    transition table, and both release the same reservation.
    """
    from shopagent.api.services import orders as order_service

    order_id = make_order(authed_client, session, [("CHK-TRANS-SQL", 600, 1)])
    order = session.get(Order, uuid.UUID(order_id))

    with recorded_sql() as statements:
        order_service.apply_transition(session, order, OrderStatus.PAID)

    normalised = [" ".join(s.lower().split()) for s in statements]
    assert [s for s in normalised if "from orders" in s and "for update" in s]


def test_cancelling_expires_the_session_before_committing(
    authed_client, session, monkeypatch
):
    """Ordering, asserted rather than described.

    Committing first and expiring after fails badly: the cancellation stands,
    the expiry times out, and a terminally cancelled order keeps a payable URL
    with no way back — a retry is refused by the transition table.
    """
    order_id = make_order(authed_client, session, [("CHK-ORDER-SQL", 800, 1)])
    session.execute(
        Order.__table__.update()
        .where(Order.id == uuid.UUID(order_id))
        .values(stripe_checkout_session_id="cs_test_order_check")
    )
    session.commit()

    seen: list[str] = []

    class Open:
        id = "cs_test_order_check"
        status = "open"

    monkeypatch.setattr(
        stripe_svc, "retrieve_checkout_session", lambda sid: seen.append("retrieve") or Open()
    )

    def expire(sid):
        # The status must still be pending at this point: if the cancellation
        # had already been committed, this would read `cancelled`.
        session.refresh(order)
        seen.append(f"expire:{order.status.value}")

    monkeypatch.setattr(stripe_svc, "expire_checkout_session", expire)

    from shopagent.api.services import orders as order_service

    order = session.get(Order, uuid.UUID(order_id))
    order_service.cancel_order(session, uuid.UUID(order_id))

    assert seen == ["retrieve", "expire:pending"]
    session.refresh(order)
    assert order.status is OrderStatus.CANCELLED


def test_cancelling_refuses_when_the_session_cannot_be_closed(
    authed_client, session, monkeypatch
):
    """A stored session id proves Stripe was configured when the order was placed.

    Swallowing a missing key now would cancel the order and leave a payable
    link behind in silence.
    """
    from shopagent.api.services import orders as order_service

    order_id = make_order(authed_client, session, [("CHK-NOKEY", 800, 1)])
    session.execute(
        Order.__table__.update()
        .where(Order.id == uuid.UUID(order_id))
        .values(stripe_checkout_session_id="cs_test_orphan")
    )
    session.commit()

    def no_key(session_id):
        raise stripe_svc.MissingStripeKey("STRIPE_SECRET_KEY is not set")

    monkeypatch.setattr(stripe_svc, "retrieve_checkout_session", no_key)

    with pytest.raises(order_service.SessionExpiryFailed):
        order_service.cancel_order(session, uuid.UUID(order_id))

    order = session.get(Order, uuid.UUID(order_id))
    session.refresh(order)
    assert order.status is OrderStatus.PENDING, "the order was cancelled anyway"


def test_an_order_survives_stripe_being_down_when_attaching_a_customer(
    authed_client, session, monkeypatch
):
    """`place_order` has already committed by the time the buyer is attached.

    An exception here would surface as a 500 for an order that exists, and the
    retry would be refused with 409 because the cart is already ordered —
    leaving the caller without the order id at all.
    """
    from shopagent.payments import customers as customer_service

    def boom(*args, **kwargs):
        raise RuntimeError("Stripe is unreachable")

    monkeypatch.setattr(customer_service, "get_or_create_customer", boom)

    variant = make_variant(session, sku="CHK-CUST-DOWN", amount_cents=900)
    cart_id = authed_client.post("/cart").json()["cart_id"]
    authed_client.post(f"/cart/{cart_id}/items", json={"variant_id": variant.id, "quantity": 1})

    response = authed_client.post(
        "/orders", json={"cart_id": cart_id, "customer_email": "down@example.com"}
    )

    assert response.status_code == 201
    order_id = response.json()["order_id"]
    order = session.get(Order, uuid.UUID(order_id))
    session.refresh(order)
    # The email is ours and is kept; the Stripe Customer is the part that failed.
    assert order.customer_email == "down@example.com"
    assert order.stripe_customer_id is None
