"""What a webhook delivery does to an order (D8, step 3).

Separate from `tests/test_webhooks.py`, which is about the door: signature,
idempotency, status codes. This file is about the room behind it — whether a
completed checkout makes an order paid, and, far more importantly, whether an
expired one is ever allowed to cancel an order that was actually paid.

That last case is the reason this file is careful. `cancelled` is terminal and
this is the only path in the project where stock is released without a person
deciding to, so a mistake here is unrecoverable in both directions at once:
the money is real and the reservation is gone.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from shopagent.config import get_settings
from shopagent.api.lifecycle import OrderStatus
from shopagent.api.models import Order, ProcessedEvent
from shopagent.api.routers import webhooks
from shopagent.api.services import events as event_service
from shopagent.catalog.models import Inventory, Price, Product, Variant
from shopagent.config import Settings
from shopagent.payments import stripe_svc


# The shop's currency, read rather than written. A test that creates a price
# row in a literal currency and then asks a service to find it is testing its
# own literal: the service filters on `settings.currency`, so the two have to
# agree by construction, and the day they stopped agreeing is the day 125 of
# them failed at once. What a specific currency *is* still gets pinned, in the
# tests that are actually about that — `format_amount`, and the two that prove
# a foreign currency is treated as foreign.
CURRENCY = get_settings().currency

pytestmark = pytest.mark.db

TEST_SECRET = "whsec_offline_test_secret_not_a_real_one"


# --- plumbing ------------------------------------------------------------


def sign(payload: bytes, secret: str = TEST_SECRET) -> str:
    """The same hand-rolled signer `tests/test_webhooks.py` documents."""
    timestamp = int(time.time())
    signed = f"{timestamp}.".encode() + payload
    return f"t={timestamp},v1={hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()}"


@pytest.fixture
def client(authed_client, monkeypatch):
    """The app on the test's transaction, with a known signing secret.

    `authed_client` rather than `api_client` because these tests build a real
    order through `POST /cart` and `POST /orders`, which need the API key. The
    webhook route itself ignores it — Stripe has none to send.
    """
    monkeypatch.setattr(
        webhooks, "get_settings", lambda: Settings(stripe_webhook_secret=TEST_SECRET)
    )
    return authed_client


def deliver(client, body: dict) -> object:
    payload = json.dumps(body).encode()
    return client.post(
        "/webhooks/stripe",
        content=payload,
        headers={"Stripe-Signature": sign(payload), "Content-Type": "application/json"},
    )


def event(event_type: str, obj: dict, *, event_id: str | None = None) -> dict:
    return {
        "id": event_id or f"evt_{uuid.uuid4().hex[:24]}",
        "object": "event",
        "type": event_type,
        "created": 1_756_000_000,
        "livemode": False,
        "api_version": stripe_svc.STRIPE_API_VERSION,
        "data": {"object": obj},
    }


def completed_event(order_id, *, session_id="cs_test_handler", payment_intent="pi_test_handler", **kw):
    return event(
        "checkout.session.completed",
        {
            "id": session_id,
            "object": "checkout.session",
            "status": "complete",
            "payment_status": "paid",
            "payment_intent": payment_intent,
            "metadata": {"order_id": str(order_id)},
        },
        **kw,
    )


def expired_event(order_id, *, session_id="cs_test_handler", **kw):
    return event(
        "checkout.session.expired",
        {
            "id": session_id,
            "object": "checkout.session",
            "status": "expired",
            "payment_status": "unpaid",
            "metadata": {"order_id": str(order_id)},
        },
        **kw,
    )


def payment_intent_event(event_type: str, order_id, **kw):
    return event(
        event_type,
        {
            "id": "pi_test_handler",
            "object": "payment_intent",
            "metadata": {"order_id": str(order_id)},
        },
        **kw,
    )


class FakeSession:
    """What `retrieve_checkout_session` returns, to the extent this code reads it.

    `status` is here for `cancel_order`, which reads it to decide whether the
    session still needs expiring; the handlers under test read
    `payment_status`.
    """

    def __init__(self, payment_status: str, status: str = "expired") -> None:
        self.payment_status = payment_status
        self.status = status


def stripe_says(monkeypatch, payment_status: str, status: str = "expired") -> None:
    """Point `retrieve_checkout_session` at a fake, and refuse every other call.

    `expire_checkout_session` is stubbed rather than left alone because
    `POST /orders/{id}/cancel` calls it, and the autouse guard in
    `conftest.py` turns any real Stripe request into a failed test — correctly,
    since none of these tests mean to leave the process.
    """
    monkeypatch.setattr(
        stripe_svc,
        "retrieve_checkout_session",
        lambda session_id: FakeSession(payment_status, status),
    )
    monkeypatch.setattr(
        stripe_svc, "expire_checkout_session", lambda session_id: FakeSession("unpaid")
    )


# --- fixtures that build a real order ------------------------------------


def make_variant(session, *, sku: str, quantity: int = 20, reserved: int = 0) -> Variant:
    product = Product(
        name=f"Webhook Fixture {sku}",
        description="A product that exists only for a webhook handler test.",
        category="shoes",
        brand="Fleetfoot",
        variants=[
            Variant(
                size="42",
                color="blue",
                sku=sku,
                prices=[Price(currency=CURRENCY, amount_cents=1500, active=True)],
                inventory=Inventory(quantity=quantity, reserved=reserved),
            )
        ],
    )
    session.add(product)
    session.commit()
    return product.variants[0]


def stock(session, variant_id: int) -> tuple[int, int]:
    row = session.execute(
        select(Inventory.quantity, Inventory.reserved).where(
            Inventory.variant_id == variant_id
        )
    ).one()
    return int(row[0]), int(row[1])


def make_order(client, variant, quantity: int = 2, *, session_id="cs_test_handler"):
    """A real order through the API, with a checkout session id attached.

    The session id is set directly rather than by calling the checkout route,
    which would reach Stripe. What matters to these tests is that the order
    points at a session, because the expiry guard compares the two.
    """
    cart_id = client.post("/cart").json()["cart_id"]
    client.post(
        f"/cart/{cart_id}/items", json={"variant_id": variant.id, "quantity": quantity}
    )
    order_id = client.post("/orders", json={"cart_id": cart_id}).json()["order_id"]
    return uuid.UUID(order_id)


def attach_session(session, order_id, session_id="cs_test_handler") -> Order:
    order = session.get(Order, order_id)
    order.stripe_checkout_session_id = session_id
    session.commit()
    return order


# --- checkout.session.completed ------------------------------------------


def test_a_completed_checkout_marks_the_order_paid(client, session):
    """The deliverable of the day, at the smallest scale that shows it."""
    variant = make_variant(session, sku="WH-PAID")
    order_id = make_order(client, variant)
    attach_session(session, order_id)

    response = deliver(client, completed_event(order_id))

    assert response.status_code == 200
    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PAID


def test_a_completed_checkout_records_the_payment_intent(client, session):
    """The column D6 declared and nothing has ever written to.

    Step 4's refund needs a PaymentIntent to refund against, and the session is
    the first object in the chain that names one.
    """
    variant = make_variant(session, sku="WH-PI")
    order_id = make_order(client, variant)
    attach_session(session, order_id)

    deliver(client, completed_event(order_id, payment_intent="pi_3ABC_real_looking"))

    session.expire_all()
    assert session.get(Order, order_id).stripe_payment_intent_id == "pi_3ABC_real_looking"


def test_a_completed_checkout_does_not_release_stock(client, session):
    """`paid` is not in `RELEASES_RESERVATION`, and this says so from outside.

    The units were sold. They stay reserved until a fulfilment flow moves them
    out of `quantity`, which this project does not have.
    """
    variant = make_variant(session, sku="WH-PAID-STOCK", quantity=30)
    order_id = make_order(client, variant, quantity=4)
    attach_session(session, order_id)
    before = stock(session, variant.id)

    deliver(client, completed_event(order_id))

    session.expire_all()
    assert stock(session, variant.id) == before


def test_a_completed_checkout_for_a_cancelled_order_changes_nothing(
    client, session, monkeypatch
):
    """The order moved on, and the transition table refuses to move it back.

    Answered 200 anyway: the refusal is permanent, so a retry produces the same
    answer, and a 500 would have Stripe redeliver for three days. Logged at
    ERROR rather than INFO, because money settled against an order that is
    terminally cancelled is a real problem for a person to look at even though
    no code can fix it.
    """
    stripe_says(monkeypatch, "unpaid", status="open")
    variant = make_variant(session, sku="WH-CANCELLED", quantity=30)
    order_id = make_order(client, variant, quantity=3)
    attach_session(session, order_id)
    client.post(f"/orders/{order_id}/cancel")
    session.expire_all()
    after_cancel = stock(session, variant.id)

    response = deliver(client, completed_event(order_id))

    assert response.status_code == 200
    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.CANCELLED
    assert stock(session, variant.id) == after_cancel


def test_the_error_for_a_cancelled_order_names_the_money(
    client, session, caplog, monkeypatch
):
    """A log line nobody can act on is a log line nobody reads."""
    caplog.set_level(logging.INFO, logger="shopagent.api.services.events")
    stripe_says(monkeypatch, "unpaid", status="open")
    variant = make_variant(session, sku="WH-CANCELLED-LOG", quantity=30)
    order_id = make_order(client, variant, quantity=2)
    attach_session(session, order_id)
    client.post(f"/orders/{order_id}/cancel")

    deliver(client, completed_event(order_id))

    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert str(order_id) in errors[0]
    assert f"3000 {CURRENCY}" in errors[0]


def test_two_different_events_both_asking_for_paid_leave_one_transition(client, session):
    """`processed_events` cannot help here, so the transition table has to.

    Deduplication is by `event.id`. A manual replay under a *fresh* id — or
    genuinely two events describing one payment — gets past it and reaches the
    handler, where `paid -> paid` is not in the table. The second is a no-op
    and a 200, logged at INFO because this is the backstop working rather than
    something going wrong.
    """
    variant = make_variant(session, sku="WH-TWICE")
    order_id = make_order(client, variant)
    attach_session(session, order_id)

    first = deliver(client, completed_event(order_id, event_id="evt_paid_one"))
    second = deliver(client, completed_event(order_id, event_id="evt_paid_two"))

    assert (first.status_code, second.status_code) == (200, 200)
    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PAID
    # Both were recorded — they are different deliveries, and the second was
    # legitimately handled and legitimately did nothing.
    assert session.get(ProcessedEvent, "evt_paid_one") is not None
    assert session.get(ProcessedEvent, "evt_paid_two") is not None


# --- checkout.session.expired --------------------------------------------


def test_an_expired_unpaid_session_cancels_the_order(client, session, monkeypatch):
    """The ordinary case: nobody paid, the page timed out, give the stock back."""
    stripe_says(monkeypatch, "unpaid")
    variant = make_variant(session, sku="WH-EXPIRED", quantity=30, reserved=5)
    before = stock(session, variant.id)
    order_id = make_order(client, variant, quantity=4)
    attach_session(session, order_id)
    session.expire_all()
    assert stock(session, variant.id)[1] == before[1] + 4

    response = deliver(client, expired_event(order_id))

    assert response.status_code == 200
    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.CANCELLED
    # Back exactly where it started, not merely lower.
    assert stock(session, variant.id) == before


def test_an_expired_session_that_was_actually_paid_cancels_nothing(
    client, session, monkeypatch
):
    """**The most important test in this file.**

    Stripe can expire a session whose payment is in flight, and delivery order
    is not guaranteed — an `expired` can arrive before its `completed`. If the
    handler trusted the event, the order would be terminally `cancelled` with
    the money real and the stock handed back, and nothing could undo it.

    So the event is not trusted: the session is fetched and `payment_status`
    read from Stripe's answer. Here Stripe says `paid`, and the correct
    behaviour is to do nothing at all and wait for `completed`.
    """
    stripe_says(monkeypatch, "paid")
    variant = make_variant(session, sku="WH-EXPIRED-PAID", quantity=30)
    order_id = make_order(client, variant, quantity=3)
    attach_session(session, order_id)
    session.expire_all()
    reserved_before = stock(session, variant.id)

    response = deliver(client, expired_event(order_id))

    assert response.status_code == 200
    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PENDING
    assert stock(session, variant.id) == reserved_before


@pytest.mark.parametrize("payment_status", ["paid", "no_payment_required", None])
def test_only_an_unpaid_session_may_cancel(client, session, monkeypatch, payment_status):
    """Allow-list rather than deny-list, checked against every other value.

    `!= "unpaid"` and `== "paid"` differ exactly when Stripe returns a third
    thing, and it has three: `no_payment_required` is real, and `None` is what
    a field this code did not expect looks like. The safe reading of anything
    unrecognised is "do not cancel".
    """
    stripe_says(monkeypatch, payment_status)
    variant = make_variant(session, sku=f"WH-PS-{payment_status}", quantity=30)
    order_id = make_order(client, variant, quantity=2)
    attach_session(session, order_id)

    assert deliver(client, expired_event(order_id)).status_code == 200

    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PENDING


def test_an_old_session_expiring_does_not_cancel_a_newer_checkout(
    client, session, monkeypatch
):
    """The second guard, and the one that is easy not to think of.

    An order whose first session expired and which then started a new checkout
    would otherwise be cancelled by the *old* session's expiry — while the
    shopper is sitting on the new payment page. The event's session must be the
    one the order currently points at.

    `retrieve_checkout_session` is made to fail here: if the guard is ever
    removed, this test does not merely assert the wrong status, it explodes,
    which is harder to paper over.
    """
    def refuse(session_id):
        raise AssertionError(
            f"the handler asked Stripe about {session_id}, which is not the "
            "session this order points at — the guard is gone"
        )

    monkeypatch.setattr(stripe_svc, "retrieve_checkout_session", refuse)
    variant = make_variant(session, sku="WH-OLD-SESSION", quantity=30)
    order_id = make_order(client, variant, quantity=2)
    attach_session(session, order_id, session_id="cs_test_the_new_one")

    response = deliver(client, expired_event(order_id, session_id="cs_test_the_old_one"))

    assert response.status_code == 200
    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PENDING


def test_a_stripe_failure_while_checking_expiry_is_a_500(client, session, monkeypatch):
    """Transient, so Stripe should retry rather than be told everything is fine.

    The alternative is guessing "probably unpaid" and cancelling, which is the
    one irreversible way to be wrong. A 500 leaves the order exactly as it was.
    """
    def boom(session_id):
        raise RuntimeError("Stripe is unreachable")

    monkeypatch.setattr(stripe_svc, "retrieve_checkout_session", boom)
    variant = make_variant(session, sku="WH-EXPIRED-BOOM", quantity=30)
    order_id = make_order(client, variant, quantity=2)
    attach_session(session, order_id)

    with pytest.raises(RuntimeError):
        deliver(client, expired_event(order_id, event_id="evt_expired_boom"))

    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PENDING
    # And the claim was rolled back with it, so the retry is a retry.
    assert session.get(ProcessedEvent, "evt_expired_boom") is None


def test_an_expired_session_for_an_already_cancelled_order_is_a_no_op(
    client, session, monkeypatch
):
    """`cancelled` is terminal, so a second release is refused before stock moves.

    The same protection D7 built for a double cancel, reached from a different
    direction.
    """
    stripe_says(monkeypatch, "unpaid", status="open")
    variant = make_variant(session, sku="WH-EXPIRED-TWICE", quantity=30)
    order_id = make_order(client, variant, quantity=3)
    attach_session(session, order_id)
    client.post(f"/orders/{order_id}/cancel")
    session.expire_all()
    after_cancel = stock(session, variant.id)

    assert deliver(client, expired_event(order_id)).status_code == 200

    session.expire_all()
    assert stock(session, variant.id) == after_cancel


# --- the events that deliberately do nothing ------------------------------


def test_payment_intent_succeeded_changes_nothing(client, session):
    """Two events describe one payment; only one may drive the transition.

    If this moved the order too, whichever of the pair arrived second would be
    refused by the transition table on every successful payment — a permanent
    stream of warnings describing the system working correctly.
    """
    variant = make_variant(session, sku="WH-PI-SUCCESS")
    order_id = make_order(client, variant)
    attach_session(session, order_id)

    response = deliver(
        client, payment_intent_event("payment_intent.succeeded", order_id)
    )

    assert response.status_code == 200
    session.expire_all()
    order = session.get(Order, order_id)
    assert order.status == OrderStatus.PENDING
    assert order.stripe_payment_intent_id is None


def test_payment_failed_leaves_the_order_pending_and_says_so(client, session, caplog):
    """A declined card is not the end of a checkout.

    The shopper is usually still on the page and can try another card, and the
    session stays open until it expires. Cancelling here would release the
    stock out from under them — and `cancelled` is terminal, so they could not
    simply try again.
    """
    caplog.set_level(logging.INFO, logger="shopagent.api.services.events")
    variant = make_variant(session, sku="WH-PI-FAIL", quantity=30)
    order_id = make_order(client, variant, quantity=2)
    attach_session(session, order_id)
    before = stock(session, variant.id)

    body = payment_intent_event("payment_intent.payment_failed", order_id)
    body["data"]["object"]["last_payment_error"] = {
        "code": "card_declined",
        "message": "Your card was declined.",
    }
    response = deliver(client, body)

    assert response.status_code == 200
    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PENDING
    assert stock(session, variant.id) == before

    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "card_declined" in logged
    assert str(order_id) in logged


def test_the_failure_log_carries_the_code_and_not_the_shopper_s_message(
    client, session, caplog
):
    """The decline message is written for a shopper and can name their bank.

    The code is what a log is for, and it is the half that means the same thing
    to everyone reading it.
    """
    caplog.set_level(logging.INFO, logger="shopagent.api.services.events")
    variant = make_variant(session, sku="WH-PI-FAIL-MSG")
    order_id = make_order(client, variant)
    attach_session(session, order_id)

    body = payment_intent_event("payment_intent.payment_failed", order_id)
    body["data"]["object"]["last_payment_error"] = {
        "code": "card_declined",
        "message": "Contact First National Bank of Somewhere on 555-0100.",
    }
    deliver(client, body)

    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "First National Bank" not in logged
    assert "555-0100" not in logged


def test_an_unhandled_event_type_changes_nothing(client, session):
    variant = make_variant(session, sku="WH-UNHANDLED")
    order_id = make_order(client, variant)

    response = deliver(
        client,
        event("invoice.payment_succeeded", {"id": "in_1", "object": "invoice"}),
    )

    assert response.status_code == 200
    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PENDING


def test_the_handled_types_are_the_ones_that_were_decided_on():
    """A table rather than a chain of `if`s, so it can be asserted.

    Everything this server acts on, in one place. Anything not listed is
    answered 200 and ignored, which is what keeps Stripe from retrying an
    event nothing was ever going to handle.
    """
    assert set(event_service.HANDLERS) == {
        "checkout.session.completed",
        "checkout.session.async_payment_succeeded",
        "checkout.session.async_payment_failed",
        "checkout.session.expired",
        "payment_intent.payment_failed",
        "payment_intent.succeeded",
        "charge.refunded",
    }


# --- attribution ---------------------------------------------------------


def test_an_event_naming_an_order_that_does_not_exist_is_accepted(client, session):
    """Permanent, so 200. A retry cannot conjure the order into this database.

    It happens for real: a `stripe listen` left running while the database is
    rebuilt delivers events for orders that no longer exist.
    """
    response = deliver(client, completed_event(uuid.uuid4()))

    assert response.status_code == 200


def test_an_event_with_no_order_id_is_accepted(client, session):
    """Metadata is fixed at creation; no redelivery carries more than the first."""
    body = completed_event(uuid.uuid4())
    body["data"]["object"]["metadata"] = {}

    assert deliver(client, body).status_code == 200


def test_a_malformed_order_id_is_accepted_rather_than_crashing(client, session):
    """Asking Postgres to look up a non-UUID raises `DataError`.

    That would surface as a 500 and be retried for three days against a string
    that is never going to parse, so it is caught where it is recognisable.
    """
    body = completed_event(uuid.uuid4())
    body["data"]["object"]["metadata"] = {"order_id": "not-a-uuid"}

    assert deliver(client, body).status_code == 200


def test_the_order_id_is_read_the_same_way_from_all_three_object_types():
    """One function, because D7 made one possible.

    Stripe propagates nothing: a session's metadata stays on the session, and
    the PaymentIntent and Charge come back empty unless the checkout passes
    `payment_intent_data` as well. It does, so all three carry `order_id` and
    the reader needs no knowledge of which arrived. This is that dependency,
    written down as a test.
    """
    order_id = uuid.uuid4()

    # Built by hand rather than through the SDK: what is under test is the
    # reader, and the SDK's own parsing is exercised throughout the rest of the
    # suite. What matters is the shape — `event.data.object.metadata._data` —
    # which is identical for a Session, a PaymentIntent and a Charge.
    class Metadata:
        def __init__(self, data):
            self._data = data

    class Obj:
        def __init__(self, data):
            self.metadata = Metadata(data)

    class Event:
        id = "evt_reader"

        def __init__(self, data):
            self.data = type("Data", (), {"object": Obj(data)})()

    for object_type in ("checkout.session", "payment_intent", "charge"):
        found = event_service.order_id_from(Event({"order_id": str(order_id)}))
        assert found == order_id, f"could not read order_id off a {object_type}"

    assert event_service.order_id_from(Event({})) is None
    assert event_service.order_id_from(Event({"order_id": "nope"})) is None


# --- the transaction, now that there is work in it ------------------------


def test_a_failure_in_a_handler_leaves_the_order_and_the_claim_untouched(
    client, session, monkeypatch
):
    """Step 2's invariant, re-checked now that the handler actually does work.

    That test used an empty seam and a stub that raised. This one fails
    *inside* a real handler, after it has already touched the order, and
    asserts both halves come back: the status is unchanged and the
    `processed_events` row is gone, so Stripe's retry is handled rather than
    skipped.
    """
    variant = make_variant(session, sku="WH-ROLLBACK", quantity=30)
    order_id = make_order(client, variant, quantity=2)
    attach_session(session, order_id)
    before = stock(session, variant.id)

    def explode(session_, order, target, event_, **kwargs):
        raise RuntimeError("the database went away mid-transition")

    monkeypatch.setattr(event_service, "_move", explode)

    with pytest.raises(RuntimeError):
        deliver(client, completed_event(order_id, event_id="evt_handler_exploded"))

    session.expire_all()
    order = session.get(Order, order_id)
    assert order.status == OrderStatus.PENDING
    # The column the handler had already set before the failure.
    assert order.stripe_payment_intent_id is None
    assert stock(session, variant.id) == before
    assert session.get(ProcessedEvent, "evt_handler_exploded") is None


def test_the_retry_after_a_handler_failure_is_processed_normally(
    client, session, monkeypatch
):
    """The consequence, end to end. This is what the rollback is for."""
    variant = make_variant(session, sku="WH-RETRY", quantity=30)
    order_id = make_order(client, variant, quantity=2)
    attach_session(session, order_id)

    def explode(session_, order, target, event_, **kwargs):
        raise RuntimeError("transient")

    monkeypatch.setattr(event_service, "_move", explode)
    body = completed_event(order_id, event_id="evt_retry_me")

    with pytest.raises(RuntimeError):
        deliver(client, body)

    monkeypatch.undo()
    monkeypatch.setattr(
        webhooks, "get_settings", lambda: Settings(stripe_webhook_secret=TEST_SECRET)
    )

    retry = deliver(client, body)

    assert retry.status_code == 200
    assert retry.json()["duplicate"] is False
    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PAID


# --- the field the whole expiry guard rests on ---------------------------


@pytest.mark.stripe
def test_a_real_expired_session_reports_payment_status_unpaid():
    """Every test above decides what Stripe says. This one asks it.

    The expiry guard is a string comparison against `payment_status`, so if
    that field were spelled differently, nested, or absent on a real session,
    the guard would be comparing `None` to `"unpaid"` — and the safe branch
    would be taken every time, quietly turning the cancel path off. D7
    collected four SDK shapes that were not what they looked like, each found
    by a real call, so this is the call.

    Skips rather than fails on an account with no expired sessions, which says
    nothing about the field.
    """
    events = list(
        stripe_svc.get_client().v1.events.list(
            params={"type": "checkout.session.expired", "limit": 1}
        )
    )
    if not events:
        pytest.skip("this Stripe account has no expired checkout sessions")

    expired = events[0].data.object

    assert expired.status == "expired"
    assert expired.payment_status == "unpaid"
    # And the value the handler compares against is reachable the same way on a
    # session fetched fresh, which is what it actually reads.
    live = stripe_svc.retrieve_checkout_session(expired.id)
    assert live.payment_status == "unpaid"


@pytest.mark.stripe
def test_a_real_completed_session_carries_a_string_payment_intent():
    """`stripe_payment_intent_id` is a VARCHAR, and step 4 refunds against it.

    An expanded PaymentIntent object here would be stored as a repr and the
    refund would fail on a value that looks almost right.
    """
    events = list(
        stripe_svc.get_client().v1.events.list(
            params={"type": "checkout.session.completed", "limit": 1}
        )
    )
    if not events:
        pytest.skip("this Stripe account has no completed checkout sessions")

    completed = events[0].data.object

    assert isinstance(completed.payment_intent, str)
    assert completed.payment_intent.startswith("pi_")
    assert completed.payment_status == "paid"


# --- which Stripe account an event came from ------------------------------


def test_an_event_from_another_account_is_accepted_and_warned_about(
    client, session, monkeypatch, caplog
):
    """A warning, never a refusal.

    A Connect platform legitimately receives events from every account
    connected to it, so refusing one would break Connect outright. The line has
    to name both ids, because "wrong account" without them is a sentence
    nobody can act on.
    """
    caplog.set_level(logging.INFO, logger="shopagent.api.services.events")
    monkeypatch.setattr(
        stripe_svc, "configured_account_id", lambda: "acct_ours"
    )
    variant = make_variant(session, sku="WH-ACCOUNT")
    order_id = make_order(client, variant)
    attach_session(session, order_id)

    body = completed_event(order_id)
    body["account"] = "acct_somebody_elses"
    response = deliver(client, body)

    assert response.status_code == 200
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "account" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert "acct_somebody_elses" in warnings[0]
    assert "acct_ours" in warnings[0]

    # And it really is only a warning: the order still moved.
    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PAID


def test_an_event_from_our_own_account_warns_about_nothing(
    client, session, monkeypatch, caplog
):
    """The other half, without which the test above passes for any event."""
    caplog.set_level(logging.INFO, logger="shopagent.api.services.events")
    monkeypatch.setattr(stripe_svc, "configured_account_id", lambda: "acct_ours")
    variant = make_variant(session, sku="WH-ACCOUNT-OK")
    order_id = make_order(client, variant)
    attach_session(session, order_id)

    body = completed_event(order_id)
    body["account"] = "acct_ours"
    deliver(client, body)

    assert not [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and "account" in r.getMessage()
    ]


def test_an_ordinary_event_never_asks_stripe_which_account_this_is(monkeypatch):
    """The cost argument, asserted rather than promised.

    `account` is a Connect-only field: an ordinary event does not carry the key
    at all, checked against five real events from this account. So the check
    must return before it looks anything up, and no webhook may pay for a
    network call it cannot use.

    Counted rather than raised from the stub, and that distinction was found by
    falsifying this test. An earlier version made the stub raise
    `AssertionError` — which `warn_on_account_mismatch` catches by design, so
    that it can never break a delivery. The test therefore passed with the
    early return deleted, which is precisely the regression it exists to
    catch. A counter cannot be swallowed.
    """
    calls: list[int] = []

    def counted():
        calls.append(1)
        return "acct_ours"

    monkeypatch.setattr(stripe_svc, "configured_account_id", counted)

    class Event:
        id = "evt_ordinary"
        type = "checkout.session.completed"

    event_service.warn_on_account_mismatch(Event())

    assert calls == [], (
        "the account check reached Stripe for an event with no `account` "
        "field — every ordinary delivery would now pay for a network call"
    )


def test_a_failed_account_lookup_does_not_break_the_delivery(
    client, session, monkeypatch
):
    """Diagnostics must not be able to cause the outage they describe.

    If Stripe is unreachable when the check runs, the check is skipped and the
    event is handled. The alternative — failing a webhook because an advisory
    comparison could not be made — turns a log line into lost payments.
    """
    def boom():
        raise RuntimeError("Stripe is unreachable")

    monkeypatch.setattr(stripe_svc, "configured_account_id", boom)
    variant = make_variant(session, sku="WH-ACCOUNT-BOOM")
    order_id = make_order(client, variant)
    attach_session(session, order_id)

    body = completed_event(order_id)
    body["account"] = "acct_somebody_elses"

    assert deliver(client, body).status_code == 200
    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PAID


# --- charge.refunded (D8 step 4) -----------------------------------------


def refunded_event(
    order_id,
    *,
    amount=3000,
    amount_refunded=3000,
    refunded=None,
    payment_intent="pi_test_handler",
    **kw,
):
    """A `charge.refunded` shaped the way Stripe really sends one.

    `refunded` defaults to whatever the amounts imply, because that is the
    invariant Stripe maintains; tests that want the two to disagree pass it
    explicitly.

    `payment_intent` defaults to the one `completed_event` writes, so the
    charge belongs to the order the way a real one does. That it was missing
    here entirely is why review had to point out that nothing checked it: the
    fixture did not model the field, so no test could have noticed its absence.
    """
    if refunded is None:
        refunded = amount_refunded >= amount
    return event(
        "charge.refunded",
        {
            "id": "ch_test_handler",
            "object": "charge",
            "amount": amount,
            "amount_refunded": amount_refunded,
            "refunded": refunded,
            "currency": CURRENCY,
            "payment_intent": payment_intent,
            "metadata": {"order_id": str(order_id)},
        },
        **kw,
    )


def test_a_refund_of_a_different_charge_does_not_refund_the_order(
    client, session, caplog
):
    """The double-charge case, which this project documented before guarding it.

    Two Charges can carry the same `order_id` — one from a superseded Checkout
    Session that was paid anyway. Refunding the duplicate is what a person
    reconciling does first, and its own `amount`/`amount_refunded` balance
    perfectly, so without an attribution check it reads as a full refund of
    the order: terminal status, whole reservation released, and the payment
    the order actually holds still charged.
    """
    caplog.set_level(logging.INFO, logger="shopagent.api.services.events")
    variant, order_id = paid_order(client, session, sku="RV-OTHER-CHARGE")
    before = stock(session, variant.id)

    other = refunded_event(order_id, payment_intent="pi_the_duplicate_charge")
    assert deliver(client, other).status_code == 200

    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PAID
    assert stock(session, variant.id) == before
    assert any(
        "not the payment this order is holding" in r.getMessage()
        and r.levelno == logging.ERROR
        for r in caplog.records
    )


def test_a_refund_with_no_payment_intent_on_the_charge_changes_nothing(
    client, session
):
    """Unverifiable attribution is not acted on, because `refunded` is terminal."""
    variant, order_id = paid_order(client, session, sku="RV-NO-PI-CHARGE")
    before = stock(session, variant.id)

    assert deliver(
        client, refunded_event(order_id, payment_intent=None)
    ).status_code == 200

    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PAID
    assert stock(session, variant.id) == before


def paid_order(client, session, *, sku: str, quantity: int = 2):
    """An order taken all the way to `paid` through the webhook, as a fixture."""
    variant = make_variant(session, sku=sku, quantity=30)
    order_id = make_order(client, variant, quantity=quantity)
    attach_session(session, order_id)
    deliver(client, completed_event(order_id))
    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PAID
    return variant, order_id


def test_a_full_refund_moves_the_order_and_releases_the_stock(client, session):
    """The deliverable of the step.

    `refunded` is in `RELEASES_RESERVATION`, which D7 built and this is the
    first caller to reach: the units were paid for and then not, so they go
    back on sale.
    """
    variant = make_variant(session, sku="RF-FULL", quantity=30, reserved=4)
    before = stock(session, variant.id)
    order_id = make_order(client, variant, quantity=3)
    attach_session(session, order_id)
    deliver(client, completed_event(order_id))
    session.expire_all()
    assert stock(session, variant.id)[1] == before[1] + 3

    response = deliver(client, refunded_event(order_id, amount=4500, amount_refunded=4500))

    assert response.status_code == 200
    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.REFUNDED
    # Back exactly where it started, not merely lower.
    assert stock(session, variant.id) == before


def test_a_partial_refund_changes_nothing_and_says_so_loudly(client, session, caplog):
    """The decision this handler exists to make.

    `charge.refunded` fires for a partial refund too — measured, not assumed:
    a $1 refund on a $190 charge produced `amount_refunded=100, refunded=False`
    under this same event type. Acting on it would drive the order to a
    terminal `refunded` and hand back the entire reservation for a fraction of
    the money.
    """
    caplog.set_level(logging.INFO, logger="shopagent.api.services.events")
    variant, order_id = paid_order(client, session, sku="RF-PARTIAL", quantity=3)
    session.expire_all()
    before = stock(session, variant.id)

    response = deliver(
        client, refunded_event(order_id, amount=4500, amount_refunded=100)
    )

    assert response.status_code == 200
    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PAID
    assert stock(session, variant.id) == before

    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "PARTIALLY" in errors[0]
    assert str(order_id) in errors[0]
    assert "100 of 4500" in errors[0]


def test_amounts_and_the_refunded_flag_disagreeing_changes_nothing(
    client, session, caplog
):
    """Neither half is trusted alone when they contradict each other.

    The arithmetic decides in the ordinary case, because `amount` and
    `amount_refunded` are numbers Stripe cannot quietly redefine while
    `refunded` is a flag that could be deprecated into absence. But a
    contradiction is not a case to resolve by picking a favourite — it is a
    shape this code does not understand, and `refunded` is terminal.
    """
    caplog.set_level(logging.INFO, logger="shopagent.api.services.events")
    variant, order_id = paid_order(client, session, sku="RF-DISAGREE")

    response = deliver(
        client,
        refunded_event(order_id, amount=3000, amount_refunded=3000, refunded=False),
    )

    assert response.status_code == 200
    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PAID
    assert any(
        "disagree" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.ERROR
    )


def test_a_charge_with_no_amounts_changes_nothing(client, session, caplog):
    """`getattr` returning None must not read as "fully refunded".

    `amount_refunded >= amount` on two `None`s would raise; defaulting either
    to zero would make an absent field mean "partial" or "full" by accident.
    Both are answered by refusing to decide.
    """
    caplog.set_level(logging.INFO, logger="shopagent.api.services.events")
    variant, order_id = paid_order(client, session, sku="RF-NOAMOUNT")

    body = refunded_event(order_id)
    del body["data"]["object"]["amount_refunded"]

    assert deliver(client, body).status_code == 200
    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PAID


def test_a_second_refund_event_under_a_new_id_is_a_no_op(client, session):
    """`processed_events` cannot help; the transition table has to.

    `refunded` is terminal, so the second event is refused before any stock
    moves — which is what stops a double release handing back units the order
    never held.
    """
    variant = make_variant(session, sku="RF-TWICE", quantity=30, reserved=2)
    before = stock(session, variant.id)
    order_id = make_order(client, variant, quantity=3)
    attach_session(session, order_id)
    deliver(client, completed_event(order_id))

    first = deliver(client, refunded_event(order_id, amount=4500, amount_refunded=4500,
                                           event_id="evt_refund_one"))
    second = deliver(client, refunded_event(order_id, amount=4500, amount_refunded=4500,
                                            event_id="evt_refund_two"))

    assert (first.status_code, second.status_code) == (200, 200)
    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.REFUNDED
    assert stock(session, variant.id) == before


def test_a_failure_while_handling_a_refund_leaves_the_order_and_the_claim(
    client, session, monkeypatch
):
    """The transaction invariant, on the path where stock moves."""
    variant, order_id = paid_order(client, session, sku="RF-ROLLBACK", quantity=3)
    session.expire_all()
    before = stock(session, variant.id)

    def explode(session_, order, target, event_, **kwargs):
        raise RuntimeError("the database went away mid-release")

    monkeypatch.setattr(event_service, "_move", explode)

    with pytest.raises(RuntimeError):
        deliver(client, refunded_event(order_id, amount=4500, amount_refunded=4500,
                                       event_id="evt_refund_exploded"))

    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PAID
    assert stock(session, variant.id) == before
    assert session.get(ProcessedEvent, "evt_refund_exploded") is None


# --- POST /orders/{id}/refund --------------------------------------------


class FakeRefund:
    id = "re_test_handler"
    status = "succeeded"
    amount = 4500
    currency = CURRENCY


def stripe_refunds(monkeypatch, calls: list | None = None):
    def create_refund(payment_intent_id, *, idempotency_key=None):
        if calls is not None:
            calls.append((payment_intent_id, idempotency_key))
        return FakeRefund()

    monkeypatch.setattr(stripe_svc, "create_refund", create_refund)


def test_refunding_a_paid_order_is_202_and_leaves_it_paid(
    client, session, monkeypatch
):
    """202, and the order has not moved. Both halves matter.

    A 200 would say the refund is done; what is done is that Stripe accepted
    it. The order stays `paid` until `charge.refunded` arrives, and a caller
    that polled after a 200 and saw `paid` would reasonably conclude the
    refund had failed.
    """
    stripe_refunds(monkeypatch)
    variant, order_id = paid_order(client, session, sku="RF-ROUTE", quantity=3)

    response = client.post(f"/orders/{order_id}/refund")

    assert response.status_code == 202
    body = response.json()
    assert body["refund_id"] == "re_test_handler"
    # Named in the payload precisely because it is the field that would
    # otherwise be assumed to have changed.
    assert body["order_status"] == "paid"

    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PAID


def test_the_refund_reaches_stripe_with_the_recorded_payment_intent(
    client, session, monkeypatch
):
    """The column step 3 started filling is what step 4 spends."""
    calls: list = []
    stripe_refunds(monkeypatch, calls)
    variant = make_variant(session, sku="RF-PI", quantity=30)
    order_id = make_order(client, variant, quantity=2)
    attach_session(session, order_id)
    deliver(client, completed_event(order_id, payment_intent="pi_recorded_here"))

    client.post(f"/orders/{order_id}/refund")

    assert len(calls) == 1
    assert calls[0][0] == "pi_recorded_here"


def test_the_idempotency_key_is_derived_from_the_order(client, session, monkeypatch):
    """The lock cannot serialise this, so the key has to.

    `refund_order` locks the order row, but the row does not change — the
    status stays `paid` until the webhook lands — so two requests seconds
    apart both read a refundable order and both reach Stripe. A key derived
    from the order makes the second return Stripe's record of the first
    instead of moving money twice; a random key would make every retry a fresh
    refund, which is the exact failure the key exists to prevent.
    """
    calls: list = []
    stripe_refunds(monkeypatch, calls)
    variant, order_id = paid_order(client, session, sku="RF-IDEM")

    client.post(f"/orders/{order_id}/refund")
    client.post(f"/orders/{order_id}/refund")

    assert len(calls) == 2
    assert calls[0][1] == calls[1][1] == f"shopagent-refund-v1-{order_id}"


def test_refunding_a_pending_order_is_409(client, session, monkeypatch):
    """Never charged, so there is nothing to send back."""
    stripe_refunds(monkeypatch)
    variant = make_variant(session, sku="RF-PENDING", quantity=30)
    order_id = make_order(client, variant, quantity=2)

    response = client.post(f"/orders/{order_id}/refund")

    assert response.status_code == 409


def test_refunding_an_already_refunded_order_is_409(client, session, monkeypatch):
    """`refunded` is terminal, so the lifecycle refuses the second attempt."""
    stripe_refunds(monkeypatch)
    variant, order_id = paid_order(client, session, sku="RF-AGAIN", quantity=3)
    deliver(client, refunded_event(order_id, amount=4500, amount_refunded=4500))
    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.REFUNDED

    assert client.post(f"/orders/{order_id}/refund").status_code == 409


def test_refunding_an_unknown_order_is_404(client, monkeypatch):
    stripe_refunds(monkeypatch)

    assert client.post(f"/orders/{uuid.uuid4()}/refund").status_code == 404


def test_a_paid_order_with_no_payment_intent_is_409_and_never_reaches_stripe(
    client, session, monkeypatch, caplog
):
    """A state the system is not supposed to be able to reach.

    An order becomes `paid` only through `checkout.session.completed`, which
    writes the PaymentIntent id in the same transaction as the status. So this
    means a defect here or an edited row — logged at ERROR because of that,
    and answered 409 because from the caller's side the order genuinely cannot
    be refunded and no retry changes it.

    The important half is the second assertion: Stripe must not be called with
    a null. `create_refund` is replaced by something that fails the test if it
    runs, rather than by a fake that would let a `None` through unnoticed.
    """
    caplog.set_level(logging.INFO, logger="shopagent.api.services.orders")

    def refuse(payment_intent_id, *, idempotency_key=None):
        raise AssertionError(
            f"Stripe was asked to refund payment_intent={payment_intent_id!r} "
            "for an order that has none recorded"
        )

    monkeypatch.setattr(stripe_svc, "create_refund", refuse)
    variant, order_id = paid_order(client, session, sku="RF-NOPI", quantity=2)
    order = session.get(Order, order_id)
    order.stripe_payment_intent_id = None
    session.commit()

    response = client.post(f"/orders/{order_id}/refund")

    assert response.status_code == 409
    assert any(
        "should be impossible" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.ERROR
    )


def test_a_zero_total_order_has_nothing_to_refund_and_is_not_a_defect(
    client, session, monkeypatch, caplog
):
    """The branch above called this impossible, and this file allows it.

    `total_amount_cents >= 0` and `prices.amount_cents >= 0` both permit zero,
    and Stripe settles a zero-total checkout with
    `payment_status="no_payment_required"` — which `SETTLED_PAYMENT_STATUSES`
    accepts on purpose. Such a session names no PaymentIntent, so the order is
    legitimately paid with nothing to refund. Answering 409 is right; calling
    it a defect and telling the caller to refund a payment that never existed
    is not. Raised in review on PR #8.
    """
    caplog.set_level(logging.INFO, logger="shopagent.api.services.orders")

    def refuse(payment_intent_id, *, idempotency_key=None):
        raise AssertionError("Stripe was asked to refund a zero-total order")

    monkeypatch.setattr(stripe_svc, "create_refund", refuse)
    variant, order_id = paid_order(client, session, sku="RF-ZERO", quantity=2)
    order = session.get(Order, order_id)
    order.stripe_payment_intent_id = None
    order.total_amount_cents = 0
    session.commit()

    response = client.post(f"/orders/{order_id}/refund")

    assert response.status_code == 409
    assert "nothing to refund" in response.json()["detail"]
    # No ERROR: nothing here is broken.
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)


def test_a_missing_stripe_key_makes_the_refund_route_503(client, session, monkeypatch):
    """The capability is absent; the server is not broken.

    The same answer the checkout route gives, for the same reason.
    """
    def no_key(payment_intent_id, *, idempotency_key=None):
        raise stripe_svc.MissingStripeKey("STRIPE_SECRET_KEY is not set")

    monkeypatch.setattr(stripe_svc, "create_refund", no_key)
    variant, order_id = paid_order(client, session, sku="RF-NOKEY")

    assert client.post(f"/orders/{order_id}/refund").status_code == 503


def test_the_refund_route_needs_the_api_key(api_client, session):
    """Mounted behind `require_api_key` by where it lives, not by a decorator.

    The sweep in `tests/test_api_auth.py` picks this route up on its own; this
    says the same thing from the route's own file, which is where somebody
    reading the refund code would look.
    """
    assert api_client.post(f"/orders/{uuid.uuid4()}/refund").status_code == 401


# --- the shapes the refund handler rests on, asked of Stripe --------------


@pytest.mark.stripe
def test_a_real_charge_refunded_event_distinguishes_partial_from_full():
    """Every offline test above decides what a refund looks like. This asks.

    The whole handler turns on one claim: `charge.refunded` fires for a
    partial refund as well as a full one, and only the amounts tell them
    apart. That was established by issuing two real refunds against one charge
    and reading the events they produced —

        partial   amount=18998  amount_refunded=100    refunded=False
        full      amount=18998  amount_refunded=18998  refunded=True

    — and this keeps it true. If Stripe ever stopped sending the event for
    partial refunds, the handler's careful branch would become dead code and
    nobody would notice; if it stopped sending `amount_refunded`, the handler
    would refuse every refund and orders would silently stay `paid`.

    Skips on an account with no refunds, which says nothing about the shape.
    """
    events = list(
        stripe_svc.get_client().v1.events.list(
            params={"type": "charge.refunded", "limit": 10}
        )
    )
    if not events:
        pytest.skip("this Stripe account has no charge.refunded events")

    for delivered in events:
        charge = delivered.data.object
        assert charge.object == "charge"
        assert isinstance(charge.amount, int)
        assert isinstance(charge.amount_refunded, int)
        # The invariant the handler cross-checks against: Stripe's own flag
        # agrees with the arithmetic. A disagreement here would mean the
        # handler's ERROR branch is reachable in normal operation.
        assert bool(charge.refunded) == (charge.amount_refunded >= charge.amount), (
            f"{delivered.id}: refunded={charge.refunded} but "
            f"amount_refunded={charge.amount_refunded} of amount={charge.amount}"
        )

    partial = [e for e in events if not e.data.object.refunded]
    if not partial:
        pytest.skip("no partial refund in the recent events to check against")

    # The claim the handler is built around, stated as an assertion: this event
    # type really does arrive for refunds that are not complete.
    assert partial[0].type == "charge.refunded"
    assert partial[0].data.object.amount_refunded < partial[0].data.object.amount


@pytest.mark.stripe
def test_a_real_charge_carries_the_order_id_in_its_metadata():
    """`order_id_from` reads a Charge the same way it reads a Session.

    True only because `payments/checkout.py` passes `payment_intent_data`
    explicitly — Stripe propagates nothing on its own. A refund handler is the
    first thing that depends on it in production, since `charge.refunded`
    carries a Charge and nothing else.
    """
    events = list(
        stripe_svc.get_client().v1.events.list(
            params={"type": "charge.refunded", "limit": 5}
        )
    )
    if not events:
        pytest.skip("this Stripe account has no charge.refunded events")

    with_order = [
        e for e in events
        if (e.data.object.metadata._data.get("order_id") if e.data.object.metadata else None)
    ]
    assert with_order, (
        "no recent charge.refunded event carries metadata.order_id — the "
        "payment_intent_data copy in payments/checkout.py has stopped working "
        "and no refund can be attributed to an order"
    )


# --- what review on PR #8 found ------------------------------------------


def test_a_live_mode_event_is_recorded_but_never_acted_on(client, session, caplog):
    """`config.py` refuses a live API key; it does not police this path.

    `STRIPE_WEBHOOK_SECRET` is a separate credential and a live endpoint's
    signing secret begins `whsec_` exactly like a test one, so it verifies just
    as well. Without this guard a live secret pasted into `.env` would let real
    customer events move this database's orders and release its stock.

    Recorded and answered 200: the delivery is genuine and no retry improves
    it — the configuration is what has to change.
    """
    caplog.set_level(logging.INFO, logger="shopagent.api.services.events")
    variant, order_id = paid_order(client, session, sku="RV-LIVE", quantity=2)
    session.expire_all()
    before = stock(session, variant.id)

    body = refunded_event(order_id, amount=3000, amount_refunded=3000,
                          event_id="evt_live_mode")
    body["livemode"] = True

    response = deliver(client, body)

    assert response.status_code == 200
    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PAID
    assert stock(session, variant.id) == before
    # Recorded, so a redelivery is not processed either.
    assert session.get(ProcessedEvent, "evt_live_mode") is not None
    assert any(
        "LIVE-MODE" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.ERROR
    )


def test_an_unsettled_checkout_does_not_mark_the_order_paid(client, session, caplog):
    """`completed` does not mean paid for delayed-notification methods.

    Stripe sends this event with `payment_status="unpaid"` for those, and
    settles later through `async_payment_succeeded`. The first version of the
    handler ignored the field, which would mark an order paid against a payment
    that had not happened — found in review on PR #8.
    """
    caplog.set_level(logging.INFO, logger="shopagent.api.services.events")
    variant = make_variant(session, sku="RV-UNPAID", quantity=30)
    order_id = make_order(client, variant, quantity=2)
    attach_session(session, order_id)

    body = completed_event(order_id)
    body["data"]["object"]["payment_status"] = "unpaid"
    response = deliver(client, body)

    assert response.status_code == 200
    session.expire_all()
    order = session.get(Order, order_id)
    assert order.status == OrderStatus.PENDING
    assert order.stripe_payment_intent_id is None
    assert any("has not settled" in r.getMessage() for r in caplog.records)


def test_a_no_payment_required_checkout_still_marks_the_order_paid(client, session):
    """The allow-list's second member, so it is not just `== "paid"`.

    A zero-amount checkout completes with `no_payment_required`, and that
    order is as paid as it will ever be.
    """
    variant = make_variant(session, sku="RV-FREE", quantity=30)
    order_id = make_order(client, variant, quantity=1)
    attach_session(session, order_id)

    body = completed_event(order_id)
    body["data"]["object"]["payment_status"] = "no_payment_required"

    assert deliver(client, body).status_code == 200
    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PAID


def test_a_delayed_payment_that_settles_later_marks_the_order_paid(client, session):
    """Without this the guard above would strand such orders at `pending`.

    `checkout.session.completed` has already been and gone by then, so nothing
    else would ever move the order.
    """
    variant = make_variant(session, sku="RV-ASYNC-OK", quantity=30)
    order_id = make_order(client, variant, quantity=2)
    attach_session(session, order_id)

    unsettled = completed_event(order_id)
    unsettled["data"]["object"]["payment_status"] = "unpaid"
    deliver(client, unsettled)

    settled = event(
        "checkout.session.async_payment_succeeded",
        {
            "id": "cs_test_handler",
            "object": "checkout.session",
            "payment_status": "paid",
            "payment_intent": "pi_settled_late",
            "metadata": {"order_id": str(order_id)},
        },
    )
    assert deliver(client, settled).status_code == 200

    session.expire_all()
    order = session.get(Order, order_id)
    assert order.status == OrderStatus.PAID
    assert order.stripe_payment_intent_id == "pi_settled_late"


def async_failed_event(order_id, *, session_id="cs_test_handler"):
    return event(
        "checkout.session.async_payment_failed",
        {
            "id": session_id,
            "object": "checkout.session",
            "payment_status": "unpaid",
            "metadata": {"order_id": str(order_id)},
        },
    )


def test_a_delayed_payment_that_fails_cancels_the_order(client, session, caplog):
    """The second review round: `pending` here was a permanent leak.

    This event only follows `checkout.session.completed`, so the session is
    `complete` — it will never expire, and `payments/checkout.py` refuses to
    start a new checkout while the order holds one. Leaving the order pending
    therefore left it and its reserved units stuck with no way out at all.
    """
    caplog.set_level(logging.INFO, logger="shopagent.api.services.events")
    variant = make_variant(session, sku="RV-ASYNC-FAIL", quantity=30)
    order_id = make_order(client, variant, quantity=2)
    attach_session(session, order_id)
    quantity, reserved_before = stock(session, variant.id)

    assert deliver(client, async_failed_event(order_id)).status_code == 200

    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.CANCELLED
    # The units go back on sale. `quantity` never moves — nothing shipped.
    assert stock(session, variant.id) == (quantity, reserved_before - 2)
    assert any("delayed payment" in r.getMessage() for r in caplog.records)


def test_a_delayed_failure_on_an_old_session_cancels_nothing(client, session, caplog):
    """The same guard the expiry handler has, and for the same reason.

    An order that has moved on to a second checkout must not be cancelled by a
    failure on the first — the shopper is looking at the newer payment page.
    """
    caplog.set_level(logging.INFO, logger="shopagent.api.services.events")
    variant = make_variant(session, sku="RV-ASYNC-FAIL-OLD", quantity=30)
    order_id = make_order(client, variant, quantity=2)
    attach_session(session, order_id, "cs_test_current")
    before = stock(session, variant.id)

    failed = async_failed_event(order_id, session_id="cs_test_superseded")
    assert deliver(client, failed).status_code == 200

    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PENDING
    assert stock(session, variant.id) == before
    assert any("not cancelling" in r.getMessage() for r in caplog.records)


def test_a_delayed_failure_cannot_cancel_an_order_that_was_paid(client, session):
    """The backstop the handler relies on instead of asking Stripe again.

    If the money did arrive, the order is `paid`, and `paid -> cancelled` is
    not in the transition table — so the refusal happens before any stock
    moves rather than being a check this handler has to remember to make.
    """
    variant = make_variant(session, sku="RV-ASYNC-FAIL-PAID", quantity=30)
    order_id = make_order(client, variant, quantity=2)
    attach_session(session, order_id)
    assert deliver(client, completed_event(order_id)).status_code == 200
    session.expire_all()
    paid_stock = stock(session, variant.id)

    assert deliver(client, async_failed_event(order_id)).status_code == 200

    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PAID
    assert stock(session, variant.id) == paid_stock


# --- the session that paid vs the session the order points at ------------
#
# Raised in the second review round on PR #8. An order whose first session
# expired with a payment in flight is left pending by the expiry guard, then
# gets a second session — and when the first one completes, the order is paid
# while the second is still open and chargeable.


def record_expiries(monkeypatch) -> list[str]:
    expired: list[str] = []
    monkeypatch.setattr(
        stripe_svc,
        "expire_checkout_session",
        lambda session_id: expired.append(session_id),
    )
    return expired


def test_a_payment_through_a_superseded_session_closes_the_newer_one(
    client, session, monkeypatch, caplog
):
    caplog.set_level(logging.INFO, logger="shopagent.api.services.events")
    expired = record_expiries(monkeypatch)
    variant = make_variant(session, sku="RV-SUPERSEDED", quantity=30)
    order_id = make_order(client, variant, quantity=2)
    attach_session(session, order_id, "cs_test_second")

    paid_through = completed_event(
        order_id, session_id="cs_test_first", payment_intent="pi_from_the_first"
    )
    assert deliver(client, paid_through).status_code == 200

    # The open one is closed, so the same order cannot be paid a second time.
    assert expired == ["cs_test_second"]

    session.expire_all()
    order = session.get(Order, order_id)
    assert order.status == OrderStatus.PAID
    assert order.stripe_payment_intent_id == "pi_from_the_first"
    # Repointed at the session the money actually came through, so the refund
    # endpoint and the dashboard agree about which payment this order is.
    assert order.stripe_checkout_session_id == "cs_test_first"
    # Closed cleanly, so nothing asks for a person.
    assert not any("refund by hand" in r.getMessage() for r in caplog.records)
    assert any(
        "a second checkout was open" in r.getMessage()
        and r.levelno == logging.ERROR
        for r in caplog.records
    )


def test_the_ordinary_payment_closes_nothing(client, session, monkeypatch):
    """The path every real payment takes must not make a Stripe call."""
    expired = record_expiries(monkeypatch)
    variant = make_variant(session, sku="RV-NO-EXPIRY", quantity=30)
    order_id = make_order(client, variant, quantity=2)
    attach_session(session, order_id, "cs_test_only")

    assert deliver(
        client, completed_event(order_id, session_id="cs_test_only")
    ).status_code == 200

    assert expired == []
    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PAID


def test_a_payment_that_cannot_be_accepted_closes_nothing(
    client, session, monkeypatch
):
    """`_may_become` before the reconciliation, not after.

    An order that cannot become paid must not cause somebody else's Checkout
    Session to be expired on the way to being told so.
    """
    stripe_says(monkeypatch, "unpaid", status="open")
    expired = record_expiries(monkeypatch)
    variant = make_variant(session, sku="RV-NO-EXPIRY-REFUSED", quantity=30)
    order_id = make_order(client, variant, quantity=2)
    attach_session(session, order_id, "cs_test_current")
    assert client.post(f"/orders/{order_id}/cancel").status_code == 200
    expired.clear()

    late = completed_event(order_id, session_id="cs_test_first")
    assert deliver(client, late).status_code == 200

    assert expired == []
    session.expire_all()
    order = session.get(Order, order_id)
    assert order.status == OrderStatus.CANCELLED
    assert order.stripe_checkout_session_id == "cs_test_current"


def test_a_session_that_will_not_close_is_reported_and_the_money_still_lands(
    client, session, monkeypatch, caplog
):
    """Stripe refuses to expire a session that is not open.

    Already expired is harmless; already complete means this order has been
    paid twice and no code here can undo that. Either way the payment that did
    arrive is real, so the order becomes paid and the log carries the warning.
    """
    caplog.set_level(logging.INFO, logger="shopagent.api.services.events")

    def refuses(session_id):
        raise stripe_svc.InvalidRequestError(
            "You may only expire a session that is in the open state", "session"
        )

    monkeypatch.setattr(stripe_svc, "expire_checkout_session", refuses)

    variant = make_variant(session, sku="RV-CLOSE-FAILS", quantity=30)
    order_id = make_order(client, variant, quantity=2)
    attach_session(session, order_id, "cs_test_second")

    assert deliver(
        client, completed_event(order_id, session_id="cs_test_first")
    ).status_code == 200

    session.expire_all()
    assert session.get(Order, order_id).status == OrderStatus.PAID
    assert any(
        "refund by hand" in r.getMessage() and r.levelno == logging.ERROR
        for r in caplog.records
    )


def test_a_transport_failure_while_closing_the_other_session_is_a_500(
    client, session, monkeypatch
):
    """The narrow catch, checked from the other side.

    `InvalidRequestError` is permanent and swallowed; anything else is not, and
    has to become a 500 so Stripe redelivers rather than the payment being
    recorded against a session nobody closed.
    """
    def unreachable(session_id):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(stripe_svc, "expire_checkout_session", unreachable)

    variant = make_variant(session, sku="RV-CLOSE-DOWN", quantity=30)
    order_id = make_order(client, variant, quantity=2)
    attach_session(session, order_id, "cs_test_second")

    with pytest.raises(RuntimeError):
        deliver(client, completed_event(order_id, session_id="cs_test_first"))


def test_a_transition_refused_under_the_lock_writes_no_column_either(
    client, session, monkeypatch
):
    """The same bug one layer down, found by the second review round.

    Moving the assignment into `_move` fixed the ordinary refusal and left the
    raced one: the unlocked preflight passes, the attribute is set, and
    `apply_transition`'s `SELECT ... FOR UPDATE` autoflushes it before the
    authoritative check runs. `session.expire()` afterwards drops the attribute
    but not the UPDATE, so the router's commit persisted it against an order
    whose status never changed.

    The race is simulated rather than threaded: neutering the preflight is
    exactly the state a caller is in when a concurrent delivery moved the order
    between the two checks. `updates` now travel to `apply_transition` and are
    assigned only after the locked check, so nothing is dirty when the flush
    happens.
    """
    variant = make_variant(session, sku="RV-AUTOFLUSH", quantity=30)
    order_id = make_order(client, variant, quantity=2)
    order = attach_session(session, order_id)

    order.status = OrderStatus.CANCELLED.value
    session.flush()
    monkeypatch.setattr(event_service, "check_transition", lambda a, b: None)

    moved = event_service._move(
        session,
        order,
        OrderStatus.PAID,
        SimpleNamespace(id="evt_raced"),
        updates={"stripe_payment_intent_id": "pi_MUST_NOT_BE_WRITTEN"},
    )
    assert moved is False

    session.commit()
    stored = session.execute(
        text(
            "SELECT stripe_payment_intent_id, status FROM orders WHERE id = :i"
        ),
        {"i": order_id},
    ).one()
    assert stored.status == OrderStatus.CANCELLED.value
    assert stored.stripe_payment_intent_id is None


def test_a_refused_transition_never_writes_the_payment_intent(client, session):
    """The bug review found: the column was assigned before the check.

    A second `checkout.session.completed` for an order already `paid` would
    overwrite the PaymentIntent the refund endpoint spends — with the one from
    a session that may never have been charged. The write now travels through
    `_move` and happens only after the transition is cleared.
    """
    variant = make_variant(session, sku="RV-PI-GUARD", quantity=30)
    order_id = make_order(client, variant, quantity=2)
    attach_session(session, order_id)

    deliver(client, completed_event(order_id, payment_intent="pi_the_real_one"))
    session.expire_all()
    assert session.get(Order, order_id).stripe_payment_intent_id == "pi_the_real_one"

    # A second completed event, under a fresh id so `processed_events` lets it
    # through, naming a different PaymentIntent.
    deliver(
        client,
        completed_event(
            order_id, payment_intent="pi_a_later_session", event_id="evt_second_completed"
        ),
    )

    session.expire_all()
    order = session.get(Order, order_id)
    assert order.status == OrderStatus.PAID
    assert order.stripe_payment_intent_id == "pi_the_real_one", (
        "a refused transition overwrote the PaymentIntent; the refund endpoint "
        "would now refund a session that was never charged"
    )


def test_a_cancelled_order_receiving_a_completed_event_keeps_no_payment_intent(
    client, session, monkeypatch
):
    """The same guard from the other direction, where nothing was set before."""
    stripe_says(monkeypatch, "unpaid", status="open")
    variant = make_variant(session, sku="RV-PI-CANCELLED", quantity=30)
    order_id = make_order(client, variant, quantity=2)
    attach_session(session, order_id)
    client.post(f"/orders/{order_id}/cancel")

    deliver(client, completed_event(order_id, payment_intent="pi_too_late"))

    session.expire_all()
    order = session.get(Order, order_id)
    assert order.status == OrderStatus.CANCELLED
    assert order.stripe_payment_intent_id is None


def test_a_second_checkout_started_during_the_stripe_call_stops_the_expiry(
    engine, monkeypatch
):
    """The expiry guard was an unlocked read deciding a write.

    `handle_checkout_expired` compares the event's session against the order's,
    then asks Stripe whether the session was really unpaid — a network call,
    which is long enough for a shopper to start a second checkout.
    `_reusable_session` sees an expired session, builds a new one and writes it
    to the order. Cancelling on the strength of the earlier read then releases
    the stock of an order whose new payment page is open and chargeable.
    Raised in review on PR #8.

    The race is made deterministic rather than threaded: the Stripe stub is
    where the concurrent write happens, from its own connection, which is
    exactly the window the real call opens. Committed rather than run in the
    `session` fixture's transaction, for the reason the test below gives.
    """
    from shopagent.api.services import events as ev
    from shopagent.api.services.cart import add_item, create_cart
    from shopagent.api.services.orders import place_order

    with Session(engine) as setup:
        variant_id, quantity, reserved_before = setup.execute(
            select(Inventory.variant_id, Inventory.quantity, Inventory.reserved)
            .where(Inventory.quantity - Inventory.reserved >= 3)
            .order_by(Inventory.variant_id)
            .limit(1)
        ).one()
        cart = create_cart(setup)
        add_item(setup, cart.id, variant_id, 3)
        order = place_order(setup, cart.id)
        order_id, cart_id = order.id, cart.id
        order.stripe_checkout_session_id = "cs_test_expired_one"
        setup.commit()

    def stripe_answers_while_a_second_checkout_starts(session_id):
        with Session(engine) as other:
            other.execute(
                text(
                    "UPDATE orders SET stripe_checkout_session_id = "
                    "'cs_test_the_new_one' WHERE id = :o"
                ),
                {"o": order_id},
            )
            other.commit()
        return FakeSession("unpaid", "expired")

    monkeypatch.setattr(
        stripe_svc,
        "retrieve_checkout_session",
        stripe_answers_while_a_second_checkout_starts,
    )

    try:
        with Session(engine) as handling:
            ev.handle_checkout_expired(
                handling,
                SimpleNamespace(
                    id="evt_expiry_race",
                    livemode=False,
                    data=SimpleNamespace(
                        object=SimpleNamespace(
                            id="cs_test_expired_one",
                            metadata=SimpleNamespace(
                                _data={"order_id": str(order_id)}
                            ),
                        )
                    ),
                ),
            )
            handling.commit()

        with Session(engine) as check:
            order = check.get(Order, order_id)
            assert order.status == OrderStatus.PENDING, (
                "the old session's expiry cancelled an order that had already "
                "moved to a new, payable checkout"
            )
            assert order.stripe_checkout_session_id == "cs_test_the_new_one"
            after = check.execute(
                select(Inventory.reserved).where(Inventory.variant_id == variant_id)
            ).scalar()
        # Still reserved: the shopper is on the new payment page.
        assert int(after) == int(reserved_before) + 3
    finally:
        with Session(engine) as cleanup:
            cleanup.execute(
                text("DELETE FROM order_items WHERE order_id = :o"), {"o": order_id}
            )
            cleanup.execute(text("DELETE FROM orders WHERE id = :o"), {"o": order_id})
            cleanup.execute(
                text("DELETE FROM cart_items WHERE cart_id = :c"), {"c": cart_id}
            )
            cleanup.execute(text("DELETE FROM carts WHERE id = :c"), {"c": cart_id})
            cleanup.execute(
                text("UPDATE inventory SET reserved = :r WHERE variant_id = :v"),
                {"r": int(reserved_before), "v": int(variant_id)},
            )
            cleanup.commit()


def test_two_concurrent_transitions_release_the_reservation_once(engine):
    """The third idempotency layer, with two real sessions instead of a claim.

    Review on PR #8 argued this was broken: `_load_order` puts the `Order` in
    the identity map, and an ORM `SELECT ... FOR UPDATE` was said to return
    that instance without refreshing, so a request waiting behind another
    transition would evaluate a stale status and release the reservation twice.

    Measured against SQLAlchemy 2.0.52 that is not what happens — the second
    session reads the settled status and the lifecycle refuses it. But the
    layer had only ever been asserted by reading SQL for `FOR UPDATE`, which
    would pass just as happily if the behaviour changed, so the claim is now
    made the only way it can be: two connections, one order, both told to move
    it, exactly one succeeding. `apply_transition` also asks for
    `populate_existing` now, so this does not rest on a default staying put.

    Everything here is committed rather than run inside the `session` fixture's
    transaction, because two connections cannot share one — a variant created
    in that transaction is invisible to the other thread. It uses a seeded
    variant and puts `reserved` back by hand.
    """
    import threading

    from shopagent.api.lifecycle import IllegalTransition
    from shopagent.api.services.cart import add_item, create_cart
    from shopagent.api.services.orders import apply_transition, place_order

    with Session(engine) as setup:
        variant_id, quantity, reserved_before = setup.execute(
            select(Inventory.variant_id, Inventory.quantity, Inventory.reserved)
            .where(Inventory.quantity - Inventory.reserved >= 3)
            .order_by(Inventory.variant_id)
            .limit(1)
        ).one()
        cart = create_cart(setup)
        add_item(setup, cart.id, variant_id, 3)
        order = place_order(setup, cart.id)
        order_id, cart_id = order.id, cart.id
        apply_transition(setup, order, OrderStatus.PAID)

    outcomes: dict[str, str] = {}

    def move(name: str) -> None:
        with Session(engine) as own:
            loaded = own.get(Order, order_id)  # both cache `paid` first
            try:
                apply_transition(own, loaded, OrderStatus.REFUNDED)
                outcomes[name] = "applied"
            except IllegalTransition:
                outcomes[name] = "refused"
            except Exception as exc:  # pragma: no cover - diagnostic only
                outcomes[name] = f"{type(exc).__name__}: {exc}"

    threads = [threading.Thread(target=move, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    try:
        assert sorted(outcomes.values()) == ["applied", "refused"], outcomes

        with Session(engine) as check:
            assert check.get(Order, order_id).status == OrderStatus.REFUNDED
            after = check.execute(
                select(Inventory.quantity, Inventory.reserved).where(
                    Inventory.variant_id == variant_id
                )
            ).one()
        # Released once, not twice: back where it started rather than three
        # units below it.
        assert (int(after[0]), int(after[1])) == (int(quantity), int(reserved_before))
    finally:
        with Session(engine) as cleanup:
            cleanup.execute(text("DELETE FROM order_items WHERE order_id = :o"),
                            {"o": order_id})
            cleanup.execute(text("DELETE FROM orders WHERE id = :o"), {"o": order_id})
            cleanup.execute(text("DELETE FROM cart_items WHERE cart_id = :c"),
                            {"c": cart_id})
            cleanup.execute(text("DELETE FROM carts WHERE id = :c"), {"c": cart_id})
            cleanup.execute(
                text("UPDATE inventory SET reserved = :r WHERE variant_id = :v"),
                {"r": int(reserved_before), "v": int(variant_id)},
            )
            cleanup.commit()
