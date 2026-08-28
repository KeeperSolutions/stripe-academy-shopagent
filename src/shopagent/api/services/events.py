"""What a Stripe webhook delivery means, and remembering that it arrived (D8).

Step 2 built the memory; step 3 built the meaning. Both live here rather than
in the router because `api/services/` is where domain rules go — a router
parses a request, calls one function and maps an exception to a status code,
and "a completed checkout makes an order paid" is not a fact about HTTP.

No FastAPI, by the rule the rest of `api/services/` follows. Nothing here
raises an `HTTPException`, and nothing here decides a status code: the router
turns "already seen" into a 200, and a reconciliation pass that runs outside
any request would turn it into a log line.

**One event type moves an order and the rest do not**, which is the decision
worth reading before the code. `checkout.session.completed` is the one that
marks an order paid. `payment_intent.succeeded` describes the same money and
deliberately does nothing: a single payment produces both, they arrive in an
order nobody controls, and having two events race for one transition would
mean the second is refused by the transition table every time — a permanent
warning in the log describing normal operation. `completed` is the primary
because it says the *checkout* finished, not merely that a charge settled,
and it is the event that carries the session this project created.

**Errors split two ways, and the split is what the status code means to
Stripe.** A permanent failure — an event type nothing handles, an order id
that is not in the metadata, an order that does not exist, a transition the
lifecycle refuses — is logged and returns normally, so the delivery is
answered 200 and Stripe stops. None of those improve on a retry, and a 4xx or
5xx would have Stripe redeliver for three days against a failure that is
already final. A transient failure — the database is unreachable, Stripe times
out, anything unexpected — is allowed to propagate, becomes a 500, and gets
retried. That is the whole reason `handle_event` catches so little.

**The whole module exists because Stripe delivers at least once.** A delivery
that did not get a 2xx is retried with backoff for up to three days, and even
a successful one can arrive twice — that is documented behaviour, not a fault.
Handling `checkout.session.completed` twice would reserve stock twice and send
a second confirmation, so the second arrival has to do nothing at all.

**Insert first, then work.** The shape the plan describes — read the id, and
if it is absent do the work — is check-then-act, and two concurrent
redeliveries both read "absent" before either writes. Retries are the case
this code exists for, so the race is not hypothetical: it is the normal
operating condition. The `INSERT` goes first instead, and the primary key
arbitrates. Postgres decides, and it decides once.

The insert is not committed here. It joins whatever transaction the caller is
in, so the record and the work it guards commit together or not at all — see
`record_event`'s docstring for why that is not merely tidy.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from shopagent.api.lifecycle import IllegalTransition, OrderStatus, check_transition
from shopagent.api.models import Order, ProcessedEvent
from shopagent.api.services.orders import apply_transition
from shopagent.payments import stripe_svc

logger = logging.getLogger(__name__)


def record_event(session: Session, event: Any) -> bool:
    """Claim this event, or report that somebody already did.

    Returns `True` when the row was written and the caller should go on to do
    the work, `False` when this event has been handled before and the caller
    should stop.

    **Wrapped in a SAVEPOINT, and that is load-bearing rather than stylistic.**
    A unique violation aborts the *entire* Postgres transaction: verified
    against the local database, the next statement on that connection comes
    back `InFailedSqlTransaction — current transaction is aborted, commands
    ignored`, and SQLAlchemy raises `PendingRollbackError` for anything the
    session tries afterwards. A duplicate is the one outcome this function
    treats as ordinary and expects to continue past, so catching the error
    without unwinding to a savepoint would leave the session unusable — and
    the symptom would not be here. It would be the *next* thing the request
    touched, failing for a reason that has nothing to do with it.
    `begin_nested()` rolls back the savepoint alone; everything the session did
    before the insert survives, which is what makes returning `False` a state
    the caller can actually act on.

    **Nothing is committed here.** The caller's transaction is what makes this
    permanent, and that is the point: if the work the caller then does raises,
    the rollback takes this row with it. A row that outlived a failed handler
    would tell the next retry "already processed" about an order nothing was
    ever done to — a payment that silently never lands, which is worse than
    doing the work twice.

    `event` is read with `getattr` for the same reason `describe_event` is:
    anybody holding the signing secret can sign `{"id": "evt_x"}`, and
    `StripeObject` raises `AttributeError` for an absent key rather than
    returning `None`. The `id` is required — an event without one cannot be
    deduplicated at all and is refused as a `ValueError` rather than stored
    under a null.
    """
    event_id = getattr(event, "id", None)
    if not event_id:
        raise ValueError(
            "this event carries no id, so it cannot be recorded as processed "
            "and cannot be recognised if Stripe delivers it again"
        )

    record = ProcessedEvent(
        event_id=event_id,
        # Falls back rather than raising: an event with no `type` is useless to
        # a handler but perfectly recordable, and refusing it here would mean
        # the same delivery is retried forever against a table that could have
        # absorbed it.
        event_type=getattr(event, "type", None) or "unknown",
        # Defaulted to False rather than nullable. The column answers "was this
        # live data", and a null answers "nobody knows", which is not a state
        # worth being able to represent for a field Stripe always sends.
        livemode=bool(getattr(event, "livemode", False)),
    )

    try:
        with session.begin_nested():
            session.add(record)
        return True
    except IntegrityError:
        # The savepoint is already rolled back by the time this runs, so the
        # session is clean and the caller can carry on. Deliberately not
        # inspecting the constraint name: the only unique constraint on this
        # table is its primary key, and matching on driver error text is how a
        # check starts silently passing after a library upgrade.
        return False


def has_been_processed(session: Session, event_id: str) -> bool:
    """Whether this event id is already recorded.

    **Not what the webhook uses**, and it must not become that — a read
    followed by a write is the race `record_event` exists to avoid. This is for
    tests and for reading the table by hand, where there is no second writer
    and the question really is just "is it in there".
    """
    return session.get(ProcessedEvent, event_id) is not None


# --- attribution (D8 step 3) ---------------------------------------------


ORDER_ID_KEY = "order_id"


def order_id_from(event: Any) -> uuid.UUID | None:
    """The order this event is about, or `None` if it does not say.

    **One function for every event type**, which is possible only because D7
    made it possible. Stripe propagates nothing down the object chain: a
    session's `metadata` stays on the session, and the PaymentIntent and Charge
    it produced come back with `metadata: {}` unless the checkout explicitly
    passes `payment_intent_data={"metadata": ...}` as well. It does — verified
    against two real payments — so `order_id` is on the session, the
    PaymentIntent *and* the Charge, and reading it needs no knowledge of which
    kind of object arrived.

    That property is worth stating as a dependency rather than an observation:
    if the copy in `payments/checkout.py` were ever dropped, every handler
    driven by a `payment_intent.*` event would silently stop finding its order.

    `metadata._data.get(...)` rather than anything nicer, and this is not
    stylistic. `StripeObject` overrides `__getattr__`, so `.get` raises
    `AttributeError: get`; `dict(metadata)` raises `KeyError: 0`; and
    `metadata["order_id"]` raises `KeyError` for an object with no such key.
    `_data` is the mapping underneath, which `routers/checkout_pages.py`
    already reaches for. All four were tried against a real event before this
    line was written.

    Returns `None` for every way of not having an id — absent metadata, absent
    key, or a value that is not a UUID. Callers treat that as a permanent
    failure, because it is: no retry produces metadata the event never carried.
    """
    obj = getattr(getattr(event, "data", None), "object", None)
    metadata = getattr(obj, "metadata", None)
    raw = getattr(metadata, "_data", {}).get(ORDER_ID_KEY) if metadata else None

    if not raw:
        return None

    try:
        return uuid.UUID(str(raw))
    except ValueError:
        # A malformed id is not a database question — asking Postgres to look
        # up a non-UUID raises `DataError`, which would surface as a 500 and be
        # retried for three days against a string that will never parse.
        logger.warning(
            "stripe event %s carries metadata.order_id=%r, which is not a "
            "UUID; nothing can be attributed to it",
            getattr(event, "id", "<unknown>"),
            raw,
        )
        return None


def warn_on_account_mismatch(event: Any) -> None:
    """Say so when an event was produced by a Stripe account that is not ours.

    **This costs nothing on the events this project actually receives, and that
    is the whole design.** `account` is a Connect field: Stripe sets it only on
    events forwarded from a connected account, and an ordinary event does not
    carry the key at all — checked against five real events from this account,
    where `getattr(event, "account", None)` is `None` and `"account"` is absent
    from the payload entirely. So the early return below is the normal path,
    the account id is never fetched, and no webhook pays for a network call.

    When the field *is* present, `configured_account_id()` answers from a cache
    filled once per process, so the cost is one call across the lifetime of the
    server rather than one per delivery.

    **A warning, never a refusal.** A platform legitimately receives events from
    every account connected to it; refusing them would break Connect outright,
    and this project's own future use of it.

    A caveat worth stating plainly, because it limits what this can be trusted
    for: **it does not catch the setup mistake that prompted it.** A local
    `stripe listen` logged into one account while `STRIPE_SECRET_KEY` belongs
    to another produces no mismatch here — the forwarded events are ordinary
    ones, carrying no `account`, and they simply never mention the orders this
    server created. The check for that lives in `tests/test_stripe_svc.py` and
    compares the CLI's configured account against the key's, because the
    difference is a fact about two local configurations rather than about any
    event.
    """
    event_account = getattr(event, "account", None)
    if event_account is None:
        return

    try:
        ours = stripe_svc.configured_account_id()
    except Exception as exc:
        # Diagnostics must not break delivery. This function exists to make a
        # misconfiguration visible; failing the webhook because the check
        # itself could not run would turn an advisory line into an outage.
        logger.debug(
            "could not read this key's Stripe account to compare against "
            "event %s (%s)",
            getattr(event, "id", "<unknown>"),
            exc,
        )
        return

    if event_account != ours:
        logger.warning(
            "stripe event %s was produced by account %s, but this server's key "
            "belongs to %s. That is expected for Stripe Connect; if this "
            "project is not using Connect it means checkout sessions are being "
            "created on one account while the webhook is fed by another, and "
            "the orders this event mentions will not exist here",
            getattr(event, "id", "<unknown>"),
            event_account,
            ours,
        )


def _load_order(session: Session, event: Any) -> Order | None:
    """The order this event names, or `None` with a line saying why not."""
    event_id = getattr(event, "id", "<unknown>")
    order_id = order_id_from(event)

    if order_id is None:
        # Permanent. The metadata is fixed at creation and no redelivery of
        # this event will carry more than the first one did.
        logger.warning(
            "stripe event %s (%s) carries no usable metadata.%s, so it cannot "
            "be attributed to an order — accepted and ignored. Every event this "
            "server creates a checkout for carries one, so a stream of these "
            "usually means the deliveries are coming from a different Stripe "
            "account than STRIPE_SECRET_KEY belongs to, or from `stripe "
            "trigger` fixtures rather than from real checkouts",
            event_id,
            getattr(event, "type", None),
            ORDER_ID_KEY,
        )
        return None

    order = session.get(Order, order_id)
    if order is None:
        # Also permanent, and worth its own message: this one means the event
        # is well formed and points at an order this database does not have —
        # a different environment's Stripe account, or a database that was
        # rebuilt under a running `stripe listen`.
        logger.warning(
            "stripe event %s (%s) names order %s, which does not exist in this "
            "database — accepted and ignored",
            event_id,
            getattr(event, "type", None),
            order_id,
        )
        return None

    return order


# --- handlers ------------------------------------------------------------


def _move(
    session: Session,
    order: Order,
    target: OrderStatus,
    event: Any,
    *,
    updates: dict[str, Any] | None = None,
) -> bool:
    """Move an order, or explain in the log why it stayed put. Never raises.

    Returns whether the order moved.

    `updates` are columns to write **as part of the same move** — today only
    `stripe_payment_intent_id`, from `checkout.session.completed`. They are
    handed to `apply_transition`, which assigns them to the locked row once
    the transition is allowed, so they commit with the status or not at all.
    Nothing is assigned here, and that is the point rather than a detail: an
    assignment made in this function is dirty on the Session by the time
    `apply_transition` issues its `SELECT ... FOR UPDATE`, SQLAlchemy
    autoflushes it, and a refusal under the lock can no longer take it back.

    Two rounds of review on PR #8 landed on that. The first version had the
    caller assign the column and then call this function, on the reasoning
    that the check inside would refuse first — it does not, the assignment has
    already happened, and the router's `commit()` writes it to an order whose
    status never changed. The second moved the assignment here, in front of
    the preflight check, which fixed the ordinary refusal and left the raced
    one: the preflight passes, the autoflush fires, the locked check refuses.
    The damaging case is the same in both — a second
    `checkout.session.completed` for an order already `paid` overwriting the
    PaymentIntent the refund endpoint spends, with one from a session that may
    never have been charged.

    **Checked before anything is written, then applied under a lock.** The
    check is not the authority — `apply_transition` locks the row and consults
    the transition table again, which is what makes it safe against a
    concurrent mover. What the early check buys is that nothing is written at
    all on the ordinary refusal. The same shape `cancel_order` uses before it
    expires a Stripe session, and for the same reason.

    **An `IllegalTransition` is a 200, not a 500, and the log level says which
    kind it was.** Two very different things reach this branch:

      * `paid -> paid`, from a second event asking for a state the order is
        already in. Ordinary — a manual replay under a fresh `event.id` gets
        past `processed_events` and lands here, which is exactly the backstop
        the transition table is for. Logged at INFO.
      * `cancelled -> paid`, where the money is real and the order is
        terminally cancelled. Nothing here can fix that, but it is not routine
        and must not read like it. Logged at ERROR, with the amount, so it is
        visible to whoever reconciles.

    Both are 200 because both are permanent: the transition table is a fixed
    rule and no redelivery changes what it allows. A 500 would have Stripe
    retry for three days and produce the same refusal every time.
    """
    event_id = getattr(event, "id", "<unknown>")
    current = OrderStatus(order.status)

    try:
        check_transition(current, target)
    except IllegalTransition as exc:
        if current == target:
            logger.info(
                "stripe event %s: order %s is already %s — nothing to do",
                event_id,
                order.id,
                target.value,
            )
        else:
            logger.error(
                "stripe event %s: order %s is %s and cannot become %s (%s). "
                "The event is accepted so Stripe stops retrying, but this "
                "order and Stripe now disagree and a person has to look: %s",
                event_id,
                order.id,
                current.value,
                target.value,
                f"{order.total_amount_cents} {order.currency}",
                exc,
            )
        return False

    try:
        apply_transition(session, order, target, updates=updates)
    except IllegalTransition as exc:
        # Lost a race between the check above and the lock inside
        # `apply_transition`: another delivery moved this order in between.
        #
        # `session.expunge` rather than `session.rollback`, and that changed in
        # review. Rolling back discarded this event's `processed_events` claim
        # while the caller still returned normally and the router still
        # answered 200 — so the delivery was recorded nowhere and Stripe was
        # told to stop, which is precisely the pair of things that must never
        # both be true. The old log line even promised a retry that a 200
        # forbids.
        #
        # Nothing of this event's needs undoing — `updates` were never
        # assigned, because `apply_transition` applies them only after the
        # locked check it just failed. What expiring buys is that this
        # instance stops carrying the status this handler read before the
        # wait, so anything reading it afterwards sees the row as it now is.
        # The claim stays intact to commit, and nothing needs retrying: the
        # order is already where this event wanted to take it, or somewhere
        # the lifecycle considers further along.
        session.expire(order)
        logger.warning(
            "stripe event %s: order %s changed status while this event was "
            "being handled, so %s was refused under the lock (%s). Nothing "
            "further is needed — the order has already settled",
            event_id,
            order.id,
            target.value,
            exc,
        )
        return False

    logger.info(
        "stripe event %s: order %s is now %s", event_id, order.id, target.value
    )
    return True


# What `checkout.session.completed` may report and still mean "the money is
# there". An allow-list rather than `!= "unpaid"`, the same shape the expiry
# guard uses: `payment_status` has three values and a fourth would otherwise
# default to "paid", which is the wrong way for this to fail.
SETTLED_PAYMENT_STATUSES = frozenset({"paid", "no_payment_required"})


def _payment_intent_updates(event: Any, checkout_session: Any) -> dict[str, Any]:
    """`stripe_payment_intent_id`, if the session names one as a string.

    Checked against a real `checkout.session.completed` from this account
    rather than assumed: `payment_intent` is a **string**, not an expanded
    object. It is null on a session completed without a payment, which a
    zero-amount checkout produces, so this can legitimately return nothing.
    """
    payment_intent = getattr(checkout_session, "payment_intent", None)

    if isinstance(payment_intent, str) and payment_intent:
        return {"stripe_payment_intent_id": payment_intent}

    if payment_intent is not None:
        # Not a string means the object was expanded, which nothing here asks
        # for. Worth a line rather than a silent `str()`: the column is a
        # VARCHAR the refund endpoint spends, and a repr in it would fail on
        # a value that looks almost right.
        logger.warning(
            "stripe event %s: session.payment_intent is %s rather than a "
            "string; stripe_payment_intent_id was left unset",
            getattr(event, "id", "<unknown>"),
            type(payment_intent).__name__,
        )

    return {}


def _may_become(order: Order, target: OrderStatus) -> bool:
    """Whether the transition table would allow this move right now.

    Unlocked and therefore not authoritative — `apply_transition` asks again
    under the lock, and that answer is the one that counts. What this buys is
    the right to do work *before* calling `_move` without doing it pointlessly:
    `_reconcile_paying_session` below expires a Stripe session, which must not
    happen for an order the move is going to be refused for anyway.
    """
    try:
        check_transition(OrderStatus(order.status), target)
    except IllegalTransition:
        return False
    return True


def _reconcile_paying_session(
    order: Order, event: Any, checkout_session: Any
) -> dict[str, Any]:
    """Make sure the session that paid is the only one that still can.

    Normally the event's session is the one the order points at and this does
    nothing. The case it exists for is narrow and expensive: an order whose
    first session expired with a payment already in flight. The expiry handler
    correctly refuses to cancel such an order — Stripe says the payment is not
    `unpaid` — so the order stays pending, `_reusable_session` sees an expired
    session and creates a second one, and the order now points at a session
    nobody has paid while the first one is still going to complete.

    When that first session completes, the order is genuinely paid and must be
    marked so. But the newer session is open and chargeable, and a shopper
    looking at that payment page would pay for the same order twice. So it is
    expired here, before the transition, and the order is repointed at the
    session the money actually came through — which is also what makes the
    refund endpoint and the dashboard agree about which payment this order is.
    Raised in review on PR #8.

    Returns the columns this implies, for `_move` to apply with the status.
    """
    event_id = getattr(event, "id", "<unknown>")
    paying_id = getattr(checkout_session, "id", None)
    current_id = order.stripe_checkout_session_id

    if not isinstance(paying_id, str) or paying_id == current_id:
        return {}

    if current_id is None:
        # Nothing to close. Worth a line because every order this project
        # creates has a session id by the time it can be paid, so an order
        # without one has been reached some other way.
        logger.warning(
            "stripe event %s: order %s was paid through session %s but had no "
            "session recorded; recording this one",
            event_id,
            order.id,
            paying_id,
        )
        return {"stripe_checkout_session_id": paying_id}

    logger.error(
        "stripe event %s: order %s was paid through session %s but points at "
        "%s, so a second checkout was open when this payment landed. Closing "
        "the other one so the same order cannot be paid twice",
        event_id,
        order.id,
        paying_id,
        current_id,
    )

    try:
        stripe_svc.expire_checkout_session(current_id)
    except stripe_svc.InvalidRequestError as exc:
        # Stripe refuses to expire anything that is not open, so this means the
        # other session is already complete or already expired. Expired is
        # harmless. Complete is the one case this whole function was trying to
        # prevent and did not reach in time, and no code here can undo a second
        # charge — so it is stated plainly, at ERROR, and a person refunds it.
        # Caught narrowly on purpose: a connection failure is not this, and it
        # must keep propagating into a 500 so Stripe retries the delivery.
        logger.error(
            "stripe event %s: session %s for order %s could not be closed "
            "(%s). If it was completed rather than expired, this order has "
            "been paid twice and needs a refund by hand",
            event_id,
            current_id,
            order.id,
            exc,
        )

    return {"stripe_checkout_session_id": paying_id}


def _pay(session: Session, order: Order, event: Any, checkout_session: Any) -> None:
    """Mark an order paid. Shared by the two events that can mean that.

    `checkout.session.completed` and `checkout.session.async_payment_succeeded`
    differ in what they have to check before they get here and in nothing after
    it, so the reconciliation and the move live in one place.
    """
    if not _may_become(order, OrderStatus.PAID):
        # Asked before reconciling, not after. `_move` is still what reports
        # the refusal — with the amount when the case is serious — but this
        # way an order that cannot become paid never causes somebody else's
        # Checkout Session to be expired on the way to being told so.
        _move(session, order, OrderStatus.PAID, event)
        return

    updates = _payment_intent_updates(event, checkout_session)
    updates |= _reconcile_paying_session(order, event, checkout_session)

    _move(session, order, OrderStatus.PAID, event, updates=updates)


def handle_checkout_completed(session: Session, event: Any) -> None:
    """A shopper finished checkout. Mark the order paid — if the money is there.

    **`completed` does not mean paid.** For delayed-notification payment
    methods this event arrives with `payment_status="unpaid"` and settles later
    through `checkout.session.async_payment_succeeded`, or fails through
    `checkout.session.async_payment_failed`. The first version of this handler
    ignored the field and moved the order regardless, which would mark an order
    paid against a payment that had not happened and might never — raised in
    review on PR #8.

    Nothing in `payments/checkout.py` restricts `payment_method_types`, so
    which methods are offered is a dashboard setting this code does not
    control and should not assume. Reading the field costs nothing and removes
    the assumption entirely.

    Also the only place `orders.stripe_payment_intent_id` is filled. The column
    has existed since D6 and nothing wrote to it; the refund endpoint needs a
    PaymentIntent to refund against, and the session is where it first appears.
    It is passed through `_move` rather than assigned here, so it cannot be
    written to an order whose transition was then refused.

    The move itself is `_pay`, which also reconciles the session that paid
    against the one the order points at — see `_reconcile_paying_session`.
    """
    order = _load_order(session, event)
    if order is None:
        return

    event_id = getattr(event, "id", "<unknown>")
    checkout_session = getattr(getattr(event, "data", None), "object", None)
    payment_status = getattr(checkout_session, "payment_status", None)

    if payment_status not in SETTLED_PAYMENT_STATUSES:
        # Not a failure and not permanent: the shopper completed a checkout
        # whose payment settles asynchronously. The order stays `pending` and
        # `async_payment_succeeded` moves it when the money actually arrives.
        logger.info(
            "stripe event %s: checkout completed for order %s with "
            "payment_status=%r, so the payment has not settled. The order "
            "stays %s until checkout.session.async_payment_succeeded",
            event_id,
            order.id,
            payment_status,
            order.status,
        )
        return

    _pay(session, order, event, checkout_session)


def handle_async_payment_succeeded(session: Session, event: Any) -> None:
    """A delayed payment finally settled. This is the other way to become paid.

    The pair to the `payment_status` guard above: without this handler an order
    paid by a delayed-notification method would sit at `pending` for ever,
    because `checkout.session.completed` has already been and gone and nothing
    else would ever move it.

    Stripe sends this only once the funds are confirmed, so unlike `completed`
    there is no payment state left to check — but the PaymentIntent is written
    here too, since `completed` deliberately wrote nothing for this order, and
    the session reconciliation applies just the same.
    """
    order = _load_order(session, event)
    if order is None:
        return

    checkout_session = getattr(getattr(event, "data", None), "object", None)

    _pay(session, order, event, checkout_session)


def handle_async_payment_failed(session: Session, event: Any) -> None:
    """A delayed payment did not settle. Cancel the order and release the stock.

    **Not the same answer as a declined card**, and the difference is the state
    the Checkout Session is left in. `payment_intent.payment_failed` happens on
    a session that is still `open`: the shopper can try another card, and if
    they never do, the session expires and `checkout.session.expired` cancels
    the order. Neither is true here. This event only arrives *after*
    `checkout.session.completed`, so the session is `complete` — and a complete
    session never expires, so `checkout.session.expired` will never come, while
    `_reusable_session` in `payments/checkout.py` refuses to start a new
    checkout for an order holding a complete one. The first version of this
    handler logged and left the order `pending`, which meant no retry path and
    no release path: the order and its reserved units were stuck for ever.
    Raised in review on PR #8.

    So the explicit policy is that this payment is over. `cancelled` releases
    the reservation, and the shopper who wants the goods places a new order.

    Two things make that safe to do without asking Stripe anything, which is
    the round trip `handle_checkout_expired` does make. The event type is
    itself Stripe's verdict on the payment — unlike `expired`, which says
    nothing about money and is why that handler cannot trust it. And if the
    money did arrive, the order is already `paid` and `paid -> cancelled` is
    not in the transition table, so `_move` refuses before any stock moves.

    The session guard is the same one the expiry handler uses, for the same
    reason: a failure on a session the order has already moved on from must not
    cancel the checkout the shopper is currently looking at.
    """
    order = _load_order(session, event)
    if order is None:
        return

    event_id = getattr(event, "id", "<unknown>")
    failed_session = getattr(getattr(event, "data", None), "object", None)
    failed_session_id = getattr(failed_session, "id", None)

    if order.stripe_checkout_session_id != failed_session_id:
        logger.info(
            "stripe event %s: the delayed payment on session %s failed, but "
            "order %s has since moved to session %s — not cancelling",
            event_id,
            failed_session_id,
            order.id,
            order.stripe_checkout_session_id,
        )
        return

    logger.warning(
        "stripe event %s: the delayed payment for order %s failed on session "
        "%s, which is complete and can neither be paid again nor expire. "
        "Cancelling the order so its stock goes back on sale",
        event_id,
        order.id,
        failed_session_id,
    )

    _move(session, order, OrderStatus.CANCELLED, event)


def handle_checkout_expired(session: Session, event: Any) -> None:
    """A checkout session timed out. Cancel the order — but only if it is safe.

    **This is the only path in the project where stock is released without a
    person deciding to**, which is why it is the most defensive function here.
    `cancelled` is terminal: get this wrong and an order that was actually paid
    is unrecoverable, with the money real and the reservation handed back.

    Two guards, and neither is optional.

    The first is that the event is not trusted about payment. Stripe can expire
    a session whose payment is in flight, and delivery order is not guaranteed
    — an `expired` arriving before its `completed` is exactly the shape of
    D7's `cancel_order` bug that review caught. So the session is fetched from
    Stripe and `payment_status` read from the answer. Anything but `unpaid`
    means stop and wait for `completed`.

    The second is that the event's session must be the one the order is
    currently pointing at. An order whose first session expired and which then
    started a new checkout would otherwise be cancelled by the *old* session's
    expiry, while the shopper is on the new payment page.

    The network call runs inside the transaction holding this event's
    `processed_events` row, which is a deliberate exception to the rule
    `cancel_order` follows about not holding a transaction across a round trip.
    The row it holds is a lock on one primary key that only a concurrent
    redelivery of this same event would contend for — and making that
    redelivery wait is the correct behaviour, not a cost. No order or inventory
    row is locked until `apply_transition`, after the call has returned.
    """
    order = _load_order(session, event)
    if order is None:
        return

    event_id = getattr(event, "id", "<unknown>")
    expired_session = getattr(getattr(event, "data", None), "object", None)
    expired_session_id = getattr(expired_session, "id", None)

    if order.stripe_checkout_session_id != expired_session_id:
        logger.info(
            "stripe event %s: session %s expired, but order %s has since moved "
            "to session %s — not cancelling",
            event_id,
            expired_session_id,
            order.id,
            order.stripe_checkout_session_id,
        )
        return

    # Asked of Stripe rather than read off the event. A transport failure here
    # propagates on purpose: it becomes a 500, Stripe retries, and the order
    # stays as it is. Guessing "probably unpaid" and cancelling would be the
    # one irreversible way to be wrong.
    live_session = stripe_svc.retrieve_checkout_session(expired_session_id)
    payment_status = getattr(live_session, "payment_status", None)

    if payment_status != "unpaid":
        logger.warning(
            "stripe event %s: session %s is reported expired but Stripe says "
            "payment_status=%r, so order %s is NOT being cancelled — waiting "
            "for checkout.session.completed",
            event_id,
            expired_session_id,
            payment_status,
            order.id,
        )
        return

    _move(session, order, OrderStatus.CANCELLED, event)


def handle_payment_failed(session: Session, event: Any) -> None:
    """A payment attempt failed. Recorded, and nothing else.

    The order stays `pending` on purpose. A failed attempt is not the end of a
    checkout — the shopper is usually still on the page and can try another
    card, and the session stays open until it expires. Cancelling here would
    release the stock out from under somebody in the middle of paying, and
    `cancelled` is terminal, so they could not simply try again.

    Expiry is what ends an unpaid order, and `checkout.session.expired` is the
    event for it.
    """
    order = _load_order(session, event)
    if order is None:
        return

    intent = getattr(getattr(event, "data", None), "object", None)
    error = getattr(intent, "last_payment_error", None)

    logger.warning(
        "stripe event %s: payment failed for order %s (%s); the order stays "
        "%s and the shopper may try again until the session expires",
        getattr(event, "id", "<unknown>"),
        order.id,
        # The decline code, not the message: the message is written for a
        # shopper and can name their bank, while the code is what a log is
        # for. Absent on failures that never reached an issuer.
        getattr(error, "code", None) or getattr(error, "type", None) or "no code",
        order.status,
    )


def handle_payment_succeeded(session: Session, event: Any) -> None:
    """The money settled. Deliberately changes nothing.

    One payment produces `payment_intent.succeeded` *and*
    `checkout.session.completed`, in an order nobody controls, and only one of
    them may drive the transition — two would race, and whichever lost would
    be refused by the transition table on every single successful payment,
    filling the log with warnings that describe the system working.

    `completed` is the one that wins, because it says the checkout finished
    rather than only that a charge settled: it carries the session this project
    created, and its `payment_status` is the field the expiry guard reads.

    Kept as a handler rather than left to the unknown-type branch so the log
    says the event was recognised and skipped on purpose. "Nothing happened
    because nothing handles this" and "nothing happened because nothing should"
    look identical from the outside, and the difference is the first thing
    anybody debugging a payment wants to know.
    """
    logger.info(
        "stripe event %s: payment_intent.succeeded acknowledged; "
        "checkout.session.completed is what marks the order paid",
        getattr(event, "id", "<unknown>"),
    )


def handle_charge_refunded(session: Session, event: Any) -> None:
    """Money went back. Move the order to `refunded` — but only if all of it did.

    **`charge.refunded` fires for a partial refund too**, which is the fact
    this handler is built around and which was measured rather than assumed.
    Two real refunds against one charge produced two events with these shapes:

        partial   amount=18998  amount_refunded=100    refunded=False
        full      amount=18998  amount_refunded=18998  refunded=True

    So the event type says nothing about completeness and the amounts say
    everything.

    **Only a full refund moves the order.** `refunded` is terminal, and a
    partially refunded order is not finished: the shopper still has the goods
    and the shop still has most of the money. There is no status between
    `paid` and `refunded` in this project, so representing the partial case
    would mean inventing one — and the wrong half of the choice is
    unrecoverable, because a terminal status cannot be corrected once written.
    Releasing the whole reservation for a $1 refund on a $190 order would be
    the concrete damage.

    So a partial refund is logged at ERROR and changes nothing. ERROR rather
    than WARNING because this system genuinely cannot represent what happened:
    somebody has moved money in a way the order will never reflect, and the
    only way anyone finds out is this line.

    **The decision is arithmetic, with the boolean as a cross-check.**
    `amount_refunded >= amount` is the primary test: those are numbers Stripe
    has always sent and cannot quietly redefine, whereas `refunded` is a flag
    that could be deprecated into absence — and `getattr` on an absent flag
    would silently mean "never full". When the two disagree, nothing moves and
    the disagreement is logged: that combination is not something this code
    understands, and the safe reading of anything unrecognised is to leave a
    terminal status alone.
    """
    order = _load_order(session, event)
    if order is None:
        return

    event_id = getattr(event, "id", "<unknown>")
    charge = getattr(getattr(event, "data", None), "object", None)
    amount = getattr(charge, "amount", None)
    amount_refunded = getattr(charge, "amount_refunded", None)
    flagged_full = getattr(charge, "refunded", None)

    if amount is None or amount_refunded is None:
        logger.error(
            "stripe event %s: charge for order %s carries amount=%r "
            "amount_refunded=%r, so this server cannot tell a full refund from "
            "a partial one and is leaving the order %s. Check the refund in "
            "the Stripe dashboard",
            event_id,
            order.id,
            amount,
            amount_refunded,
            order.status,
        )
        return

    fully_refunded = amount_refunded >= amount

    if flagged_full is not None and bool(flagged_full) != fully_refunded:
        logger.error(
            "stripe event %s: charge for order %s reports refunded=%r while "
            "amount_refunded=%d of amount=%d says %s. Those disagree, so "
            "nothing is being changed — a terminal status is not worth "
            "guessing at",
            event_id,
            order.id,
            flagged_full,
            amount_refunded,
            amount,
            "full" if fully_refunded else "partial",
        )
        return

    if not fully_refunded:
        logger.error(
            "stripe event %s: order %s was PARTIALLY refunded, %d of %d %s. "
            "This server has no status for that — `refunded` is terminal and "
            "would release the whole reservation — so the order stays %s and "
            "this line is the only record. Reconcile by hand",
            event_id,
            order.id,
            amount_refunded,
            amount,
            order.currency,
            order.status,
        )
        return

    # Releases the reservation, because `refunded` is in
    # `lifecycle.RELEASES_RESERVATION` — the units were paid for and then not,
    # so they go back on sale. D7 built that; this is the first caller to
    # reach it.
    _move(session, order, OrderStatus.REFUNDED, event)


HANDLERS = {
    "checkout.session.completed": handle_checkout_completed,
    "checkout.session.async_payment_succeeded": handle_async_payment_succeeded,
    "checkout.session.async_payment_failed": handle_async_payment_failed,
    "checkout.session.expired": handle_checkout_expired,
    "payment_intent.payment_failed": handle_payment_failed,
    "payment_intent.succeeded": handle_payment_succeeded,
    "charge.refunded": handle_charge_refunded,
}


def handle_event(session: Session, event: Any) -> None:
    """Do whatever this event means. The seam the router calls.

    A table rather than a chain of `if`s, so the set of events this server acts
    on can be read in one place and asserted in a test — `charge.refunded`
    joins it in step 4.

    An unhandled type returns normally and is therefore answered 200. Stripe
    sends every event type an endpoint is subscribed to, and a `stripe listen`
    with no filter forwards the lot; refusing them would mean an endpoint that
    answers an error to perfectly good deliveries and is retried for three days
    over each one.

    Nothing is caught here. A handler that raises is reporting a transient
    failure, and the router turns that into a 500 with this event's
    `processed_events` row rolled back — which is what makes Stripe's retry a
    retry rather than a duplicate.
    """
    warn_on_account_mismatch(event)

    if getattr(event, "livemode", False):
        # Recorded (the caller already claimed it) but never dispatched. This
        # project is test-mode only — `config.py` refuses an `sk_live_` key —
        # but that check does not cover this path: `STRIPE_WEBHOOK_SECRET` is a
        # separate credential and the `whsec_` prefix is identical for test and
        # live endpoints, so a live signing secret pasted into `.env` verifies
        # perfectly and real customer events would start mutating this
        # database's orders and inventory.
        #
        # 200 rather than an error, because the delivery is genuine and no
        # retry improves it; the configuration is what has to change. ERROR
        # rather than WARNING because nothing about this is routine.
        logger.error(
            "stripe event %s (%s) is a LIVE-MODE event. This server is test "
            "mode only, so it was recorded and NOT acted on. Check "
            "STRIPE_WEBHOOK_SECRET — a live endpoint's signing secret starts "
            "with whsec_ exactly like a test one and verifies just as well",
            getattr(event, "id", "<unknown>"),
            getattr(event, "type", None),
        )
        return

    event_type = getattr(event, "type", None)
    handler = HANDLERS.get(event_type)

    if handler is None:
        logger.info(
            "stripe event %s: nothing handles %s — accepted and ignored",
            getattr(event, "id", "<unknown>"),
            event_type,
        )
        return

    handler(session, event)
