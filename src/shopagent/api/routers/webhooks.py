"""The Stripe webhook endpoint (D8, step 1).

A delivery is authenticated, claimed in `processed_events`, and answered 200.
No order changes state yet — dispatching on event type is step 3, and
`process_event` below is the empty seam it fills. Separating that from
verification and idempotency is deliberate: both are easy to get wrong while
attention is on what the events *mean*.

**A duplicate is answered 200 and does nothing.** Stripe delivers at least
once, so a second arrival of an event is ordinary rather than an error, and
`services/events.py` decides which one this is by inserting rather than
looking — see there for why the difference matters under concurrent retries.

**Unauthenticated, and that is not an exception being made twice.** Stripe does
not send this server's `X-API-Key`; it signs the body with a shared secret and
puts the result in `Stripe-Signature`. The signature *is* the authentication,
and a stronger one than the header every other route uses: it proves the sender
holds the secret and that this exact byte sequence is what they sent. The
checkout pages next door are public for a different reason — a browser arrives
there carrying nothing at all — and their safety rests on reading only. This
route writes, from step 2 onwards, and is safe because nothing unsigned gets
past the first ten lines of the handler.

**The status codes are chosen for a client that retries.** Stripe redelivers
anything it does not get a 2xx for, with backoff, for up to three days. So the
question behind every code here is "should Stripe try this again":

    valid signature          200   received, whatever it turns out to say
    already processed        200   Stripe delivers at least once; this is normal
    bad or stale signature   400   not Stripe; retrying changes nothing
    no Stripe-Signature      400   likewise
    body is not JSON         400   signed, but unusable, and it will not improve
    no signing secret        503   *this server* is unconfigured — retry later
    processing raised        500   nothing was committed; retrying is the fix

The one to be careful with is the future case of an event type nothing handles:
that is a **200**, because the sender is legitimate and the delivery arrived
intact. A 400 there tells Stripe the request was malformed, which it was not,
and it means an endpoint answers an error to a perfectly good event forever.

**An event rendered at a different API version is accepted and warned about.**
`event.api_version` is fixed when the event is created and never changes, so a
real endpoint receives several versions at once as an account is upgraded over
the years — Stripe does not re-render old events and does not move an existing
endpoint's version. Refusing a mismatch would therefore lose payments to a
difference that verification does not even depend on. What it can affect is a
handler reading a field, which is why the warning exists and why it names both
versions: when step 3 misreads something, the log already said why.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from shopagent.api.db import get_session
from shopagent.api.services import events as event_service
from shopagent.config import get_settings
from shopagent.payments import stripe_svc
from shopagent.payments.stripe_svc import SignatureVerificationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# The header Stripe signs with. Named rather than typed inline because two
# places read it — the handler and the log line that says whether it arrived.
SIGNATURE_HEADER = "Stripe-Signature"


class MissingWebhookSecret(RuntimeError):
    """Raised when a delivery arrives and there is no secret to verify it with.

    A distinct type for the same reason `MissingStripeKey` is one: the handler
    turns it into 503 rather than 400, and the difference matters to whoever
    reads the log. 400 would blame the sender for a gap on this side.
    """


def webhook_signing_secret() -> str:
    """Return the configured signing secret, or refuse to hand one back.

    `Settings` already rejects a blank line and anything not beginning
    `whsec_`, so what reaches here is either a plausible secret or `None`. The
    absent case is not a startup failure — see the field's comment in
    `config.py` — which is why it has to be answered per request instead.
    """
    secret = get_settings().stripe_webhook_secret

    if not secret or not secret.strip():
        raise MissingWebhookSecret(
            "STRIPE_WEBHOOK_SECRET is not set, so an incoming webhook cannot "
            "be verified and this endpoint refuses to trust it. Run `stripe "
            "listen --forward-to localhost:8000/webhooks/stripe` and copy the "
            "whsec_... it prints into .env — see .env.example."
        )
    return secret


def describe_event(event: object) -> dict[str, object]:
    """The identifiers worth logging, and deliberately nothing else.

    The rule this follows is the one D5 set for the MCP server's argument log:
    record what makes the log answer questions, and leave out what a shopper
    would not want written down. There the free-text `query` was the line; here
    it is the whole of `data.object`, which for the events this endpoint will
    subscribe to carries the buyer's email, the amount, the billing address and
    the last four digits of a card.

    None of that helps debug a webhook. The questions a webhook log actually
    has to answer are "did it arrive", "which one was it", "had we seen it
    before" and "which Stripe object was it about", and every one of those is
    answered by an identifier. So the line carries `event.id`, `event.type`,
    `created`, `livemode`, and the id and type of the object the event is
    about — opaque Stripe references, each of which can be looked up in the
    dashboard by whoever is entitled to see the details behind it. That
    indirection is the point: the log says which record to go and read, rather
    than being a second copy of it in a file with different access rules.

    Read with `getattr` throughout because a verified event is not necessarily
    a well-formed one. Anyone holding the signing secret can sign
    `{"id": "evt_x"}`, and `StripeObject` raises `AttributeError` for a field
    that is not there rather than returning `None` — so the naive version of
    this function would crash inside the logging call, on a request that had
    already been authenticated.
    """
    data = getattr(event, "data", None)
    obj = getattr(data, "object", None)

    return {
        "id": getattr(event, "id", None),
        "type": getattr(event, "type", None),
        "created": getattr(event, "created", None),
        "livemode": getattr(event, "livemode", None),
        "api_version": getattr(event, "api_version", None),
        "object_id": getattr(obj, "id", None),
        "object_type": getattr(obj, "object", None),
    }


def warn_on_api_version_mismatch(api_version: str | None, event_id: object) -> None:
    """Say so when an event was rendered at a version this code does not expect.

    `event.api_version` is the version `data` was serialised at, fixed when the
    event was created and never updated afterwards. `STRIPE_API_VERSION` is
    what this repo pins its client to. When the two differ, verification is
    entirely unaffected — an HMAC is computed over bytes and knows nothing
    about object shape — but a field a handler reads may be named differently,
    nested differently, or absent.

    **This warns and does not refuse.** A differently rendered event is a valid
    event: Stripe never changes the version of an existing endpoint, so a
    long-lived one legitimately receives several versions at once, and events
    created before an upgrade keep their original rendering forever. Rejecting
    would turn a cosmetic mismatch into lost payments, which is far worse than
    reading a field that is not there. The point of the line is that when a
    handler does misread something, the log already said why.

    Locally the mismatch is expected and cannot be fixed from the command
    line, which is worth stating because the obvious assumption is that it can.
    `stripe listen` renders events at the *account's* default API version;
    `--latest` renders them at Stripe's newest. Measured against this account
    on 2026-08-28: the default is `2026-06-24.dahlia` and `--latest` is
    `2026-08-26.dahlia`, while this repo pins `2026-07-29.dahlia` — one older,
    one newer, neither equal. There is no `--stripe-version` flag in Stripe
    CLI 1.50.6 to close the gap with. So the line below is expected to appear
    during local development, and it is still worth having: it is the record
    that says which rendering a handler was actually given.

    Read with `getattr` upstream, in `describe_event`, and that is not
    defensive habit. `Event.api_version` is typed `Optional[str]` and Stripe
    documents it as populated only for events created after October 2014 — but
    `StripeObject` raises `AttributeError` for an absent key rather than
    returning `None`, so a plain `event.api_version` would raise on any payload
    lacking it. That is a 500 on a request that had already passed
    authentication, which Stripe reads as retryable and would redeliver for
    three days.
    """
    if api_version is None:
        # Distinct from a mismatch, and worth its own line rather than silence:
        # the shape cannot be checked at all, so a handler misreading a field
        # later has nothing in the log to explain it.
        logger.warning(
            "stripe webhook %s carries no api_version, so the rendering of its "
            "data cannot be compared against the pinned %s — field names and "
            "nesting may differ from what handlers expect",
            event_id,
            stripe_svc.STRIPE_API_VERSION,
        )
        return

    if api_version != stripe_svc.STRIPE_API_VERSION:
        logger.warning(
            "stripe webhook %s was rendered at API version %s, but this client "
            "is pinned to %s — the event is valid and is being accepted, but "
            "the shape of its data object may differ from what handlers "
            "expect. Locally this is normal: stripe listen renders at the "
            "account's default version and --latest at Stripe's newest, and "
            "the CLI has no flag for an arbitrary one",
            event_id,
            api_version,
            stripe_svc.STRIPE_API_VERSION,
        )


def process_event(session: Session, event: object) -> None:
    """Act on a verified, not-yet-seen event.

    One line, and kept as a function rather than inlined because the
    transaction boundary around it is the contract: it runs inside the
    transaction that already holds this event's `processed_events` row, and
    anything it raises unwinds both. `tests/test_webhooks.py` replaces this
    name with a stub that raises to prove that, which is a claim about the
    router and is best made where the router can be pointed at.

    What an event *means* is not here. Dispatching on type, refusing an unsafe
    cancellation, moving an order — all of that is domain and lives in
    `services/events.py`, by the rule that a router parses, calls one service
    function, and maps an exception to a status code.
    """
    event_service.handle_event(session, event)


@router.post("/stripe", status_code=status.HTTP_200_OK)
async def stripe_webhook(
    request: Request, session: Session = Depends(get_session)
) -> dict[str, object]:
    """Verify a delivery, record it, and do nothing else (step 1).

    **This handler takes no body parameter and must never take one.** FastAPI
    reads the request body once and hands a declared parameter the *parsed*
    result; a signature is computed over the bytes that arrived. Re-serialising
    a parsed body produces a string that differs in whitespace and key order,
    so verification would then run against something Stripe never signed. The
    reason that is dangerous rather than merely wrong is that it would mostly
    work — `json.dumps` reproduces many payloads byte for byte — and fail
    later, intermittently, on whichever event happened to contain a float or a
    non-ASCII character. `tests/test_webhooks.py` fails if a parameter is ever
    added.

    So: `Request` in, `await request.body()` for the raw bytes, and the parsed
    event comes back from the SDK's verification rather than from FastAPI.
    """
    try:
        secret = webhook_signing_secret()
    except MissingWebhookSecret as exc:
        # Answered before the body is read: with no secret there is nothing to
        # check it against, and reading it would only lend it credibility.
        # 503 rather than 400 or 500 — the capability is absent and Stripe
        # should try again once somebody configures it, which is exactly what
        # `MissingStripeKey` means at the checkout route.
        logger.error("stripe webhook refused: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    payload = await request.body()
    signature = request.headers.get(SIGNATURE_HEADER)

    # The plan's rule — log every delivery before doing anything with it,
    # because when something breaks the log is the only trace. This is the
    # earliest honest line: at this point the request is still *unverified*, so
    # it says how many bytes arrived and whether a signature came with them,
    # and repeats nothing the sender wrote. A body logged here would be a body
    # anybody on the internet could put in this server's log.
    logger.info(
        "stripe webhook received: %d bytes, signature %s",
        len(payload),
        "present" if signature else "absent",
    )

    if signature is None:
        # Logged as its own line rather than left to the "signature absent"
        # note above, so every refusal has a `rejected` record with a reason.
        # Reading the log for a delivery that never arrived is how this gets
        # used, and a case that only appears as an absence is one somebody has
        # to already suspect before they can see it.
        logger.warning(
            "stripe webhook rejected: no %s header", SIGNATURE_HEADER
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{SIGNATURE_HEADER} header is required",
        )

    try:
        event = stripe_svc.construct_webhook_event(payload, signature, secret)
    except SignatureVerificationError as exc:
        # Covers three cases the SDK does not distinguish and neither should
        # we: a wrong signature, a header that will not parse, and a timestamp
        # older than the tolerance. All three mean "this did not come from
        # Stripe, or did not come from Stripe *now*", and the answer to each is
        # to refuse without reading further.
        logger.warning("stripe webhook rejected: signature invalid (%s)", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="signature verification failed",
        ) from exc
    except (ValueError, AttributeError) as exc:
        # Signed correctly, so the sender holds the secret — but the body is
        # not JSON, or is JSON that is not an object. `ValueError` covers
        # `json.JSONDecodeError`; `AttributeError` is what the SDK raises when
        # the payload parses to a list. Narrow on purpose: anything else is a
        # bug here and must not be reported as the sender's fault.
        logger.warning("stripe webhook rejected: body is not a usable event (%s)", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="request body is not a JSON event",
        ) from exc

    described = describe_event(event)
    logger.info(
        "stripe webhook verified: id=%s type=%s created=%s livemode=%s "
        "api_version=%s object=%s(%s)",
        described["id"],
        described["type"],
        described["created"],
        described["livemode"],
        described["api_version"],
        described["object_type"],
        described["object_id"],
    )

    # After the line above rather than before it, so the record identifying the
    # delivery is already in the log when the warning refers to it by id.
    warn_on_api_version_mismatch(described["api_version"], described["id"])

    # The claim, and it is a write rather than a read on purpose — see
    # `services/events.py`. A `False` here means another delivery of this same
    # event already did whatever there was to do.
    if not event_service.record_event(session, event):
        # Deliberately not rolled back and not committed: nothing was written,
        # because the savepoint inside `record_event` has already unwound. A
        # duplicate is the system working, so this is INFO rather than a
        # warning — an endpoint that logged every retry at WARNING would be
        # noisy on exactly the behaviour Stripe documents.
        logger.info(
            "stripe webhook %s (%s) was already processed — no action taken",
            described["id"],
            described["type"],
        )
        return {
            "received": True,
            "duplicate": True,
            "id": described["id"],
            "type": described["type"],
        }

    try:
        # The seam step 3 fills. Everything it does joins the transaction that
        # already holds the `processed_events` row, so the record of the work
        # and the work itself commit together.
        process_event(session, event)
        session.commit()
    except Exception:
        # Rolled back here rather than left to `get_session`, and the reason is
        # not belt-and-braces. The invariant — the row and the work survive
        # together or neither does — belongs to the code that owns both, and a
        # dependency that happens to roll back on the way out is a different
        # module's implementation detail. Without it, a handler that failed
        # after the insert could leave the claim behind, and the retry that
        # should have fixed it would be told the event was already handled.
        session.rollback()
        logger.exception(
            "stripe webhook %s (%s) failed while being processed; its "
            "processed_events row was rolled back, so Stripe's retry will be "
            "handled rather than skipped",
            described["id"],
            described["type"],
        )
        raise

    logger.info(
        "stripe webhook %s (%s) recorded as processed",
        described["id"],
        described["type"],
    )

    # The same keys on both paths, so a reader diffing two deliveries of one
    # event sees `duplicate` flip and nothing else move. Stripe reads the
    # status code alone; this body is for whoever is watching `stripe listen`.
    return {
        "received": True,
        "duplicate": False,
        "id": described["id"],
        "type": described["type"],
    }
