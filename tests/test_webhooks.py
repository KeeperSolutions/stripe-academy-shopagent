"""The Stripe webhook endpoint: verification and idempotency (D8, steps 1-2).

Marked `db` for the reason `tests/test_api_auth.py` gives — the fixture chain,
not the assertions. Signature verification is still pure HMAC and needs
nothing outside the process, but step 2 made the endpoint write to
`processed_events`, so every request through it now needs a session. The few
tests here that check configuration or the signing scheme alone would run
offline; splitting the file to keep that true would put two halves of one
endpoint in two places, which is a worse trade than a marker that is honest
about the fixtures.

Nothing here reaches Stripe. The one exception is marked `stripe` and says so.

**The signatures are computed here, by hand.** Not copied from a real delivery,
which would be a secret in a repository and would rot the moment the tolerance
window passed; and not produced with the SDK's own helper either, for the
reason D7 wrote down after a test compared a variable to itself: an assertion
whose two sides come from the same code proves only that the code agrees with
itself. `sign()` below is an independent implementation of the scheme Stripe
documents, so a change in the SDK's signing would fail these tests instead of
being invisible to them. `test_the_sdk_signs_the_way_the_documentation_says`
is the one place the two are deliberately compared, which pins the scheme.
"""

from __future__ import annotations

import ast
import hashlib
import hmac
import inspect
import json
import logging
import textwrap
import time

import pytest
import stripe
from fastapi import Depends, FastAPI, Request, params
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect
from pydantic import BaseModel, ValidationError

from sqlalchemy import func, text
from sqlalchemy import select as sqlalchemy_select
from sqlalchemy.exc import IntegrityError, OperationalError

from shopagent.api.main import app
from shopagent.api.models import ProcessedEvent
from shopagent.api.routers import webhooks
from shopagent.api.services import events as event_service
from shopagent.config import REPO_ROOT, Settings
from shopagent.payments import stripe_svc

# Not a real secret and never was. The real one lives in `.env`, is read
# through `get_settings()`, and is patched out of the way in every test below —
# a signing secret in a test file is a signing secret in a git history.
pytestmark = pytest.mark.db

TEST_SECRET = "whsec_offline_test_secret_not_a_real_one"


def sign(payload: bytes, secret: str = TEST_SECRET, timestamp: int | None = None) -> str:
    """Build a `Stripe-Signature` header the way Stripe documents it.

    The signed string is `"{timestamp}.{body}"`, hashed with HMAC-SHA256 under
    the signing secret, hex-encoded, and presented as `t=...,v1=...`. Written
    out here rather than called from the SDK on purpose — see this module's
    docstring.
    """
    if timestamp is None:
        timestamp = int(time.time())

    signed = f"{timestamp}.".encode() + payload
    digest = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return f"t={timestamp},v1={digest}"


def event_body(**overrides) -> bytes:
    """A `checkout.session.completed` shaped like the real one, as bytes.

    Carries the fields a delivery actually has, including the ones the log must
    *not* repeat — an email and an amount — so the redaction test has something
    to look for.
    """
    body = {
        "id": "evt_offline_1",
        "object": "event",
        "type": "checkout.session.completed",
        "created": 1_756_000_000,
        "livemode": False,
        # The pinned version by default, so a warning in any test below is
        # something that test asked for rather than background noise.
        "api_version": stripe_svc.STRIPE_API_VERSION,
        "data": {
            "object": {
                "id": "cs_test_offline_1",
                "object": "checkout.session",
                "amount_total": 28497,
                "currency": "usd",
                "customer_details": {"email": "shopper@example.com"},
                "metadata": {"order_id": "eb268d01-0000-0000-0000-000000000000"},
            }
        },
    }
    body.update(overrides)
    return json.dumps(body).encode()


def use_secret(monkeypatch, secret: str | None) -> None:
    """Point the router at a known signing secret for the rest of a test.

    `get_settings` is patched inside the router's namespace rather than
    `webhook_signing_secret` being stubbed out, so the code that decides what
    counts as configured runs for real on every request.
    """
    monkeypatch.setattr(
        webhooks, "get_settings", lambda: Settings(stripe_webhook_secret=secret)
    )


@pytest.fixture
def client(api_client, monkeypatch):
    """The real app, on the test's own transaction, with a known secret.

    Built on `api_client` rather than a bare `TestClient` since step 2: the
    handler writes to `processed_events`, and without that fixture's dependency
    override the write would land in a session of the app's own — committed
    outside whatever transaction the test opened, invisible to the test's own
    reads, and left behind afterwards. See `tests/conftest.py`.
    """
    use_secret(monkeypatch, TEST_SECRET)
    return api_client


def post(client: TestClient, payload: bytes, signature: str | None) -> object:
    headers = {"Content-Type": "application/json"}
    if signature is not None:
        headers["Stripe-Signature"] = signature
    return client.post("/webhooks/stripe", content=payload, headers=headers)


# --- the happy path ------------------------------------------------------


def test_a_correctly_signed_event_is_accepted(client):
    payload = event_body()

    response = post(client, payload, sign(payload))

    assert response.status_code == 200
    assert response.json()["received"] is True
    assert response.json()["id"] == "evt_offline_1"
    assert response.json()["type"] == "checkout.session.completed"


def test_an_unknown_event_type_is_still_a_200(client):
    """The rule that keeps Stripe from retrying forever.

    An event nothing handles is not a malformed request — the sender is
    legitimate and the delivery arrived intact. Answering 400 would have Stripe
    redeliver it with backoff for three days and then give up, and the endpoint
    would report an error for a perfectly good event every time.

    Nothing is dispatched on type yet, so today this passes trivially. It is
    written now because step 3 is where the temptation to `raise
    HTTPException(400)` in an `else` branch arrives.
    """
    payload = event_body(type="invoice.payment_succeeded")

    response = post(client, payload, sign(payload))

    assert response.status_code == 200
    assert response.json()["type"] == "invoice.payment_succeeded"


def test_the_endpoint_records_what_it_processed(client, session):
    """Replaces `test_the_endpoint_needs_no_database` from step 1.

    That test broke the session factory and asserted the route still answered,
    which was true and worth pinning while "verify and log" was the whole
    contract. Step 2 ends it: the endpoint claims each event in
    `processed_events` before doing anything with it, so it now needs a session
    on every request and the old assertion would be asserting a bug.

    Deleted rather than adapted, and named here so the change is visible in a
    diff instead of looking like a test that quietly stopped existing. The
    marker it left behind — "this is exactly where verification-only stopped
    being true" — is the reason it was written that way.
    """
    payload = event_body()

    assert post(client, payload, sign(payload)).status_code == 200
    assert event_service.has_been_processed(session, "evt_offline_1")


# --- idempotency (step 2) ------------------------------------------------


def recorded_ids(session) -> list[str]:
    return sorted(
        session.scalars(sqlalchemy_select(ProcessedEvent.event_id)).all()
    )


def test_a_first_delivery_is_recorded(client, session):
    payload = event_body()

    response = post(client, payload, sign(payload))

    assert response.status_code == 200
    assert response.json()["duplicate"] is False
    assert recorded_ids(session) == ["evt_offline_1"]


def test_the_same_event_twice_is_one_row_and_two_200s(client, session):
    """The heart of the step.

    Stripe delivers at least once, so this is the ordinary case rather than
    the exceptional one: the second arrival has to be told "yes, thank you"
    and do nothing. A 4xx would have Stripe retry for three days; a second
    round of work would reserve stock twice.
    """
    payload = event_body()
    signature = sign(payload)

    first = post(client, payload, signature)
    second = post(client, payload, signature)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["duplicate"] is False
    assert second.json()["duplicate"] is True
    assert recorded_ids(session) == ["evt_offline_1"]


def test_two_different_events_are_two_rows(client, session):
    """The other half: the guard is not simply refusing everything after the first."""
    for event_id in ("evt_offline_a", "evt_offline_b"):
        payload = event_body(id=event_id)
        assert post(client, payload, sign(payload)).status_code == 200

    assert recorded_ids(session) == ["evt_offline_a", "evt_offline_b"]


def test_a_duplicate_leaves_the_session_usable(client, session):
    """Aimed squarely at `PendingRollbackError`, which is how this goes wrong.

    A unique violation aborts the whole Postgres transaction — the next
    statement on that connection comes back `InFailedSqlTransaction`, and
    SQLAlchemy raises `PendingRollbackError` for anything the session tries
    afterwards. So the failure would not appear on the duplicate at all. It
    would appear on whatever the request touched *next*, or on the next
    request sharing the session, for reasons that have nothing to do with it.

    `record_event` unwinds to a SAVEPOINT instead of letting the error reach
    the outer transaction, and this is what says so: a duplicate, then a
    perfectly ordinary read and a further request, all on the same session.
    """
    payload = event_body()
    signature = sign(payload)

    post(client, payload, signature)
    duplicate = post(client, payload, signature)
    assert duplicate.status_code == 200

    # The session the handler used is this one. If the savepoint had not been
    # there, this read is where it would surface.
    assert session.scalar(sqlalchemy_select(func.count()).select_from(ProcessedEvent)) == 1

    third = event_body(id="evt_offline_after_duplicate")
    assert post(client, third, sign(third)).status_code == 200
    assert len(recorded_ids(session)) == 2


def test_a_failure_after_the_insert_leaves_no_row(client, session, monkeypatch):
    """The most important test of the step, and the reason the seam exists.

    The record and the work it guards have to commit together. If the row
    could outlive a failed handler, Stripe's retry — the mechanism that exists
    precisely to recover from this — would be told the event was already
    processed and would do nothing. A payment that silently never lands, and
    a `processed_events` row that says everything is fine.

    `process_event` is the empty seam step 3 fills, so a stub that raises is
    exactly the shape of a handler that fails halfway. What is asserted is not
    the 500 but the absence: no row, so the next delivery is handled.
    """
    def explode(session, event) -> None:
        raise RuntimeError("step 3 blew up halfway through")

    monkeypatch.setattr(webhooks, "process_event", explode)

    payload = event_body()
    with pytest.raises(RuntimeError):
        post(client, payload, sign(payload))

    assert recorded_ids(session) == []


def test_the_delivery_after_a_failure_is_handled_rather_than_skipped(
    client, session, monkeypatch
):
    """The consequence of the test above, spelled out end to end.

    Asserting "no row" is a statement about a table; this is the behaviour that
    matters — the retry Stripe sends next is processed normally, which is only
    true because the failed attempt left nothing claiming to have handled it.
    """
    def explode(session, event) -> None:
        raise RuntimeError("step 3 blew up halfway through")

    monkeypatch.setattr(webhooks, "process_event", explode)
    payload = event_body()
    signature = sign(payload)

    with pytest.raises(RuntimeError):
        post(client, payload, signature)

    monkeypatch.undo()
    use_secret(monkeypatch, TEST_SECRET)

    retry = post(client, payload, signature)

    assert retry.status_code == 200
    assert retry.json()["duplicate"] is False
    assert recorded_ids(session) == ["evt_offline_1"]


def test_the_row_records_what_is_worth_reading_at_three_in_the_morning(
    client, session
):
    """Type and livemode alongside the id, and no part of the event body.

    The id alone can only confirm something somebody already suspects. The
    question that actually gets asked is "did the paid event ever arrive", and
    that is a `GROUP BY event_type` rather than a trip to the dashboard.
    """
    payload = event_body()
    post(client, payload, sign(payload))

    record = session.get(ProcessedEvent, "evt_offline_1")

    assert record.event_type == "checkout.session.completed"
    assert record.livemode is False
    assert record.processed_at is not None

    stored = {column.name for column in ProcessedEvent.__table__.columns}
    assert stored == {"event_id", "event_type", "livemode", "processed_at"}, (
        "processed_events grew a column. It exists to deduplicate deliveries, "
        "not to be a second copy of the event — the body lives in Stripe, "
        "under access rules this table does not have."
    )


def test_a_livemode_event_is_recorded_as_such(client, session):
    """`config.py` refuses live *API keys*; it does not police this path.

    `STRIPE_WEBHOOK_SECRET` is a separate credential, and a live endpoint's
    signing secret would verify here perfectly well. This column is the only
    place that would be recorded, which is the argument for keeping it in a
    project that is otherwise test-mode-only.
    """
    payload = event_body(livemode=True)
    post(client, payload, sign(payload))

    assert session.get(ProcessedEvent, "evt_offline_1").livemode is True


def test_an_event_with_no_id_cannot_be_recorded(session):
    """There is nothing to deduplicate on, so storing it would be a lie.

    A `ValueError` rather than a silent skip: an event without an id is either
    forged by somebody holding the signing secret or a payload shape nobody
    has seen, and both are worth a stack trace.
    """
    class Anonymous:
        type = "checkout.session.completed"
        livemode = False

    with pytest.raises(ValueError):
        event_service.record_event(session, Anonymous())


def test_an_event_with_no_type_is_still_recorded(session):
    """Useless to a handler, perfectly recordable, and refusing it costs more.

    A delivery this server cannot deduplicate is one Stripe retries forever.
    `unknown` is honest and takes the event out of the retry loop.
    """
    class Untyped:
        id = "evt_untyped"
        livemode = False

    assert event_service.record_event(session, Untyped()) is True
    assert session.get(ProcessedEvent, "evt_untyped").event_type == "unknown"


def test_the_second_claim_on_one_event_is_refused_by_the_database(session):
    """`record_event` returns False rather than raising, and the PK is why.

    Called directly rather than through the app, so what is being checked is
    the service's contract and not the router's mapping of it.
    """
    class Event:
        id = "evt_claimed_twice"
        type = "checkout.session.completed"
        livemode = False

    assert event_service.record_event(session, Event()) is True
    assert event_service.record_event(session, Event()) is False
    assert recorded_ids(session) == ["evt_claimed_twice"]


# --- the race the insert-first design exists to close ---------------------


def test_a_second_connection_is_made_to_wait_rather_than_told_absent(engine):
    """Why this is an INSERT and not a SELECT, shown rather than argued.

    Two real connections, because a race needs two — the rest of this file
    shares one session and can only observe the outcome, never the mechanism.
    Concurrent redeliveries of one event are not a corner case here: retries
    are the reason the table exists at all.

    The claim is what happens *while the first writer is still uncommitted*.
    A `SELECT` from the second connection would return nothing — the row is
    not visible yet — and check-then-act would have both deliveries proceed to
    do the work. The `INSERT` instead blocks: Postgres makes the second writer
    wait on the first transaction's outcome, which is what the `lock_timeout`
    below turns into something a test can assert. Then the first commits, and
    the second finds out it lost.

    Writes outside the fixture's transaction, because two connections cannot
    share one, so it cleans up after itself in `finally` — and uses an id no
    other test claims.
    """
    event_id = "evt_race_between_two_connections"
    insert = text(
        "INSERT INTO processed_events (event_id, event_type, livemode) "
        f"VALUES ('{event_id}', 'checkout.session.completed', false)"
    )

    first = engine.connect()
    second = engine.connect()
    try:
        first_transaction = first.begin()
        first.execute(insert)

        second_transaction = second.begin()
        # Without this the assertion below would hang until the other
        # transaction ends, which in a test means forever.
        second.execute(text("SET LOCAL lock_timeout = '400ms'"))

        with pytest.raises(OperationalError) as blocked:
            second.execute(insert)
        assert "lock timeout" in str(blocked.value.orig)

        second_transaction.rollback()
        first_transaction.commit()

        # And once the winner has committed, the loser is refused outright.
        second_transaction = second.begin()
        with pytest.raises(IntegrityError):
            second.execute(insert)
        second_transaction.rollback()
    finally:
        with engine.begin() as cleanup:
            cleanup.execute(
                text("DELETE FROM processed_events WHERE event_id = :id"),
                {"id": event_id},
            )
        first.close()
        second.close()


# --- everything that must be refused -------------------------------------


def test_a_wrong_signature_is_refused(client):
    """400 rather than 401, and rather than a retryable code.

    Whoever sent this does not hold the signing secret, so they are not Stripe
    and a redelivery would fail identically. 401 would invite them to try again
    with a credential, which is not how this endpoint is entered at all.
    """
    payload = event_body()

    response = post(client, payload, sign(payload, secret="whsec_a_different_secret"))

    assert response.status_code == 400
    assert "signature" in response.json()["detail"]


def test_a_signature_over_a_different_body_is_refused(client):
    """The case the raw body exists for.

    A signature that is valid for *some* payload is not valid for this one.
    This is what a handler that verified against a re-serialised body would
    start getting wrong — see `test_the_handler_takes_no_body_parameter`.
    """
    signature = sign(event_body())
    tampered = event_body(id="evt_someone_elses")

    assert post(client, tampered, signature).status_code == 400


def test_a_missing_signature_header_is_refused(client, caplog):
    """Refused, and refused audibly.

    Every other rejection writes a line saying why; without one here the case
    shows up in the log only as a `received` with nothing after it, which is
    visible to somebody who already suspects it and to nobody else.
    """
    caplog.set_level(logging.INFO, logger="shopagent.api.routers.webhooks")

    assert post(client, event_body(), None).status_code == 400
    assert any(
        "rejected" in record.getMessage() and "Stripe-Signature" in record.getMessage()
        for record in caplog.records
    )


def test_a_malformed_signature_header_is_refused(client):
    """`t=1,v1=deadbeef` — the shape, none of the substance.

    Parsed by the SDK and rejected at the digest, and the endpoint must not
    distinguish it from any other bad signature in what it returns.
    """
    assert post(client, event_body(), "t=1,v1=deadbeef").status_code == 400


def test_a_signature_header_that_does_not_parse_is_refused(client):
    assert post(client, event_body(), "not-a-signature").status_code == 400


def test_a_body_that_is_not_json_is_refused(client):
    """Signed correctly, so the sender is Stripe — and still unusable.

    This is the one refusal that is not about authenticity, which is why the
    handler catches `ValueError` separately from
    `SignatureVerificationError`. It is still a 400: the body will not become
    JSON on a retry.
    """
    payload = b"{not json"

    response = post(client, payload, sign(payload))

    assert response.status_code == 400
    assert "JSON" in response.json()["detail"]


def test_a_body_that_is_json_but_not_an_object_is_refused(client):
    """A signed JSON array reaches the SDK and raises `AttributeError`.

    Caught explicitly, because the alternative is a 500 on a request that was
    correctly signed — and a 500 is retryable, so Stripe would redeliver it for
    three days.
    """
    payload = b"[1, 2, 3]"

    assert post(client, payload, sign(payload)).status_code == 400


# --- the replay window ---------------------------------------------------


def test_a_stale_signature_is_refused(client):
    """A captured delivery, replayed later, must not still be accepted.

    The signature stays valid forever — it is an HMAC over fixed bytes — so
    what expires is the timestamp inside the signed string. Five minutes past
    the tolerance is well outside it and cannot be a slow network.
    """
    payload = event_body()
    stale = int(time.time()) - webhooks.stripe_svc.WEBHOOK_TOLERANCE_SECONDS - 300

    response = post(client, payload, sign(payload, timestamp=stale))

    assert response.status_code == 400


def test_a_signature_inside_the_tolerance_is_accepted(client):
    """The other half, without which the test above passes for any tolerance.

    Ten seconds short of the window: still valid, and it would fail if the
    tolerance were ever narrowed to something a slow request could exceed.
    """
    payload = event_body()
    recent = int(time.time()) - stripe_svc.WEBHOOK_TOLERANCE_SECONDS + 10

    assert post(client, payload, sign(payload, timestamp=recent)).status_code == 200


def test_the_tolerance_is_the_sdk_default_and_is_five_minutes():
    """Pinned in both directions, for the same reason the API version is.

    The constant exists so the tolerance is a value tests can reason about
    instead of a default buried in a call. Asserting only that it equals
    `stripe.Webhook.DEFAULT_TOLERANCE` would pass however that default moved,
    so the number is written out too — an SDK upgrade that changes the replay
    window then has to be noticed by a person.
    """
    assert stripe_svc.WEBHOOK_TOLERANCE_SECONDS == stripe.Webhook.DEFAULT_TOLERANCE
    assert stripe_svc.WEBHOOK_TOLERANCE_SECONDS == 300


# --- an unconfigured server ----------------------------------------------


def test_a_missing_signing_secret_is_503(api_client, monkeypatch):
    """The server is unconfigured; the sender did nothing wrong.

    503 rather than 400 because Stripe should retry this — somebody setting
    `STRIPE_WEBHOOK_SECRET` fixes it, and the delivery is still good. It is the
    same answer `MissingStripeKey` gets at the checkout route, for the same
    reason: an absent capability is not a broken server and not a bad request.
    """
    use_secret(monkeypatch, None)

    payload = event_body()
    response = post(api_client, payload, sign(payload))

    assert response.status_code == 503
    assert "STRIPE_WEBHOOK_SECRET" in response.json()["detail"]


def test_a_whitespace_signing_secret_reads_as_absent(api_client, monkeypatch):
    """Written expecting a `ValidationError`, and it does not raise — correctly.

    `_empty_to_none` is a `BeforeValidator`, so it runs first and turns any
    whitespace-only value into `None` before the `whsec_` check ever sees it.
    A secret of spaces is therefore "not configured" rather than "configured
    wrongly", which is the right of the two answers: it comes from a blank-ish
    line in `.env`, and the fix is to set the variable, not to correct it.

    That makes the `not secret.strip()` branch in `webhook_signing_secret()`
    unreachable through `Settings` — kept anyway, because it also guards a
    `Settings` built in code, and because a defensive check on the value a
    signature is verified against is not where to save two lines.
    """
    assert Settings(stripe_webhook_secret="   ").stripe_webhook_secret is None

    use_secret(monkeypatch, "   ")
    payload = event_body()
    assert post(api_client, payload, sign(payload)).status_code == 503


def test_the_secret_is_looked_up_per_request_rather_than_captured_at_import(
    client, monkeypatch
):
    """The router calls `get_settings()` on each delivery, not once at import.

    Written first as "a rotated secret needs no restart", which review on PR #8
    correctly called out as a claim this test cannot make: `get_settings` is
    `@lru_cache`d, so editing `.env` really does require a restart, and
    replacing the getter with a fresh lambda steps around the very cache that
    makes that true.

    What it does establish is narrower and still worth pinning — the module
    reads through `get_settings` per request rather than binding the secret to
    a module-level constant. That is what makes every other test in this file
    able to choose a secret, and it is the property that would silently break
    if somebody hoisted the lookup to import time.
    """
    payload = event_body()
    assert post(client, payload, sign(payload)).status_code == 200

    use_secret(monkeypatch, "whsec_rotated_to_something_else")

    assert post(client, payload, sign(payload)).status_code == 400


def test_get_settings_is_cached_so_env_changes_need_a_restart():
    """The other half of the correction: state the real behaviour, once.

    Recorded as a test rather than a comment because it is a live operational
    fact — rotating `STRIPE_WEBHOOK_SECRET` in `.env` does nothing until the
    process restarts — and because the test above must not be read as
    promising otherwise.
    """
    from shopagent.config import get_settings

    assert hasattr(get_settings, "cache_clear"), (
        "get_settings is no longer cached; the note about restarts in "
        "test_the_secret_is_looked_up_per_request_rather_than_captured_at_import "
        "and in README needs revisiting"
    )
    assert get_settings() is get_settings()


# --- the structural guard ------------------------------------------------


class ProbeEvent(BaseModel):
    """A body model for the falsification probe below, at module scope.

    Not defined inside the test that uses it, and the reason is worth keeping:
    `from __future__ import annotations` turns every annotation into a string,
    and FastAPI resolves those against the *module's* globals. A model declared
    inside a function is invisible there, so the annotation stays an
    unresolved string, FastAPI treats the parameter as an ordinary one, and the
    probe reports no body — making the falsification pass for the wrong reason
    and the guard it is checking look sound. That is exactly the failure mode
    this pair of tests exists to rule out, and it happened here first.
    """

    id: str


def webhook_route() -> APIRoute:
    from tests.test_api_auth import walk_api_routes

    return next(
        route for route in walk_api_routes(app.routes)
        if route.path == "/webhooks/stripe"
    )


def test_the_handler_takes_no_body_parameter():
    """The failure this prevents cannot be caught by testing behaviour.

    Someone will eventually add `event: StripeEvent` to the handler "for
    typing". FastAPI would then read the body, parse it, and hand over a model;
    the signature would have to be verified against a re-serialised version of
    that model, and `json.dumps` reproduces most payloads byte for byte. So it
    would work — in the tests, in `stripe listen`, in a demo — and then fail on
    the first event containing a float, a non-ASCII character or a key order
    the encoder does not preserve. An intermittent 400 on real payments,
    traceable to a parameter added months earlier.

    Two assertions rather than one, because they fail for different reasons.
    The signature check names the offending parameter, and allows exactly two
    things: the raw `Request`, and anything supplied by `Depends` — step 2's
    database session is the latter, and a dependency is resolved by FastAPI
    rather than read out of the body. `body_field` is FastAPI's own conclusion
    about whether this route consumes a body, which is the thing that actually
    matters: a parameter form the name check did not anticipate is still caught
    there.
    """
    parameters = inspect.signature(webhooks.stripe_webhook).parameters

    for name, parameter in parameters.items():
        if name == "request":
            continue
        assert isinstance(parameter.default, params.Depends), (
            f"stripe_webhook takes `{name}`, which is neither the raw "
            "`Request` nor a dependency. FastAPI reads a parameter it does not "
            "recognise as the request body, and a signature is only valid over "
            "the bytes that arrived — never over a parsed body re-encoded."
        )
    assert webhook_route().body_field is None, (
        "FastAPI thinks /webhooks/stripe consumes a request body. The "
        "signature is computed over raw bytes, so a parsed-and-re-encoded body "
        "verifies against something Stripe never signed."
    )


def test_the_body_parameter_guard_actually_catches_one():
    """Falsified rather than trusted, because a guard that cannot fail is a comment.

    A throwaway route with a Pydantic parameter — the exact mistake — and the
    assertion above has to notice it. If FastAPI ever stopped populating
    `body_field`, the real test would keep passing while protecting nothing,
    and this is what turns that into a failure.
    """
    probe = FastAPI()

    @probe.post("/probe")
    async def with_a_body(event: ProbeEvent) -> dict[str, str]:
        return {"id": event.id}

    route = next(r for r in probe.routes if isinstance(r, APIRoute))
    assert route.body_field is not None, (
        "the guard in test_the_handler_takes_no_body_parameter is vacuous: "
        "FastAPI no longer reports a body parameter through `body_field`"
    )


def test_the_handler_reads_the_raw_body():
    """The positive half: raw bytes off the stream are what the handler uses.

    Read from the source rather than mocked, so it stays true of whatever the
    handler grows into. Together with the guard above this pins both ends —
    no parsed body in, raw bytes used.

    The route delegates to `read_capped_body`, so the claim is checked in two
    places: the handler calls it, and it is the thing that touches the request.
    """
    assert "read_capped_body(request)" in inspect.getsource(webhooks.stripe_webhook)

    reader = ast.parse(textwrap.dedent(inspect.getsource(webhooks.read_capped_body)))
    reached = {
        node.func.attr
        for node in ast.walk(reader)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "request"
    }

    # Walked rather than grepped: this function's docstring names `body()` in
    # order to explain why it is not used, and a substring check cannot tell
    # the explanation from the call. `body()` would buffer the whole delivery
    # before the cap could refuse it, which is what the cap exists to prevent.
    assert "stream" in reached
    assert "body" not in reached


# --- how much of an anonymous body this endpoint will read ---------------
#
# Raised in the second review round on PR #8. A signature is the credential
# here, and it cannot be checked until the body has arrived — so everything
# read before that point was sent by somebody who holds nothing.


def oversized() -> bytes:
    return b"x" * (webhooks.MAX_WEBHOOK_BODY_BYTES + 1)


def test_a_body_over_the_cap_is_refused(client):
    response = post(client, oversized(), sign(b"whatever"))

    assert response.status_code == 413
    assert str(webhooks.MAX_WEBHOOK_BODY_BYTES) in response.json()["detail"]


def test_the_oversized_body_is_refused_before_any_signature_work(client, monkeypatch):
    """The point of the cap, and the only assertion that shows it.

    Refusing after verification would still have buffered the whole body and
    done the HMAC over it, which is the work being denied. So verification must
    not be reached at all.
    """
    def must_not_run(*args, **kwargs):
        raise AssertionError("the signature was verified over an oversized body")

    monkeypatch.setattr(stripe_svc, "construct_webhook_event", must_not_run)

    assert post(client, oversized(), sign(b"whatever")).status_code == 413


def test_a_chunked_body_over_the_cap_is_refused(client):
    """`Content-Length` is the fast path, not the enforcement.

    It is a header the sender writes, and a chunked request need not carry one
    at all — so the cap has to hold while the stream is being read. Passing an
    iterator makes httpx send `Transfer-Encoding: chunked` with no length.
    """
    chunk = b"x" * 8192
    count = webhooks.MAX_WEBHOOK_BODY_BYTES // len(chunk) + 2

    response = client.post(
        "/webhooks/stripe",
        content=(chunk for _ in range(count)),
        headers={"Content-Type": "application/json", "Stripe-Signature": sign(b"x")},
    )

    assert response.status_code == 413


def test_the_cap_holds_when_the_declared_length_is_wrong(client):
    """The stream check enforces the cap on its own, header or no header.

    Not a smuggling scenario — checked with a raw socket against the running
    server, and HTTP/1.1 framing already prevents that: a request declaring
    `Content-Length: 10` *is* ten bytes long, and whatever follows is not part
    of it. What this pins is narrower and still worth pinning, because it is
    the reason the fast path is safe to have at all: the header is consulted
    and then not relied upon, so a wrong one cannot raise the limit.
    """
    response = client.post(
        "/webhooks/stripe",
        content=(c for c in [oversized()]),
        headers={
            "Content-Type": "application/json",
            "Stripe-Signature": sign(b"x"),
            "Content-Length": "10",
        },
    )

    assert response.status_code == 413


def test_a_caller_that_disconnects_mid_body_is_a_refusal_not_a_crash(client, monkeypatch):
    """Found by the live run, not by a test — which is why it is one now.

    Starlette raises `ClientDisconnect` out of `request.stream()` when a caller
    goes away mid-upload. Left alone it reaches uvicorn as an unhandled ASGI
    exception and prints a full traceback for a request that has no client left
    to answer. `request.body()` had the same edge, so this is not new; what is
    new is how reachable it became once the body is streamed under a cap, and
    an unauthenticated endpoint whose log length a stranger controls is worth
    closing.
    """
    async def disconnects():
        yield b'{"partial":'
        raise ClientDisconnect()

    monkeypatch.setattr(
        webhooks.Request, "stream", lambda self: disconnects(), raising=False
    )

    response = post(client, b"ignored", sign(b"ignored"))

    assert response.status_code == 400
    assert "incomplete" in response.json()["detail"]


def test_a_body_at_the_cap_is_read_normally(client):
    """The limit is not off by one, and padding is not what refuses a delivery.

    A real event this large would be extraordinary — the biggest measured on
    this account is 4,145 bytes — but the assertion that matters is that a body
    right up against the cap reaches verification rather than the 413.
    """
    padded = event_body()
    padding = webhooks.MAX_WEBHOOK_BODY_BYTES - len(padded) - len(b', "pad": ""')
    payload = padded[:-1] + b', "pad": "' + b"x" * padding + b'"}'
    assert len(payload) == webhooks.MAX_WEBHOOK_BODY_BYTES

    assert post(client, payload, sign(payload)).status_code == 200


# --- what the log says ---------------------------------------------------


def test_the_log_records_the_event_before_saying_anything_about_it(client, caplog):
    """The plan's rule: log every delivery, and log it before processing.

    Two lines, in order. The first is written while the request is still
    unverified and therefore says only how much arrived and whether it was
    signed — a body logged there would let anyone on the internet write into
    this server's log. The second is written once the signature holds, and is
    the one that names the event.
    """
    caplog.set_level(logging.INFO, logger="shopagent.api.routers.webhooks")
    payload = event_body()

    post(client, payload, sign(payload))

    messages = [record.getMessage() for record in caplog.records]
    assert any("webhook received" in message for message in messages)
    assert any("webhook verified" in message for message in messages)

    received = next(i for i, m in enumerate(messages) if "webhook received" in m)
    verified = next(i for i, m in enumerate(messages) if "webhook verified" in m)
    assert received < verified


def test_the_log_names_the_event_and_the_object_it_is_about(client, caplog):
    caplog.set_level(logging.INFO, logger="shopagent.api.routers.webhooks")
    payload = event_body()

    post(client, payload, sign(payload))

    verified = next(
        record.getMessage()
        for record in caplog.records
        if "webhook verified" in record.getMessage()
    )
    assert "evt_offline_1" in verified
    assert "checkout.session.completed" in verified
    assert "cs_test_offline_1" in verified


def test_the_log_repeats_none_of_the_shopper_s_details(client, caplog):
    """The D5 redaction decision, applied to a body that carries much more.

    `data.object` on these events holds the buyer's email, the amount, the
    billing address and a card's last four. None of it helps answer the
    questions a webhook log exists for — did it arrive, which one was it, what
    was it about — and all of it is a second copy of a record that already
    lives in Stripe under different access rules. So the log carries
    identifiers and the details are looked up by whoever is entitled to see
    them.
    """
    caplog.set_level(logging.DEBUG, logger="shopagent.api.routers.webhooks")
    payload = event_body()

    post(client, payload, sign(payload))

    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "shopper@example.com" not in logged
    assert "28497" not in logged
    # The order id is not secret and step 3 will have every reason to log it.
    # It is absent today only because nothing reads `metadata` yet, so this
    # asserts the amount and the email rather than everything in the object.
    assert "billing" not in logged


def test_the_unverified_line_does_not_echo_the_body(client, caplog):
    """The first line is written before anything is known about the sender."""
    caplog.set_level(logging.INFO, logger="shopagent.api.routers.webhooks")
    payload = json.dumps({"anyone_can_write_this": "into your log"}).encode()

    post(client, payload, "t=1,v1=deadbeef")

    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "anyone_can_write_this" not in logged
    assert "into your log" not in logged


def test_describe_event_survives_a_sparsely_populated_event():
    """A verified event is not necessarily a well-formed one.

    Anyone holding the signing secret can sign `{"id": "evt_x"}`, and
    `StripeObject` raises `AttributeError` for an absent field rather than
    returning `None`. The naive version of `describe_event` would therefore
    crash inside a logging call, on a request that had already passed
    authentication — a 500 where the log should have said what arrived.
    """
    payload = b'{"id": "evt_bare"}'
    event = stripe_svc.construct_webhook_event(payload, sign(payload), TEST_SECRET)

    described = webhooks.describe_event(event)

    assert described["id"] == "evt_bare"
    assert described["type"] is None
    assert described["object_id"] is None


# --- the API version an event was rendered at ----------------------------


def test_an_event_from_another_api_version_is_accepted(client):
    """Warned about, never refused.

    `event.api_version` is fixed at creation and Stripe does not move an
    existing endpoint's version, so a real endpoint receives several at once as
    an account is upgraded. Refusing would lose payments over a difference that
    signature verification does not depend on at all.
    """
    payload = event_body(api_version="2026-06-24.dahlia")

    assert post(client, payload, sign(payload)).status_code == 200


def test_an_event_from_another_api_version_warns_and_names_both(client, caplog):
    """The warning has to be actionable on its own, so it carries both numbers.

    A line saying only "version mismatch" sends whoever reads it looking for
    the pin. Naming both versions and the event makes the log the whole
    answer — and it deliberately does *not* name a flag that would reconcile
    them, because there is none: Stripe CLI 1.50.6 offers the account default
    or `--latest` and nothing in between. An earlier draft of this message
    advertised `stripe listen --stripe-version`, which the CLI rejects with
    `unknown flag`.
    """
    caplog.set_level(logging.INFO, logger="shopagent.api.routers.webhooks")
    # An unhandled type, so the only thing that can warn is the version check.
    # Since step 3 a handled event also logs about the order it is attributed
    # to, and counting both would make this test fail for an unrelated reason.
    payload = event_body(type="product.created", api_version="2026-06-24.dahlia")

    post(client, payload, sign(payload))

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "2026-06-24.dahlia" in warnings[0]
    assert stripe_svc.STRIPE_API_VERSION in warnings[0]
    assert "evt_offline_1" in warnings[0]
    assert "stripe listen" in warnings[0]


def test_an_event_from_the_pinned_version_warns_about_nothing(client, caplog):
    """The other half, without which the test above passes for any event.

    A check that only ever fires is a check nobody can trust to be quiet, and a
    warning on every delivery is one people learn to scroll past.
    """
    caplog.set_level(logging.INFO, logger="shopagent.api.routers.webhooks")
    # Unhandled, for the reason given above: this asserts *nothing* warned, and
    # a handled event legitimately warns about an order this test never made.
    payload = event_body(type="product.created", api_version=stripe_svc.STRIPE_API_VERSION)

    response = post(client, payload, sign(payload))

    assert response.status_code == 200
    assert [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
    ] == []


def test_an_event_with_no_api_version_is_accepted_and_says_so(client, caplog):
    """Absent is a third case, and it is not silence.

    Nothing can be compared, so nothing can be promised about the shape of the
    data — which is exactly the sentence a handler misreading a field later
    needs to find in the log.
    """
    caplog.set_level(logging.INFO, logger="shopagent.api.routers.webhooks")
    payload = event_body(type="product.created")
    without = json.loads(payload)
    del without["api_version"]
    payload = json.dumps(without).encode()

    response = post(client, payload, sign(payload))

    assert response.status_code == 200
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "no api_version" in warnings[0]


def test_a_null_api_version_is_read_as_absent_rather_than_crashing(client):
    """`None` and "not there" are two shapes and only one of them is a read.

    `StripeObject` raises `AttributeError` for a key that is missing and
    returns `None` for one explicitly set to null — probed rather than assumed,
    because a direct `event.api_version` would then raise on the first shape.
    That would be a 500 on a request that had already been authenticated, and
    Stripe treats a 500 as retryable: the same delivery for three days.
    """
    payload = event_body(api_version=None)

    assert post(client, payload, sign(payload)).status_code == 200


def test_describe_event_reads_a_missing_api_version_as_none():
    """The `getattr` in `describe_event`, checked at the seam it protects."""
    payload = b'{"id": "evt_bare"}'
    event = stripe_svc.construct_webhook_event(payload, sign(payload), TEST_SECRET)

    assert webhooks.describe_event(event)["api_version"] is None


@pytest.mark.stripe
def test_a_real_event_carries_an_api_version():
    """The claim the offline tests rest on, put to the account itself.

    Every test above decides what `api_version` looks like, which makes them a
    check on this code and not on Stripe. D7 collected four SDK objects that
    were not the shape they appeared to be, each found by a real call, so the
    assumption that this field is populated is worth one.

    Skips rather than fails on an account with no events in the last thirty
    days: that says nothing about the field, and a test that fails for being
    run on a fresh account teaches people to ignore it.
    """
    events = list(stripe_svc.get_client().v1.events.list(params={"limit": 5}))

    if not events:
        pytest.skip("this Stripe account has no events in the last 30 days")

    for event in events:
        assert getattr(event, "api_version", None), (
            f"{event.id} ({event.type}) carries no api_version, so the "
            "mismatch check in the webhook router can never fire"
        )


# --- the scheme itself ---------------------------------------------------


def test_the_sdk_signs_the_way_the_documentation_says():
    """The one place the hand-rolled signer and the SDK are compared.

    Everywhere else `sign()` is used alone, so these tests do not rest on the
    SDK agreeing with itself. Here the two implementations meet: if a future
    stripe-python changed the signed string or the digest, this fails and says
    so, instead of every test above quietly starting to verify a scheme Stripe
    no longer uses.
    """
    payload = event_body()
    timestamp = 1_756_000_000

    theirs = stripe.WebhookSignature.generate_signature_header(
        payload.decode(), TEST_SECRET, timestamp=timestamp
    )

    assert theirs == sign(payload, timestamp=timestamp)


# --- configuration -------------------------------------------------------


def test_a_signing_secret_must_look_like_one():
    """The typo this catches costs every payment until somebody notices.

    Pasting the API key into `STRIPE_WEBHOOK_SECRET` fails silently from the
    server's side: verification never succeeds, every delivery is answered 400,
    Stripe retries and gives up, and no order is ever marked paid. Nothing logs
    a cause, because the endpoint only ever saw badly signed requests.
    """
    with pytest.raises(ValidationError) as excinfo:
        Settings(stripe_webhook_secret="sk_test_51Nxxxxxxxxxxxxxxxxxxxxxx")

    assert "whsec_" in str(excinfo.value)


def test_a_real_looking_signing_secret_is_accepted():
    """The other half — the validator is not simply always raising."""
    settings = Settings(stripe_webhook_secret="whsec_abc123")

    assert settings.stripe_webhook_secret == "whsec_abc123"


def test_no_signing_secret_is_a_valid_configuration():
    """Absent is a state a developer chose, and the API still has to start.

    Payments are one part of this system: a cart that could not be browsed
    because webhooks are unconfigured would be the wrong failure. The refusal
    happens at the endpoint, as a 503, not at import.
    """
    assert Settings(stripe_webhook_secret=None).stripe_webhook_secret is None


def test_the_example_env_ships_the_variable_and_no_value():
    """`.env.example` is what somebody copies; a plausible secret there is a bug."""
    example = (REPO_ROOT / ".env.example").read_text()

    line = next(
        raw for raw in example.splitlines()
        if raw.startswith("STRIPE_WEBHOOK_SECRET=")
    )
    assert line == "STRIPE_WEBHOOK_SECRET="
