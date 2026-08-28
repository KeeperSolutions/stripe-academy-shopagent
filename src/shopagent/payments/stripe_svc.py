"""The Stripe SDK, and the one call that proves it is wired up (D7).

No FastAPI, by the same rule `api/services/` follows: D8's webhook handler and
D9's agent tools reach payments outside any HTTP request, where an
`HTTPException` would have nobody to catch it.

**The client is built lazily and is not the module-level `stripe.api_key`.**
Two reasons, and they are different problems. Setting `stripe.api_key` mutates
process-global state, so every caller in the process shares one configuration
and a test that changes it leaks into the next; `StripeClient` is an object,
which makes the configuration explicit and disposable. Building it lazily is
what lets this module be imported with no key at all — `api/main.py` will
import a checkout router on D7 step 3, and a cart API that refuses to start
because payments are unconfigured would be the wrong failure. The key is
required at the moment somebody actually wants to charge, and `MissingStripeKey`
says so in a sentence.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import stripe

from shopagent.config import get_settings

# Pinned, never left to the default. Stripe advances the default API version
# per account and per signup date, so an unpinned client means the same code
# returns a differently shaped object one morning with nothing in this repo
# having changed — the failure would look like a bug in our parsing. This is
# the version stripe-python 15.5.0 is generated against, so the SDK's own
# types and this string agree.
#
#   pinned 2026-08-27, against stripe-python 15.5.0
#
# Changing it is a deliberate act with a changelog to read first:
# https://docs.stripe.com/upgrades
STRIPE_API_VERSION = "2026-07-29.dahlia"

# Two retries, not zero and not five. Stripe's own guidance is to retry, and
# the SDK adds an idempotency key to every retried request itself, so a retried
# create cannot double-charge. Two is chosen for who is waiting: a checkout
# session is built inside a request a shopper is sitting in front of, and a
# third and fourth attempt against an outage that is not transient only turns a
# fast error into a slow one. Reads — like `retrieve_account` below — are
# naturally idempotent and would tolerate more; there is no reason to configure
# them separately.
MAX_NETWORK_RETRIES = 2


class MissingStripeKey(RuntimeError):
    """Raised when something needs Stripe and no key is configured.

    A distinct type rather than a bare `RuntimeError` because D7 step 3 has to
    turn this into a specific HTTP status — a checkout that cannot be built
    because the server is unconfigured is not the shopper's fault and must not
    be reported as if it were.
    """


@lru_cache(maxsize=1)
def get_client() -> stripe.StripeClient:
    """Return this process's Stripe client, built on first use.

    Cached the same way `get_settings()` and `get_engine()` are: the client
    owns an HTTP connection pool, and a second one would quietly double the
    connections held against Stripe. Tests drop it with
    `get_client.cache_clear()`.

    Raises `MissingStripeKey` rather than building a client with `None`, which
    would fail later with Stripe's own authentication error — accurate, but it
    describes a rejected credential rather than an absent one, and those have
    different fixes.
    """
    key = get_settings().stripe_secret_key

    if not key or not key.strip():
        raise MissingStripeKey(
            "STRIPE_SECRET_KEY is not set, so this operation cannot reach "
            "Stripe. Add a test-mode key (sk_test_...) to .env — see "
            ".env.example. The cart and order API works without it; only "
            "payments do not."
        )

    return stripe.StripeClient(
        api_key=key,
        stripe_version=STRIPE_API_VERSION,
        max_network_retries=MAX_NETWORK_RETRIES,
    )


def retrieve_account() -> Any:
    """Fetch the Stripe account this key belongs to.

    Half of the connectivity check: it answers "whose key is this", which is
    what makes a wrong-account mistake visible. It reads, it creates nothing,
    and it is safe to run at any time.

    It does **not** answer whether the key is a test key. `GET /v1/account`
    returns no `livemode` field — verified against this SDK and API version,
    where the object carries `charges_enabled`, `details_submitted` and
    friends and nothing else. `in_test_mode()` below is the function for that.
    """
    # `retrieve_current`, not `retrieve` — the latter takes an account id and
    # is for platforms reading their connected accounts. This is `GET
    # /v1/account`, "whoever this key belongs to", which is the question a
    # connectivity check is actually asking.
    #
    # Reached through the `v1` namespace: `client.accounts` still works but is
    # deprecated in stripe-python 15, and the warning is worth heeding now
    # rather than at the point where it becomes an error.
    return get_client().v1.accounts.retrieve_current()


@lru_cache(maxsize=1)
def configured_account_id() -> str:
    """The Stripe account this process's key belongs to, fetched once.

    Cached because the answer cannot change while the process runs: the key is
    read at configuration time and an account id is not something Stripe
    reassigns. One network call per process, and only if something asks — see
    `api/services/events.py`, which asks only for events that carry an
    `account` field.

    `lru_cache` does not cache exceptions, which is the behaviour wanted here
    rather than a detail to work around: a failed lookup is a transient Stripe
    problem, and caching it would mean one bad moment at startup disables the
    check for the life of the process.
    """
    return retrieve_account().id


def in_test_mode() -> bool:
    """Whether this key is operating against Stripe's test data.

    The second layer under the `sk_test_` prefix check in `config.py`, and the
    one that cannot be fooled: the prefix is a string this repo compares, while
    `livemode` is Stripe's own answer about the request it just served.

    Reads the balance rather than the account, because `GET /v1/account` does
    not carry `livemode` and `GET /v1/balance` does. Balance is the right
    second choice on its own merits — it always exists, it is read-only, it
    needs no arguments, and it is what Stripe's own documentation reaches for
    to prove a key works.
    """
    balance = get_client().v1.balance.retrieve()
    return balance.livemode is False


# --- catalog sync (D7 step 2) --------------------------------------------
#
# Everything below writes Products and Prices into Stripe so the catalog is
# visible in the dashboard and the Products/Prices API gets exercised.
#
# **No part of the checkout reads these objects.** D7 step 3 builds
# `line_items` from the `order_items` snapshot via `price_data`, because a
# Stripe Price id would be a second source of truth for a number D6 already
# froze at order time — the shopper would be charged Stripe's amount while
# `orders.total_amount_cents` claimed another, and the two would diverge
# silently the first time a local price changed without a re-sync. The sync is
# a learning deliverable and a dashboard convenience, not a billing path.


def create_product(
    *, name: str, description: str, sku_group: str, idempotency_key: str
) -> Any:
    """Create a Stripe Product for one local product.

    `idempotency_key` is not optional here and is derived from the local row
    rather than generated: it is what makes a script that died halfway safe to
    re-run. Stripe replays the original response for 24 hours instead of
    creating a second Product, which covers exactly the window this script
    cannot — the object existed in Stripe but its id never reached our
    database.

    `metadata.sku_group` carries the local identity so a row in the dashboard
    can be traced back here without a lookup table.
    """
    return get_client().v1.products.create(
        params={
            "name": name,
            "description": description,
            "metadata": {"sku_group": sku_group, "source": "shopagent-catalog"},
        },
        options={"idempotency_key": idempotency_key},
    )


def create_price(
    *,
    product_id: str,
    unit_amount_cents: int,
    currency: str,
    sku: str,
    idempotency_key: str,
) -> Any:
    """Create a Stripe Price under a Product.

    `unit_amount` takes `amount_cents` unchanged. This is the payoff of D3's
    decision to store money as an integer of minor units: Stripe wants the
    smallest unit, we have the smallest unit, and there is no conversion here
    to get wrong.
    """
    return get_client().v1.prices.create(
        params={
            "product": product_id,
            "unit_amount": unit_amount_cents,
            "currency": currency,
            "metadata": {"sku": sku, "source": "shopagent-catalog"},
        },
        options={"idempotency_key": idempotency_key},
    )


def archive_product(product_id: str) -> Any:
    """Deactivate a Product. Stripe has no delete for one that has a Price.

    Archiving is the supported way to retire a Stripe object: `active=false`
    hides it from the dashboard's default view and from new purchases while
    leaving it readable, because anything that was ever bought has to stay
    resolvable. Used by the `stripe`-marked test to clean up after itself.
    """
    return get_client().v1.products.update(product_id, params={"active": False})


def archive_price(price_id: str) -> Any:
    """Deactivate a Price. Prices cannot be deleted at all, only archived."""
    return get_client().v1.prices.update(price_id, params={"active": False})


def list_prices(limit: int = 100) -> list[Any]:
    """Every Price on the account, following pagination.

    Used by the sync to answer "has anything drifted" in one or two round trips
    rather than one retrieve per variant.
    """
    return list(get_client().v1.prices.list(params={"limit": limit}).auto_paging_iter())


# --- checkout (D7 step 3) ------------------------------------------------


def create_checkout_session(
    *,
    line_items: list[Any],
    metadata: dict[str, str],
    client_reference_id: str,
    success_url: str,
    cancel_url: str,
    buyer: dict[str, str] | None = None,
    payment_intent_metadata: dict[str, str] | None = None,
) -> Any:
    """Create a Checkout Session. The SDK call and nothing else.

    Every decision about *what* goes in here — that the lines come from the
    order snapshot, that `metadata.order_id` is mandatory, that the totals must
    agree — lives in `payments/checkout.py`. This function exists so that layer
    can be tested by replacing one name.

    `mode="payment"` because these are one-off purchases. `subscription` and
    `setup` are the other two, and neither describes a cart.

    No idempotency key, unlike the catalog sync. Sessions are deduplicated by
    `orders.stripe_checkout_session_id`, which is durable, where a key expires
    after 24 hours — and a shopper coming back the next day to an order that is
    still pending should reach the session that already exists, not a replay.
    """
    # `buyer` carries at most one of `customer` / `customer_email`; Stripe
    # rejects a session with both. Deciding which is the caller's job — see
    # `payments/checkout.py`.
    return get_client().v1.checkout.sessions.create(
        params={
            "mode": "payment",
            "line_items": line_items,
            "metadata": metadata,
            "client_reference_id": client_reference_id,
            "success_url": success_url,
            "cancel_url": cancel_url,
            # Copied onto the PaymentIntent the session will create. Metadata
            # does not flow down the object chain on its own: a session's
            # `metadata` stays on the session, and the PaymentIntent and Charge
            # it produces arrive with `metadata: {}`. Verified against a real
            # payment. Without this, a webhook on `payment_intent.succeeded`
            # receives a successful charge it cannot attribute to an order.
            "payment_intent_data": {"metadata": payment_intent_metadata or {}},
            **(buyer or {}),
        }
    )


def retrieve_checkout_session(session_id: str) -> Any:
    """Read a Checkout Session back, to see whether it can still be paid."""
    return get_client().v1.checkout.sessions.retrieve(session_id)


def expire_checkout_session(session_id: str) -> Any:
    """Close an open session early.

    The closest thing to a delete: Stripe keeps Checkout Sessions permanently
    and offers no way to remove one, so `expire` is what a test can do to stop
    a session it created from remaining payable. A session that is already
    complete or expired cannot be expired again.
    """
    return get_client().v1.checkout.sessions.expire(session_id)


# --- customers (D7 step 4) -----------------------------------------------


def create_customer(
    *, email: str, name: str | None = None, idempotency_key: str | None = None
) -> Any:
    """Create a Stripe Customer.

    Stripe stores as many Customers with the same email as it is asked to — it
    treats the field as data, not identity — so deduplication is
    `payments/customers.py`'s job and is done by looking first.

    The key closes the window that looking cannot. Two first orders for the
    same address arriving together both find nothing and both create, because
    look-then-create is check-then-act; a key derived from the address makes
    Stripe answer the second with the first one's Customer. It covers 24 hours,
    which is the concurrent case rather than the returning-shopper case — that
    one is answered by the local lookup, which is durable.
    """
    params: dict[str, Any] = {"email": email}
    if name:
        params["name"] = name

    options = {"idempotency_key": idempotency_key} if idempotency_key else None
    return get_client().v1.customers.create(params=params, options=options)


def find_customers_by_email(email: str, limit: int = 1) -> list[Any]:
    """Customers with exactly this email.

    `customers.list(email=...)` rather than `customers.search(...)`. Search is
    backed by an index that lags writes by up to a minute, so a customer
    created and then searched for in the same script is frequently not found —
    which is precisely the case deduplication has to get right. `list` filters
    on the field directly and is immediately consistent.
    """
    return list(get_client().v1.customers.list(params={"email": email, "limit": limit}))


def delete_customer(customer_id: str) -> Any:
    """Delete a Customer. Unlike Products, Prices and Sessions, this one is real.

    Stripe permits deleting a Customer outright, which is what lets the
    `stripe`-marked tests leave nothing behind.
    """
    return get_client().v1.customers.delete(customer_id)


# --- webhooks (D8 step 1) ------------------------------------------------


# Re-exported so a caller can name the failure without importing the SDK. The
# router has to tell "this was not signed by Stripe" (400) apart from every
# other exception, and `except Exception` there would swallow bugs in our own
# code and answer 400 to them — a response Stripe reads as "malformed", never
# retries, and which therefore loses the delivery for good.
SignatureVerificationError = stripe.SignatureVerificationError

# Re-exported for the same reason: `events.py` has to tell "Stripe refused this
# request" apart from "Stripe could not be reached", because the first is
# permanent and the second is worth a retry — and `payments/stripe_svc.py` is
# the only module allowed to import `stripe`.
InvalidRequestError = stripe.InvalidRequestError


# What Stripe allows between the timestamp it signed and the moment we verify.
# Five minutes, and it is the SDK's own default rather than a number chosen
# here — named so the tolerance is a value this repo can assert on instead of
# one buried in a call. Its job is to stop a captured delivery being replayed
# later: the signature stays valid forever, the timestamp inside it does not.
#
# Note what it does *not* do — `verify_header` compares `timestamp < now -
# tolerance` and nothing else, so a timestamp in the future passes however far
# ahead it is. That is Stripe's behaviour, not an oversight here, and it is
# harmless for the same reason the rest works: forging a future timestamp still
# needs the signing secret.
WEBHOOK_TOLERANCE_SECONDS = stripe.Webhook.DEFAULT_TOLERANCE


def construct_webhook_event(payload: bytes, sig_header: str, secret: str) -> Any:
    """Verify a webhook's signature and return the event it carries.

    One SDK call, wrapped for the reason every other function in this module
    is: `stripe` is imported here and nowhere else, so the router that handles
    deliveries can be exercised by replacing one name. It holds no decision —
    what to do with an event, and which failure becomes which status code,
    belongs to the caller.

    `payload` must be the bytes that arrived. Verification recomputes an HMAC
    over `timestamp + "." + body`, so a body that was parsed and re-serialised
    is a different string — different whitespace, different key order — and
    signs to a different digest. It would fail some of the time rather than
    all of the time, which is worse than failing outright.

    Raises `stripe.SignatureVerificationError` for a bad, stale or unparseable
    signature header, and `ValueError` (`json.JSONDecodeError`) when the body
    signed correctly but is not JSON.
    """
    return stripe.Webhook.construct_event(payload, sig_header, secret)


# --- refunds (D8 step 4) -------------------------------------------------


def create_refund(
    payment_intent_id: str, *, idempotency_key: str | None = None
) -> Any:
    """Refund a PaymentIntent in full.

    No `amount` parameter, and that absence is the interface. Stripe supports
    partial refunds; this project has nowhere to put one — `orders.status` has
    a `refunded` value and no notion of "partly refunded", and inventing one
    from a route argument would produce orders in a state nothing else can
    read. A partial refund issued from the dashboard still *arrives* here as a
    `charge.refunded` event, and `api/services/events.py` refuses to act on it
    and says so loudly. This is the same boundary drawn from the other side.

    The idempotency key is what stops a double refund, and it is doing real
    work rather than being defensive. `refund_order` locks the order row, but
    the row does not change — the status stays `paid` until the webhook
    arrives — so two requests seconds apart both read a refundable order and
    both reach this call. A key derived from the order makes the second one
    return Stripe's record of the first instead of moving money twice.

    Twenty-four hours is the window Stripe honours, which is exactly the gap
    that needs covering: after it, the webhook has long since moved the order
    to `refunded` and the lifecycle refuses a second attempt outright.
    """
    return get_client().v1.refunds.create(
        params={"payment_intent": payment_intent_id},
        options={"idempotency_key": idempotency_key} if idempotency_key else None,
    )
