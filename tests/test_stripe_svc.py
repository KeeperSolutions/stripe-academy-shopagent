"""Tests for shopagent.payments.stripe_svc (D7, step 1).

Almost all of this is offline. Building a `StripeClient` makes no network call,
so the pin, the retry setting and the missing-key path are all answerable
without an account — and the autouse guard in `tests/conftest.py` blocks the
request funnel rather than the constructor precisely so they can be.

The single `stripe`-marked test is the one claim no fake can settle: that the
configured key reaches a real account, and that the account is not live.
"""

from __future__ import annotations

import pytest
import stripe

from shopagent.config import Settings, get_settings
from shopagent.payments import stripe_svc
from shopagent.payments.stripe_svc import (
    MAX_NETWORK_RETRIES,
    STRIPE_API_VERSION,
    MissingStripeKey,
    get_client,
)


@pytest.fixture(autouse=True)
def _fresh_client_cache():
    """`get_client` is `lru_cache`d, so a test that changes settings must clear it."""
    get_client.cache_clear()
    yield
    get_client.cache_clear()


def _with_key(monkeypatch, key: str | None) -> None:
    patched = get_settings().model_copy(update={"stripe_secret_key": key})
    monkeypatch.setattr(stripe_svc, "get_settings", lambda: patched)


def _options(client: stripe.StripeClient):
    """What the client will actually send, read off the client itself."""
    return client._requestor._options


# --- a live key never gets that far --------------------------------------


@pytest.mark.parametrize("key", ["sk_live_abc123", "rk_live_abc123"])
def test_a_live_key_is_refused_by_configuration(key):
    """The cheap layer, and the one that works offline.

    Test mode is the only mode this project runs in. A live key would charge
    real cards against invented prices from a fictional seed catalog, and
    configuration time is the last moment the mistake is still free.
    """
    with pytest.raises(ValueError) as excinfo:
        Settings(stripe_secret_key=key)

    message = str(excinfo.value)
    assert "live key" in message
    assert "sk_test_" in message, "the message must say what to use instead"


def test_a_test_key_is_accepted():
    assert Settings(stripe_secret_key="sk_test_abc123").stripe_secret_key


def test_no_key_at_all_is_accepted_by_configuration():
    """Payments are one part of the system, not a precondition for the rest.

    `shopagent_api_key` gates every request and the API refuses to start
    without it. Stripe does not: a cart that cannot be browsed because payments
    are unconfigured would be the wrong failure.
    """
    assert Settings(stripe_secret_key=None).stripe_secret_key is None


# --- a missing key fails on use, not on import ---------------------------


def test_importing_the_module_needs_no_key(monkeypatch):
    """If this ever fails, `api/main.py` stops booting without Stripe.

    Loaded as a fresh module object under its own name rather than reloaded in
    place. `importlib.reload` re-executes the module and rebinds
    `MissingStripeKey` to a *new* class, which this file has already imported
    the old one of — every `pytest.raises(MissingStripeKey)` below then stops
    matching, for reasons that look nothing like the cause. Found the hard way.
    """
    import importlib.util

    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)

    spec = importlib.util.spec_from_file_location(
        "shopagent_stripe_svc_probe", stripe_svc.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.STRIPE_API_VERSION == STRIPE_API_VERSION


@pytest.mark.parametrize("key", [None, "", "   "])
def test_using_stripe_without_a_key_raises_a_named_error(monkeypatch, key):
    _with_key(monkeypatch, key)

    with pytest.raises(MissingStripeKey) as excinfo:
        get_client()

    message = str(excinfo.value)
    assert "STRIPE_SECRET_KEY" in message
    # Written for whoever is reading a traceback at 2am: what is missing, where
    # it goes, and what still works without it.
    assert ".env" in message
    assert "cart and order API works without it" in message


def test_the_missing_key_error_is_its_own_type(monkeypatch):
    """D7 step 3 maps this to a status code; an unconfigured server is not a
    shopper's bad request and must not be reported as one."""
    _with_key(monkeypatch, None)

    with pytest.raises(RuntimeError):  # the base class, so the subtype is real
        get_client()
    assert issubclass(MissingStripeKey, RuntimeError)


# --- the client is configured, and that is read back off the client ------


def test_the_client_sends_the_pinned_api_version(monkeypatch):
    """Read from the client, not from the constant.

    Asserting `STRIPE_API_VERSION == STRIPE_API_VERSION` would pass against a
    client that was never given the pin at all.
    """
    _with_key(monkeypatch, "sk_test_abc123")

    assert _options(get_client()).stripe_version == STRIPE_API_VERSION


def test_the_pin_matches_the_sdk_it_was_written_against():
    """The assertion with actual teeth.

    An unpinned client currently falls back to exactly this string, so the test
    above cannot by itself prove the pin is being passed — the two coincide
    today. What does not coincide is the future: upgrading `stripe` moves the
    SDK's generated version while `STRIPE_API_VERSION` stays put, and the
    request would then be made against a version the installed types no longer
    describe. Failing here forces the upgrade to be a decision with a changelog
    read first, rather than a silent divergence.
    """
    assert STRIPE_API_VERSION == stripe.api_version, (
        f"stripe-python is generated against {stripe.api_version}, but this "
        f"repo pins {STRIPE_API_VERSION}. Re-pin deliberately after reading "
        "https://docs.stripe.com/upgrades, or hold the SDK back."
    )


def test_the_client_sets_max_network_retries(monkeypatch):
    """This one does distinguish configured from default: unpinned is `None`."""
    _with_key(monkeypatch, "sk_test_abc123")

    assert _options(get_client()).max_network_retries == MAX_NETWORK_RETRIES
    assert stripe.StripeClient("sk_test_x")._requestor._options.max_network_retries is None


def test_the_client_is_cached(monkeypatch):
    """One client per process — a second owns a second connection pool."""
    _with_key(monkeypatch, "sk_test_abc123")

    assert get_client() is get_client()


def test_the_module_does_not_set_the_global_api_key(monkeypatch):
    """`stripe.api_key` is process-global state.

    Setting it would make every caller in the process share one configuration
    and let a test leak into the next. The client is an object instead.
    """
    _with_key(monkeypatch, "sk_test_abc123")
    monkeypatch.setattr(stripe, "api_key", None)

    get_client()

    assert stripe.api_key is None


# --- the one call that actually reaches Stripe ---------------------------


@pytest.mark.stripe
def test_the_key_reaches_a_real_test_mode_account():
    """Connectivity, and the second layer of the live-key guard.

    The `sk_test_` prefix check in `config.py` is a string comparison and can
    be fooled by a key that merely looks right. `livemode` comes from Stripe
    itself and cannot. Nothing here creates an object.
    """
    if not get_settings().stripe_secret_key:
        pytest.skip("STRIPE_SECRET_KEY is not set; add a test-mode key to .env")

    account = stripe_svc.retrieve_account()

    assert account.id.startswith("acct_")
    assert account.object == "account"
    # Not `account.livemode` — `GET /v1/account` does not return that field.
    # Asserting it raised AttributeError the first time this ran, which is the
    # kind of thing only a real call finds.
    assert "livemode" not in account._data

    # The claim this test exists for, from the object that does carry it.
    assert stripe_svc.in_test_mode(), (
        "this key is operating in LIVE mode. Stop and replace it with a "
        "test-mode key before running anything else in this project."
    )
