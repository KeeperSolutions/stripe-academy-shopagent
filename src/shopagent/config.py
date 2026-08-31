"""Centralised configuration (D1).

The only place in the project allowed to read the environment — nothing else
reads env variables directly. Every new variable is added here as a typed
field, then to `.env.example`, and only then used in code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Optional

from pydantic import BeforeValidator, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py lives in src/shopagent/, .env sits at the repo root. An absolute
# path means configuration works no matter which directory the app starts from.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _empty_to_none(value: object) -> object:
    """Treat a blank line in .env (`STRIPE_SECRET_KEY=`) as "not set".

    Without this, pydantic yields `""` instead of `None`, so `is None` checks
    in payments/ and obs/ would quietly fail to do what they appear to do.
    """
    if isinstance(value, str) and not value.strip():
        return None
    return value


OptionalStr = Annotated[Optional[str], BeforeValidator(_empty_to_none)]


class Settings(BaseSettings):
    """Every env variable in the project, typed and defaulted."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (D1-D3, D9) ---
    # min_length=1 because an empty OPENAI_API_KEY= in .env would otherwise
    # pass validation, and the app would only fail on the first API call with
    # a useless error message.
    openai_api_key: str = Field(min_length=1)
    openai_model: str = "gpt-5.6-luna"
    embedding_model: str = "text-embedding-3-small"
    # Sent only alongside function tools. gpt-5.6-luna rejects tools on
    # /v1/chat/completions with 400 unless this is 'none' ("use /v1/responses
    # or set reasoning_effort to 'none'"), and this project stays on Chat
    # Completions on purpose. Blank means the parameter is not sent at all,
    # which is what a model that does not know it needs (e.g. gpt-4o-mini).
    openai_reasoning_effort: OptionalStr = "none"

    # --- MCP (D4-D5) ---
    # The off switch for the catalog. The agent loop registers the MCP tools
    # alongside the local ones when this is true; set it false and the same
    # binary runs with the two local tools only, which is what makes "the
    # catalog answers are coming from MCP" a claim that can be tested rather
    # than asserted.
    mcp_catalog_enabled: bool = True
    # The catalog server logs every tool call with its arguments, and `query`
    # is the one argument that is free text a shopper wrote. Default true, so
    # the safe setting is the one nobody has to remember to type: a log that
    # over-redacts costs a debugging session, a log that under-redacts cannot
    # be un-leaked. Set false deliberately, on a developer's own machine, when
    # reading back what the model actually searched for.
    mcp_log_redact_query: bool = True
    # Whether the catalog server advertises its `ping` diagnostic in
    # `tools/list`. Off, because the tool list is what the model reads to
    # decide what it can do, and a name in it that means nothing commercially
    # is a name it has to rule out on every turn. `ping` keeps its value for a
    # person — it separates "the server is unreachable" from "the catalog is
    # broken" — and this is the switch that hands it back, on the machine of
    # whoever is debugging. Deliberately a server-side switch rather than a
    # filter in `mcp_client/`: the client registers whatever the server lists,
    # which is the property D5 exists to demonstrate, and a name check there
    # would be this project's own client knowing about this project's own
    # server.
    mcp_expose_ping: bool = False

    # --- Infrastructure (D3, D6) ---
    database_url: str = (
        "postgresql+psycopg://shopagent:shopagent@localhost:5432/shopagent"
    )
    app_base_url: str = "http://localhost:8000"

    # --- Commerce (D6-D7) ---
    # The shop's currency, ISO-4217 and lowercase, which is the form Stripe
    # sends and expects. Every price in the catalog is seeded in it and every
    # cart, order and Checkout Session is denominated in it.
    #
    # Not the currency this project's *costs* are in: OpenAI bills in USD and
    # `llm/usage.py` reports dollars, which is a real exchange rate away from
    # this and deliberately unrelated. Two currencies, one of them invented.
    currency: str = "eur"
    # Where Stripe sends the shopper after Checkout. Left as None so the
    # fallbacks below can derive them from `app_base_url` — one place to change
    # when the app moves, instead of three that drift apart. Set them
    # explicitly the moment the success page stops living on this host.
    checkout_success_url: OptionalStr = None
    checkout_cancel_url: OptionalStr = None
    # Required, with no default, for the same reason `openai_api_key` is: it is
    # the API's only authentication secret. A default here would be a published
    # one — every deployment that forgot the variable would be protected by a
    # string anybody can read in this file, and would look correctly configured
    # while doing it. `min_length=1` additionally rejects a blank
    # `SHOPAGENT_API_KEY=` in .env, which would otherwise validate as the empty
    # string and have `require_api_key` compare every request against nothing.
    # Both failures happen when configuration is read, not at the first request
    # that should have been refused.
    shopagent_api_key: str = Field(min_length=1)
    # Where the agent's commerce tools (D9) reach that API. Deliberately not
    # `app_base_url`, which is the URL a *browser* is sent to — Stripe puts it
    # in a redirect, so it has to be public, and the day this runs behind ngrok
    # or in compose the two stop being the same string: the shopper returns to
    # https://shopagent.example while the agent process still has to call
    # http://api:8000. One field serving both would make that a choice between
    # a redirect the shopper cannot follow and a call the agent cannot make.
    # Same default, because on one machine they do coincide.
    commerce_api_base_url: str = "http://localhost:8000"

    # --- Stripe (D7-D8) ---
    # Optional, and deliberately not treated the way `shopagent_api_key` is.
    # That key gates every request, so the API refuses to start without it;
    # payments are one part of the system rather than a precondition for the
    # rest, and a cart that cannot be browsed because Stripe is unconfigured
    # would be the wrong failure. A missing key therefore surfaces at the
    # moment something needs it — see `payments/stripe_svc.py` — not at import.
    stripe_secret_key: OptionalStr = None
    # The signing secret `stripe listen` prints, or the one a dashboard
    # endpoint shows. Optional for the same reason the key above is: a shop
    # that cannot receive webhooks is a shop with a gap in its payment flow,
    # not a broken process — `POST /webhooks/stripe` answers 503 when this is
    # absent, and every other route is unaffected.
    stripe_webhook_secret: OptionalStr = None

    @field_validator("stripe_secret_key")
    @classmethod
    def _refuse_a_live_key(cls, value: str | None) -> str | None:
        """Test mode is the only mode this project runs in.

        A live key here would charge real cards against a real balance, and no
        part of this repo is built to be trusted with that: the seed catalog is
        fiction, the prices are invented, and D8 will replay webhooks by hand.
        Refusing at configuration time is the one place the mistake is still
        free — after the first charge it is a refund, a statement line and a
        conversation with somebody's bank.

        Deliberately a prefix check rather than a call to Stripe. It costs no
        network, works offline, and answers before the SDK is ever built.
        `livemode` on the account is the second layer, asserted by the one test
        that actually talks to Stripe.
        """
        if value is None:
            return None
        if value.startswith("sk_live_") or value.startswith("rk_live_"):
            raise ValueError(
                "STRIPE_SECRET_KEY is a live key. This project runs in test "
                "mode only — a live key here charges real cards. Use the key "
                "beginning sk_test_ from the Stripe dashboard's test mode."
            )
        return value

    @field_validator("stripe_webhook_secret")
    @classmethod
    def _refuse_a_secret_that_is_not_a_signing_secret(cls, value: str | None) -> str | None:
        """A webhook secret that is not one fails in the quietest way available.

        Every signing secret Stripe issues begins `whsec_` — the one
        `stripe listen` prints and the one a dashboard endpoint shows. The
        mistake this catches is pasting the API key into the wrong line, which
        is easy because both live in the same block of `.env` and both are
        opaque strings.

        Without the check the consequence is invisible from the server's side:
        verification fails, every delivery is answered 400, Stripe retries and
        eventually gives up, and no order is ever marked paid. Nothing logs an
        error that names the cause, because from the endpoint's point of view
        it merely received a stream of badly signed requests.

        Refusing here is deliberately harsher than a missing secret, which
        leaves the app running and answers 503 at the endpoint. Absent is a
        state a developer chose — payments are one part of this system and the
        cart works without them. Present-and-wrong is a typo, and the last
        moment it is free is configuration time. Same reasoning as the live-key
        check above, and the same prefix-not-network mechanism.
        """
        if value is None:
            return None
        if not value.startswith("whsec_"):
            raise ValueError(
                "STRIPE_WEBHOOK_SECRET does not look like a signing secret. "
                "Stripe's begin with whsec_ — this is probably the API key, or "
                "an endpoint id. Run `stripe listen --forward-to "
                "localhost:8000/webhooks/stripe` and copy the whsec_... it "
                "prints, or read it from the endpoint's page in the dashboard."
            )
        return value

    # --- Langfuse (D10) --- same, only populated on D10
    langfuse_public_key: OptionalStr = None
    langfuse_secret_key: OptionalStr = None
    langfuse_host: str = "https://cloud.langfuse.com"


    @property
    def success_url(self) -> str:
        """Where Checkout returns a shopper who paid.

        Carries `{CHECKOUT_SESSION_ID}`, which Stripe substitutes on redirect.
        That is what lets the success page look up what actually happened
        instead of taking the redirect as proof — the redirect is a URL anybody
        can open, and D8's webhook is the only thing that may flip an order to
        `paid`.
        """
        if self.checkout_success_url:
            return self.checkout_success_url
        return f"{self.app_base_url}/checkout/success?session_id={{CHECKOUT_SESSION_ID}}"

    @property
    def cancel_url(self) -> str:
        """Where Checkout returns a shopper who backed out."""
        if self.checkout_cancel_url:
            return self.checkout_cancel_url
        return f"{self.app_base_url}/checkout/cancel"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return this process's configuration.

    Cached: `.env` is read and validated once, and every caller shares the same
    object. Tests can clear it with `get_settings.cache_clear()`.
    """
    return Settings()
