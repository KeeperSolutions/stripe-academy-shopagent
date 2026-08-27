"""Centralised configuration (D1).

The only place in the project allowed to read the environment — nothing else
reads env variables directly. Every new variable is added here as a typed
field, then to `.env.example`, and only then used in code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Optional

from pydantic import BeforeValidator, Field
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

    # --- Infrastructure (D3, D6) ---
    database_url: str = (
        "postgresql+psycopg://shopagent:shopagent@localhost:5432/shopagent"
    )
    app_base_url: str = "http://localhost:8000"

    # --- Commerce (D6-D7) ---
    currency: str = "usd"
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

    # --- Stripe (D7-D8) --- no values yet, so these must be allowed to be None
    stripe_secret_key: OptionalStr = None
    stripe_webhook_secret: OptionalStr = None

    # --- Langfuse (D10) --- same, only populated on D10
    langfuse_public_key: OptionalStr = None
    langfuse_secret_key: OptionalStr = None
    langfuse_host: str = "https://cloud.langfuse.com"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return this process's configuration.

    Cached: `.env` is read and validated once, and every caller shares the same
    object. Tests can clear it with `get_settings.cache_clear()`.
    """
    return Settings()
