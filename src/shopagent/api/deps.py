"""Shared dependencies for the API — today, authentication (D6).

One API key, sent as `X-API-Key`. This is a training project with exactly one
client (the agent on D9), so a shared secret is the honest amount of auth: per
user credentials would be scaffolding around a concept the project is not
here to teach. What it does have to get right is *failing* correctly.
"""

from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from shopagent.config import get_settings

API_KEY_HEADER = "X-API-Key"

# auto_error=False on purpose, and this is the whole reason the scheme is
# declared rather than the header read by hand. With auto_error=True FastAPI
# raises 403 when the header is missing entirely, and 403 is the wrong answer:
# it means "I know who you are and you may not", when what happened is "you
# did not say who you are". That is 401, and it is the code a client uses to
# decide whether retrying with a credential is worth it. Declaring the scheme
# is also what puts the Authorize button in /docs.
_api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


class MissingAPIKey(RuntimeError):
    """Raised at import time when the server has no key to check against."""


def configured_api_key() -> str:
    """Return the configured key, or refuse to hand one back.

    `Settings` already rejects an empty `SHOPAGENT_API_KEY=` with
    `min_length=1`, so the usual blank-in-.env case never reaches here. This
    catches the two it cannot: a key that is only whitespace, which passes a
    length check and authenticates nobody, and a `Settings` built in code
    rather than read from the environment.

    Called at module scope in `api/main.py`, so the process dies while uvicorn
    is importing the app rather than on the first request. A server that starts
    happily and then rejects everything — or worse, accepts everything — is a
    failure that looks like uptime.
    """
    key = get_settings().shopagent_api_key
    if not key or not key.strip():
        raise MissingAPIKey(
            "SHOPAGENT_API_KEY is empty. The API refuses to start without a key "
            "to check requests against. Set it in .env — see .env.example."
        )
    return key


def require_api_key(presented: str | None = Depends(_api_key_header)) -> None:
    """Refuse the request unless it carries the configured key.

    Attached to routers rather than to individual routes, so a route added
    later is protected by where it is mounted rather than by whoever added it
    remembering a decorator. See `api/main.py`.

    `compare_digest` rather than `==` because `==` on strings returns as soon
    as two bytes differ, and how long that takes is a measurement of how much
    of the key was right. That leaks the secret one byte at a time to anyone
    patient enough to time the responses.
    """
    expected = configured_api_key()

    if presented is None or not secrets.compare_digest(presented, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            # Named for whoever is holding a curl command, not for a browser:
            # the header is the thing they have to add.
            detail=f"a valid {API_KEY_HEADER} header is required",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )
