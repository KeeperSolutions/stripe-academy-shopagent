"""The one place the agent's tools speak HTTP (D9).

`httpx` is imported here and nowhere else under `tools/`, for the reason
`openai` lives only in `llm/client.py` and `stripe` only in
`payments/stripe_svc.py`: five tools each building their own request would be
five places to change a header, a timeout or a base URL, and the first symptom
of a drift is one tool authenticating and another not.

The other half of this module is the failure taxonomy. HTTP is a new source of
breakage on D9 — the API is a separate process now — and the tool layer above
has to say something different to the model for each kind. A refused
connection is not a customer's problem and no retry fixes it; a 409 is prose
the API already wrote for a reader and must reach the model untouched; a 401
is our own misconfiguration and saying "out of stock" about it would be a lie.
So the transport raises one exception per kind and writes no user-facing text
of its own: the wording belongs to `tools/commerce.py`, which knows which tool
was being called.
"""

from __future__ import annotations

from typing import Any

import httpx

from shopagent.config import get_settings

# Two numbers rather than one, because they are two different failures.
#
# Connect: the commerce API is a local process (or, deployed, one on the other
# side of a load balancer that answers immediately or not at all). A TCP
# connect that has not completed in two seconds is not going to — the common
# case by far is that uvicorn is simply not running, where the OS refuses at
# once and this ceiling never applies. What it bounds is the pathological case,
# a host that accepts packets and never answers.
#
# Read: ten seconds, because one of these calls is not local. `POST
# /orders/{id}/checkout` creates a Stripe Checkout Session, which is a round
# trip to Stripe from inside the request — measured at roughly a second on D7,
# and Stripe's own timeout is longer than anything worth waiting for here.
#
# The ceiling matters because this blocks the agent loop: `run_tool_loop` is
# synchronous, a hung request freezes the conversation with no output, and the
# loop allows eight rounds. Twelve seconds a call is the worst case a person
# waits at the prompt, and it is deliberately shorter than the patience of
# whoever is typing.
CONNECT_TIMEOUT_SECONDS = 2.0
READ_TIMEOUT_SECONDS = 10.0


class CommerceAPIError(Exception):
    """Base for every way a call to the commerce API can fail."""


class CommerceAPIUnreachable(CommerceAPIError):
    """The request was never delivered: no connection was established.

    The distinction from `CommerceAPIInterrupted` below is about money, not
    about tidiness. This one is the only failure the layer above may describe
    as "nothing was charged", because it is the only one where nothing can have
    reached the API.
    """


class CommerceAPIInterrupted(CommerceAPIError):
    """The connection broke after the request went out, so the outcome is unknown.

    A read error, a write error, a protocol violation mid-exchange: the socket
    was open, the bytes may have arrived, and the API may have committed the
    write before the answer was lost. Telling the model "nothing was charged"
    here would be a guess presented as a fact — and on `create_checkout` the
    guess is wrong exactly when an order was placed. Raised in review on PR #9.
    """


class CommerceAPITimeout(CommerceAPIError):
    """A connection was made and the answer did not arrive in time.

    Same ambiguity as `CommerceAPIInterrupted` and a different cause, which is
    why it keeps its own class: the message the model gets can name the number
    of seconds, and that sentence would be false for a broken socket.
    """


class CommerceAPIRefused(CommerceAPIError):
    """The API understood the request and said no, in prose meant for a reader.

    404 and 409 both land here. `detail` is the API's own sentence — "cart is
    empty", "only 2 units of ... are available" — written for whoever reads it
    and therefore fit to hand to the model unchanged.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"{status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class CommerceAPIUnauthorized(CommerceAPIError):
    """The key was missing or wrong. Ours to fix, never the customer's."""


class CommerceAPIBroken(CommerceAPIError):
    """A 5xx, or an answer that is not the JSON this client was promised.

    Carries no detail on purpose: a 500 body is an internal message, and the
    layer above must not pass it to the model.
    """


class CommerceAPI:
    """A thin, authenticated client for the commerce API.

    `transport` exists so tests can answer requests inside the process; it is
    the seam the whole offline suite for these tools rests on, because a
    refused connection and a 401 are unreachable against a healthy local API.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        settings = get_settings()
        self._client = httpx.Client(
            base_url=base_url or settings.commerce_api_base_url,
            headers={"X-API-Key": api_key or settings.shopagent_api_key},
            timeout=httpx.Timeout(
                connect=CONNECT_TIMEOUT_SECONDS,
                read=READ_TIMEOUT_SECONDS,
                write=READ_TIMEOUT_SECONDS,
                pool=CONNECT_TIMEOUT_SECONDS,
            ),
            transport=transport,
        )

    def request(
        self, method: str, path: str, *, json: dict[str, Any] | None = None
    ) -> Any:
        """Make one call and return the decoded body, or raise a typed failure.

        Returns `None` for a 204, which is what `DELETE /cart/{id}/items/{id}`
        answers — a body-less success is still a success.
        """
        try:
            response = self._client.request(method, path, json=json)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            # Nothing was sent: no connection was established, or none was
            # available. These are the only failures that can honestly be
            # reported as "this did not go through".
            raise CommerceAPIUnreachable(str(exc)) from exc
        except httpx.TimeoutException as exc:
            # A read or write timeout. The request is out; the answer is not
            # back. Whether it took effect is unknown.
            raise CommerceAPITimeout(str(exc)) from exc
        except httpx.HTTPError as exc:
            # `ReadError`, `WriteError`, `RemoteProtocolError` and anything
            # else httpx raises once the exchange has started. Grouped with the
            # unknown outcome rather than with "unreachable", which is where
            # they used to land: a socket that broke mid-exchange may well have
            # delivered the request first, and an order placed behind a lost
            # answer is the case that makes the difference matter.
            raise CommerceAPIInterrupted(str(exc)) from exc

        return self._read(response)

    def _read(self, response: httpx.Response) -> Any:
        if response.status_code in (401, 403):
            raise CommerceAPIUnauthorized(f"{response.status_code} from the commerce API")
        if response.status_code >= 500:
            raise CommerceAPIBroken(f"{response.status_code} from the commerce API")
        if response.status_code >= 400:
            raise CommerceAPIRefused(response.status_code, _detail(response))

        if response.status_code == 204 or not response.content:
            return None

        try:
            return response.json()
        except ValueError as exc:
            # A 200 that is not JSON means something is answering on this port
            # that is not our API. Broken rather than refused: there is no
            # sentence in it worth showing anybody.
            raise CommerceAPIBroken("the commerce API returned a body that is not JSON") from exc

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "CommerceAPI":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _detail(response: httpx.Response) -> str:
    """FastAPI's `detail`, flattened to one sentence.

    A hand-raised `HTTPException` puts a string there, which is the case that
    matters — every 404 and 409 in `api/routers/` is a sentence somebody wrote
    for a reader. A 422 puts a list of field errors there instead, and those
    are rendered rather than dropped: the model can act on "quantity: Input
    should be greater than 0" and cannot act on "422".
    """
    try:
        body = response.json()
    except ValueError:
        return response.reason_phrase or f"HTTP {response.status_code}"

    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        parts = []
        for error in detail:
            if not isinstance(error, dict):
                continue
            field = ".".join(str(part) for part in error.get("loc", []) if part != "body")
            message = error.get("msg", "is invalid")
            parts.append(f"{field}: {message}" if field else message)
        if parts:
            return "; ".join(parts)
    return response.reason_phrase or f"HTTP {response.status_code}"
