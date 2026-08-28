"""The commerce API (D6).

Carts and orders over HTTP, while the catalog stays on MCP. The split is
deliberate rather than incidental: the agent on D9 reaches products through one
protocol and its basket through another, which is what makes the difference
between them something this project can point at.

    uvicorn shopagent.api.main:app --reload --port 8000

Routes arrive in steps 3 and 4. What is here is the shape they arrive into:
an app, an unauthenticated health check, and a rule about where authentication
is attached.
"""

from __future__ import annotations

import logging

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shopagent.api.deps import configured_api_key, require_api_key
from shopagent.api.routers import cart, checkout_pages, orders, webhooks

def configure_logging(level: int = logging.INFO) -> None:
    """Give this project's loggers somewhere to write when uvicorn is the host.

    D8's webhook logs every delivery before doing anything with it, which is
    the only trace of what arrived when something goes wrong — and without
    this it would go nowhere. Uvicorn installs handlers on its own `uvicorn.*`
    loggers and leaves the root logger bare, so an `INFO` record from
    `shopagent.*` propagates to a root with no handler and is dropped by the
    fallback, which only emits `WARNING` and above. The endpoint would work
    perfectly and appear to log nothing.

    Attached to the `shopagent` logger rather than through `basicConfig`,
    which configures the *root* logger and would therefore also start printing
    every library's records through this format. Guarded on `handlers`, so
    importing the app twice does not double every line, and so a process that
    has already configured logging deliberately — a test, or a deployment with
    its own dictConfig — keeps what it set up. `propagate` is left alone:
    `caplog` works by attaching to the root logger, and severing propagation
    here would leave every logging assertion in the suite passing vacuously.
    """
    package_logger = logging.getLogger("shopagent")
    if package_logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    package_logger.addHandler(handler)
    package_logger.setLevel(level)


configure_logging()

# Read at import, so a server with no usable key dies while uvicorn is loading
# the module instead of starting and refusing every request afterwards. The
# return value is discarded — this is the check, not the lookup, and
# `require_api_key` reads the setting again per request so a test can patch it.
configured_api_key()

app = FastAPI(
    title="ShopAgent Commerce API",
    # The cart and order endpoints. Product search is not here and will not be:
    # it lives behind the MCP server, which is the point of the split.
    description="Carts and orders for the ShopAgent conversational assistant.",
    version="0.1.0",
)

# Wide open, and only because nothing here is reached by a browser: the client
# is the agent process, and the credential is a header the browser same-origin
# rules were never protecting. The day a real front end exists this becomes a
# list of origins — `allow_origins=["*"]` with credentials is a combination
# browsers refuse anyway, which is the reminder built into the setting.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    """Report that the process is up, and nothing else.

    Deliberately does not touch the database. A health check that runs a query
    answers a different question than the one it is asked — it reports the
    database's latency as the process's liveness, so a slow Postgres reads as a
    dead API and whatever is watching this endpoint restarts the wrong thing.
    Whether the database is reachable is a readiness question, and it belongs
    at a separate path if it is ever wanted.

    Unauthenticated on purpose: a probe that needs a secret is a probe that
    fails for the wrong reason the day the secret rotates.
    """
    return {"status": "ok", "service": "shopagent-commerce-api"}


# Authentication is attached to the mount, never to a route. The difference is
# what happens when someone adds a route: decorating each handler means a new
# one is unprotected until remembered, while mounting means a new one is
# protected because of where it was put. `tests/test_api_auth.py` sweeps
# `app.routes` to keep that true, and it now has routes to sweep.
app.include_router(cart.router, dependencies=[Depends(require_api_key)])
app.include_router(orders.router, dependencies=[Depends(require_api_key)])

# Mounted without authentication, unlike everything above. Stripe redirects a
# *browser* to these two, and a browser carries no `X-API-Key`. Safe only
# because they read and never write: see `routers/checkout_pages.py`.
app.include_router(checkout_pages.router)

# Also without authentication, and for a different reason worth keeping
# distinct from the one above. Stripe does not send this server's key; it signs
# the body and puts the digest in `Stripe-Signature`, so the signature is the
# credential. Mounting this behind `require_api_key` would refuse every real
# delivery with a 401 while accepting nothing extra — Stripe has no key to
# send. This route does write, from step 2 onwards, which is why the
# verification in `routers/webhooks.py` runs before anything else in the
# handler rather than being a property of where it is mounted.
app.include_router(webhooks.router)
