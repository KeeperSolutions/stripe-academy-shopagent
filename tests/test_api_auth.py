"""Tests for the API skeleton and its authentication (D6).

Marked `db` for one reason only: `api_client` builds on the `session` fixture,
which needs Postgres. Nothing here queries anything — `/health` deliberately
does not touch the database, and a request refused for a missing key never
reaches a handler. The marker is honest about the fixture chain rather than
about the assertions.

The sweep at the bottom is the test that will matter later. Today there are no
protected routes and it passes on the sentinel branch; from step 3 it fans out
over every route the app mounts, so a route added without authentication fails
here without anyone having to remember to write a test for it.
"""

from __future__ import annotations

import contextlib
import importlib.util

import pytest
from fastapi import Depends, FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from shopagent.api.deps import (
    API_KEY_HEADER,
    MissingAPIKey,
    configured_api_key,
    require_api_key,
)
from shopagent.api.main import app
from pydantic import ValidationError

from shopagent.config import REPO_ROOT, Settings, get_settings

pytestmark = pytest.mark.db

# Routes that are public by design. `/docs`, `/redoc` and `/openapi.json` are
# not listed: FastAPI mounts them as plain Starlette routes rather than
# `APIRoute`s, so the isinstance filter below already leaves them out, and
# naming them here would suggest a decision that is not being made.
# `/checkout/success` and `/checkout/cancel` are public because Stripe
# redirects a *browser* to them and a browser carries no `X-API-Key`. That is
# acceptable only because they read and never write — the worst a stranger with
# a session id can do is read the status of a payment they would have had to
# make. Adding a route here has to be this deliberate: the sweep below found
# both of them on the day they were mounted.
PUBLIC_PATHS = {"/health", "/checkout/success", "/checkout/cancel"}

# Any UUID will do — the sweep asserts the request is refused before a handler
# ever looks at it, so the value only has to parse.
PLACEHOLDER = "00000000-0000-0000-0000-000000000000"


def walk_api_routes(routes) -> list[APIRoute]:
    """Every `APIRoute` reachable from the app, mounted routers included.

    FastAPI 0.141 stopped flattening `include_router` into `app.routes` and
    wraps each mount in an `_IncludedRouter` instead, so a sweep that only
    filtered the top level would have found `/health` and nothing else — and
    would have gone on passing while every cart route sat unprotected. That is
    the exact failure this sweep exists to catch, so it recurses, and
    `test_the_route_walk_sees_everything_openapi_does` keeps it honest if the
    internals move again.
    """
    found: list[APIRoute] = []
    for route in routes:
        if isinstance(route, APIRoute):
            found.append(route)
            continue
        nested = getattr(route, "original_router", None)
        if nested is not None:
            found.extend(walk_api_routes(nested.routes))
    return found


def protected_routes() -> list[tuple[str, str]]:
    """Every (method, path) the app mounts that is not public by design."""
    return sorted(
        (method, route.path)
        for route in walk_api_routes(app.routes)
        if route.path not in PUBLIC_PATHS
        for method in route.methods - {"HEAD", "OPTIONS"}
    )


def valid_key() -> str:
    return get_settings().shopagent_api_key


@contextlib.contextmanager
def guarded_client():
    """A throwaway app with one route behind `require_api_key`.

    Built here rather than in `api/main.py` because the app has no protected
    route until step 3, and a route that exists only to be tested would be a
    permanent hole in the real surface. This exercises the dependency itself,
    which is what the sweep further down cannot do while the app is bare.
    """
    probe = FastAPI()

    @probe.get("/guarded", dependencies=[Depends(require_api_key)])
    def guarded() -> dict[str, bool]:
        return {"reached": True}

    with TestClient(probe) as client:
        yield client


# --- health --------------------------------------------------------------


def test_health_answers_without_a_key(api_client):
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_does_not_touch_the_database(api_client, monkeypatch):
    """A liveness probe that queries reports the database's latency as its own.

    Enforced by breaking the session factory outright: if `/health` still
    answers 200, it never asked for a session.
    """
    def refuse() -> None:
        raise AssertionError("/health asked for a database session")

    monkeypatch.setattr("shopagent.api.db.get_sessionmaker", refuse)
    monkeypatch.setattr("shopagent.db.get_sessionmaker", refuse)

    assert api_client.get("/health").status_code == 200


# --- the key itself ------------------------------------------------------


def test_a_missing_key_is_401_not_403():
    """401, because the client never said who it was.

    403 is the answer to a known caller who may not do this, and it tells a
    client that retrying with a credential is pointless. `auto_error=False` on
    the `APIKeyHeader` is what buys the difference — the default raises 403.
    """
    with guarded_client() as client:
        assert client.get("/guarded").status_code == 401


def test_the_configured_key_is_accepted():
    with guarded_client() as client:
        response = client.get("/guarded", headers={API_KEY_HEADER: valid_key()})

    assert response.status_code == 200
    assert response.json() == {"reached": True}


def test_the_401_names_the_header_a_client_has_to_send():
    with guarded_client() as client:
        response = client.get("/guarded")

    assert API_KEY_HEADER in response.json()["detail"]
    assert response.headers["WWW-Authenticate"] == API_KEY_HEADER


@pytest.mark.parametrize(
    "presented",
    [
        pytest.param("", id="empty-string"),
        pytest.param("wrong", id="wrong-key"),
        pytest.param("  ", id="whitespace"),
    ],
)
def test_a_key_that_is_not_the_key_is_refused(presented):
    with guarded_client() as client:
        response = client.get("/guarded", headers={API_KEY_HEADER: presented})

    assert response.status_code == 401


def test_a_prefix_of_the_real_key_is_refused():
    """The case `==` would still get right, and `compare_digest` is here for.

    Built from the configured key rather than hard-coded, so it stays a real
    prefix whatever the key happens to be.
    """
    almost = valid_key()[:-1]
    assert almost and almost != valid_key()

    with guarded_client() as client:
        assert client.get("/guarded", headers={API_KEY_HEADER: almost}).status_code == 401


# --- the sweep -----------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    protected_routes() or [pytest.param(None, None, id="no-protected-routes-yet")],
)
def test_every_non_public_route_refuses_a_request_without_a_key(api_client, method, path):
    """Authentication is a property of where a route is mounted.

    The sentinel branch runs while the app has no protected routes — and it is
    not a skip, because the thing worth checking today is that the *sweep* is
    sound. An empty result has two causes: no routes yet, or a discovery bug
    that would keep this test quiet forever once routes arrive. Asserting the
    app is exactly its public surface tells them apart.
    """
    if method is None:
        mounted = {route.path for route in walk_api_routes(app.routes)}
        assert mounted == PUBLIC_PATHS, (
            f"the app mounts {sorted(mounted - PUBLIC_PATHS)}, which this sweep "
            "should have picked up — protected_routes() is not seeing them"
        )
        return

    concrete = path
    while "{" in concrete:
        head, _, rest = concrete.partition("{")
        _, _, tail = rest.partition("}")
        concrete = f"{head}{PLACEHOLDER}{tail}"

    response = api_client.request(method, concrete)
    assert response.status_code == 401, (
        f"{method} {path} answered {response.status_code} without a key. "
        "Mount its router with dependencies=[Depends(require_api_key)]."
    )


def test_the_public_surface_is_health_and_the_docs_only():
    """A second route added to PUBLIC_PATHS should be a deliberate edit."""
    api_paths = {route.path for route in walk_api_routes(app.routes)}
    assert PUBLIC_PATHS <= api_paths

    starlette_routes = {
        route.path
        for route in app.routes
        if not isinstance(route, APIRoute) and hasattr(route, "path")
    }
    assert starlette_routes == {
        "/openapi.json",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
    }


def test_the_public_checkout_pages_only_read():
    """Their safety rests entirely on this, so it is asserted rather than assumed.

    An unauthenticated route that could write would be a hole; these two look
    a session up and render HTML. If either ever gains a write, this fails and
    the decision to leave them unauthenticated has to be made again.
    """
    from fastapi.routing import APIRoute as _APIRoute

    for route in walk_api_routes(app.routes):
        if route.path.startswith("/checkout/"):
            assert route.methods - {"HEAD", "OPTIONS"} == {"GET"}, (
                f"{route.path} accepts {sorted(route.methods)} while mounted "
                "without authentication"
            )


def test_the_route_walk_sees_everything_openapi_does():
    """The sweep's traversal, checked against a public API rather than internals.

    `walk_api_routes` reads `_IncludedRouter.original_router`, which is not a
    documented attribute. If a FastAPI upgrade renames it the walk silently
    shrinks to `/health` and every sweep below starts passing vacuously. The
    OpenAPI document is built by a different code path, so comparing the two
    turns that into a failure here.
    """
    walked = {(method.lower(), route.path) for route in walk_api_routes(app.routes)
              for method in route.methods - {"HEAD", "OPTIONS"}}
    documented = {
        (method, path)
        for path, operations in app.openapi()["paths"].items()
        for method in operations
    }

    assert documented <= walked, (
        f"the route walk missed {sorted(documented - walked)} — it is no longer "
        "seeing mounted routers, and every auth sweep below is passing vacuously"
    )


# --- refusing to start without a key -------------------------------------


def test_settings_reject_a_blank_api_key():
    """First layer: `.env` holding `SHOPAGENT_API_KEY=` is a config error."""
    with pytest.raises(ValueError):
        Settings(shopagent_api_key="")


def test_the_api_key_has_no_default(monkeypatch):
    """Raised in review on PR #6, and it was right.

    The field used to default to `"dev-local-key"`, so a deployment that simply
    forgot `SHOPAGENT_API_KEY` started successfully and authenticated every
    request against a string published in `.env.example`. It would have looked
    correctly configured while being open. There is no safe default for the
    only secret the API has, so there is now no default at all — the same
    treatment `openai_api_key` already had.
    """
    monkeypatch.delenv("SHOPAGENT_API_KEY", raising=False)

    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None, openai_api_key="sk-test")

    missing = {error["loc"][0] for error in excinfo.value.errors()}
    assert "shopagent_api_key" in missing


def test_the_example_env_ships_no_usable_key():
    """`.env.example` is copied verbatim by whoever sets this up next.

    A plausible-looking value there becomes a real credential on somebody's
    machine, so the placeholder has to be one that cannot be mistaken for a key.
    """
    example = (REPO_ROOT / ".env.example").read_text()

    line = next(
        raw for raw in example.splitlines() if raw.startswith("SHOPAGENT_API_KEY=")
    )
    value = line.split("=", 1)[1]
    assert value == "CHANGE-ME", f"the example ships a usable-looking key: {value!r}"


def test_configured_api_key_rejects_a_whitespace_key(monkeypatch):
    """Second layer: what `min_length=1` lets through and authenticates nobody."""
    monkeypatch.setattr(
        "shopagent.api.deps.get_settings",
        lambda: Settings(shopagent_api_key="   "),
    )

    with pytest.raises(MissingAPIKey):
        configured_api_key()


def test_the_app_module_refuses_to_import_without_a_usable_key(monkeypatch):
    """Third layer, and the one that matters: uvicorn never gets an app.

    A server that starts and then refuses every request is a failure that
    looks like uptime; one that starts and accepts every request is worse.
    `api/main.py` calls `configured_api_key()` at module scope, so the process
    dies during import instead.

    Loaded as a fresh module object under its own name rather than reloaded in
    place: a reload that raises halfway would leave `shopagent.api.main`
    gutted for every test after this one.
    """
    monkeypatch.setattr(
        "shopagent.api.deps.get_settings",
        lambda: Settings(shopagent_api_key="   "),
    )

    spec = importlib.util.spec_from_file_location(
        "shopagent_api_main_probe", app_module_path()
    )
    module = importlib.util.module_from_spec(spec)

    with pytest.raises(MissingAPIKey):
        spec.loader.exec_module(module)


def app_module_path() -> str:
    import shopagent.api.main as main_module

    return main_module.__file__


def test_a_good_key_still_imports_the_app_module():
    """The other half: the guard is not simply always raising."""
    spec = importlib.util.spec_from_file_location(
        "shopagent_api_main_probe_ok", app_module_path()
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.app.title == "ShopAgent Commerce API"
