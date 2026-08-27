"""Proof that the API's session override contains what a handler writes (D6).

This file exists because the failure it guards against is invisible. FastAPI's
`get_session` builds its own session from the shared factory, so a handler that
commits writes straight to the database — outside whatever transaction a test
opened. Nothing raises. The rows simply stay behind after the test, the test's
own session reads an older snapshot and never sees them, and the suite starts
depending on the order it ran in. That is the kind of bug found on the day
step 4 has thirty cart tests and three of them fail only in CI.

So the claim is checked from both sides:

  * the test's session sees what the handler wrote — the override is in force,
    and the handler is not on some other connection;
  * a *separate* connection does not — the write is still inside the test's
    uncommitted transaction, which is what makes the final rollback able to
    undo it.

The route is built here, on a throwaway app, rather than added to
`api/main.py`. A write endpoint that exists only for a test would be a real
endpoint on the real surface.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from shopagent.api.db import get_session
from shopagent.api.main import app
from shopagent.api.models import Cart, CartStatus

pytestmark = pytest.mark.db


def writing_app() -> FastAPI:
    """A throwaway app with one route that creates and commits a cart."""
    probe = FastAPI()

    @probe.post("/probe-carts")
    def create_cart(db: Session = Depends(get_session)) -> dict[str, str]:
        cart = Cart()
        db.add(cart)
        # The commit is the point. Without `join_transaction_mode=
        # "create_savepoint"` on the test's session this would end the outer
        # transaction, and the rollback at the end of the test would have
        # nothing left to undo.
        db.commit()
        return {"cart_id": str(cart.id)}

    return probe


# --- the override is in force -------------------------------------------


def test_a_handler_writes_into_the_test_session(session):
    probe = writing_app()
    probe.dependency_overrides[get_session] = lambda: session

    with TestClient(probe) as client:
        response = client.post("/probe-carts")

    assert response.status_code == 200
    cart_id = uuid.UUID(response.json()["cart_id"])

    # Visible from the test's own session, which is only true if the handler
    # ran on the same connection.
    written = session.get(Cart, cart_id)
    assert written is not None
    assert written.status is CartStatus.OPEN


def test_the_write_does_not_escape_the_test_transaction(session, engine):
    """The half that catches a broken override.

    Read from a connection of its own: the handler's commit landed on a
    SAVEPOINT inside the test's still-open transaction, so no other session in
    the database can see it. If the override were missing, the handler would
    have committed for real and this row would be visible here — and would
    still be in the table after the test.
    """
    probe = writing_app()
    probe.dependency_overrides[get_session] = lambda: session

    with TestClient(probe) as client:
        cart_id = uuid.UUID(client.post("/probe-carts").json()["cart_id"])

    assert session.get(Cart, cart_id) is not None

    with engine.connect() as outside:
        visible = outside.execute(
            text("SELECT count(*) FROM carts WHERE id = :id"), {"id": cart_id}
        ).scalar_one()

    assert visible == 0, (
        "the handler's write is visible outside the test transaction, so the "
        "dependency override is not in force and this row will survive the test"
    )


def test_two_requests_share_one_transaction(session):
    """Consecutive requests accumulate, the way step 3's cart flow will."""
    probe = writing_app()
    probe.dependency_overrides[get_session] = lambda: session

    with TestClient(probe) as client:
        ids = {uuid.UUID(client.post("/probe-carts").json()["cart_id"]) for _ in range(3)}

    assert len(ids) == 3
    found = session.scalars(select(Cart).where(Cart.id.in_(ids))).all()
    assert len(found) == 3


# --- the shared fixture does the same thing ------------------------------


def test_the_api_client_fixture_installs_the_override(api_client, session):
    """`api_client` is what steps 3 and 4 use; this is the same claim for it."""
    assert app.dependency_overrides[get_session]() is session


def test_the_api_client_fixture_removes_its_override_afterwards(session):
    """Leaking an override would silently rewire every later test's database.

    The fixture is stepped through by hand rather than requested, because
    asserting "no override is installed" at the top of a test only proves
    cleanup if some earlier test happened to install one — which makes the
    proof depend on collection order. Driving the generator shows both edges
    inside one test.
    """
    # pytest imports conftest.py as a top-level module named `conftest`;
    # `__wrapped__` is the undecorated generator function underneath the
    # fixture marker.
    import conftest

    generator = conftest.api_client.__wrapped__(session)

    assert get_session not in app.dependency_overrides
    client = next(generator)
    assert app.dependency_overrides[get_session]() is session
    assert client.get("/health").status_code == 200

    with pytest.raises(StopIteration):
        next(generator)
    assert get_session not in app.dependency_overrides
