"""Shared database fixtures for the schema tests (D3, extended on D6).

`tests/test_models.py`, `tests/test_seed.py` and `tests/test_commerce_models.py`
use these. Every other test file in the project still reaches nothing outside
the process, which is the rule in CLAUDE.md; these are the documented
exception, because a schema and a seed are claims about what Postgres holds and
only Postgres can settle them.

Both fixtures skip rather than fail when the database is unreachable, so
`pytest tests/` stays useful without `docker compose up -d`.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

# Imported for the side effect of registering the commerce tables on the
# shared `Base.metadata`, the same reason `scripts/create_schema.py` imports
# it. Without this the `engine` fixture creates the catalog's four tables and
# none of D6's, and every commerce test fails on a missing relation.
from shopagent.api import models as commerce_models  # noqa: F401
from shopagent.catalog.models import Base
from shopagent.db import ensure_vector_extension, get_engine


@pytest.fixture(autouse=True)
def no_accidental_api_calls(request, monkeypatch):
    """Make an unmarked test that reaches the OpenAI API fail loudly.

    `search_products` embeds its query by default, so a test that passes one
    and forgets `mode="keyword"` or `query_embedding=` would quietly spend
    tokens on every run. That happened once while step 4 was being written,
    which is why this exists rather than a note asking people to be careful.
    """
    if request.node.get_closest_marker("network"):
        return

    def refuse() -> None:
        raise AssertionError(
            "this test tried to call the OpenAI API. Mark it @pytest.mark.network, "
            'or pass mode="keyword" / query_embedding= to keep it offline.'
        )

    monkeypatch.setattr("shopagent.catalog.embeddings.default_client", refuse)


@pytest.fixture(scope="session")
def engine():
    """The shared engine, with the schema guaranteed to exist."""
    engine = get_engine()
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except OperationalError as exc:
        pytest.skip(f"Postgres is not reachable ({exc.orig}). Run: docker compose up -d")
    ensure_vector_extension(engine)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(engine):
    """A session whose every write is undone when the test ends.

    `join_transaction_mode="create_savepoint"` lets a test call `commit()` —
    which most must, since a constraint fires on flush and the seeder commits
    on its own — without that commit escaping the outer transaction. The
    rollback at the end restores whatever the database held before, so these
    tests can run against a seeded catalog without disturbing it.
    """
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


# --- the API (D6) --------------------------------------------------------


@pytest.fixture
def api_client(session):
    """A `TestClient` whose handlers write into the test's own transaction.

    This is the fixture the whole D6 test suite rests on, and the reason it
    exists is a mismatch that fails silently. `api/db.py`'s `get_session`
    builds a session of its own from the shared factory, which means a handler
    commits to the database directly — outside the transaction the `session`
    fixture opened and will roll back. The rows survive the test, the test's
    own session never sees them (it is reading an older snapshot), and the
    suite starts passing or failing on the order it happened to run in.

    Overriding the dependency with the *same* session object the test holds is
    what closes the gap. Two details make it work:

      * the `session` fixture binds to one connection with
        `join_transaction_mode="create_savepoint"`, so a `commit()` inside a
        handler lands on a SAVEPOINT and the outer transaction stays open to be
        rolled back at the end;
      * the override is a plain function, not a generator. FastAPI closes a
        generator dependency when the response is done, and closing this
        session would leave the test holding a dead one after its first
        request.

    The override is popped by key rather than cleared wholesale, so a fixture
    that layers another override on top does not lose it here.
    """
    from shopagent.api.db import get_session
    from shopagent.api.main import app

    app.dependency_overrides[get_session] = lambda: session
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_session, None)


@pytest.fixture
def authed_client(api_client):
    """`api_client` with the configured API key on every request.

    Separate from `api_client` rather than replacing it, because the auth
    sweep in `tests/test_api_auth.py` needs a client that sends no key at all.
    Everything testing behaviour behind the auth wants this one.
    """
    from shopagent.api.deps import API_KEY_HEADER
    from shopagent.config import get_settings

    api_client.headers[API_KEY_HEADER] = get_settings().shopagent_api_key
    return api_client
