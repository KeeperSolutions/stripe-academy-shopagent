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
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session

# Imported for the side effect of registering the commerce tables on the
# shared `Base.metadata`, the same reason `scripts/create_schema.py` imports
# it. Without this the `engine` fixture creates the catalog's four tables and
# none of D6's, and every commerce test fails on a missing relation.
from shopagent.agent import profile as agent_profile  # noqa: F401
from shopagent.api import models as commerce_models  # noqa: F401
from shopagent.catalog.models import Base
from shopagent.db import ensure_vector_extension, get_engine


# --- the state a manual run leaves behind (D10, step 1) ------------------
#
# The tables a manual run writes to and that some tests assume are empty. Not
# `carts`: five of them can sit in this database with nothing failing, and a
# guard that stops the suite over rows nothing reads would be a guard people
# learn to work around.
DIRTY_TABLES = ("orders", "processed_events")

CLEAN_UP_COMMAND = "python scripts/manual_test_state.py restore"


def pytest_collection_modifyitems(config, items):
    """Stop before the first test if a manual run is still in the database.

    Five times now, a leftover order has turned ~29 tests red for a reason with
    nothing to do with what changed: `test_api_orders` and
    `test_commerce_models` assert `orders` is empty, `test_seed` then dies on
    the `ON DELETE RESTRICT` that stops a reset taking order history with it,
    and `test_webhooks` asserts the same thing about `processed_events`. The
    defect appears a long way from its cause, which is the shape of failure D8
    already recorded for `InFailedSqlTransaction`.

    **Not a skip.** A skip reports "nothing to see here" in green, and D9
    measured what that costs: a run that said `452 passed, 380 skipped` in
    green while the Docker daemon was down, correct in every detail and
    unreadable. A dirty database is not a reason to report success.

    **Not twenty-nine failures.** That is the thing being removed.

    **Not one failure plus twenty-eight skips**, which would read best of all
    and needs a marker on twenty-nine tests across four files — a refactor, and
    a second record in `conftest.py` of which tests assume what. A partly green
    run over a database known to be lying is also a worse report than one that
    did not start.

    So the run stops with one sentence naming the counts and the command. It is
    deliberately broader than the tests that actually assert emptiness: it fires
    whenever *any* `db` test is collected, because narrowing it means listing
    those four modules here, and that list goes stale the first time somebody
    writes a fifth. The cost of being broad is one command; the cost of being
    stale is this paragraph again.
    """
    if not any(item.get_closest_marker("db") for item in items):
        # Nothing collected touches Postgres — do not even connect. This is
        # what keeps `pytest tests/test_money.py` free and offline.
        return

    engine = get_engine()
    try:
        with engine.connect() as connection:
            leftovers = {
                table: connection.execute(
                    text(f"SELECT count(*) FROM {table}")  # noqa: S608 - a literal from DIRTY_TABLES
                ).scalar_one()
                for table in DIRTY_TABLES
            }
    except (OperationalError, ProgrammingError):
        # Unreachable, or the schema is not built yet. Both are already handled
        # — the `engine` fixture skips with its own explanation — and neither is
        # this guard's business.
        return

    dirty = {table: count for table, count in leftovers.items() if count}
    if not dirty:
        return

    counts = ", ".join(f"{count} {table}" for table, count in dirty.items())
    pytest.exit(
        f"\nThe database still holds a manual run: {counts}.\n"
        f"This is not a regression — around 29 db tests assert these tables are "
        f"empty and would fail for that reason alone.\n"
        f"Clean up with:  {CLEAN_UP_COMMAND}\n"
        f"(That deletes rows created since the last snapshot. Anything older "
        f"has to go by hand, and an order holds stock: decrement "
        f"inventory.reserved by its lines rather than zeroing the column.)",
        returncode=1,
    )


@pytest.fixture(autouse=True)
def no_accidental_api_calls(request, monkeypatch):
    """Make an unmarked test that reaches a paid or external API fail loudly.

    Two providers, one mechanism: replace the module attribute every call has
    to pass through with something that raises, unless the test carries the
    marker that says it meant it.

    OpenAI, because `search_products` embeds its query by default — a test that
    passes one and forgets `mode="keyword"` or `query_embedding=` would quietly
    spend tokens on every run. That happened once while D3 step 4 was being
    written, which is why this exists rather than a note asking people to be
    careful.

    Stripe, because D7 introduces a second way to leave the process. The seam
    is the SDK's request funnel rather than our own `get_client()`: building a
    client makes no network call, and offline tests legitimately build one to
    read back the pinned API version. Blocking construction would fail those
    for no reason, while blocking the funnel catches every call however it was
    reached — including one that bypasses `payments/` and uses the SDK
    directly.

    `_APIRequestor.request` is private, and `monkeypatch.setattr` with a string
    target raises `AttributeError` when the attribute is missing. That is
    deliberate: if a future SDK moves it, every test fails loudly instead of
    the guard quietly protecting nothing.
    """
    if not request.node.get_closest_marker("network"):

        def refuse_openai() -> None:
            raise AssertionError(
                "this test tried to call the OpenAI API. Mark it "
                '@pytest.mark.network, or pass mode="keyword" / '
                "query_embedding= to keep it offline."
            )

        monkeypatch.setattr("shopagent.catalog.embeddings.default_client", refuse_openai)

    if not request.node.get_closest_marker("stripe"):

        def refuse_stripe(*args, **kwargs):
            raise AssertionError(
                "this test tried to call the Stripe API. Mark it "
                "@pytest.mark.stripe if it is meant to, or keep it offline — "
                "building a client is free, only requests are guarded."
            )

        monkeypatch.setattr(
            "stripe._api_requestor._APIRequestor.request", refuse_stripe
        )
        monkeypatch.setattr(
            "stripe._api_requestor._APIRequestor.request_async", refuse_stripe
        )


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
