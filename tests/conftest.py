"""Shared database fixtures for the catalog tests (D3).

Only `tests/test_models.py` and `tests/test_seed.py` use these. Every other
test file in the project still reaches nothing outside the process, which is
the rule in CLAUDE.md; these two are the documented exception, because a
schema and a seed are claims about what Postgres holds and only Postgres can
settle them.

Both fixtures skip rather than fail when the database is unreachable, so
`pytest tests/` stays useful without `docker compose up -d`.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

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
