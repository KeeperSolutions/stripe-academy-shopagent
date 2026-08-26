"""Database access for the whole project (D3).

Deliberately at the top of the package rather than inside `catalog/`. On D6 the
FastAPI cart and order routers talk to the same Postgres, and if each module
built its own engine there would be two connection pools against one database:
double the connections, and a pool exhaustion bug that reproduces in only half
the code. One engine per process, shared.

Configuration is read through `get_settings()` and nowhere else, per CLAUDE.md.

pgvector needs no registration here. `pgvector.sqlalchemy.VECTOR` carries its
own bind and result processors, so values cross as text through whatever driver
SQLAlchemy is using; the psycopg-level `register_vector()` is for raw psycopg
connections, which this project does not open.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from shopagent.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return this process's engine, built on first use.

    Cached for the same reason `get_settings()` is: an engine owns a connection
    pool, and a second one would quietly double the connections held against
    the same database. Tests can drop it with `get_engine.cache_clear()`.
    """
    settings = get_settings()
    # pool_pre_ping because Postgres runs in a Docker container that gets
    # stopped and started between working sessions. Without it the first query
    # after a restart fails on a connection the pool still believes is live.
    return create_engine(settings.database_url, pool_pre_ping=True)


@lru_cache(maxsize=1)
def get_sessionmaker() -> sessionmaker[Session]:
    """Return the session factory bound to this process's engine."""
    return sessionmaker(
        bind=get_engine(),
        # expire_on_commit defaults to True, which marks every attribute stale
        # after a commit and reloads it on next access. That costs an extra
        # SELECT at best, and raises DetachedInstanceError once the session has
        # closed — which is exactly what an API handler does before serialising
        # the object it just wrote (D6).
        expire_on_commit=False,
    )


@contextmanager
def session_scope() -> Iterator[Session]:
    """Yield a session that commits on success and rolls back on failure.

    The session is always closed, so its connection returns to the pool even
    when the body raises. The exception is re-raised rather than swallowed:
    callers decide what a failed write means, this only guarantees the
    transaction does not stay open.
    """
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ensure_vector_extension(engine: Engine | None = None) -> None:
    """Create the pgvector extension if the database does not already have it.

    Must run before `create_all`: `products.embedding` is declared `VECTOR(1536)`
    and Postgres cannot create a column of a type it does not know yet.
    """
    engine = engine or get_engine()
    with engine.connect() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # DDL is transactional in Postgres, and SQLAlchemy 2.0 does not
        # autocommit — without this the extension disappears with the block.
        connection.commit()
