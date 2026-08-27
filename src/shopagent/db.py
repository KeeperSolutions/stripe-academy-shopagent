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
from dataclasses import dataclass
from functools import lru_cache

from sqlalchemy import Engine, create_engine, inspect, text
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


# --- schema integrity ----------------------------------------------------


@dataclass(frozen=True)
class ForeignKeyGap:
    """One declared foreign key the live database does not match."""

    table: str
    columns: tuple[str, ...]
    expected_ondelete: str
    actual_ondelete: str | None  # None means the constraint is absent entirely

    @property
    def is_missing(self) -> bool:
        return self.actual_ondelete is None

    def describe(self) -> str:
        columns = ", ".join(self.columns)
        if self.is_missing:
            return (
                f"{self.table}({columns}) has no foreign key at all; "
                f"the models declare ON DELETE {self.expected_ondelete}"
            )
        return (
            f"{self.table}({columns}) is ON DELETE {self.actual_ondelete}, "
            f"but the models declare ON DELETE {self.expected_ondelete}"
        )


def _declared_ondelete(constraint) -> str:
    """`ondelete=None` in SQLAlchemy means the SQL default, NO ACTION."""
    return (constraint.ondelete or "NO ACTION").upper()


def find_foreign_key_gaps(engine: Engine | None = None) -> list[ForeignKeyGap]:
    """Compare the foreign keys the models declare against the ones that exist.

    This exists because of a failure with no symptom. `DROP TABLE ... CASCADE`
    on the catalog also drops the foreign keys the *commerce* tables hold into
    it — including the `ON DELETE RESTRICT` on `order_items.variant_id`, which
    is what stops a catalog reset from taking order history with it. Running
    `create_all` afterwards does not put them back: `cart_items` and
    `order_items` already exist, and `create_all` never touches a table it did
    not create. The guard in `scripts/seed_catalog.py` still refuses to run
    while orders exist, so the protection *looks* intact from the one direction
    anybody checks, while the layer that held against every client — psql
    included — is silently gone.

    Derived from `Base.metadata` rather than a hard-coded list, so a foreign
    key added later is covered without anybody remembering to add it here.
    Both model modules have to be imported for the metadata to be complete;
    the callers already import them for `create_all`.
    """
    engine = engine or get_engine()
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    # Imported here rather than at module scope: `db.py` is imported by the
    # models themselves, and importing them back would be a cycle.
    from shopagent.catalog.models import Base

    gaps: list[ForeignKeyGap] = []

    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            # A table that does not exist is a different problem, and
            # `create_all` reports it. Not this function's business.
            continue

        actual = {
            tuple(fk["constrained_columns"]): (fk.get("options") or {}).get(
                "ondelete", "NO ACTION"
            ).upper()
            for fk in inspector.get_foreign_keys(table.name)
        }

        for constraint in table.foreign_key_constraints:
            columns = tuple(c.name for c in constraint.columns)
            expected = _declared_ondelete(constraint)
            found = actual.get(columns)

            if found is None or found != expected:
                gaps.append(
                    ForeignKeyGap(
                        table=table.name,
                        columns=columns,
                        expected_ondelete=expected,
                        actual_ondelete=found,
                    )
                )

    return gaps
