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
    expected_target: str
    expected_ondelete: str
    # None on both means the constraint is absent entirely.
    actual_target: str | None
    actual_ondelete: str | None

    @property
    def is_missing(self) -> bool:
        return self.actual_ondelete is None and self.actual_target is None

    def describe(self) -> str:
        columns = ", ".join(self.columns)
        expected = f"-> {self.expected_target} ON DELETE {self.expected_ondelete}"

        if self.is_missing:
            return (
                f"{self.table}({columns}) has no foreign key at all; "
                f"the models declare {expected}"
            )
        actual = f"-> {self.actual_target} ON DELETE {self.actual_ondelete}"
        return f"{self.table}({columns}) is {actual}, but the models declare {expected}"


def _target(table: str, columns: list[str]) -> str:
    """`variants(id)` — what a foreign key points at, as one comparable string."""
    return f"{table}({', '.join(columns)})"


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

        # Keyed by the constrained columns, carrying both what the constraint
        # points at and how it deletes. Comparing the delete action alone would
        # call a foreign key correct while it referenced the wrong table
        # entirely — which is a worse fault than a wrong ON DELETE and would
        # have been reported as a match.
        actual = {
            tuple(fk["constrained_columns"]): (
                _target(fk["referred_table"], fk["referred_columns"]),
                ((fk.get("options") or {}).get("ondelete") or "NO ACTION").upper(),
            )
            for fk in inspector.get_foreign_keys(table.name)
        }

        for constraint in table.foreign_key_constraints:
            columns = tuple(c.name for c in constraint.columns)
            expected_ondelete = _declared_ondelete(constraint)
            expected_target = _target(
                constraint.referred_table.name,
                [element.column.name for element in constraint.elements],
            )
            found = actual.get(columns)

            if found is None:
                gaps.append(
                    ForeignKeyGap(
                        table=table.name,
                        columns=columns,
                        expected_target=expected_target,
                        expected_ondelete=expected_ondelete,
                        actual_target=None,
                        actual_ondelete=None,
                    )
                )
                continue

            found_target, found_ondelete = found
            if found_target != expected_target or found_ondelete != expected_ondelete:
                gaps.append(
                    ForeignKeyGap(
                        table=table.name,
                        columns=columns,
                        expected_target=expected_target,
                        expected_ondelete=expected_ondelete,
                        actual_target=found_target,
                        actual_ondelete=found_ondelete,
                    )
                )

    return gaps


@dataclass(frozen=True)
class ColumnGap:
    """A column the models declare that the live database does not have."""

    table: str
    column: str
    declared_type: str

    def describe(self) -> str:
        return (
            f"{self.table}.{self.column} is declared in the models "
            f"({self.declared_type}) but does not exist in the database"
        )


def find_column_gaps(engine: Engine | None = None) -> list[ColumnGap]:
    """Columns the models declare and the database lacks.

    The commerce tables are not disposable, so a schema change to them is an
    `ALTER`, applied by hand from `migrations/`. Nothing enforces that it was
    applied: `create_all` builds missing *tables* and never alters an existing
    one, so it reports a table as "already present" while it is missing the
    column added last week. The first symptom is `UndefinedColumn` from an
    ordinary read, at whatever moment the ORM first selects that column — which
    is a long way from the change that caused it.

    Reported rather than repaired, for the same reason `find_foreign_key_gaps`
    is: a script that issues its own `ALTER` statements against a table holding
    real orders is a migration tool, and this project has deliberately not
    built one. Naming the gap and pointing at `migrations/` is the whole job.

    Extra columns in the database are not reported. They are what a rolled-back
    deployment looks like, they break nothing, and flagging them would make the
    check noisy in exactly the situation where it needs to be trusted.
    """
    engine = engine or get_engine()
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    from shopagent.catalog.models import Base

    gaps: list[ColumnGap] = []

    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue

        actual = {column["name"] for column in inspector.get_columns(table.name)}

        for column in table.columns:
            if column.name not in actual:
                gaps.append(
                    ColumnGap(
                        table=table.name,
                        column=column.name,
                        declared_type=str(column.type),
                    )
                )

    return gaps
