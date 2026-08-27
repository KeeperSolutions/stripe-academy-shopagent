"""The request-scoped database session (D6).

A thin dependency over the factory in `shopagent.db`, and deliberately nothing
more. That module's docstring already makes the argument: one engine per
process, because a second one would be a second connection pool against the
same database — double the connections, and a pool exhaustion bug that
reproduces in only half the code. This file owns the *lifetime* of a session,
not its construction.

`get_session` does not commit. A handler that wrote something commits it,
because only the handler knows whether the unit of work finished; a dependency
that committed on the way out would turn a half-written request into a
half-written database. It does roll back on an exception escaping the handler,
so a failed request cannot leave its transaction open holding locks, and it
always closes, so the connection returns to the pool either way.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy.orm import Session

from shopagent.db import get_sessionmaker


def get_session() -> Iterator[Session]:
    """Yield a session for the lifetime of one request.

    FastAPI runs the part after `yield` once the response has been produced.
    The rollback is therefore a safety net rather than the normal path: an
    unhandled exception in a handler unwinds through here, and without it the
    session would go back to the pool mid-transaction.

    Tests replace this whole function through `app.dependency_overrides`, which
    is what lets them hand the app a session already enlisted in a transaction
    they can roll back. See `tests/conftest.py`.
    """
    session = get_sessionmaker()()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
