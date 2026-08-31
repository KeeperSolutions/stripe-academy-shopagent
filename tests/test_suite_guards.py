"""The guards that protect the suite from its own environment (D10, step 1).

`tests/conftest.py` holds two of them now. `no_accidental_api_calls` has been
there since D3 and is exercised by every unmarked test that runs; the one added
on D10 is not, because on a clean database it is a no-op — which is exactly the
state a guard is hardest to have confidence in.

So the hook is called directly here, with a fake engine standing in for
Postgres. That is the whole point: the condition it fires on is a database
somebody has to dirty by hand, and a test that needed one would be a test
nobody could run twice.

`conftest` is imported as a module, which works because pytest puts this
directory on `sys.path`. It is the only file in the project that does this, and
it is worth a sentence: the alternative was moving the guard into `src/`, where
a rule about how the *tests* run has no business being.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from _pytest.outcomes import Exit  # what `pytest.exit` raises

import conftest


class FakeConnection:
    """Answers `SELECT count(*) FROM <table>` from a dict."""

    def __init__(self, counts):
        self.counts = counts
        self.asked = []

    def execute(self, statement):
        table = str(statement).rsplit(" ", 1)[-1]
        self.asked.append(table)
        return SimpleNamespace(scalar_one=lambda: self.counts[table])

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeEngine:
    def __init__(self, counts):
        self.connection = FakeConnection(counts)

    def connect(self):
        return self.connection


def item(is_db=True):
    """A collected test, as far as the hook is concerned."""
    return SimpleNamespace(
        get_closest_marker=lambda name: object() if (name == "db" and is_db) else None
    )


def run_hook(monkeypatch, counts, items):
    monkeypatch.setattr(conftest, "get_engine", lambda: FakeEngine(counts))
    return conftest.pytest_collection_modifyitems(None, items)


CLEAN = {"orders": 0, "processed_events": 0}
DIRTY = {"orders": 2, "processed_events": 27}


def test_a_clean_database_lets_the_suite_run(monkeypatch):
    assert run_hook(monkeypatch, CLEAN, [item()]) is None


def test_a_leftover_manual_run_stops_the_suite_before_the_first_test(monkeypatch):
    with pytest.raises(Exit) as stopped:
        run_hook(monkeypatch, DIRTY, [item()])

    message = str(stopped.value)
    assert "2 orders" in message
    assert "27 processed_events" in message


def test_the_message_says_it_is_not_a_regression_and_how_to_fix_it(monkeypatch):
    """The whole criterion this guard was built against.

    A reader has to know in the first second that nothing they changed caused
    this, and what to type. Everything else about the guard is bookkeeping.
    """
    with pytest.raises(Exit) as stopped:
        run_hook(monkeypatch, DIRTY, [item()])

    message = str(stopped.value)
    assert "not a regression" in message
    assert conftest.CLEAN_UP_COMMAND in message


def test_one_dirty_table_is_enough(monkeypatch):
    """Either table alone turns tests red, so either alone has to stop the run."""
    for counts, named, unnamed in (
        ({"orders": 1, "processed_events": 0}, "1 orders", "processed_events"),
        ({"orders": 0, "processed_events": 1}, "1 processed_events", "orders"),
    ):
        with pytest.raises(Exit) as stopped:
            run_hook(monkeypatch, counts, [item()])
        assert named in str(stopped.value)
        assert unnamed not in str(stopped.value), "a clean table must not be named"


def test_a_run_with_no_db_tests_never_opens_a_connection(monkeypatch):
    """`pytest tests/test_money.py` must stay free and offline.

    Not a nicety: the guard would otherwise make every offline run depend on
    Docker being up, which is the thing `db`'s skip exists to avoid.
    """

    def explode():
        raise AssertionError("the guard connected to Postgres for an offline run")

    monkeypatch.setattr(conftest, "get_engine", explode)

    assert conftest.pytest_collection_modifyitems(None, [item(is_db=False)]) is None


def test_an_unreachable_database_is_not_this_guard_s_problem(monkeypatch):
    """The `engine` fixture already skips with its own explanation.

    Two mechanisms answering "Postgres is down" would mean the message a reader
    gets depends on which fired first.
    """
    from sqlalchemy.exc import OperationalError

    class RefusingEngine:
        def connect(self):
            raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    monkeypatch.setattr(conftest, "get_engine", RefusingEngine)

    assert conftest.pytest_collection_modifyitems(None, [item()]) is None
