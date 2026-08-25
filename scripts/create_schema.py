"""Create the catalog schema in Postgres (D3, step 1).

`create_all`, not Alembic. D6 adds new tables to the same metadata rather than
altering these ones, so a migration tool would carry cost without yet carrying
its weight. The day a column has to change shape on a database holding real
rows is the day to introduce it.

Safe to run repeatedly: the extension is created IF NOT EXISTS and `create_all`
checks for each table before creating it, so a second run reports everything as
already present and changes nothing.

    python scripts/create_schema.py
"""

from __future__ import annotations

import sys

from sqlalchemy import inspect, text
from sqlalchemy.exc import OperationalError

from shopagent.catalog.models import Base
from shopagent.db import ensure_vector_extension, get_engine


def main() -> int:
    engine = get_engine()

    try:
        ensure_vector_extension(engine)
    except OperationalError as exc:
        # The common failure by far is Postgres not running. Say so plainly
        # rather than printing a driver traceback.
        print(f"Cannot reach the database at {engine.url}.", file=sys.stderr)
        print("Is Postgres up? Try: docker compose up -d", file=sys.stderr)
        print(f"\n{exc.orig}", file=sys.stderr)
        return 1

    with engine.connect() as connection:
        version = connection.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one()
    print(f"pgvector extension: ready (version {version})")

    # Snapshot before and after, so the script can report what it actually did
    # rather than what it attempted. This is the whole idempotency story: on a
    # second run both sets match and nothing is listed as created.
    before = set(inspect(engine).get_table_names())
    Base.metadata.create_all(engine)
    after = set(inspect(engine).get_table_names())

    created = after - before
    for table in Base.metadata.sorted_tables:
        state = "created" if table.name in created else "already present"
        print(f"  {table.name:<12} {state}")

    print(f"\n{len(created)} table(s) created, {len(after) - len(created)} unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
