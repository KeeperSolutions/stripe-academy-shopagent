"""Create the whole schema in Postgres (D3, step 1; extended on D6).

`create_all`, not Alembic. D6 added its four commerce tables to the same
metadata rather than altering the catalog's four, so a migration tool would
carry cost without yet carrying its weight. That stops being true for `carts`
and `orders` the moment they hold a real order — see CLAUDE.md.

Safe to run repeatedly: the extension is created IF NOT EXISTS and `create_all`
checks for each table before creating it, so a second run reports everything as
already present and changes nothing.

    python scripts/create_schema.py
"""

from __future__ import annotations

import sys

from sqlalchemy import inspect, text
from sqlalchemy.exc import OperationalError

# Both model modules are imported for their side effect: a table joins
# `Base.metadata` when its class is defined, so `create_all` builds only what
# has been imported. Importing `Base` alone would silently create the four
# catalog tables and none of the commerce ones.
from shopagent.api import models as commerce_models  # noqa: F401
from shopagent.catalog import models as catalog_models  # noqa: F401
from shopagent.catalog.models import Base
from shopagent.db import (
    ensure_vector_extension,
    find_column_gaps,
    find_foreign_key_gaps,
    get_engine,
)


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

    # `create_all` builds missing tables and never touches an existing one, so
    # it cannot notice a foreign key that was dropped out from under a table it
    # is now reporting as "already present". `DROP TABLE ... CASCADE` on the
    # catalog does exactly that to the commerce tables — see
    # `find_foreign_key_gaps`. Reporting is the whole job here: repairing would
    # mean issuing ALTER statements from a script whose entire premise is that
    # this project has no migration path, and a script that quietly rewrites
    # constraints is worse than one that names the problem.
    # `create_all` never alters an existing table, so it cannot notice a column
    # the models gained after that table was built. On the commerce tables that
    # change arrives as a hand-applied ALTER from `migrations/`, and nothing
    # else checks it was run.
    column_gaps = find_column_gaps(engine)
    if column_gaps:
        print(
            f"\nWARNING: {len(column_gaps)} column(s) declared in the models are "
            "missing from the database:",
            file=sys.stderr,
        )
        for gap in column_gaps:
            print(f"  {gap.describe()}", file=sys.stderr)
        print(
            "\nThese tables are not disposable, so the fix is an ALTER rather "
            "than a drop.\nApply the matching file from migrations/ and run "
            "this again.",
            file=sys.stderr,
        )
        return 2

    gaps = find_foreign_key_gaps(engine)
    if gaps:
        print(f"\nWARNING: {len(gaps)} foreign key(s) do not match the models:", file=sys.stderr)
        for gap in gaps:
            print(f"  {gap.describe()}", file=sys.stderr)
        print(
            "\nThis is what a DROP TABLE ... CASCADE on the catalog leaves "
            "behind: create_all\nrebuilds the catalog tables but cannot restore "
            "the constraints the commerce\ntables held into them. Restore them "
            "with ALTER TABLE ... ADD CONSTRAINT before\nrelying on the "
            "protection they provide.",
            file=sys.stderr,
        )
        return 2

    print("Columns and foreign keys: all match the models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
