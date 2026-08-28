"""Snapshot the database before a manual run, and restore it afterwards (D8).

Written after a cleanup went wrong in exactly the way this prevents. Driving
the API by hand leaves carts, orders and `processed_events` behind, and raises
`inventory.reserved`. The obvious tidy-up is `UPDATE inventory SET reserved =
0` — and it is wrong, because `reserved` is not zero to begin with:
`catalog/seed.py` ships `FF-TRLGTX-42-BLK` with `reserved=2`, deliberately, so
that `check_stock` has something to subtract. Zeroing it broke
`tests/test_search.py`, the repair was a reseed, and the reseed discarded every
embedding in the catalog.

None of those steps was unreasonable on its own. The mistake was restoring to a
remembered constant rather than to what was actually there, so this reads the
value first.

    python scripts/manual_test_state.py snapshot   # before driving the API
    ... place orders, pay, replay webhooks ...
    python scripts/manual_test_state.py restore    # after

`restore` deletes commerce rows created since the snapshot and puts every
`inventory.reserved` back to the number it held. It **never** touches the
catalog: no reseed, no re-embed, nothing that spends money or rebuilds a
vector. If the catalog is what got damaged, this says so and stops, because
repairing it costs an OpenAI call and that is a decision for a person.

The snapshot lives in `.manual-test-state.json` at the repository root, which
is gitignored — it is a scratch file about one machine at one moment, not a
record of anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sqlalchemy import text

from shopagent.config import REPO_ROOT
from shopagent.db import get_engine

# Imported for the side effect of registering the commerce tables on the shared
# metadata, the same reason `scripts/create_schema.py` does it.
import shopagent.api.models  # noqa: F401

SNAPSHOT_PATH = REPO_ROOT / ".manual-test-state.json"

# The tables a manual run writes to. Ordered so deletes respect the foreign
# keys: children before parents.
COMMERCE_TABLES = ["order_items", "orders", "cart_items", "carts", "processed_events"]


def read_state(connection) -> dict:
    """What the database holds right now, in the terms restore needs."""
    reserved = {
        str(variant_id): int(reserved)
        for variant_id, reserved in connection.execute(
            text("SELECT variant_id, reserved FROM inventory ORDER BY variant_id")
        )
    }
    counts = {
        table: connection.execute(text(f"SELECT count(*) FROM {table}")).scalar()
        for table in COMMERCE_TABLES
    }
    embedded = connection.execute(
        text("SELECT count(*) FROM products WHERE embedding IS NOT NULL")
    ).scalar()
    products = connection.execute(text("SELECT count(*) FROM products")).scalar()

    return {
        "reserved": reserved,
        "counts": counts,
        "products": products,
        "embedded": embedded,
    }


def snapshot() -> int:
    with get_engine().connect() as connection:
        state = read_state(connection)

    SNAPSHOT_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))

    print(f"snapshot written to {SNAPSHOT_PATH.name}")
    print(f"  products     {state['products']} ({state['embedded']} embedded)")
    print(f"  reserved     {sum(state['reserved'].values())} across {len(state['reserved'])} variants")
    for table, count in state["counts"].items():
        print(f"  {table:16} {count}")
    return 0


def restore() -> int:
    if not SNAPSHOT_PATH.exists():
        print(
            f"no {SNAPSHOT_PATH.name} — run `snapshot` before the manual test, "
            "not after it.",
            file=sys.stderr,
        )
        return 2

    before = json.loads(SNAPSHOT_PATH.read_text())
    engine = get_engine()

    with engine.connect() as connection:
        now = read_state(connection)

    # Refused rather than repaired. Rebuilding embeddings costs an OpenAI call
    # and reseeding rewrites the catalog; both are decisions for a person, and
    # a cleanup script that quietly spends money is one nobody can trust to run.
    if now["products"] != before["products"] or now["embedded"] != before["embedded"]:
        print(
            "the catalog changed during this run and restore will not touch it.\n"
            f"  products  {before['products']} -> {now['products']}\n"
            f"  embedded  {before['embedded']} -> {now['embedded']}\n"
            "Nothing has been undone. Repairing this means "
            "`python scripts/seed_catalog.py --reset` and then "
            "`python scripts/embed_catalog.py`, which calls the OpenAI API and "
            "costs money — run them deliberately.",
            file=sys.stderr,
        )
        return 2

    with engine.begin() as connection:
        deleted = {}
        for table in COMMERCE_TABLES:
            result = connection.execute(text(f"DELETE FROM {table}"))
            deleted[table] = result.rowcount

        # Back to the recorded number, not to zero. `reserved` is seeded
        # non-zero on purpose for at least one variant, and assuming otherwise
        # is the bug this script exists to prevent.
        changed = 0
        for variant_id, reserved in before["reserved"].items():
            result = connection.execute(
                text(
                    "UPDATE inventory SET reserved = :reserved "
                    "WHERE variant_id = :variant_id AND reserved <> :reserved"
                ),
                {"variant_id": int(variant_id), "reserved": reserved},
            )
            changed += result.rowcount

    print("restored")
    for table, count in deleted.items():
        print(f"  {table:16} deleted {count}")
    print(f"  inventory        {changed} variant(s) put back to their snapshot value")

    with engine.connect() as connection:
        after = read_state(connection)

    if after["reserved"] != before["reserved"]:
        print(
            "reserved does not match the snapshot after restore — a variant was "
            "added or removed during the run.",
            file=sys.stderr,
        )
        return 2

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["snapshot", "restore"])
    action = parser.parse_args().action

    return snapshot() if action == "snapshot" else restore()


if __name__ == "__main__":
    raise SystemExit(main())
