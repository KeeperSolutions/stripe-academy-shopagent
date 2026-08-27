"""Load the catalog into Postgres (D3, step 2).

    python scripts/seed_catalog.py            # write what is missing
    python scripts/seed_catalog.py --reset    # clear the catalog first

Safe to run twice: `seed_catalog` matches on sku and writes only rows that are
absent, so a second run reports thirty products skipped and touches nothing.

`--reset` deletes every product first, and the cascade takes the variants,
prices and stock with them. That is the supported way to pick up an edit to
`catalog/seed.py`, because the seeder never updates a row it already stored.

It refuses to run once any order exists (D6). The cascade reaches `cart_items`,
which is fine — carts are as disposable as the catalog — but `order_items` is
history, and `ON DELETE RESTRICT` on its `variant_id` would stop the delete
anyway. The guard turns that IntegrityError into a sentence.
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError, ProgrammingError

from shopagent.api.models import OrdersExist, assert_no_orders
from shopagent.catalog.models import Inventory, Price, Product, Variant
from shopagent.catalog.seed import reset_catalog, seed_catalog
from shopagent.db import get_engine, session_scope


def print_totals(session) -> None:
    """Report what is in the database now, not what this run wrote."""
    for label, model in (
        ("products", Product),
        ("variants", Variant),
        ("prices", Price),
        ("inventory", Inventory),
    ):
        count = session.scalar(select(func.count()).select_from(model))
        print(f"  {label:<12} {count}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete the existing catalog before seeding",
    )
    args = parser.parse_args(argv)

    try:
        with session_scope() as session:
            if args.reset:
                assert_no_orders(session, operation="reset the catalog")
                deleted = reset_catalog(session)
                print(f"Reset: {deleted} product(s) deleted, cascade took the rest.\n")

            summary = seed_catalog(session)
            print("Seed run")
            for line in summary.as_lines():
                print(line)

            print("\nIn the database now")
            print_totals(session)
    except OrdersExist as exc:
        print(exc, file=sys.stderr)
        return 1
    except OperationalError as exc:
        print(f"Cannot reach the database at {get_engine().url}.", file=sys.stderr)
        print("Is Postgres up? Try: docker compose up -d", file=sys.stderr)
        print(f"\n{exc.orig}", file=sys.stderr)
        return 1
    except ProgrammingError as exc:
        # Reaching Postgres but not the tables means step 1 has not run here.
        print("The catalog tables are missing.", file=sys.stderr)
        print("Run: python scripts/create_schema.py", file=sys.stderr)
        print(f"\n{exc.orig}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
