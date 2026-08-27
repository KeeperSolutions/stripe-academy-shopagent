"""Mirror the local catalog into Stripe Products and Prices (D7, step 2).

    python scripts/sync_stripe_catalog.py --dry-run   # report, write nothing
    python scripts/sync_stripe_catalog.py             # create what is missing

Safe to run twice: a product that already carries a `stripe_product_id` is
skipped without an API call, so a second run reports everything skipped and
touches nothing. See `payments/catalog_sync.py` for the second, shorter-lived
half of that guarantee — the Stripe idempotency key that covers a crash between
creating an object and storing its id.

**Nothing charges from what this writes.** D7's checkout builds `line_items`
from the `order_items` snapshot via `price_data`, never from a Stripe Price id,
because that id would be a second source of truth for a price D6 already froze
at order time. This script exists so the catalog is visible in the dashboard
and so the Products/Prices API is exercised.
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy.exc import OperationalError, ProgrammingError

from shopagent.db import get_engine, session_scope
from shopagent.payments.catalog_sync import build_plan, run_sync
from shopagent.payments.stripe_svc import MissingStripeKey


def print_plan(plan) -> None:
    """What a dry run has to show: the intent, itemised enough to check."""
    print("Planned")
    print(f"  products to create   {len(plan.products_to_create)}")
    print(f"  products already synced {len(plan.products) - len(plan.products_to_create)}")
    print(f"  prices to create     {len(plan.prices_to_create)}")
    print(f"  prices already synced {len(plan.prices) - len(plan.prices_to_create)}")

    for planned in plan.products_to_create[:5]:
        print(f"    + product  {planned.name}")
    if len(plan.products_to_create) > 5:
        print(f"    … and {len(plan.products_to_create) - 5} more products")

    for planned in plan.prices_to_create[:5]:
        amount = planned.amount_cents / 100
        print(f"    + price    {planned.sku:<24} {amount:>8.2f} {planned.currency}")
    if len(plan.prices_to_create) > 5:
        print(f"    … and {len(plan.prices_to_create) - 5} more prices")


def print_summary(summary) -> None:
    print("\nSync run" + (" (dry run — nothing was written)" if summary.dry_run else ""))
    for line in summary.as_lines():
        print(line)

    if summary.skipped:
        print(f"\nSkipped {len(summary.skipped)} variant(s):")
        for skipped in summary.skipped:
            print(f"  {skipped.sku:<24} {skipped.reason}")

    if summary.drifted:
        print(f"\nDrifted {len(summary.drifted)} price(s) — reported, not repaired:")
        for drift in summary.drifted:
            print(
                f"  {drift.sku:<24} local {drift.local_amount_cents} "
                f"vs Stripe {drift.stripe_amount_cents} ({drift.stripe_price_id})"
            )
        print(
            "\n  A Stripe Price is immutable: fixing one means creating a new "
            "Price and\n  archiving the old. This script will not do that on "
            "its own — nothing is\n  charged from these objects, so the cost of "
            "drift is a stale dashboard."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be created without calling any write API",
    )
    args = parser.parse_args(argv)

    try:
        with session_scope() as session:
            plan = build_plan(session)
            print_plan(plan)
            summary = run_sync(session, dry_run=args.dry_run)
            print_summary(summary)
    except MissingStripeKey as exc:
        print(exc, file=sys.stderr)
        return 1
    except OperationalError as exc:
        print(f"Cannot reach the database at {get_engine().url}.", file=sys.stderr)
        print("Is Postgres up? Try: docker compose up -d", file=sys.stderr)
        print(f"\n{exc.orig}", file=sys.stderr)
        return 1
    except ProgrammingError as exc:
        print("The catalog tables are missing or out of date.", file=sys.stderr)
        print("Run: python scripts/create_schema.py", file=sys.stderr)
        print(f"\n{exc.orig}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
