"""The foreign keys the schema depends on, asserted against Postgres (D7).

This file exists because of a failure that leaves no trace. `DROP TABLE ...
CASCADE` on the catalog also drops the foreign keys the *commerce* tables hold
into it, and `create_all` afterwards does not restore them — `cart_items` and
`order_items` already exist, and it never touches a table it did not create.
What is left is a database where `assert_no_orders()` still refuses a
`--reset`, so the protection looks intact from the one direction anybody
checks, while `ON DELETE RESTRICT` on `order_items.variant_id` — the layer that
held against every client, psql included — is gone.

It happened here, during D7 step 2, on a live database.

**Read from `pg_constraint`, and expect from a hard-coded table.** Both halves
are deliberate. Reading the catalog rather than SQLAlchemy metadata is the
difference between "what the source says should be true" and "what the database
will actually do" — metadata would have described a RESTRICT that was not
there. Hard-coding the expectations rather than deriving them from the models
is the other half: a test generated from the same source it is checking passes
whenever the two agree, including when both are wrong. `find_foreign_key_gaps`
in `db.py` does the derived comparison and catches a different thing — drift on
any table, including ones nobody listed here. The two are complementary, not
redundant.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.db

# What Postgres records in `pg_constraint.confdeltype`.
DELETE_ACTIONS = {
    "a": "NO ACTION",
    "r": "RESTRICT",
    "c": "CASCADE",
    "n": "SET NULL",
    "d": "SET DEFAULT",
}

# (table, column) -> (referenced table, confdeltype, why it is that letter)
#
# Written out by hand on purpose. Every line is a decision D3 or D6 made, and
# changing one should require editing this file and saying why.
EXPECTED_FOREIGN_KEYS = {
    # --- commerce: the half that holds order history ---------------------
    ("order_items", "variant_id"): (
        "variants",
        "r",
        "RESTRICT: Postgres refuses DELETE FROM products while any order line "
        "points into the catalog. This is the load-bearing one — the guard in "
        "seed_catalog.py is only a courtesy in front of it.",
    ),
    ("cart_items", "variant_id"): (
        "variants",
        "c",
        "CASCADE: the opposite answer to the same question. A product that no "
        "longer exists cannot be bought and has no business in a basket.",
    ),
    ("order_items", "order_id"): (
        "orders",
        "c",
        "CASCADE: an order's lines have no meaning without the order.",
    ),
    ("cart_items", "cart_id"): (
        "carts",
        "c",
        "CASCADE: a cart's lines have no meaning without the cart.",
    ),
    ("orders", "cart_id"): (
        "carts",
        "a",
        "NO ACTION: an order records which cart it came from, and a cart an "
        "order points at is not something to delete quietly.",
    ),
    # --- catalog: what makes `--reset` a single DELETE --------------------
    ("variants", "product_id"): (
        "products",
        "c",
        "CASCADE: reset_catalog issues one DELETE FROM products and relies on "
        "the cascade to clear the rest.",
    ),
    ("prices", "variant_id"): ("variants", "c", "CASCADE: same cascade."),
    ("inventory", "variant_id"): ("variants", "c", "CASCADE: same cascade."),
}

FOREIGN_KEY_QUERY = text(
    """
    SELECT c.conrelid::regclass::text  AS tbl,
           a.attname                   AS col,
           c.confrelid::regclass::text AS refs,
           c.confdeltype               AS ondelete,
           c.conname                   AS name
    FROM pg_constraint c
    JOIN unnest(c.conkey) WITH ORDINALITY AS k(attnum, ord) ON true
    JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum
    WHERE c.contype = 'f'
    """
)


@pytest.fixture(scope="module")
def actual_foreign_keys(engine):
    """Every foreign key the database actually has, straight from the catalog."""
    with engine.connect() as connection:
        rows = connection.execute(FOREIGN_KEY_QUERY).all()
    return {(row.tbl, row.col): row for row in rows}


@pytest.mark.parametrize(
    ("table", "column"), sorted(EXPECTED_FOREIGN_KEYS), ids=lambda v: str(v)
)
def test_the_foreign_key_exists_with_the_delete_action_it_was_given(
    actual_foreign_keys, table, column
):
    referenced, ondelete, why = EXPECTED_FOREIGN_KEYS[(table, column)]

    found = actual_foreign_keys.get((table, column))
    assert found is not None, (
        f"{table}.{column} has no foreign key in this database. It was "
        f"declared ON DELETE {DELETE_ACTIONS[ondelete]} — {why}\n"
        "A DROP TABLE ... CASCADE on the catalog removes these, and create_all "
        "does not put them back. Restore it with ALTER TABLE ... ADD CONSTRAINT."
    )

    assert found.refs == referenced, (
        f"{table}.{column} points at {found.refs}, expected {referenced}"
    )

    assert found.ondelete == ondelete, (
        f"{table}.{column} is ON DELETE {DELETE_ACTIONS.get(found.ondelete, found.ondelete)}, "
        f"expected ON DELETE {DELETE_ACTIONS[ondelete]}.\n{why}"
    )


def test_no_foreign_key_exists_that_this_file_has_not_accounted_for(
    actual_foreign_keys,
):
    """Drift in the other direction: a constraint added without a decision.

    Keeps the table above honest. A new foreign key is fine — it just has to be
    written down here with the reason its delete action is what it is, which is
    the same standard every existing line was held to.
    """
    tracked_tables = {table for table, _ in EXPECTED_FOREIGN_KEYS}
    unaccounted = {
        key
        for key in actual_foreign_keys
        if key[0] in tracked_tables and key not in EXPECTED_FOREIGN_KEYS
    }

    assert not unaccounted, (
        f"these foreign keys exist but are not listed in this file: "
        f"{sorted(unaccounted)}. Add them with the reason for their ON DELETE "
        "action, or drop them."
    )


def test_restrict_is_not_quietly_no_action(actual_foreign_keys):
    """The single most consequential letter in the schema, on its own.

    `a` and `r` differ by one character and by whether a customer's order
    history survives a catalog reset. Worth a test that names it rather than
    leaving it as one row of a parametrised sweep.
    """
    found = actual_foreign_keys.get(("order_items", "variant_id"))

    assert found is not None, "order_items.variant_id has no foreign key at all"
    assert found.ondelete == "r", (
        f"order_items.variant_id is ON DELETE "
        f"{DELETE_ACTIONS.get(found.ondelete, found.ondelete)}. "
        "DELETE FROM products would take order line items with it."
    )


# --- the derived check, which covers what this file does not -------------


def test_the_models_and_the_database_agree_about_every_foreign_key(engine):
    """`find_foreign_key_gaps` is what `scripts/create_schema.py` reports with.

    Derived from `Base.metadata`, so it covers foreign keys nobody has written
    into `EXPECTED_FOREIGN_KEYS` yet — the cost being that it can only find
    disagreement between the models and the database, never a case where both
    are wrong together. That is the case the hard-coded table above is for.
    """
    import shopagent.api.models  # noqa: F401  (registers the commerce tables)

    from shopagent.db import find_foreign_key_gaps

    gaps = find_foreign_key_gaps(engine)

    assert not gaps, "\n".join(gap.describe() for gap in gaps)
