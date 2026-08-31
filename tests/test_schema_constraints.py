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


# --- what review on PR #7 turned up --------------------------------------


def test_the_gap_check_notices_a_foreign_key_pointing_somewhere_else(engine):
    """Comparing the delete action alone called a wrong target a match.

    A constraint on the right columns with the right ON DELETE but referencing
    the wrong table is a worse fault than a wrong ON DELETE, and the derived
    check reported it as fine. It now compares what the key points at too.
    """
    import shopagent.api.models  # noqa: F401

    from shopagent.db import ForeignKeyGap, find_foreign_key_gaps

    # The healthy state, so the assertion below means something.
    assert find_foreign_key_gaps(engine) == []

    gap = ForeignKeyGap(
        table="cart_items",
        columns=("variant_id",),
        expected_target="variants(id)",
        expected_ondelete="CASCADE",
        actual_target="products(id)",
        actual_ondelete="CASCADE",
    )
    assert not gap.is_missing
    described = gap.describe()
    assert "products(id)" in described
    assert "variants(id)" in described


def test_a_missing_column_is_reported_rather_than_discovered_at_runtime(engine):
    """`create_all` never alters an existing table.

    A commerce table that gained a column since it was built stays "already
    present", and the first symptom is `UndefinedColumn` from an ordinary read.
    `migrations/` holds the ALTER; this is what says whether it was applied.
    """
    import shopagent.api.models  # noqa: F401

    from shopagent.db import find_column_gaps

    assert find_column_gaps(engine) == [], (
        "this database is missing a column the models declare — apply the "
        "matching file from migrations/"
    )


def test_the_recorded_migration_covers_the_columns_day_7_added():
    """The ALTER has to exist in the repo, not only in somebody's shell history.

    These two columns were added to `orders` by hand during D7. Without a file
    recording that, a database built from Day 6 would lack them and nothing
    would say so until a query failed.
    """
    import pathlib

    migrations = pathlib.Path(__file__).resolve().parents[1] / "migrations"
    sql = "\n".join(path.read_text() for path in migrations.glob("*.sql"))

    for column in ("customer_email", "stripe_customer_id"):
        assert column in sql, f"no migration records adding orders.{column}"
    # Re-runnable, because nothing in this project tracks which have been run.
    assert "IF NOT EXISTS" in sql


def test_the_recorded_migration_creates_the_table_day_8_added():
    """The first *table* the convention creates, rather than columns it adds.

    Easy to think unnecessary, because `create_all` checks tables one at a
    time and builds any it does not find — a fresh clone and a pre-D8 database
    alike. An earlier version of this docstring said `create_all` would leave
    such a database alone, which is wrong; review on PR #8 corrected it.

    What the migration is for is the record: a file saying what changed, which
    a deployment can read before running it. `create_all` builds whatever it
    finds missing and says nothing about which change was needed. This test
    only asserts the file exists and is idempotent, which is the part a
    convention can enforce.
    """
    import pathlib

    migrations = pathlib.Path(__file__).resolve().parents[1] / "migrations"
    sql = "\n".join(path.read_text() for path in migrations.glob("*.sql"))

    assert "CREATE TABLE IF NOT EXISTS processed_events" in sql


def test_the_migration_and_the_model_declare_the_same_columns():
    """Two artifacts describing one table, kept from drifting apart.

    `api/models.py` is what `create_all` builds on a fresh clone;
    `migrations/0002` is what an existing database gets. A column added to one
    and not the other produces two schemas that are both "correct" depending
    on when the database was created — and `find_column_gaps` only catches it
    on a machine that actually ran the migration, which is not the machine the
    mistake is made on.

    Compared by name against the migration's text rather than by parsing SQL:
    a real parser here would be a second implementation of Postgres, and the
    failure being caught is a forgotten column, not a subtle type difference.
    """
    import pathlib

    from shopagent.api.models import ProcessedEvent

    sql = (
        pathlib.Path(__file__).resolve().parents[1]
        / "migrations"
        / "0002_d8_processed_events.sql"
    ).read_text()

    for column in ProcessedEvent.__table__.columns:
        assert column.name in sql, (
            f"the model declares processed_events.{column.name} and migration "
            "0002 does not create it — a database built from the migration "
            "would be missing it"
        )


def test_every_migration_can_be_applied_twice(session):
    """Idempotency is the whole reason this project needs no migrations table.

    A ledger of what has run is a second record of the schema, and its failure
    mode is the one this area exists to prevent: the ledger says applied while
    the column is not there. The alternative only works if re-running every
    migration is always safe — so that is asserted rather than assumed, by
    running each file twice inside a transaction that is rolled back.

    DDL is transactional in Postgres, so nothing here survives the test.
    """
    import pathlib

    migrations = sorted(
        (pathlib.Path(__file__).resolve().parents[1] / "migrations").glob("*.sql")
    )
    assert migrations, "no migrations found — this test would pass vacuously"

    for path in migrations:
        sql = text(path.read_text())
        session.execute(sql)
        # The second pass is the claim. A non-idempotent statement raises here.
        session.execute(sql)

    session.rollback()


def test_migrations_are_numbered_so_their_order_is_readable():
    """The number is the order they were written, not a version anything reads.

    Without it the directory is a set rather than a sequence, and the next
    person cannot tell which change came first.
    """
    import pathlib
    import re

    migrations = sorted(
        (pathlib.Path(__file__).resolve().parents[1] / "migrations").glob("*.sql")
    )

    for path in migrations:
        assert re.match(r"^\d{4}_[a-z0-9_]+\.sql$", path.name), (
            f"{path.name} does not match NNNN_short_description.sql"
        )

    numbers = [path.name[:4] for path in migrations]
    assert len(numbers) == len(set(numbers)), f"duplicate migration numbers: {numbers}"


# --- every model module has to be imported, or its table does not exist ---


def _modules_declaring_tables():
    """Modules under `src/shopagent` that define a mapped class.

    Found by walking the AST rather than by importing everything: importing is
    what this test is checking has been arranged, so an import here would be
    the test performing the thing it is meant to detect the absence of.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "shopagent"
    found = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        declares = any(
            isinstance(node, ast.ClassDef)
            and any(isinstance(base, ast.Name) and base.id == "Base" for base in node.bases)
            for node in ast.walk(tree)
        )
        if declares:
            found.append(".".join(path.relative_to(root.parent).with_suffix("").parts))
    return found


def _modules_imported_by(source: str) -> set[str]:
    """Every module one file imports, whichever spelling it used.

    `import a.b.c` and `from a.b import c` name the same module and share no
    substring, which is why this reads the AST instead of the text: the first
    version of this test matched on the dotted path and failed against every
    existing import in the repository, all of which use the second form.
    """
    import ast

    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


@pytest.mark.parametrize("importer", ["scripts/create_schema.py", "tests/conftest.py"])
def test_every_model_module_is_imported_where_the_schema_is_built(importer):
    """A table joins `Base.metadata` when its class is imported, and not before.

    CLAUDE.md states this and both call sites already import
    `shopagent.api.models` for the side effect alone. D9 added a third module —
    `agent/profile.py` — and the cost of forgetting it is the failure that rule
    warns about: a schema that looks complete until the first missing relation,
    with `create_all` cheerfully reporting every table it *did* know about as
    present.

    Enforced by reading the file rather than by remembering, the same way
    `test_lifecycle.py` walks the AST to prove nothing calls `transition()`
    outside `apply_transition`. A fourth model module fails this test on the
    day it is written.
    """
    import pathlib

    source = (pathlib.Path(__file__).resolve().parents[1] / importer).read_text()
    imported = _modules_imported_by(source)
    modules = _modules_declaring_tables()

    assert modules, "no model modules found — the AST walk is broken, not the imports"
    for module in modules:
        assert module in imported, (
            f"{importer} does not import {module}, so the tables it declares "
            f"are absent from Base.metadata when the schema is built"
        )


def test_the_recorded_migration_creates_the_table_day_9_added():
    """`shopper_profiles` holds what a person typed about themselves.

    No script regenerates it, so it is on the commerce side of the line and
    needs a recorded change like every other table there.
    """
    import pathlib

    migrations = pathlib.Path(__file__).resolve().parents[1] / "migrations"
    sql = "\n".join(path.read_text() for path in migrations.glob("*.sql"))

    assert "CREATE TABLE IF NOT EXISTS shopper_profiles" in sql


def test_the_day_9_migration_and_the_model_declare_the_same_columns():
    """Two artifacts describing one table, kept from drifting apart.

    Same check as `0002`, and it matters more here: every width in that file is
    part of an argument about what can be written into a system prompt, so a
    column the model declares and the migration does not create is a column
    that exists with different rules depending on when the database was built.
    """
    import pathlib

    from shopagent.agent.profile import ShopperProfile

    sql = (
        pathlib.Path(__file__).resolve().parents[1]
        / "migrations"
        / "0003_d9_shopper_profiles.sql"
    ).read_text()

    for column in ShopperProfile.__table__.columns:
        assert column.name in sql, (
            f"the model declares shopper_profiles.{column.name} and migration "
            "0003 does not create it"
        )
