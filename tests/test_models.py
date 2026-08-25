"""Tests for shopagent.catalog.models (D3).

These are the first tests in the project that touch a database. Everything up
to D2 could be verified against fakes, but a schema is a claim about what
Postgres will accept, and only Postgres can settle it: a CHECK constraint, an
ON DELETE CASCADE and a column type are all things the ORM merely *requests*.
Asserting them against SQLAlchemy's own metadata would be asserting that the
source file says what it says.

So: a local Postgres (`docker compose up -d`), and no network beyond it. If it
is not reachable the module skips with an explanation rather than failing —
the other 164 tests still have nothing to do with a database.

The `engine` and `session` fixtures live in `tests/conftest.py`, shared with
`tests/test_seed.py`: every test here runs inside a transaction that is rolled
back afterwards, so nothing it writes survives.
"""

from __future__ import annotations

import pytest
from sqlalchemy import delete, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from shopagent.catalog.models import EMBEDDING_DIM, Inventory, Price, Product, Variant

# Every test in this file talks to Postgres. See the marker table in
# pyproject.toml: `pytest tests/` runs these, and skips nothing offline except
# the API-backed ones.
pytestmark = pytest.mark.db


def make_product(session: Session, *, sku: str = "SKU-RUN-42") -> Product:
    """A product with one variant, one price and stock — the common fixture body."""
    product = Product(
        name="Trail Runner 3",
        description="Lightweight road and trail running shoe.",
        category="shoes",
        brand="Fleetfoot",
        variants=[
            Variant(
                size="42",
                color="blue",
                sku=sku,
                prices=[Price(currency="usd", amount_cents=8999)],
                inventory=Inventory(quantity=12, reserved=2),
            )
        ],
    )
    session.add(product)
    session.commit()
    return product


# --- relationships -----------------------------------------------------


def test_product_variants_prices_and_inventory_are_reachable_through_relationships(session):
    product = make_product(session)
    session.expire_all()

    stored = session.get(Product, product.id)
    assert stored is not None
    assert len(stored.variants) == 1

    variant = stored.variants[0]
    assert variant.size == "42"
    assert variant.sku == "SKU-RUN-42"
    assert variant.product.id == stored.id
    assert [price.amount_cents for price in variant.prices] == [8999]
    assert variant.inventory is not None
    assert variant.inventory.quantity - variant.inventory.reserved == 10


def test_price_defaults_apply_without_being_passed(session):
    product = make_product(session)
    price = product.variants[0].prices[0]

    assert price.currency == "usd"
    assert price.active is True


# --- cascade -----------------------------------------------------------


def test_deleting_a_product_through_the_orm_deletes_its_variants_and_prices(session):
    product = make_product(session)
    variant_id = product.variants[0].id

    session.delete(product)
    session.commit()

    assert session.get(Variant, variant_id) is None
    assert session.scalars(select(Price).where(Price.variant_id == variant_id)).first() is None
    assert session.get(Inventory, variant_id) is None


def test_a_delete_that_bypasses_the_orm_still_cascades(session):
    """The database enforces it too, not only the session.

    `passive_deletes=True` means the ORM stops issuing child DELETEs itself, so
    this is the half of the cascade that has to hold: a bulk DELETE, or anyone
    reaching the table with psql, must not leave orphaned variants behind.
    """
    product = make_product(session)
    variant_id = product.variants[0].id

    session.execute(delete(Product).where(Product.id == product.id))
    session.commit()
    session.expire_all()

    assert session.get(Variant, variant_id) is None


# --- constraints -------------------------------------------------------


def test_a_duplicate_sku_is_rejected(session):
    make_product(session, sku="SKU-DUP-1")

    duplicate = Product(
        name="Trail Runner 3 (relisted)",
        description="The same shoe, entered twice by mistake.",
        category="shoes",
        brand="Fleetfoot",
        variants=[Variant(size="43", color="black", sku="SKU-DUP-1")],
    )
    session.add(duplicate)

    with pytest.raises(IntegrityError) as excinfo:
        session.commit()

    assert "sku" in str(excinfo.value).lower()


def test_a_negative_price_is_rejected(session):
    product = make_product(session)
    product.variants[0].prices.append(Price(currency="usd", amount_cents=-1))

    with pytest.raises(IntegrityError):
        session.commit()


# --- money is an integer -----------------------------------------------


def test_amount_cents_is_an_integer_column_in_the_database(engine):
    """Introspect the live column, do not trust the model declaration.

    The rule this guards is in CLAUDE.md: money is minor units as `int`. A
    float or a NUMERIC here would still pass every Python-level assertion in
    this file and only surface as a rounding error at checkout on D7.
    """
    columns = {column["name"]: column for column in inspect(engine).get_columns("prices")}

    assert str(columns["amount_cents"]["type"]) == "INTEGER"


def test_amount_cents_stores_an_int_and_returns_an_int(session):
    product = make_product(session)
    session.expire_all()

    amount = session.get(Product, product.id).variants[0].prices[0].amount_cents
    assert amount == 8999
    assert isinstance(amount, int)
    assert not isinstance(amount, bool)


# --- embedding ---------------------------------------------------------


def test_embedding_is_null_until_the_embedding_pass_fills_it(session):
    product = make_product(session)
    session.expire_all()

    assert session.get(Product, product.id).embedding is None


def test_embedding_column_is_a_vector_of_the_expected_dimension(engine):
    columns = {column["name"]: column for column in inspect(engine).get_columns("products")}

    assert str(columns["embedding"]["type"]) == f"VECTOR({EMBEDDING_DIM})"
