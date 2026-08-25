"""Tests for shopagent.catalog.seed (D3, step 2).

Two jobs. The first is mechanical: seeding an empty catalog produces the counts
the spec implies, and seeding twice produces them once.

The second is to guard the D3 definition of done. Step 4 claims that a query
about running in wet weather finds the weatherproof shoes without the query
word appearing anywhere in the data. That claim is only worth something while
the data stays clear of the word, so the last test in this file checks every
name, description, category and brand for those four letters as a substring. If
someone later adds a product whose copy says it outright, the semantic proof
collapses into a keyword match — and this test fails first.

Fixtures come from `tests/conftest.py`; everything written here is rolled back.
"""

from __future__ import annotations

import pytest
from sqlalchemy import delete, func, select

from shopagent.catalog.models import Inventory, Price, Product, Variant
from shopagent.catalog.seed import CATALOG, reset_catalog, seed_catalog

# Derived from the spec rather than typed in, so adding a product does not
# break the arithmetic. The floors below are what the day actually requires.
EXPECTED_PRODUCTS = len(CATALOG)
EXPECTED_VARIANTS = sum(len(product.variants) for product in CATALOG)
EXPECTED_PRICES = sum(
    2 if variant.previous_amount_cents is not None else 1
    for product in CATALOG
    for variant in product.variants
)

# The word the catalog must never contain. Written as a substring check because
# "rainproof" and "rainy" have to fail too.
FORBIDDEN = "rain"


@pytest.fixture
def empty_session(session):
    """A session whose catalog starts empty, however the database looked.

    The delete runs inside the test transaction, so the seeded rows come back
    on rollback. That is what lets these tests assert "seeding an empty
    catalog gives thirty products" on a database that already holds thirty.
    """
    session.execute(delete(Product))
    session.commit()
    return session


def count(session, model) -> int:
    return session.scalar(select(func.count()).select_from(model))


# --- counts ------------------------------------------------------------


def test_seeding_an_empty_catalog_writes_every_row(empty_session):
    summary = seed_catalog(empty_session)

    assert summary.products_created == EXPECTED_PRODUCTS
    assert summary.variants_created == EXPECTED_VARIANTS
    assert summary.prices_created == EXPECTED_PRICES
    assert summary.inventory_created == EXPECTED_VARIANTS
    assert summary.products_skipped == 0

    assert count(empty_session, Product) == EXPECTED_PRODUCTS
    assert count(empty_session, Variant) == EXPECTED_VARIANTS
    assert count(empty_session, Price) == EXPECTED_PRICES
    assert count(empty_session, Inventory) == EXPECTED_VARIANTS


def test_seeding_twice_writes_nothing_the_second_time(empty_session):
    seed_catalog(empty_session)
    second = seed_catalog(empty_session)

    assert second.products_created == 0
    assert second.variants_created == 0
    assert second.prices_created == 0
    assert second.products_skipped == EXPECTED_PRODUCTS

    assert count(empty_session, Product) == EXPECTED_PRODUCTS
    assert count(empty_session, Variant) == EXPECTED_VARIANTS


def test_a_removed_variant_is_restored_without_duplicating_its_product(empty_session):
    """Idempotency keys on the sku, so a partial catalog fills its own gaps."""
    seed_catalog(empty_session)
    sku = "FF-TRLGTX-42-BLK"
    empty_session.execute(delete(Variant).where(Variant.sku == sku))
    empty_session.commit()

    summary = seed_catalog(empty_session)

    assert summary.products_created == 0
    assert summary.variants_created == 1
    assert count(empty_session, Product) == EXPECTED_PRODUCTS
    assert count(empty_session, Variant) == EXPECTED_VARIANTS


def test_reset_clears_the_catalog_and_everything_hanging_off_it(empty_session):
    seed_catalog(empty_session)

    deleted = reset_catalog(empty_session)

    assert deleted == EXPECTED_PRODUCTS
    assert count(empty_session, Product) == 0
    assert count(empty_session, Variant) == 0
    assert count(empty_session, Price) == 0
    assert count(empty_session, Inventory) == 0


# --- the demo scenario -------------------------------------------------


def active_price_rows(session):
    """Every (variant, price) pair the shop would actually quote."""
    return session.execute(
        select(Variant, Price).join(Price).where(Price.active.is_(True))
    ).all()


def test_there_are_running_shoes_under_a_hundred_dollars(empty_session):
    seed_catalog(empty_session)

    affordable = empty_session.scalars(
        select(Product)
        .join(Variant)
        .join(Price)
        .where(
            Product.category == "shoes",
            Product.description.ilike("%running%"),
            Price.active.is_(True),
            Price.amount_cents < 10000,
        )
        .distinct()
    ).all()

    assert len(affordable) >= 3


def test_a_running_shoe_above_a_hundred_dollars_exists_for_the_filter_to_reject(
    empty_session,
):
    seed_catalog(empty_session)

    expensive = empty_session.scalars(
        select(Product)
        .join(Variant)
        .join(Price)
        .where(
            Product.category == "shoes",
            Price.active.is_(True),
            Price.amount_cents > 10000,
        )
        .distinct()
    ).all()

    assert expensive


def test_there_is_a_blue_jacket_under_eighty_dollars(empty_session):
    seed_catalog(empty_session)

    jackets = empty_session.scalars(
        select(Product)
        .join(Variant)
        .join(Price)
        .where(
            Product.category == "jackets",
            Variant.color == "blue",
            Price.active.is_(True),
            Price.amount_cents < 8000,
        )
        .distinct()
    ).all()

    assert jackets


def test_size_42_is_available(empty_session):
    seed_catalog(empty_session)

    variants = empty_session.scalars(select(Variant).where(Variant.size == "42")).all()

    assert variants


# --- shapes the later days depend on -----------------------------------


def test_at_least_three_variants_are_out_of_stock(empty_session):
    """D9 guards checkout on stock, and can only be tested against a zero."""
    seed_catalog(empty_session)

    empty = empty_session.scalars(
        select(Inventory).where(Inventory.quantity == 0)
    ).all()

    assert len(empty) >= 3


def test_some_stock_is_reserved(empty_session):
    seed_catalog(empty_session)

    reserved = empty_session.scalars(
        select(Inventory).where(Inventory.reserved > 0)
    ).all()

    assert reserved


def test_a_variant_exists_with_neither_size_nor_colour(empty_session):
    """Both columns are nullable; the seed has to prove it means it."""
    seed_catalog(empty_session)

    plain = empty_session.scalars(
        select(Variant).where(Variant.size.is_(None), Variant.color.is_(None))
    ).all()

    assert plain


def test_exactly_one_variant_carries_a_superseded_price(empty_session):
    seed_catalog(empty_session)

    inactive = empty_session.scalars(
        select(Price).where(Price.active.is_(False))
    ).all()

    assert len(inactive) == 1
    assert count(empty_session, Price) == count(empty_session, Variant) + 1


def test_every_variant_has_exactly_one_active_price(empty_session):
    seed_catalog(empty_session)

    assert len(active_price_rows(empty_session)) == EXPECTED_VARIANTS


def test_the_catalog_spans_at_least_five_categories(empty_session):
    seed_catalog(empty_session)

    categories = set(empty_session.scalars(select(Product.category)))

    assert len(categories) >= 5
    assert all(category == category.lower() for category in categories)


def test_every_sku_is_unique_and_readable(empty_session):
    seed_catalog(empty_session)

    skus = list(empty_session.scalars(select(Variant.sku)))

    assert len(skus) == len(set(skus))
    # Readable means a person can tell what it is: brand, model, then the
    # variant. A UUID would satisfy uniqueness and nothing else.
    assert all("-" in sku and sku.isupper() for sku in skus)


# --- the guard on the definition of done --------------------------------


def test_no_stored_text_contains_the_word_the_semantic_query_will_use(empty_session):
    """The whole point of step 4 rests on this staying true.

    If a description says the word outright, then a `LIKE '%...%'` finds the
    shoes too, and the comparison between keyword and semantic search stops
    demonstrating anything.
    """
    seed_catalog(empty_session)

    products = empty_session.scalars(select(Product)).all()
    offenders = [
        (product.name, field)
        for product in products
        for field in (product.name, product.description, product.category, product.brand)
        if FORBIDDEN in field.lower()
    ]

    assert offenders == []
