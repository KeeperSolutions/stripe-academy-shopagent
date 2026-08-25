"""Tests for shopagent.catalog.search (D3, step 3).

The seed is deterministic, so these assert against named products rather than
against counts alone: "Summit Peak Pro costs 14999 and must not survive a
10000 filter" is a claim worth making, "some product was excluded" is not.

Ids are looked up by sku or name, never written down. `--reset` renumbers every
row, and a test pinned to id 1011 would pass until the day someone reseeds.

Most tests pass `session=session` so they see the rows written inside the
rolled-back transaction. One deliberately does not, to exercise the path where
the function opens its own session — the one D4's MCP server will use.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from shopagent.catalog.models import Product, Variant
from shopagent.catalog.search import (
    candidate_products_statement,
    check_stock,
    get_product,
    search_products,
)
from shopagent.catalog.seed import seed_catalog


@pytest.fixture
def catalog(session):
    """A seeded catalog inside the test transaction.

    `seed_catalog` is idempotent, so this fills the gaps on an empty database
    and does nothing on a seeded one. Either way the test sees the full
    catalog, and the rollback leaves the database as it found it.
    """
    seed_catalog(session)
    return session


def variant_id_for(session, sku: str) -> int:
    return session.scalars(select(Variant.id).where(Variant.sku == sku)).one()


def product_id_for(session, name: str) -> int:
    return session.scalars(select(Product.id).where(Product.name == name)).one()


def names(results: list[dict]) -> set[str]:
    return {result["name"] for result in results}


def every_variant(results: list[dict]):
    for result in results:
        yield from result["variants"]


# --- search_products: the basics ---------------------------------------


def test_an_unfiltered_search_returns_products_and_respects_the_limit(catalog):
    results = search_products(limit=3, session=catalog)

    assert len(results) == 3
    assert all(result["variants"] for result in results)
    assert all(isinstance(v["price_cents"], int) for v in every_variant(results))


def test_a_search_that_matches_nothing_returns_an_empty_list_not_none(catalog):
    results = search_products(
        query="Cloud Sprint 2", category="bags", max_price_cents=100, session=catalog
    )

    assert results == []


# --- price filters -----------------------------------------------------


def test_max_price_excludes_the_expensive_shoe(catalog):
    results = search_products(
        category="shoes", max_price_cents=10000, limit=20, session=catalog
    )

    assert "Summit Peak Pro" not in names(results)
    assert results
    assert all(v["price_cents"] <= 10000 for v in every_variant(results))


def test_min_price_excludes_the_cheap_shoes(catalog):
    results = search_products(
        category="shoes", min_price_cents=10000, limit=20, session=catalog
    )

    assert "Summit Peak Pro" in names(results)
    assert "Harbor Slip-On" not in names(results)
    assert all(v["price_cents"] >= 10000 for v in every_variant(results))


def test_min_and_max_together_describe_a_band(catalog):
    results = search_products(
        min_price_cents=7000, max_price_cents=9000, limit=50, session=catalog
    )

    assert results
    assert all(7000 <= v["price_cents"] <= 9000 for v in every_variant(results))


# --- category, size, colour --------------------------------------------


def test_category_filters(catalog):
    results = search_products(category="jackets", limit=50, session=catalog)

    assert results
    assert all(result["category"] == "jackets" for result in results)


def test_size_filters_the_variants_not_only_the_products(catalog):
    """The point of the whole two-pass query: no 43 in an answer about 42."""
    results = search_products(size="42", limit=50, session=catalog)

    assert results
    assert all(variant["size"] == "42" for variant in every_variant(results))


def test_colour_filters_the_variants_too(catalog):
    results = search_products(color="blue", limit=50, session=catalog)

    assert results
    assert all(variant["color"] == "blue" for variant in every_variant(results))


def test_size_and_colour_match_case_insensitively(catalog):
    lower = search_products(color="blue", limit=50, session=catalog)
    upper = search_products(color="BLUE", limit=50, session=catalog)

    assert names(lower) == names(upper)


# --- the keyword half of the comparison --------------------------------


def test_query_matches_a_product_name(catalog):
    results = search_products(query="Trail Runner", limit=50, session=catalog)

    assert names(results) == {"Trail Runner GTX"}


def test_query_matches_words_only_in_the_description(catalog):
    results = search_products(query="carbon plate", limit=50, session=catalog)

    assert names(results) == {"Summit Peak Pro"}


def test_query_matches_a_brand(catalog):
    results = search_products(query="Cobbleway", limit=50, session=catalog)

    assert results
    assert all(result["brand"] == "Cobbleway" for result in results)


def test_the_weather_word_finds_nothing_at_all(catalog):
    """The baseline step 4 has to beat.

    Keyword search cannot answer this question, because the catalog never says
    the word. When the same query returns shoes in step 4, the difference is
    the embedding and nothing else.
    """
    assert search_products(query="rain", limit=50, session=catalog) == []
    assert search_products(query="raining", limit=50, session=catalog) == []


def test_a_wildcard_in_the_query_is_matched_literally(catalog):
    """A model writing '%' must not turn the search into "everything"."""
    assert search_products(query="%", limit=50, session=catalog) == []


# --- prices that are no longer offered ---------------------------------


def test_a_superseded_price_never_reaches_the_result(catalog):
    """Cloud Sprint 2 size 42 costs 7499 and used to cost 8999."""
    results = search_products(query="Cloud Sprint", limit=50, session=catalog)

    prices = {
        variant["price_cents"]
        for variant in every_variant(results)
        if variant["sku"] == "AE-CLDSP2-42-WHT"
    }

    assert prices == {7499}


def test_the_superseded_price_does_not_duplicate_its_variant(catalog):
    results = search_products(query="Cloud Sprint", limit=50, session=catalog)

    skus = [variant["sku"] for variant in every_variant(results)]

    assert len(skus) == len(set(skus))


# --- the filters really are in SQL -------------------------------------


def test_an_expensive_filter_survives_a_small_limit(catalog):
    """Filtering in Python after LIMIT would return nothing here.

    Results are ordered cheapest first, so the two cheapest shoes are Harbor
    Slip-On at 4999 and Studio Flex at 6499. A search that fetched `limit`
    products and then dropped the ones under 13000 in Python would hand back an
    empty list. The database applies the predicate before the limit, so the one
    shoe that qualifies is found.
    """
    results = search_products(
        category="shoes", min_price_cents=13000, limit=2, session=catalog
    )

    assert names(results) == {"Summit Peak Pro"}


def test_a_variant_level_filter_survives_a_small_limit(catalog):
    """Same argument, for a filter that lives on the variant rather than price.

    The two cheapest products in the shop are an armband and a pair of socks,
    neither of them blue. Python-side filtering after a limit of two would
    therefore find no blue anything.
    """
    results = search_products(color="blue", limit=2, session=catalog)

    assert len(results) == 2
    assert all(variant["color"] == "blue" for variant in every_variant(results))


def test_the_generated_sql_carries_every_predicate_in_its_where_clause():
    """Read the statement itself, so the claim does not rest on outcomes alone."""
    statement = candidate_products_statement(
        query="running",
        category="shoes",
        max_price_cents=10000,
        min_price_cents=5000,
        size="42",
        color="black",
        limit=5,
    )
    # The PostgreSQL dialect, not the default one: the generic compiler renders
    # ILIKE as `lower(a) LIKE lower(b)`, which is not the SQL that runs here.
    sql = str(
        statement.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )
    where = sql.split("WHERE", 1)[1]

    assert "amount_cents <= 10000" in where
    assert "amount_cents >= 5000" in where
    assert "lower(variants.size) = '42'" in where
    assert "lower(variants.color) = 'black'" in where
    assert "lower(products.category) = 'shoes'" in where
    # The psycopg paramstyle doubles a literal '%', so the pattern renders as
    # '%%running%%' — assert on the operator and the word, not the escaping.
    assert "ILIKE" in where
    assert "running" in where
    assert "LIMIT 5" in sql


# --- get_product -------------------------------------------------------


def test_get_product_returns_every_variant_including_the_sold_out_one(catalog):
    product_id = product_id_for(catalog, "Storm Pace 4")

    product = get_product(product_id, session=catalog)

    assert product is not None
    assert product["name"] == "Storm Pace 4"
    sizes = {variant["size"] for variant in product["variants"]}
    assert sizes == {"41", "42", "43"}
    sold_out = [v for v in product["variants"] if v["sku"] == "FF-STRMP4-43-NVY"]
    assert sold_out[0]["available"] == 0


def test_get_product_ignores_the_filters_a_search_would_apply(catalog):
    """No price, size or stock filtering: this is the detail view."""
    product_id = product_id_for(catalog, "Cloud Sprint 2")

    product = get_product(product_id, session=catalog)

    assert len(product["variants"]) == 3
    assert all(variant["price_cents"] == 7499 for variant in product["variants"])


def test_get_product_returns_none_for_an_unknown_id(catalog):
    assert get_product(10_000_000, session=catalog) is None


# --- check_stock -------------------------------------------------------


def test_check_stock_subtracts_what_is_reserved(catalog):
    variant_id = variant_id_for(catalog, "FF-TRLGTX-42-BLK")

    stock = check_stock(variant_id, session=catalog)

    assert stock == {
        "variant_id": variant_id,
        "sku": "FF-TRLGTX-42-BLK",
        "product_name": "Trail Runner GTX",
        "size": "42",
        "color": "black",
        "quantity": 12,
        "reserved": 2,
        "available": 10,
        "in_stock": True,
    }


def test_check_stock_reports_a_sold_out_variant_as_out_of_stock(catalog):
    variant_id = variant_id_for(catalog, "FF-STRMP4-43-NVY")

    stock = check_stock(variant_id, session=catalog)

    assert stock["quantity"] == 0
    assert stock["available"] == 0
    assert stock["in_stock"] is False


def test_check_stock_returns_none_for_an_unknown_id(catalog):
    assert check_stock(10_000_000, session=catalog) is None


# --- the session argument ----------------------------------------------


def test_the_functions_work_without_a_session_being_passed(engine):
    """The path D4 takes: no session in hand, just a question.

    Reads the committed catalog rather than a test transaction, so it skips
    when the database has not been seeded.
    """
    results = search_products(limit=1)
    if not results:
        pytest.skip("The committed catalog is empty. Run: python scripts/seed_catalog.py")

    assert results[0]["variants"]
    assert get_product(results[0]["product_id"]) is not None
