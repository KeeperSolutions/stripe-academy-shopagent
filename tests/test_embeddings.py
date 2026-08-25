"""Tests for shopagent.catalog.embeddings and the semantic half of search (D3, step 4).

Split three ways, matching the markers in pyproject.toml:

* unmarked — the text builder and the compiled SQL. No database, no API.
* `db` — ranking, proved with vectors written straight into Postgres by hand.
  One-hot vectors make the expected order arithmetic rather than a guess, and
  no token is spent to check that `<=>` orders rows the way the code claims.
* `network` — the real thing: a real query, embedded by the real model, against
  the real catalog. Deselected by default; run with `pytest tests/ -m network`.

The middle group is the point. Whether the ranking mechanism works and whether
the embedding model is any good are two separate questions, and only the second
one needs to be paid for.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from shopagent.catalog.embeddings import embedding_text
from shopagent.catalog.models import EMBEDDING_DIM, Product
from shopagent.catalog.search import candidate_products_statement, search_products
from shopagent.catalog.seed import seed_catalog

# The query the whole day is built around. The catalog never contains the word.
WET_QUERY = "something to run in when it's raining"
DRY_QUERY = "keeping my gear dry on a boat"


def one_hot(index: int) -> list[float]:
    """A unit vector along one axis.

    Two of these are orthogonal, so their cosine distance is exactly 1, and a
    vector against itself is exactly 0. That makes every assertion below a
    statement about arithmetic rather than about a model's judgement.
    """
    vector = [0.0] * EMBEDDING_DIM
    vector[index] = 1.0
    return vector


# --- offline: the text that gets embedded ------------------------------


def test_embedding_text_joins_all_four_fields():
    product = Product(
        name="Trail Runner GTX",
        brand="Fleetfoot",
        category="shoes",
        description="A GORE-TEX lined running shoe for the grey months.",
    )

    text = embedding_text(product)

    assert "Trail Runner GTX" in text
    assert "Fleetfoot" in text
    assert "shoes" in text
    assert "A GORE-TEX lined running shoe for the grey months." in text


def test_embedding_text_is_not_the_description_alone():
    """The plan's requirement, stated as a test.

    Embedding the description by itself loses the brand and the category, so a
    search for "Fleetfoot" would match nothing at all.
    """
    product = Product(
        name="Storm Pace 4",
        brand="Fleetfoot",
        category="shoes",
        description="A road running shoe.",
    )

    assert embedding_text(product) != product.description
    assert embedding_text(product).endswith(product.description)


# --- offline: the SQL a semantic search generates -----------------------


def compiled(statement) -> str:
    return str(
        statement.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )


def test_semantic_sql_orders_by_cosine_distance():
    sql = compiled(
        candidate_products_statement(query_embedding=one_hot(0), limit=5)
    )

    assert "<=>" in sql
    assert "ORDER BY distance ASC NULLS LAST" in sql


def test_semantic_sql_keeps_every_filter_in_the_where_clause():
    """The trap from the plan: ranking must not step around the filters."""
    sql = compiled(
        candidate_products_statement(
            query_embedding=one_hot(0),
            category="shoes",
            max_price_cents=10000,
            min_price_cents=5000,
            size="42",
            color="black",
            limit=5,
        )
    )
    where, order_by = sql.split("WHERE", 1)[1].split("ORDER BY", 1)

    assert "amount_cents <= 10000" in where
    assert "amount_cents >= 5000" in where
    assert "lower(variants.size) = '42'" in where
    assert "lower(variants.color) = 'black'" in where
    assert "lower(products.category) = 'shoes'" in where
    # The distance expression is computed in the SELECT list and ordered by its
    # label, so the ranking shows up in two places and the filters in neither.
    assert "<=>" in sql.split("FROM", 1)[0]
    assert order_by.strip().startswith("distance ASC NULLS LAST")
    # And the LIMIT is last of all: filter, then rank, then truncate.
    assert sql.rindex("LIMIT") > sql.index("WHERE")


def test_a_semantic_query_is_not_also_matched_literally():
    """A vector search filtered by ILIKE would find only what it already knew."""
    sql = compiled(
        candidate_products_statement(query="raining", query_embedding=one_hot(0))
    )

    assert "ILIKE" not in sql


def test_a_keyword_query_does_not_rank_by_distance():
    sql = compiled(candidate_products_statement(query="running"))

    assert "<=>" not in sql
    assert "ILIKE" in sql


# --- with a database: the ranking mechanism ----------------------------


@pytest.mark.db
class TestRankingWithHandWrittenVectors:
    """Ranking, proved without spending a token.

    Every product gets a one-hot vector, so "nearest" is decided by which axis
    the query points along. The catalog's real embeddings are overwritten
    inside the test transaction and restored by the rollback.
    """

    @pytest.fixture
    def ranked(self, session):
        seed_catalog(session)
        products = list(session.scalars(select(Product).order_by(Product.id)))
        for index, product in enumerate(products):
            product.embedding = one_hot(index)
        session.commit()
        return session, {product.name: index for index, product in enumerate(products)}

    def test_the_nearest_product_comes_first(self, ranked):
        session, axis = ranked

        results = search_products(
            query=WET_QUERY,
            query_embedding=one_hot(axis["Merino Beanie"]),
            limit=3,
            session=session,
        )

        assert results[0]["name"] == "Merino Beanie"

    def test_a_mixed_query_vector_orders_by_how_close_each_product_is(self, ranked):
        session, axis = ranked
        vector = one_hot(axis["Dry Duffel 40L"])
        vector[axis["Foam Roller"]] = 0.5

        results = search_products(
            query=DRY_QUERY, query_embedding=vector, limit=3, session=session
        )

        assert [result["name"] for result in results[:2]] == [
            "Dry Duffel 40L",
            "Foam Roller",
        ]

    def test_ranking_does_not_step_around_the_price_filter(self, ranked):
        """The nearest product is too expensive, so it must not be returned.

        Summit Peak Pro at 14999 is the closest match by construction. Ranking
        first and filtering the results afterwards would either return it or
        return a short list with a hole in it; filtering in SQL returns the
        nearest product that actually qualifies.
        """
        session, axis = ranked
        vector = one_hot(axis["Summit Peak Pro"])
        vector[axis["Storm Pace 4"]] = 0.5

        results = search_products(
            query=WET_QUERY,
            query_embedding=vector,
            max_price_cents=10000,
            limit=3,
            session=session,
        )

        names = [result["name"] for result in results]
        assert "Summit Peak Pro" not in names
        assert names[0] == "Storm Pace 4"
        assert all(
            variant["price_cents"] <= 10000
            for result in results
            for variant in result["variants"]
        )

    def test_ranking_respects_a_variant_level_filter_too(self, ranked):
        session, axis = ranked

        results = search_products(
            query=WET_QUERY,
            query_embedding=one_hot(axis["Trail Runner GTX"]),
            size="42",
            limit=5,
            session=session,
        )

        assert results[0]["name"] == "Trail Runner GTX"
        assert all(
            variant["size"] == "42"
            for result in results
            for variant in result["variants"]
        )

    def test_a_product_without_a_vector_sorts_behind_every_product_with_one(
        self, ranked
    ):
        session, axis = ranked
        unembedded = session.scalars(
            select(Product).where(Product.name == "Foam Roller")
        ).one()
        unembedded.embedding = None
        session.commit()

        results = search_products(
            query=DRY_QUERY,
            query_embedding=one_hot(axis["Foam Roller"]),
            limit=30,
            session=session,
        )

        names = [result["name"] for result in results]
        assert names[-1] == "Foam Roller"


# --- with the API: is the model any good -------------------------------


@pytest.mark.network
@pytest.mark.db
def test_the_wet_weather_query_finds_the_weatherproof_shoes():
    """The D3 definition of done, run against the real model.

    Neither shoe's description contains the query's weather word — the guard in
    `tests/test_seed.py` makes sure of that — so a keyword search returns
    nothing at all here.
    """
    results = search_products(WET_QUERY, limit=3)

    names = {result["name"] for result in results}
    assert {"Trail Runner GTX", "Storm Pace 4"} <= names


@pytest.mark.network
@pytest.mark.db
def test_the_definition_of_done_holds_under_a_price_filter():
    results = search_products(WET_QUERY, max_price_cents=10000, limit=3)

    assert results
    assert all(
        variant["price_cents"] <= 10000
        for result in results
        for variant in result["variants"]
    )
    assert "shoes" in {result["category"] for result in results}


@pytest.mark.network
@pytest.mark.db
def test_the_same_query_finds_nothing_by_keyword():
    assert search_products(WET_QUERY, mode="keyword", limit=3) == []


@pytest.mark.network
@pytest.mark.db
def test_a_second_query_with_no_lexical_overlap_also_works():
    results = search_products(DRY_QUERY, limit=3)

    names = {result["name"] for result in results}
    assert names & {"Dry Duffel 40L", "Compact Dry Bag 10L"}
