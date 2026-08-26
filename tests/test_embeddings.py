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
from sqlalchemy import select, update
from sqlalchemy.dialects import postgresql

from shopagent.catalog.embeddings import (
    embed_products,
    embedded_count,
    embedding_text,
    missing_embedding_count,
)
from shopagent.catalog.models import EMBEDDING_DIM, Product
from shopagent.catalog.search import candidate_products_statement, search_products
from shopagent.catalog.seed import CATALOG, seed_catalog
from shopagent.llm.usage import UsageTracker

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
    order_by = sql.split("ORDER BY", 1)[1]

    assert "<=>" in order_by
    assert "min(" not in order_by


def test_semantic_sql_has_nothing_after_the_distance_in_its_order_by():
    """An index-eligibility guard, and it is not theoretical.

    Measured against this database: `ORDER BY embedding <=> $1` plans as an
    HNSW index scan, and `ORDER BY embedding <=> $1, products.id` plans as a
    sort of the whole table. A tie-break added here for reproducibility would
    silently cost the index, so the shape is pinned by a test rather than by a
    comment.
    """
    # Compiled with bound parameters rather than literals: a 1536-element
    # vector inlined into the SQL brings 1535 commas of its own, and the thing
    # being counted here is sort keys.
    sql = str(
        candidate_products_statement(query_embedding=one_hot(0)).compile(
            dialect=postgresql.dialect()
        )
    )
    order_by = sql.split("ORDER BY", 1)[1].split("LIMIT", 1)[0]

    assert order_by.count(",") == 0
    assert "products.id" not in order_by


def test_semantic_sql_excludes_products_that_have_no_vector():
    """Also not stylistic: an index scan never returns them, a table scan does.

    Leaving it to the planner would mean the same query returning a different
    number of rows depending on the plan it happened to choose.
    """
    sql = compiled(candidate_products_statement(query_embedding=one_hot(0)))

    assert "products.embedding IS NOT NULL" in sql


def test_semantic_sql_qualifies_variants_with_exists_not_a_group_by():
    """The restructure that made the ordering index-eligible in the first place."""
    sql = compiled(
        candidate_products_statement(query_embedding=one_hot(0), size="42")
    )

    assert "EXISTS" in sql
    assert "GROUP BY" not in sql


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
    qualification, order_by = sql.split("ORDER BY", 1)

    # The price, size and colour predicates live in the EXISTS subquery, the
    # category one in the outer WHERE. Both are qualification, and both run
    # before the ranking.
    assert "amount_cents <= 10000" in qualification
    assert "amount_cents >= 5000" in qualification
    assert "lower(variants.size) = '42'" in qualification
    assert "lower(variants.color) = 'black'" in qualification
    assert "lower(products.category) = 'shoes'" in qualification
    # Nothing is filtered by distance — it only orders.
    assert "<=>" not in qualification
    assert "<=>" in order_by
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

    def test_a_product_without_a_vector_is_left_out_of_a_semantic_search(
        self, ranked
    ):
        """Excluded, not ranked last — and the count is the same either way.

        The alternative was `NULLS LAST`, which reads more forgiving but is
        not: an HNSW index scan never returns an unembedded row, a sequential
        scan does, so the result would depend on which plan Postgres chose.
        """
        session, axis = ranked
        unembedded = session.scalars(
            select(Product).where(Product.name == "Foam Roller")
        ).one()
        unembedded.embedding = None
        session.commit()

        results = search_products(
            query=DRY_QUERY,
            query_embedding=one_hot(axis["Foam Roller"]),
            limit=50,
            session=session,
        )

        names = [result["name"] for result in results]
        assert "Foam Roller" not in names
        assert len(names) == len(axis) - 1


# --- with a database: writing the vectors ------------------------------


class FakeEmbeddingClient:
    """Stands in for LLMClient, counting what it was asked to do.

    Deterministic: the vector for a text is a one-hot at the position of that
    text in the order it was first seen. What matters here is the bookkeeping —
    how many requests, how many texts each carried, which products got which
    vector — none of which needs a real model.
    """

    def __init__(self, *, prompt_tokens_per_text: int = 10) -> None:
        self.batches: list[list[str]] = []
        self.prompt_tokens_per_text = prompt_tokens_per_text
        self._seen: list[str] = []

    def embed(self, texts):
        texts = list(texts)
        self.batches.append(texts)
        vectors = []
        for text in texts:
            if text not in self._seen:
                self._seen.append(text)
            vectors.append(one_hot(self._seen.index(text)))
        usage = UsageTracker().record(
            model="text-embedding-3-small",
            prompt_tokens=self.prompt_tokens_per_text * len(texts),
            completion_tokens=0,
        )
        return vectors, usage


@pytest.mark.db
class TestEmbedProducts:
    @pytest.fixture
    def unembedded(self, session):
        seed_catalog(session)
        session.execute(update(Product).values(embedding=None))
        session.commit()
        return session

    def test_a_first_pass_embeds_every_product_in_one_request(self, unembedded):
        client = FakeEmbeddingClient()

        summary = embed_products(unembedded, client=client)

        total = len(CATALOG)
        assert summary.products_embedded == total
        assert summary.products_skipped == 0
        assert summary.api_calls == 1
        assert len(client.batches) == 1
        assert len(client.batches[0]) == total
        assert missing_embedding_count(unembedded) == 0
        assert embedded_count(unembedded) == total

    def test_a_second_pass_makes_no_request_at_all(self, unembedded):
        client = FakeEmbeddingClient()
        embed_products(unembedded, client=client)

        second = embed_products(unembedded, client=client)

        assert second.products_embedded == 0
        assert second.api_calls == 0
        assert second.prompt_tokens == 0
        assert second.cost_usd == 0.0
        assert second.products_skipped == len(CATALOG)
        assert len(client.batches) == 1

    def test_force_re_embeds_everything(self, unembedded):
        client = FakeEmbeddingClient()
        embed_products(unembedded, client=client)

        forced = embed_products(unembedded, client=client, force=True)

        assert forced.products_embedded == len(CATALOG)
        assert forced.products_skipped == 0
        assert len(client.batches) == 2

    def test_only_the_products_missing_a_vector_are_sent(self, unembedded):
        client = FakeEmbeddingClient()
        embed_products(unembedded, client=client)
        one = unembedded.scalars(select(Product).order_by(Product.id)).first()
        one.embedding = None
        unembedded.commit()

        summary = embed_products(unembedded, client=client)

        assert summary.products_embedded == 1
        assert client.batches[-1] == [embedding_text(one)]

    def test_a_large_catalog_is_sent_in_several_requests(self, unembedded):
        client = FakeEmbeddingClient()

        summary = embed_products(unembedded, client=client, batch_size=7)

        total = len(CATALOG)
        assert summary.api_calls == len(client.batches)
        assert sum(len(batch) for batch in client.batches) == total
        assert max(len(batch) for batch in client.batches) == 7
        assert embedded_count(unembedded) == total

    def test_each_product_gets_the_vector_for_its_own_text(self, unembedded):
        """A misaligned batch would corrupt the catalog quietly, not loudly."""
        client = FakeEmbeddingClient()
        embed_products(unembedded, client=client, batch_size=7)
        unembedded.expire_all()

        texts = client._seen
        for product in unembedded.scalars(select(Product)):
            assert product.embedding == one_hot(texts.index(embedding_text(product)))

    def test_tokens_and_cost_are_summed_across_batches(self, unembedded):
        client = FakeEmbeddingClient(prompt_tokens_per_text=10)

        summary = embed_products(unembedded, client=client, batch_size=7)

        assert summary.prompt_tokens == 10 * len(CATALOG)
        assert summary.cost_usd == pytest.approx(
            summary.prompt_tokens * 0.02 / 1_000_000
        )

    def test_a_short_response_is_refused_rather_than_written(self, unembedded):
        """Never write a vector against the wrong product."""

        class ShortClient(FakeEmbeddingClient):
            def embed(self, texts):
                vectors, usage = super().embed(texts)
                return vectors[:-1], usage

        with pytest.raises(RuntimeError, match="asked for"):
            embed_products(unembedded, client=ShortClient())

    @pytest.mark.parametrize("batch_size", [0, -1])
    def test_a_batch_size_below_one_is_refused(self, unembedded, batch_size):
        """Zero raised an incidental ValueError from range(); a negative one was
        worse — an empty range, so the pass reported success having embedded
        nothing."""
        client = FakeEmbeddingClient()

        with pytest.raises(ValueError, match="batch_size"):
            embed_products(unembedded, client=client, batch_size=batch_size)

        assert client.batches == []
        assert missing_embedding_count(unembedded) == len(CATALOG)


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
