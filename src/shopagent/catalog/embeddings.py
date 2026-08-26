"""Turning products into vectors (D3, step 4).

This module does not import `openai` — that rule from CLAUDE.md holds here as
everywhere else. It goes through `LLMClient.embed`, which is the one place the
SDK lives.

What gets embedded is `name + brand + category + description` joined, which the
plan asks for and which is worth stating the reason for: the description alone
loses the brand and the category, so "Fleetfoot" as a query would match nothing
and "jackets" would rank by whatever the prose happened to say. The joining
lives in `embedding_text`, on its own, so a test can check it without a network
call.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session

from shopagent.catalog.models import Product
from shopagent.db import get_engine
from shopagent.llm.client import LLMClient

# One request per batch. Thirty products fit in a single call, so this only
# matters the day the catalog grows; the API caps a request by tokens, and a
# few hundred short product texts stay well inside it.
BATCH_SIZE = 100

# HNSW over cosine distance, because search ranks with `<=>`. An index built
# for a different operator class is simply not used by that operator, silently.
HNSW_INDEX_NAME = "ix_products_embedding_hnsw"


@lru_cache(maxsize=1)
def default_client() -> LLMClient:
    """A shared client, so a search does not build an SDK object per query."""
    return LLMClient()


def embedding_text(product: Product) -> str:
    """The string that represents a product to the embedding model.

    Written as prose rather than as `a|b|c` fields: the model was trained on
    language, and the query it will be compared against ("something to run in
    when the ground is wet") is language too. Keeping both sides in the same
    register is most of what makes the distances mean anything.
    """
    return (
        f"{product.name} by {product.brand}. "
        f"Category: {product.category}. "
        f"{product.description}"
    )


@dataclass
class EmbedSummary:
    """What one embedding pass did, and what it cost."""

    products_embedded: int = 0
    products_skipped: int = 0
    api_calls: int = 0
    prompt_tokens: int = 0
    cost_usd: float = 0.0

    def as_lines(self) -> list[str]:
        return [
            f"  products embedded  {self.products_embedded}",
            f"  products skipped   {self.products_skipped}",
            f"  API calls          {self.api_calls}",
            f"  tokens             {self.prompt_tokens:,}",
            f"  cost               ${self.cost_usd:.6f}",
        ]


def embed_products(
    session: Session,
    *,
    force: bool = False,
    client: LLMClient | None = None,
    batch_size: int = BATCH_SIZE,
) -> EmbedSummary:
    """Write an embedding for every product that lacks one.

    Idempotent in the sense that matters for a paid API: a second run finds
    nothing with a NULL embedding, makes no request and spends no tokens.
    `force=True` re-embeds everything, which is what an edited description or a
    changed embedding model needs.
    """
    # Validated before anything else, because both bad values fail badly:
    # `range(0, n, 0)` raises an incidental ValueError from deep inside the
    # loop, and a negative step yields an empty range, so the pass reports a
    # successful run of zero work while every product is still unembedded.
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1; got {batch_size}")

    client = client or default_client()
    summary = EmbedSummary()

    statement = select(Product).order_by(Product.id)
    if not force:
        statement = statement.where(Product.embedding.is_(None))
    products = list(session.scalars(statement))

    # Counted before anything is written: with `force` nothing is skipped, and
    # otherwise the skipped ones are exactly those that already have a vector.
    summary.products_skipped = 0 if force else embedded_count(session)

    for start in range(0, len(products), batch_size):
        batch = products[start : start + batch_size]
        vectors, usage = client.embed([embedding_text(product) for product in batch])
        if len(vectors) != len(batch):
            # Never write a vector against the wrong product: a mismatch here
            # would corrupt the whole catalog quietly, and every search after
            # it would be subtly wrong rather than obviously broken.
            raise RuntimeError(
                f"asked for {len(batch)} embeddings, got {len(vectors)}"
            )
        for product, vector in zip(batch, vectors, strict=True):
            product.embedding = vector

        summary.products_embedded += len(batch)
        summary.api_calls += 1
        summary.prompt_tokens += usage.prompt_tokens
        summary.cost_usd += usage.cost_usd

    session.commit()
    return summary


def embed_query(query: str, *, client: LLMClient | None = None) -> list[float]:
    """Embed one search query. One API call, one vector."""
    client = client or default_client()
    vectors, _ = client.embed([query])
    return vectors[0]


def ensure_hnsw_index(engine: Engine | None = None) -> None:
    """Build the HNSW index over `products.embedding`, if it is not there.

    Runs after the vectors exist, never before: an index over an all-NULL
    column has nothing to build and would have to be rebuilt anyway.

    On thirty products this index is decorative — Postgres will sequential-scan
    the table faster than it can descend a graph, and may well ignore the index
    entirely. It is here because the syntax is the thing worth knowing, and
    because the operator class has to match the operator the search uses.
    """
    engine = engine or get_engine()
    with engine.connect() as connection:
        connection.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS {HNSW_INDEX_NAME} "
                "ON products USING hnsw (embedding vector_cosine_ops)"
            )
        )
        connection.commit()


def embedded_count(session: Session) -> int:
    """How many products currently carry a vector."""
    return len(
        session.scalars(select(Product.id).where(Product.embedding.is_not(None))).all()
    )


def missing_embedding_count(session: Session) -> int:
    return len(
        session.scalars(select(Product.id).where(Product.embedding.is_(None))).all()
    )


__all__ = [
    "BATCH_SIZE",
    "EmbedSummary",
    "HNSW_INDEX_NAME",
    "default_client",
    "embed_products",
    "embed_query",
    "embedded_count",
    "embedding_text",
    "ensure_hnsw_index",
    "missing_embedding_count",
]
