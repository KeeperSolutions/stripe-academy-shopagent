"""Catalog search — the contract D4 and D9 consume (D3, step 3).

Three plain functions. None of them knows about MCP, HTTP or an LLM: D4 wraps
them in an MCP server, D9 calls them behind tools, and neither has to reach in
here to do it. That separation is the reason the project can swap transports on
D5 without touching search logic.

Their signatures are deliberately shaped like `ProductQuery` in
`llm/structured.py`, so D9 can hand a parsed query almost straight through. The
one difference is `keywords: list[str]` there against `query: str` here —
joining a list of words into one search string is the caller's decision, not
this module's.

Everything filters in SQL. The plan flags this as the trap of the day and it is
worth being explicit about why: `LIMIT` runs after `WHERE` and before anything
Python sees, so a price filter applied to the returned list is a filter applied
to an already-truncated set. Ask for five shoes under $50 and you get whatever
of the first five happened to qualify — usually nothing — while the shop is
full of matches. Every predicate below therefore reaches the database.

Step 4 added vector ranking, and it changed the ORDER BY of the candidate query
and nothing else. The filters, the two-pass structure and the result shape are
the same ones step 3 tested. `mode` chooses which kind of matching a query
means; every other argument behaves identically either way.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Literal

from sqlalchemy import ColumnElement, Select, func, or_, select
from sqlalchemy.orm import Session

from shopagent.catalog.embeddings import embed_query
from shopagent.catalog.models import Inventory, Price, Product, Variant
from shopagent.config import get_settings
from shopagent.db import session_scope

# How a `query` is matched. "semantic" embeds it and ranks by cosine distance;
# "keyword" keeps the ILIKE of step 3, which is what the journal comparison
# needs and what still answers an exact-name search ("Trail Runner GTX") more
# sharply than a vector does.
SearchMode = Literal["semantic", "keyword"]

# An upper bound on `limit`. The caller on D9 is a language model filling in a
# number, and "show me everything" turning into 10,000 rows of prompt is a real
# failure mode; 50 is far above any useful answer and far below a runaway.
MAX_LIMIT = 50

# LIKE treats these as wildcards. A model that puts a literal '%' in a query
# would otherwise match the entire catalog, so they are escaped and the escape
# character is declared on the operator.
_LIKE_ESCAPE = "\\"


@contextmanager
def _session_for(session: Session | None) -> Iterator[Session]:
    """Use the caller's session, or open one for the length of the call.

    Both callers exist. On D4 the MCP server has no session of its own and just
    wants an answer; on D6 FastAPI already holds one per request and passing it
    in is what keeps a read consistent with the write that preceded it. A
    borrowed session is never closed here — closing something you did not open
    is how a request handler ends up with a dead session halfway through.
    """
    if session is not None:
        yield session
    else:
        with session_scope() as own_session:
            yield own_session


def _escape_like(value: str) -> str:
    for character in (_LIKE_ESCAPE, "%", "_"):
        value = value.replace(character, _LIKE_ESCAPE + character)
    return value


def _available(inventory_alias: type[Inventory] = Inventory) -> ColumnElement[int]:
    """`quantity - reserved`, as SQL, defaulting to 0 when stock is unknown.

    Every variant the seed writes has an inventory row, but a row can be
    deleted and the outer join then yields NULL. Reporting "0 available" is the
    safe reading of "no idea"; reporting NULL would put a null into a field the
    model is told is an integer.
    """
    return func.coalesce(inventory_alias.quantity, 0) - func.coalesce(
        inventory_alias.reserved, 0
    )


def _variant_filters(
    *,
    max_price_cents: int | None,
    min_price_cents: int | None,
    size: str | None,
    color: str | None,
) -> list[ColumnElement[bool]]:
    """Predicates that decide whether a single variant qualifies.

    Kept apart from the product-level ones because they are needed twice: once
    to choose which products match, and again to choose which of a matching
    product's variants are allowed into the result. Using the same list in both
    places is what stops the search offering size 43 to someone who asked for
    42.
    """
    filters: list[ColumnElement[bool]] = []
    if max_price_cents is not None:
        filters.append(Price.amount_cents <= max_price_cents)
    if min_price_cents is not None:
        filters.append(Price.amount_cents >= min_price_cents)
    if size is not None:
        # Case-insensitive: the model writes what the user typed, and "m" and
        # "M" are the same size.
        filters.append(func.lower(Variant.size) == size.strip().lower())
    if color is not None:
        filters.append(func.lower(Variant.color) == color.strip().lower())
    return filters


def _product_filters(
    *, query: str | None, category: str | None
) -> list[ColumnElement[bool]]:
    """Predicates that decide whether a product matches, ignoring its variants."""
    filters: list[ColumnElement[bool]] = []
    if category is not None:
        filters.append(func.lower(Product.category) == category.strip().lower())
    if query is not None and query.strip():
        # Keyword search, step 3's half of the comparison. It matches only
        # where the words literally appear, which is exactly the limitation
        # step 4 exists to show: nothing here can connect "wet weather" to a
        # GORE-TEX membrane.
        pattern = f"%{_escape_like(query.strip())}%"
        filters.append(
            or_(
                Product.name.ilike(pattern, escape=_LIKE_ESCAPE),
                Product.description.ilike(pattern, escape=_LIKE_ESCAPE),
                Product.brand.ilike(pattern, escape=_LIKE_ESCAPE),
            )
        )
    return filters


def _priced_variants_of(*only_products: Any) -> Select[Any]:
    """The join every query in this module starts from.

    A variant is only sellable if it has an active price in the shop's
    currency and it is these three tables together, never `variants` alone,
    that a search reads.
    """
    currency = get_settings().currency
    return (
        select(*only_products)
        .join(Variant, Variant.product_id == Product.id)
        .join(
            Price,
            (Price.variant_id == Variant.id)
            & Price.active.is_(True)
            & (Price.currency == currency),
        )
        .outerjoin(Inventory, Inventory.variant_id == Variant.id)
    )


def _has_a_qualifying_variant(
    variant_filters: list[ColumnElement[bool]],
) -> ColumnElement[bool]:
    """EXISTS: at least one of this product's variants passes every filter.

    Semantically identical to joining and grouping — a product qualifies when
    one variant does — but it leaves the outer query as a plain scan of
    `products`, which is what a vector index needs. See
    `candidate_products_statement`.
    """
    currency = get_settings().currency
    return (
        select(Variant.id)
        .join(
            Price,
            (Price.variant_id == Variant.id)
            & Price.active.is_(True)
            & (Price.currency == currency),
        )
        .where(Variant.product_id == Product.id, *variant_filters)
        .exists()
    )


def candidate_products_statement(
    *,
    query: str | None = None,
    query_embedding: Sequence[float] | None = None,
    category: str | None = None,
    max_price_cents: int | None = None,
    min_price_cents: int | None = None,
    size: str | None = None,
    color: str | None = None,
    limit: int = 5,
) -> Select[Any]:
    """The statement that picks which products match, exposed for inspection.

    Public so the SQL can be read — by a test asserting the predicates are in
    the WHERE clause, and by a person compiling it to see what actually runs.
    It builds SQL and makes no API call: the caller embeds the query and hands
    the vector in, which is what keeps this function testable offline.

    `query_embedding` decides the kind of match. Given one, the query ranks by
    cosine distance and `query` is not used as a text filter — gating a
    semantic search on a literal match would defeat the entire point of it.
    Given none, a `query` filters with ILIKE, exactly as in step 3.

    The two branches are shaped differently on purpose:

    * Ranked by distance, the variant filters move into an `EXISTS` subquery so
      the outer statement stays a scan of `products` ordered by a bare
      `embedding <=> :vector`. That is the only form an HNSW index can serve;
      wrapped in `min()` behind a `GROUP BY`, as this first did, the distance
      is computed after aggregation and every qualifying product gets sorted no
      matter how large the catalog grows.
    * Ranked by price, the join and `GROUP BY` stay, because `min(amount_cents)`
      is the ordering key and has to be aggregated to exist.

    Either way the filters are WHERE predicates and `LIMIT` runs after them,
    which is the trap the plan warns about: rank first and filter afterwards,
    and a search for shoes under $100 quietly returns the five nearest shoes
    minus the ones that were too expensive — often nothing, while the shop is
    full of matches.
    """
    variant_filters = _variant_filters(
        max_price_cents=max_price_cents,
        min_price_cents=min_price_cents,
        size=size,
        color=color,
    )

    if query_embedding is not None:
        # `<=>` is pgvector's cosine distance: 0 is identical, 2 is opposite.
        #
        # Two details here are not stylistic. There is no `products.id`
        # tie-break, because a second sort key stops the HNSW index being used
        # at all — measured, not assumed: `ORDER BY embedding <=> $1` plans as
        # an index scan, and `ORDER BY embedding <=> $1, id` plans as a sort of
        # the whole table. Exact ties between real embeddings do not happen, so
        # the tie-break bought reproducibility that nothing needed.
        #
        # And products without a vector are excluded rather than sorted last.
        # They have to be: an index scan never returns them, since NULL vectors
        # are not in the index, while a sequential scan does. Leaving it to the
        # planner would mean the same query returning 29 or 30 rows depending
        # on which plan it picked. Excluded is the honest reading anyway —
        # nothing can rank by a similarity it has no vector for.
        distance = Product.embedding.cosine_distance(query_embedding)
        return (
            select(Product.id)
            .where(
                *_product_filters(query=None, category=category),
                Product.embedding.is_not(None),
                _has_a_qualifying_variant(variant_filters),
            )
            .order_by(distance.asc())
            .limit(limit)
        )

    cheapest = func.min(Price.amount_cents).label("cheapest_cents")
    return (
        _priced_variants_of(Product.id, cheapest)
        .where(
            *_product_filters(query=query, category=category),
            *variant_filters,
        )
        .group_by(Product.id)
        # Cheapest match first, which is both a sensible default for a shopper
        # and a total order — without it, LIMIT would return an arbitrary
        # subset that changes between runs. Product.id breaks ties so the
        # result is reproducible, which the tests depend on.
        .order_by(cheapest.asc(), Product.id.asc())
        .limit(limit)
    )


def _rows_to_products(rows: Sequence[Any], order: Sequence[int]) -> list[dict]:
    """Fold (product, variant, price, available) rows into one dict per product."""
    by_id: dict[int, dict] = {}
    for product, variant, amount_cents, available in rows:
        payload = by_id.get(product.id)
        if payload is None:
            payload = {
                "product_id": product.id,
                "name": product.name,
                "brand": product.brand,
                "category": product.category,
                "description": product.description,
                "variants": [],
            }
            by_id[product.id] = payload
        payload["variants"].append(
            {
                "variant_id": variant.id,
                "sku": variant.sku,
                "size": variant.size,
                "color": variant.color,
                # The rename that CLAUDE.md describes: `amount_cents` is the
                # column, `price_cents` is what the model reads. int, never
                # float, all the way to Stripe.
                "price_cents": int(amount_cents),
                "available": int(available),
            }
        )
    return [by_id[product_id] for product_id in order if product_id in by_id]


def search_products(
    query: str | None = None,
    category: str | None = None,
    max_price_cents: int | None = None,
    min_price_cents: int | None = None,
    size: str | None = None,
    color: str | None = None,
    limit: int = 5,
    *,
    mode: SearchMode = "semantic",
    query_embedding: Sequence[float] | None = None,
    session: Session | None = None,
) -> list[dict]:
    """Find products matching every filter given.

    With a `query`, results come back nearest first: the query is embedded and
    ranked against `products.embedding` by cosine distance. Without one, they
    come back cheapest first. `mode="keyword"` swaps the ranking for the ILIKE
    matching of step 3 — kept because the journal comparison needs both halves,
    and because an exact name is still something a substring finds better than
    a vector.

    Returns one dict per product, carrying only the variants that passed the
    variant-level filters — a search for size 42 must not hand the model a 43
    to offer. An empty list means nothing matched; it is never None.

    `query_embedding` skips the embedding call and uses the vector given. It is
    what lets a caller reuse a vector it already has, and what lets the tests
    prove the ranking works without spending a token.
    """
    limit = max(1, min(int(limit), MAX_LIMIT))

    if query_embedding is None and mode == "semantic" and query and query.strip():
        # One API call per search, before any SQL runs.
        query_embedding = embed_query(query.strip())

    variant_filters = _variant_filters(
        max_price_cents=max_price_cents,
        min_price_cents=min_price_cents,
        size=size,
        color=color,
    )

    with _session_for(session) as active_session:
        candidates = candidate_products_statement(
            query=query,
            query_embedding=query_embedding,
            category=category,
            max_price_cents=max_price_cents,
            min_price_cents=min_price_cents,
            size=size,
            color=color,
            limit=limit,
        )
        product_ids = list(active_session.scalars(candidates))
        if not product_ids:
            return []

        # Second pass, same filters. The first pass answered "which products",
        # this one answers "which of their variants", and both questions are
        # settled in SQL.
        detail = (
            _priced_variants_of(Product, Variant, Price.amount_cents, _available())
            .where(Product.id.in_(product_ids), *variant_filters)
            .order_by(Product.id.asc(), Price.amount_cents.asc(), Variant.id.asc())
        )
        rows = active_session.execute(detail).all()

    return _rows_to_products(rows, product_ids)


def get_product(product_id: int, *, session: Session | None = None) -> dict | None:
    """Return one product with every variant it has, or None if there is no such id.

    No filtering at all, on purpose: this is what the model calls once the user
    has picked something, and at that point hiding the sold-out sizes would be
    hiding the answer to "do you have it in 43?". Stock is in the payload;
    deciding what to do about a zero is the caller's job.
    """
    with _session_for(session) as active_session:
        rows = active_session.execute(
            _priced_variants_of(Product, Variant, Price.amount_cents, _available())
            .where(Product.id == product_id)
            .order_by(Variant.id.asc())
        ).all()

    products = _rows_to_products(rows, [product_id])
    return products[0] if products else None


def check_stock(variant_id: int, *, session: Session | None = None) -> dict | None:
    """Report stock for one variant, or None if there is no such id.

    `available` is `quantity - reserved`: a unit held by someone else's pending
    checkout is not one this conversation can sell. D9 calls this before it
    lets a cart become an order.
    """
    with _session_for(session) as active_session:
        row = active_session.execute(
            select(
                Variant.id,
                Variant.sku,
                Variant.size,
                Variant.color,
                Product.name,
                func.coalesce(Inventory.quantity, 0),
                func.coalesce(Inventory.reserved, 0),
                _available(),
            )
            .join(Product, Product.id == Variant.product_id)
            .outerjoin(Inventory, Inventory.variant_id == Variant.id)
            .where(Variant.id == variant_id)
        ).first()

    if row is None:
        return None

    variant_id_, sku, size, color, product_name, quantity, reserved, available = row
    return {
        "variant_id": variant_id_,
        "sku": sku,
        "product_name": product_name,
        "size": size,
        "color": color,
        "quantity": int(quantity),
        "reserved": int(reserved),
        "available": int(available),
        "in_stock": int(available) > 0,
    }
