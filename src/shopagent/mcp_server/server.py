"""The catalog MCP server (D4).

Four tools: a diagnostic `ping` and three thin wrappers over
`catalog/search.py`. No search logic lives here — no filtering, no SQL, no
reshaping of an individual product. Every question about *what* the catalog
returns is settled in `catalog/`, which is what lets D5 swap the transport
without touching search.

**The SDK is not the one the plan describes.** `notes/plans` says `FastMCP`,
which was the v1 entry point and does not exist in `mcp==2.0.0`:
`mcp.server.fastmcp` was renamed to `mcp.server.mcpserver` and `FastMCP` to
`MCPServer`. Decorators and handler signatures survived intact, so this is an
import change rather than a redesign, but a v1 tutorial will not run here.

**Two channels carry the contract, and only one of them is the docstring.** A
tool's docstring becomes its `description`. Per-parameter text does *not* come
from an `Args:` section — the SDK builds the argument schema from the type
hints alone, and a `description` per property appears only where the annotation
carries `Annotated[..., Field(description=...)]`. Both channels are used below.

**What is deliberately not exposed.** `search_products` in `catalog/search.py`
takes three arguments these tools keep to themselves. `session` is
infrastructure. `mode` is the internal keyword/semantic comparison the journal
needs; semantic is the answer for a shopper, so it stays fixed. `query_embedding`
is an optimisation for a caller that already holds a vector, which no MCP client
does. Seven arguments reach the model, and every one is a question a shopper can
actually answer.

**Nothing here may write to stdout.** stdio transport *is* stdout: a stray
`print` lands in the middle of a JSON-RPC frame and the client drops the
connection with a parse error that names neither the print nor the tool. The
logger writes to stderr, which the client leaves alone.
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Annotated, Any

from mcp.server import MCPServer
from pydantic import Field

from shopagent.catalog import search as catalog

logger = logging.getLogger(__name__)

# What the client sees in the server list. It names the surface rather than the
# project, because D5 adds a second source of tools (local HTTP commerce) and
# "shopagent" alone would not say which of the two answered.
SERVER_NAME = "shopagent-catalog"

def _package_version() -> str:
    """Report the installed package version, so it cannot drift from pyproject.

    Read at runtime rather than copied into a literal: the handshake then
    reports what is actually running. An uninstalled source tree has no
    metadata to read, which is not worth failing a server over.
    """
    try:
        return version("shopagent")
    except PackageNotFoundError:  # pragma: no cover - only outside an install
        return "0.0.0+unknown"


server = MCPServer(SERVER_NAME, version=_package_version())


@server.tool()
def ping() -> str:
    """Check that the catalog server is reachable.

    Diagnostic only, and deliberately so: it takes no arguments, touches no
    database and returns a fixed string. That is what makes it useful. When a
    catalog tool fails, `ping` separates the two explanations — if `ping`
    answers, the transport and the server process are healthy and the fault is
    in the catalog or the database behind it; if `ping` does not answer, no
    result from any other tool means anything.

    Returns the string "pong". It says nothing whatsoever about the catalog,
    and a successful call is not evidence that any product exists.
    """
    logger.info("ping")
    return "pong"


@server.tool()
def search_products(
    query: Annotated[
        str | None,
        Field(
            description=(
                "What the shopper is looking for, in their own words. Matching is "
                "semantic, so describe the need rather than guessing catalogue "
                "words: 'something to run in when it's raining' finds waterproof "
                "running shoes, and synonyms work. Omit it to browse a category "
                "or a price range instead, which returns the cheapest matches. "
                'Example: "warm jacket for commuting by bike".'
            )
        ),
    ] = None,
    category: Annotated[
        str | None,
        Field(
            description=(
                "Restrict to one catalogue section. Matched loosely, but the "
                "sections are known and lowercase: shoes, jackets, bags, "
                'accessories, equipment. Example: "shoes".'
            )
        ),
    ] = None,
    max_price_cents: Annotated[
        int | None,
        Field(
            description=(
                "Upper price bound in CENTS, not dollars. $100 is 10000, $49.99 "
                "is 4999. Passing 100 here means one dollar and will match "
                "nothing. Applies to the variant price, so a product is "
                "returned when at least one of its variants is within the bound."
            )
        ),
    ] = None,
    min_price_cents: Annotated[
        int | None,
        Field(
            description=(
                "Lower price bound in CENTS, not dollars. $50 is 5000. Use it "
                "only when the shopper asked for a floor; it is not needed to "
                'express "cheap".'
            )
        ),
    ] = None,
    size: Annotated[
        str | None,
        Field(
            description=(
                "Keep only variants in exactly this size. Matched exactly, not "
                "as a range, and sizes are strings because they are not all "
                'numbers: "42", "M", "L". Asking for "42" never returns a 43.'
            )
        ),
    ] = None,
    color: Annotated[
        str | None,
        Field(
            description=(
                "Keep only variants in exactly this colour, lowercase and "
                'singular: "black", "navy", "olive". It is an exact match on '
                'the variant colour, so "dark" matches nothing.'
            )
        ),
    ] = None,
    limit: Annotated[
        int,
        Field(
            description=(
                "How many products to return. Ask for what the answer needs — 3 "
                "to offer a short choice, 10 to survey a category. Values are "
                "clamped to 1-50, so a larger number is not an error but buys "
                "nothing beyond 50."
            )
        ),
    ] = 5,
) -> list[dict[str, Any]]:
    """Search the shop catalogue for products matching a description and filters.

    This is the tool to reach for whenever a shopper describes what they want,
    asks what is available, or names a budget. Call it before answering any
    question about what the shop sells: the catalogue is the only source of
    truth for names, prices and stock, and none of it can be recalled from
    memory.

    Every argument is optional and they combine as AND. A product is returned
    only if it satisfies all of them, and the variants attached to it are only
    those that passed the variant-level filters — a search for size 42 never
    returns a 43 to offer.

    Returns a list of products, nearest match first when `query` is given and
    cheapest first otherwise. Each product carries product_id, name, brand,
    category, description, and a list of variants; each variant carries
    variant_id, sku, size, color, price_cents (an integer number of cents) and
    available (units that can be sold right now).

    An empty list is a normal, successful answer: it means nothing in the shop
    matches these filters, not that anything went wrong. Say so and suggest
    relaxing a filter — a wider price bound or a different size — rather than
    inventing a product. Use the product_id from a result to call
    get_product_details, and a variant_id to call check_stock.
    """
    logger.info("search_products query=%r category=%r limit=%s", query, category, limit)
    return catalog.search_products(
        query=query,
        category=category,
        max_price_cents=max_price_cents,
        min_price_cents=min_price_cents,
        size=size,
        color=color,
        limit=limit,
    )



@server.tool()
def get_product_details(
    product_id: Annotated[
        int,
        Field(
            description=(
                "The product_id of a product, as returned by search_products. "
                "It is the numeric product_id field, not a sku and not a "
                "variant_id. Example: 7."
            )
        ),
    ],
) -> dict[str, Any]:
    """Fetch one product in full, with every variant it has.

    Call this once the shopper has settled on a product and wants specifics —
    which sizes exist, which colours, what each costs. Unlike search_products
    this filters nothing: every variant is returned, including the ones that
    are out of stock, because "do you have it in 43?" cannot be answered by a
    tool that hides the 43.

    Returns product_id, name, brand, category, description and a list of
    variants; each variant carries variant_id, sku, size, color, price_cents
    (an integer number of cents) and available (units sellable right now). An
    available of 0 means that variant exists but cannot be sold — report it as
    out of stock rather than omitting it.

    Fails if no product has this id. That is a different situation from a
    search that matched nothing: it means the id itself is wrong, so search
    again rather than retrying with the same number.
    """
    logger.info("get_product_details product_id=%s", product_id)
    product = catalog.get_product(product_id)

    if product is None:
        # `None` is not an answer a model can act on, and a string describing
        # the problem would arrive as a successful result. Raising is what
        # reaches the client as is_error, which is the whole point.
        raise ValueError(
            f"No product has product_id {product_id}. This id does not exist in "
            f"the catalogue, so retrying with it will fail again. Call "
            f"search_products to find the product and use the product_id from "
            f"its result."
        )

    return product


@server.tool()
def check_stock(
    variant_id: Annotated[
        int,
        Field(
            description=(
                "The variant_id of one specific buyable variant — one size and "
                "colour of a product — as returned by search_products or "
                "get_product_details. It is not the product_id and not the sku. "
                "Example: 21."
            )
        ),
    ],
) -> dict[str, Any]:
    """Check how many units of one specific product variant can be sold now.

    Call this before promising availability, and before treating any purchase
    as possible. A variant is one size and colour of a product, so this answers
    "is the black one in 42 actually in stock?" — a question search results
    hint at but do not settle.

    Returns variant_id, sku, product_name, size, color, quantity, reserved,
    available and in_stock. The number that matters is available, which is
    quantity minus reserved: units held by someone else's pending checkout are
    counted in quantity but cannot be sold to this shopper. Answer from
    available and in_stock, never from quantity alone.

    Fails if no variant has this id, which means the id is wrong rather than
    the item being sold out. A variant that exists but has none left is a
    successful result with available 0.
    """
    logger.info("check_stock variant_id=%s", variant_id)
    stock = catalog.check_stock(variant_id)

    if stock is None:
        raise ValueError(
            f"No variant has variant_id {variant_id}. This id does not exist in "
            f"the catalogue, which is not the same as the item being out of "
            f"stock. Call get_product_details on the product to list its real "
            f"variant_ids, or search_products if the product is not known yet."
        )

    return stock


def main() -> None:
    """Serve over stdio until the client disconnects.

    `run` is synchronous and opens the event loop itself, so there is no
    `anyio.run` here. Logging is configured to stderr on the way in; without a
    handler the SDK's own warnings would be invisible, which is a bad trade in
    a process whose only other output channel is a protocol stream.
    """
    logging.basicConfig(level=logging.INFO)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
