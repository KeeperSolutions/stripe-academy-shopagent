"""The catalog MCP server (D4, narrowed on D9).

Three tools, all thin wrappers over `catalog/search.py`, and a fourth the
server keeps to itself. `ping` is a diagnostic with no business meaning, and
D5 recorded the cost of advertising it: it sat in the model's tool list beside
the three that mean something, and the list is what the model reads to work
out what it can do. The same entry named the only place the fix belongs —
here, by not publishing it, rather than in a name check inside `mcp_client/`,
which registers whatever a server lists and must go on doing so. It is still a
tool, still callable, and `MCP_EXPOSE_PING=true` puts it back in `tools/list`
for whoever is debugging. No search logic lives here — no filtering, no SQL, no
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

**Three kinds of outcome, and the model has to tell them apart.** A wrong id is
an error: the model must change the argument or it will never get an answer. A
filter combination that nothing matches is a *success* carrying zero results:
the arguments were valid and the shop simply has nothing like that, so the model
should relax a filter rather than retry. A range that no catalog could ever
satisfy — a negative price, a minimum above a maximum — is an error again, because
widening will not help and the model would otherwise loop against an empty shop.

**Nothing here may write to stdout.** stdio transport *is* stdout: a stray
`print` lands in the middle of a JSON-RPC frame and the client drops the
connection with a parse error that names neither the print nor the tool. The
logger writes to stderr, which the client leaves alone.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Annotated, Any, TypedDict

from mcp.server import MCPServer
from pydantic import Field

from shopagent.catalog import search as catalog
from shopagent.config import get_settings

logger = logging.getLogger(__name__)

# The one argument that is free text somebody typed. Everything else the tools
# accept is an id, a category from a closed set, a price bound or a limit —
# values already visible in the tool schema, and the reason the log is worth
# keeping at all.
SENSITIVE_ARGUMENT = "query"

# A per-process salt, generated once at import and never logged. It is what
# makes the digests below correlatable without being readable.
#
# A bare SHA-256 of a shopping query is not a redaction: the space of things a
# shopper types is small enough that a wordlist recovers most of it, so a
# leaked log would still say what people searched for. Keying the digest with a
# secret nobody has removes that entirely.
#
# Per-process rather than configured, because the question the log has to
# answer is "did the model send the same query twice in this conversation",
# and a conversation lives inside one process. Correlating across restarts is
# not a use this server has, and a stable salt would be a long-lived secret to
# store, rotate and eventually leak.
_LOG_SALT = os.urandom(32)


def _digest(text: str) -> str:
    """Eight hex characters of a keyed digest — enough to tell calls apart.

    Truncated because the log needs an equality token, not a cryptographic
    commitment; sixty-four characters would push the interesting part of every
    line off the screen.
    """
    return hmac.new(_LOG_SALT, text.encode("utf-8"), hashlib.sha256).hexdigest()[:8]


def redact_arguments(arguments: Any) -> Any:
    """Return the arguments as they should appear in a log line.

    Replaces `query` with a salted digest and leaves every other argument
    alone. The digest is deliberately not accompanied by the text's length:
    with the salt in place, length is the only thing left that could narrow a
    guess, and knowing a query was thirty characters long has never helped
    anyone debug this server.

    What survives is the two things the log is read for. Whether a query was
    sent at all is visible from the key being present, and whether the model
    sent the same one twice is visible from two identical digests. What was
    searched for is gone.

    Returns the input untouched when redaction is off, when the payload is not
    a mapping, or when `query` is absent or not a string — a log line is not
    worth an exception, and a client is free to send nonsense.
    """
    if not get_settings().mcp_log_redact_query:
        return arguments
    if not isinstance(arguments, dict):
        return arguments

    value = arguments.get(SENSITIVE_ARGUMENT)
    if not isinstance(value, str):
        return arguments

    return {**arguments, SENSITIVE_ARGUMENT: f"<redacted:{_digest(value)}>"}

# What the client sees in the server list. It names the surface rather than the
# project, because D5 adds a second source of tools (local HTTP commerce) and
# "shopagent" alone would not say which of the two answered.
SERVER_NAME = "shopagent-catalog"

# Pydantic's rendering of a validation failure is written for a developer: it
# ends with a documentation URL and tags each error with an internal type code,
# and it names the generated argument model rather than the tool. The model is
# the only reader of this text, and neither part helps it. These two patterns
# strip exactly that much and leave the field name and the expected type, which
# are the parts it can act on.
_PYDANTIC_NOISE = re.compile(r"\s*\[type=[^\]]*\]|\n?\s*For further information visit \S+")
_PYDANTIC_MODEL_NAME = re.compile(r"(\d+ validation errors? for )\w+Arguments")


class SearchResults(TypedDict):
    """The envelope `search_products` returns.

    Declared as a type rather than built ad hoc so the shape reaches the model
    twice: the docstring explains what `count` of 0 means, and this puts
    `count` and `results` in the tool's output schema, where a client can see
    them without reading prose. `results` stays a free-form list of objects —
    the product shape belongs to `catalog/search.py`, and restating it here
    would be a second definition to keep in step with the first.
    """

    count: int
    results: list[dict[str, Any]]


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


async def log_tool_calls(ctx, call_next):
    """Log every tool call with its arguments, duration and outcome.

    This is a long-lived process that D5 drives from an agent loop, and when a
    conversation goes wrong the log is the only record of what the model
    actually asked for. Timing lives here rather than in each tool because only
    this layer sees both ends of the call, and outcome lives here because only
    this layer sees the result after the tool has returned or raised.

    Arguments go through `redact_arguments` first. Everything except `query`
    is a catalog identifier, a filter or a price — already visible in the tool
    schema, and the reason this log is worth keeping. `query` is free text a
    shopper typed, and from D6 there are real carts behind it, so it is
    replaced with a salted digest by default. See `redact_arguments`.
    """
    if ctx.method != "tools/call":
        return await call_next(ctx)

    params = ctx.params or {}
    name = params.get("name", "<unknown>")
    arguments = params.get("arguments") or {}
    started = time.perf_counter()

    try:
        result = await call_next(ctx)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        # A raise here is a protocol-level failure, not a tool returning an
        # error result — worth a distinct log line, because the client sees
        # something very different in each case.
        logger.error(
            "tool %s args=%r raised after %.1fms: %s",
            name,
            redact_arguments(arguments),
            elapsed_ms,
            exc,
        )
        raise

    elapsed_ms = (time.perf_counter() - started) * 1000
    failed = bool(result.get("isError")) if isinstance(result, dict) else False
    logger.log(
        logging.WARNING if failed else logging.INFO,
        "tool %s args=%r -> %s in %.1fms",
        name,
        redact_arguments(arguments),
        "error" if failed else "ok",
        elapsed_ms,
    )
    return result


async def readable_validation_errors(ctx, call_next):
    """Strip the developer-facing half of a Pydantic validation message.

    An argument of the wrong type never reaches a tool body: the SDK validates
    against the generated model first and the failure arrives as an error
    result. That text is Pydantic's own, and it ends with a link to
    errors.pydantic.dev and an internal type code — a reader who can follow a
    URL is not the reader this message has.

    The rewrite is deliberately narrow. It fires only on text carrying
    Pydantic's own header, so the messages the tools below write by hand pass
    through untouched, and it removes only the trailing noise: the field name
    and the expected type, which are what the model needs to correct itself,
    survive. Loosening the schema types to produce a friendlier message was the
    alternative, and it was rejected — it would make the error more likely in
    order to make it prettier.
    """
    result = await call_next(ctx)

    if ctx.method != "tools/call" or not isinstance(result, dict) or not result.get("isError"):
        return result

    for block in result.get("content", []):
        text = block.get("text")
        if not text or "validation error" not in text:
            continue
        text = _PYDANTIC_MODEL_NAME.sub(r"\1the arguments you sent", text)
        block["text"] = _PYDANTIC_NOISE.sub("", text).rstrip()

    return result


# Outermost first: the logger wraps the rewrite, so what it records as the
# outcome is the result the client actually receives.
server = MCPServer(
    SERVER_NAME,
    version=_package_version(),
    middleware=[log_tool_calls, readable_validation_errors],
)


def _reject_impossible_price_range(min_price_cents: int | None, max_price_cents: int | None) -> None:
    """Refuse a price range that no catalog could ever satisfy.

    These are errors rather than empty results, and the difference is what the
    model should do next. An empty result means relax a filter; there is no
    relaxing a negative bound, and a minimum above a maximum stays empty however
    wide the shop is. Left to return nothing, both would read as "this shop has
    no cheap shoes" and the model would keep searching for something it has
    made unfindable.
    """
    for field, value in (("min_price_cents", min_price_cents), ("max_price_cents", max_price_cents)):
        if value is not None and value < 0:
            raise ValueError(
                f"{field} was {value}, but a price cannot be negative. Prices are "
                f"in cents, so €100 is 10000 and €49.99 is 4999. Re-send with a "
                f"value of 0 or more, or leave {field} out to search without that "
                f"bound."
            )

    if min_price_cents is not None and max_price_cents is not None and min_price_cents > max_price_cents:
        raise ValueError(
            f"min_price_cents ({min_price_cents}) is greater than max_price_cents "
            f"({max_price_cents}), so no product can match whatever the shop "
            f"stocks. Both are in cents. Swap the two values if they were the "
            f"wrong way round, or drop one of them to search with a single bound."
        )


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

    **Not registered by default** — see MCP_EXPOSE_PING and this module's
    docstring. The switch decides one `add_tool` call below, which is also why
    turning it on is not a different code path: the tool a debugger reaches is
    the same object, registered the same way, and the only thing that changed
    is whether the model was told about it.
    """
    return "pong"


if get_settings().mcp_expose_ping:
    server.add_tool(ping)


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
                "Restrict to one catalogue section. Matched exactly, not as a "
                'prefix or a synonym: "shoe" does not match "shoes" and '
                '"footwear" matches nothing at all. Capitalisation and '
                "surrounding spaces do not matter. There are five sections and "
                "these are all of them: shoes, jackets, bags, accessories, "
                "equipment. If what the shopper wants is not one of those five, "
                "leave category out entirely and describe it in query instead — "
                "a section that does not exist returns nothing rather than "
                'something close. Example: "shoes".'
            )
        ),
    ] = None,
    max_price_cents: Annotated[
        int | None,
        Field(
            description=(
                "Upper price bound in CENTS, not euros. €100 is 10000, €49.99 "
                "is 4999. Passing 100 here means one dollar and will match "
                "nothing. Must be 0 or more. Applies to the variant price, so a "
                "product is returned when at least one of its variants is within "
                "the bound."
            )
        ),
    ] = None,
    min_price_cents: Annotated[
        int | None,
        Field(
            description=(
                "Lower price bound in CENTS, not euros. €50 is 5000. Must be 0 "
                "or more, and not greater than max_price_cents. Use it only when "
                'the shopper asked for a floor; it is not needed to express "cheap".'
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
                "clamped to 1-50 rather than refused, so asking for 100 returns "
                "at most 50 and `count` tells you how many actually came back."
            )
        ),
    ] = 5,
) -> SearchResults:
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

    Returns an object with two fields. `count` is how many products came back.
    `results` is the list of them, nearest match first when `query` is given and
    cheapest first otherwise. Each product carries product_id, name, brand,
    category, description, and a list of variants; each variant carries
    variant_id, sku, size, color, price_cents (an integer number of cents) and
    available (units that can be sold right now).

    A `count` of 0 is a successful answer, not a failure: the search ran and the
    shop has nothing matching this combination of filters. Do not retry the same
    arguments and do not invent a product. Say what was not found and widen
    exactly one thing — raise max_price_cents, drop size or color, drop category,
    or describe the need differently in `query` — then search again. Every filter
    given is a way the result could have been narrowed to nothing.

    Use the product_id from a result to call get_product_details, and a
    variant_id to call check_stock.
    """
    _reject_impossible_price_range(min_price_cents, max_price_cents)

    results = catalog.search_products(
        query=query,
        category=category,
        max_price_cents=max_price_cents,
        min_price_cents=min_price_cents,
        size=size,
        color=color,
        limit=limit,
    )

    # The envelope exists for the empty case. A bare list of zero products
    # serialises to zero content blocks, so a client reading `content` sees
    # nothing at all and cannot tell "no matches" from "the call did nothing".
    # One object always renders as one block, and `count` states the result in
    # words the model cannot miss. Nothing about an individual product changes.
    return SearchResults(count=len(results), results=results)


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
    stock = catalog.check_stock(variant_id)

    if stock is None:
        raise ValueError(
            f"No variant has variant_id {variant_id}. This id does not exist in "
            f"the catalogue, which is not the same as the item being out of "
            f"stock. Call get_product_details on the product to list its real "
            f"variant_ids, or search_products if the product is not known yet."
        )

    return stock


def configure_stderr_logging(level: int = logging.INFO) -> None:
    """Send plain, untruncated log lines to stderr.

    `force=True` is doing real work here. Constructing `MCPServer` calls the
    SDK's own `configure_logging`, which installs a `RichHandler` if `rich` is
    importable — and it is, as a transitive dependency. That handler formats for
    a terminal: it wraps to the console width and, when stderr is a pipe rather
    than a tty, truncates to a default width. The first casualty is the end of
    every line, which is where this module puts the arguments and the duration.
    A log that drops exactly the diagnostic payload is worse than no log,
    because it still looks like logging.

    Since `MCPServer` is constructed at import time, that handler is already in
    place by the time this runs; `force=True` removes it and installs a plain
    one. stderr, never stdout: stdout carries the protocol.
    """
    logging.basicConfig(
        level=level,
        force=True,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )


def main() -> None:
    """Serve over stdio until the client disconnects.

    `run` is synchronous and opens the event loop itself, so there is no
    `anyio.run` here.
    """
    configure_stderr_logging()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
