"""Tests for shopagent.mcp_server.server (D4, steps 1-2).

Unmarked: no database, no network, no subprocess. The server object is built at
import time, and the SDK can list and call its tools in-process, so the whole
protocol surface is testable without a transport.

Two ways in, and the difference matters. `Client(server)` runs the SDK's
in-memory transport: requests go through the same handlers a stdio client
reaches, so a failure comes back as `is_error=True` exactly as it would on the
wire. `server.call_tool(...)` is the bare Python API underneath that, and it
*raises* `ToolError` instead. Step 3 depends on that distinction, so it is
pinned here rather than discovered later.

Step 2 added the three catalog tools, and with them a split in what a test
needs. The schema is a claim about the contract and is checked offline. Whether
the tools return real rows is a claim about the database, so those carry the
`db` marker and read the seeded catalog directly — the rollback fixture in
conftest cannot reach them, because each tool opens its own session rather than
accepting one, which is exactly the behaviour step 2 set out to verify.

One trap worth naming: the MCP `search_products` does not expose `mode`, so a
`query` always embeds and always costs money. Every test below therefore passes
`query=None` and filters instead. The autouse guard in conftest turns a slip
into a failure rather than a bill.
"""

from __future__ import annotations

import anyio
import pytest
from mcp.client.client import Client
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from sqlalchemy import text

from shopagent.mcp_server.server import SERVER_NAME, ping, server

# The tool names the model sees. They are pinned as a list because the set of
# tools *is* the interface: D5 loads them dynamically, so a rename here is a
# silent change to what the agent can do.
EXPECTED_TOOLS = ["ping", "search_products", "get_product_details", "check_stock"]

# What `search_products` exposes, and nothing else. `session`, `mode` and
# `query_embedding` are deliberately absent — see the module docstring in
# server.py for why each one is withheld.
EXPECTED_SEARCH_PARAMETERS = {
    "query",
    "category",
    "max_price_cents",
    "min_price_cents",
    "size",
    "color",
    "limit",
}
WITHHELD_SEARCH_PARAMETERS = {"session", "mode", "query_embedding"}


def schema_of(tool_name):
    """The input schema one tool advertises."""
    tools = {tool.name: tool for tool in call(server.list_tools)}
    return tools[tool_name].input_schema


def call_over_transport(tool_name, arguments):
    """Call a tool the way a real client does, and hand back the result.

    Goes through the in-memory transport rather than `server.call_tool`, so a
    failure arrives as `is_error=True` instead of raising. Every assertion
    about error behaviour below depends on that difference.
    """

    async def scenario():
        async with Client(server) as client:
            return await client.call_tool(tool_name, arguments)

    return call(scenario)


def results_of(result):
    """The list a search returned, read from the structured payload."""
    return (result.structured_content or {}).get("result")


def call(fn):
    """Run one async SDK call from a sync test.

    The project has no async test plugin and this step does not justify adding
    one: `anyio` is already a transitive dependency of `mcp`, and `anyio.run`
    turns each coroutine into an ordinary function call.
    """
    return anyio.run(fn)


def test_server_is_an_mcp_server():
    assert isinstance(server, MCPServer)


def test_server_name_is_the_recognisable_one():
    """The name is what a client shows in its server list, so it is pinned."""
    assert server.name == SERVER_NAME
    assert SERVER_NAME == "shopagent-catalog"


def test_ping_is_callable_without_the_server():
    """The tool is a plain function; the decorator only registers it.

    Same rule as `tools/basic.py` in CLAUDE.md — a tool stays testable without
    the machinery that wraps it.
    """
    assert ping() == "pong"


def test_every_tool_is_registered_under_its_expected_name():
    """The four names the model sees, in the order the server declares them."""
    assert [tool.name for tool in call(server.list_tools)] == EXPECTED_TOOLS


def test_ping_advertises_a_schema_with_no_arguments():
    """An argument-free tool still needs a schema, and it must not invent one."""
    schema = schema_of("ping")

    assert schema["type"] == "object"
    assert schema.get("properties", {}) == {}
    assert schema.get("required", []) == []


def test_search_products_exposes_exactly_seven_parameters():
    """Seven, and the seven a shopper can actually answer."""
    properties = schema_of("search_products")["properties"]

    assert set(properties) == EXPECTED_SEARCH_PARAMETERS
    assert len(properties) == 7


def test_search_products_withholds_the_internal_arguments():
    """`session`, `mode` and `query_embedding` never reach the model.

    Each would be a way for the model to break something it has no way to
    reason about: a database handle, the keyword/semantic switch the journal
    uses for comparison, and a raw 1536-float vector.
    """
    properties = schema_of("search_products")["properties"]

    assert WITHHELD_SEARCH_PARAMETERS.isdisjoint(properties)


def test_search_products_takes_no_required_arguments():
    """Every filter is optional, so browsing with no arguments is legal."""
    assert schema_of("search_products").get("required", []) == []


@pytest.mark.parametrize("field", ["max_price_cents", "min_price_cents"])
def test_price_parameters_say_the_unit_is_cents(field):
    """The likeliest expensive mistake a model can make here is dollars.

    `$100` is `10000`, and a model that passes `100` silently searches for
    something under a dollar and reports an empty shop. The schema has to say
    so in the one place the model reads.
    """
    description = schema_of("search_products")["properties"][field]["description"]

    assert "cent" in description.lower()
    assert "dollar" in description.lower()


def test_the_cents_example_is_spelled_out_for_the_upper_bound():
    """Naming the unit is not enough; the conversion has to be shown."""
    description = schema_of("search_products")["properties"]["max_price_cents"]["description"]

    assert "10000" in description


@pytest.mark.parametrize(
    ("tool_name", "field"),
    [("get_product_details", "product_id"), ("check_stock", "variant_id")],
)
def test_id_parameters_are_required_and_described(tool_name, field):
    """An id has no sensible default, and the model must not confuse the two."""
    schema = schema_of(tool_name)

    assert schema["required"] == [field]
    assert schema["properties"][field]["description"]


def test_ping_description_comes_from_the_docstring():
    """The docstring is the contract the model reads; MCP derives it from here.

    Asserting the first line rather than the whole string keeps this from
    breaking every time the prose is edited, while still failing loudly if the
    docstring stops reaching the schema at all.
    """
    tools = {tool.name: tool for tool in call(server.list_tools)}
    description = tools["ping"].description

    assert description is not None
    assert description.startswith("Check that the catalog server is reachable.")


def test_ping_over_the_in_memory_transport_returns_pong():
    """The end-to-end path a stdio client takes, minus the pipe."""
    result = call_over_transport("ping", {})

    assert result.is_error is False
    assert [content.text for content in result.content] == ["pong"]


def test_unknown_tool_is_an_error_result_not_a_crash():
    """A client asking for a tool that is not there must get `is_error`.

    This is the shape step 3 has to reproduce deliberately: an error the client
    can see as an error, rather than a successful result whose text happens to
    describe a failure.
    """

    result = call_over_transport("no_such_tool", {})

    assert result.is_error is True
    assert "no_such_tool" in result.content[0].text


def test_listing_tools_over_the_transport_matches_the_direct_listing():
    """What the wire advertises is what the server holds — no adapter in between."""

    async def scenario():
        async with Client(server) as client:
            return await client.list_tools()

    assert [tool.name for tool in call(scenario).tools] == EXPECTED_TOOLS


def test_direct_call_tool_raises_where_the_transport_reports():
    """The bare API raises; the transport converts. Step 3 turns on this.

    `server.call_tool` is the layer below the protocol handler, and it lets a
    `ToolError` out. The handler above it is what catches that and produces
    `is_error=True`. A tool that wants the client to see a failure therefore
    raises — it does not return a string describing the problem.
    """
    with pytest.raises(ToolError, match="no_such_tool"):
        call(lambda: server.call_tool("no_such_tool", {}))


# --- against the real catalog -------------------------------------------
#
# These need rows. The tools open their own sessions, so conftest's rollback
# fixture cannot wrap them; `engine` is used only to decide whether to skip.
# Nothing below writes, so reading the seeded catalog directly is safe.


@pytest.fixture
def seeded(engine):
    """Skip unless the catalog actually has rows in it.

    A schema with no products would fail every assertion here for a reason
    that has nothing to do with the MCP server.
    """
    with engine.connect() as connection:
        products = connection.execute(text("SELECT count(*) FROM products")).scalar()
    if not products:
        pytest.skip("the catalog is empty. Run: python scripts/seed_catalog.py")
    return products


@pytest.fixture
def a_real_product(seeded):
    """One product straight from the catalog, as the tool would return it."""
    result = call_over_transport("search_products", {"category": "shoes", "limit": 1})
    products = results_of(result)
    assert products, "expected the seeded catalog to contain shoes"
    return products[0]


@pytest.mark.db
def test_search_products_returns_real_products_from_the_seed(seeded):
    """The wrapper reaches the database and the rows come back whole."""
    result = call_over_transport("search_products", {"category": "shoes", "limit": 3})
    products = results_of(result)

    assert result.is_error is False
    assert 1 <= len(products) <= 3
    for product in products:
        assert product["category"] == "shoes"
        assert product["name"]
        assert product["variants"]
        for variant in product["variants"]:
            # The rename CLAUDE.md insists on: the model reads `price_cents`,
            # an int, never the `amount_cents` column and never a float.
            assert isinstance(variant["price_cents"], int)
            assert "amount_cents" not in variant


@pytest.mark.db
def test_search_products_honours_a_price_ceiling_in_cents(seeded):
    """The filter runs in SQL, so nothing over the bound survives the limit."""
    ceiling = 8000
    result = call_over_transport(
        "search_products", {"category": "shoes", "max_price_cents": ceiling, "limit": 10}
    )

    products = results_of(result)
    assert result.is_error is False
    for product in products:
        assert any(variant["price_cents"] <= ceiling for variant in product["variants"])


@pytest.mark.db
def test_get_product_details_returns_the_product_for_a_real_id(a_real_product):
    result = call_over_transport(
        "get_product_details", {"product_id": a_real_product["product_id"]}
    )

    assert result.is_error is False
    product = result.structured_content
    assert product["product_id"] == a_real_product["product_id"]
    assert product["name"] == a_real_product["name"]
    assert product["variants"]


@pytest.mark.db
def test_check_stock_reports_availability_for_a_real_variant(a_real_product):
    variant = a_real_product["variants"][0]

    result = call_over_transport("check_stock", {"variant_id": variant["variant_id"]})

    assert result.is_error is False
    stock = result.structured_content
    assert stock["variant_id"] == variant["variant_id"]
    assert stock["available"] == stock["quantity"] - stock["reserved"]
    assert stock["in_stock"] is (stock["available"] > 0)


# --- the distinction step 2 exists to establish --------------------------
#
# "no such id" and "no results" are different answers, and the model has to be
# able to tell them apart. An id that does not exist is an error it should stop
# retrying; an empty result is a successful answer it should report as such.
# These two tests are the ones that would catch a regression turning either
# into the other.


@pytest.mark.db
def test_get_product_details_on_a_missing_id_is_an_error_that_names_the_way_out(seeded):
    """Not None, not a cheerful string — an error the model can act on."""
    result = call_over_transport("get_product_details", {"product_id": 10_000_000})

    assert result.is_error is True
    message = result.content[0].text
    assert "10000000" in message
    assert "search_products" in message


@pytest.mark.db
def test_check_stock_on_a_missing_variant_is_an_error_not_a_sold_out_report(seeded):
    """A wrong id must not be reported as an item that happens to be sold out."""
    result = call_over_transport("check_stock", {"variant_id": 10_000_000})

    assert result.is_error is True
    message = result.content[0].text
    assert "10000000" in message
    assert "out of stock" in message


@pytest.mark.db
def test_a_search_matching_nothing_succeeds_with_an_empty_list(seeded):
    """Nothing matched is an answer, not a failure.

    A shop with no shoes under one cent is behaving correctly. If this ever
    starts reporting `is_error`, the model will apologise for a broken tool
    instead of offering a wider price range.
    """
    result = call_over_transport(
        "search_products", {"category": "shoes", "max_price_cents": 1}
    )

    assert result.is_error is False
    assert results_of(result) == []

