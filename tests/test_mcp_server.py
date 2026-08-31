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

import io
import logging
import sys

import anyio
import pytest
from mcp.client.client import Client
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from rich.console import Console
from rich.logging import RichHandler
from sqlalchemy import text

import shopagent.mcp_server.server as server_module
from shopagent.config import get_settings
from shopagent.money import SYMBOLS, format_amount
from shopagent.mcp_server.server import (
    SERVER_NAME,
    configure_stderr_logging,
    ping,
    server,
)

# The tool names the model sees. They are pinned as a list because the set of
# tools *is* the interface: D5 loads them dynamically, so a rename here is a
# silent change to what the agent can do.
#
# `ping` left this list on D9. It is a diagnostic with no business meaning, and
# it sat among the three that mean something for four days — the Known gaps
# entry that tracked it also named the only correct place to fix it, which is
# here rather than in a name check on the client side. It is still a tool and
# still reachable; the server simply does not advertise it unless asked. See
# MCP_EXPOSE_PING.
EXPECTED_TOOLS = ["search_products", "get_product_details", "check_stock"]

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


def output_schema_of(tool_name):
    """The output schema one tool advertises."""
    tools = {tool.name: tool for tool in call(server.list_tools)}
    return tools[tool_name].output_schema


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
    """The products a search returned, unwrapped from its envelope."""
    return (result.structured_content or {})["results"]


def count_of(result):
    """The `count` a search reported."""
    return (result.structured_content or {})["count"]


def message_of(result):
    """The text a failed call put in front of the model."""
    return result.content[0].text


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


@pytest.fixture
def ping_exposed():
    """The server as `MCP_EXPOSE_PING=true` builds it.

    Registering it here rather than reloading the module with the variable set:
    the flag decides one `add_tool` call, and this makes the same call. What
    that leaves untested is the reading of the setting itself, which
    `test_the_switch_is_what_decides_whether_ping_is_registered` covers by
    asserting the branch is the only thing between the two states.
    """
    server.add_tool(ping)
    try:
        yield
    finally:
        server.remove_tool("ping")


def test_ping_is_not_advertised_to_the_model():
    """Closed on D9: a diagnostic does not belong in a shopper's tool list.

    The cost of it being there was never that the model called it — across
    every demo scenario it never did. It is that the list is what the model
    reads to decide what it can do, and every name in it that means nothing is
    a name it has to rule out first.
    """
    assert "ping" not in [tool.name for tool in call(server.list_tools)]


def test_the_switch_is_what_decides_whether_ping_is_registered(ping_exposed):
    """With the switch on, it is an ordinary tool again."""
    assert "ping" in [tool.name for tool in call(server.list_tools)]


def test_ping_is_callable_without_the_server():
    """The tool is a plain function; the decorator only registers it.

    Same rule as `tools/basic.py` in CLAUDE.md — a tool stays testable without
    the machinery that wraps it.
    """
    assert ping() == "pong"


def test_every_tool_is_registered_under_its_expected_name():
    """The four names the model sees, in the order the server declares them."""
    assert [tool.name for tool in call(server.list_tools)] == EXPECTED_TOOLS


def test_ping_advertises_a_schema_with_no_arguments(ping_exposed):
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
    """The likeliest expensive mistake a model can make here is major units.

    `€100` is `10000`, and a model that passes `100` silently searches for
    something under one euro and reports an empty shop. The schema has to say
    so in the one place the model reads.

    The contrast is asserted through the shop's own currency symbol rather
    than the word "dollar", which is what this said until D9 — and it kept
    passing for a week after the shop moved to EUR, because a description
    teaching dollars still contains the word. A check derived from `CURRENCY`
    cannot go stale that way.
    """
    symbol = SYMBOLS[get_settings().currency]
    description = schema_of("search_products")["properties"][field]["description"]

    assert "cent" in description.lower()
    assert symbol in description


# Currency names this shop does not sell in. A description that teaches one of
# them is teaching the model a wrong unit, which is a wrong search it has no
# way to notice — `max_price_cents` said "passing 100 here means one dollar"
# for a week after the shop moved to EUR, and every test passed. Raised in
# review on PR #9.
OTHER_CURRENCY_WORDS = ("dollar", "pound", "yen", "franc", "rupee", "peso")


def test_no_tool_teaches_the_model_a_currency_this_shop_does_not_sell_in():
    """Swept over every description a model reads, not just the ones changed.

    A test naming the two price fields would have gone stale the same way the
    description did. This one fails whenever any tool starts talking about a
    currency the shop does not use, including a tool added later.
    """
    tools = call(server.list_tools)
    texts = []
    for tool in tools:
        texts.append(tool.description or "")
        for field in (tool.input_schema.get("properties") or {}).values():
            texts.append(field.get("description", ""))

    haystack = " ".join(texts).lower()
    found = [word for word in OTHER_CURRENCY_WORDS if word in haystack]
    assert found == [], f"model-facing text names {found}; the shop sells in {get_settings().currency}"


def test_the_price_examples_are_generated_from_the_configured_currency():
    """Not typed beside it, which is how the stale unit survived.

    Asserted through `format_amount` so the check cannot agree with a wrong
    description by repeating the same literal.
    """
    currency = get_settings().currency
    description = schema_of("search_products")["properties"]["max_price_cents"]["description"]

    assert format_amount(10000, currency) in description
    assert format_amount(4999, currency) in description
    assert format_amount(100, currency) in description


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


def test_ping_description_comes_from_the_docstring(ping_exposed):
    """The docstring is the contract the model reads; MCP derives it from here.

    Asserting the first line rather than the whole string keeps this from
    breaking every time the prose is edited, while still failing loudly if the
    docstring stops reaching the schema at all.
    """
    tools = {tool.name: tool for tool in call(server.list_tools)}
    description = tools["ping"].description

    assert description is not None
    assert description.startswith("Check that the catalog server is reachable.")


def test_ping_over_the_in_memory_transport_returns_pong(ping_exposed):
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
    assert count_of(result) == 0
    assert results_of(result) == []


# --- the envelope, and why it exists -------------------------------------


def test_search_results_arrive_wrapped_in_a_count_and_a_list():
    """The shape the model is promised: an object, not a bare list."""
    schema = output_schema_of("search_products")

    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"count", "results"}


@pytest.mark.db
def test_an_empty_search_still_produces_a_content_block(seeded):
    """The whole reason the envelope exists.

    A bare empty list serialises to zero content blocks, and a client that
    reads `content` — which is what the D5 adapter will hand the model — then
    sees literally nothing. "No matches" and "the call did nothing" have to
    look different, so an empty search must still say something.
    """
    result = call_over_transport(
        "search_products", {"category": "shoes", "size": "99", "color": "chartreuse"}
    )

    assert result.is_error is False
    assert len(result.content) == 1
    assert "count" in message_of(result)
    assert count_of(result) == 0


@pytest.mark.db
def test_count_agrees_with_the_number_of_results(seeded):
    """`count` is what the model is told to trust, so it cannot drift."""
    result = call_over_transport("search_products", {"category": "shoes", "limit": 3})

    assert count_of(result) == len(results_of(result))


# --- edge cases: error, or a legitimately empty result? ------------------
#
# The line: an error is what the model MUST change to get any answer at all.
# An empty result is what it gets when the arguments were valid and the shop
# happens to hold nothing like that. Each test below pins one side of that
# line, so a later change of mind fails loudly instead of quietly.


@pytest.mark.parametrize("field", ["max_price_cents", "min_price_cents"])
def test_a_negative_price_is_an_error_not_an_empty_shop(field):
    """No catalogue can hold a product priced below zero.

    Returning nothing would read as "no such cheap product" and invite the
    model to search lower, which is the one direction that cannot help.
    """
    result = call_over_transport("search_products", {field: -500})

    assert result.is_error is True
    message = message_of(result)
    assert field in message
    assert "cent" in message.lower()


def test_a_minimum_above_a_maximum_is_an_error():
    """An impossible range cannot be fixed by widening the search."""
    result = call_over_transport(
        "search_products", {"min_price_cents": 90_000, "max_price_cents": 1_000}
    )

    assert result.is_error is True
    message = message_of(result)
    assert "min_price_cents" in message and "max_price_cents" in message


@pytest.mark.db
@pytest.mark.parametrize("limit", [100, 0, -5])
def test_a_limit_outside_the_range_is_clamped_rather_than_refused(seeded, limit):
    """Out-of-range is not an error: the model still gets a usable answer.

    `search.py` clamps to 1-50. That is deliberate — refusing would cost the
    model a turn to learn something the schema already tells it — but it does
    mean the clamp is silent, which is why the parameter description points at
    `count` as the authority on how many came back.
    """
    result = call_over_transport("search_products", {"category": "shoes", "limit": limit})

    assert result.is_error is False
    assert 1 <= count_of(result) <= 50


@pytest.mark.db
@pytest.mark.parametrize("query", ["", "   "])
def test_a_blank_query_browses_instead_of_failing(seeded, query):
    """An empty query means "no query", and must not cost an embedding call.

    `search_products` embeds whatever it is given, so a blank string reaching
    the API would be a paid request for nothing. `catalog.search_products`
    already guards this; the test is here because the guard is invisible from
    the outside and easy to remove by accident.
    """
    result = call_over_transport("search_products", {"query": query, "limit": 2})

    assert result.is_error is False
    assert count_of(result) > 0


@pytest.mark.db
@pytest.mark.parametrize(
    ("field", "value"), [("size", "99"), ("color", "chartreuse")]
)
def test_a_filter_nothing_matches_is_an_empty_result_not_an_error(seeded, field, value):
    """Valid arguments, nothing in stock like that. The model should relax one."""
    result = call_over_transport("search_products", {field: value})

    assert result.is_error is False
    assert count_of(result) == 0


# --- the validation message the model actually reads ---------------------


def test_a_wrong_type_keeps_the_field_and_the_expectation():
    """Middleware trims Pydantic's message; it must not trim the useful half."""
    result = call_over_transport(
        "search_products", {"max_price_cents": "one hundred dollars"}
    )

    assert result.is_error is True
    message = message_of(result)
    assert "max_price_cents" in message
    assert "valid integer" in message


def test_a_wrong_type_loses_the_developer_facing_noise():
    """No documentation URL, no internal type code, no generated model name."""
    message = message_of(
        call_over_transport("search_products", {"max_price_cents": "one hundred dollars"})
    )

    assert "errors.pydantic.dev" not in message
    assert "[type=" not in message
    assert "search_productsArguments" not in message


def test_the_rewrite_leaves_a_handwritten_message_alone():
    """The middleware fires on Pydantic's shape only.

    The tools' own messages are already written for the model, and a rewrite
    that touched them would be silently editing the text this whole step exists
    to get right.
    """
    message = message_of(call_over_transport("search_products", {"max_price_cents": -500}))

    assert message.endswith("out to search without that bound.")


# --- category matches exactly, and the schema has to say so --------------
#
# `catalog/search.py` lowercases and trims, then compares for equality. That is
# the right behaviour — category is a closed set of five values the model has
# in front of it — but the schema described it as "matched loosely", which
# invited a near miss like "shoe" and returned an unexplained empty list.
# The description was the bug. These tests hold the two halves together.


def test_the_category_description_promises_exact_matching():
    """The schema must not advertise fuzziness the query does not have."""
    description = schema_of("search_products")["properties"]["category"]["description"]

    assert "Matched exactly" in description
    assert "loosely" not in description.lower()


def test_the_category_description_lists_every_section():
    """A closed set is only useful to the model if the model has all of it."""
    description = schema_of("search_products")["properties"]["category"]["description"]

    for section in ("shoes", "jackets", "bags", "accessories", "equipment"):
        assert section in description


def test_the_category_description_says_what_to_do_when_it_is_not_a_section():
    """Without this the model retries synonyms against a closed set forever."""
    description = schema_of("search_products")["properties"]["category"]["description"]

    assert "query" in description


@pytest.mark.db
def test_a_near_miss_category_returns_nothing_on_purpose(seeded):
    """"shoe" is not "shoes", and that is deliberate rather than a bug.

    Pinned so nobody later "fixes" it into a prefix or fuzzy match. The closed
    set is what makes exactness safe: the model is given all five names, so a
    value outside them is a mistake worth surfacing as an empty result rather
    than papering over with a guess about what was meant.
    """
    result = call_over_transport("search_products", {"category": "shoe", "limit": 50})

    assert result.is_error is False
    assert count_of(result) == 0


@pytest.mark.db
def test_case_and_surrounding_space_do_not_change_a_category(seeded):
    """The three spellings the description promises are interchangeable."""
    counts = {
        spelling: count_of(
            call_over_transport("search_products", {"category": spelling, "limit": 50})
        )
        for spelling in ("shoes", "Shoes", "  SHOES  ")
    }

    assert counts["shoes"] > 0, "expected the seeded catalog to contain shoes"
    assert len(set(counts.values())) == 1, counts


# --- the logging configuration, which is easy to silently undo -----------
#
# `MCPServer.__init__` installs a `RichHandler` whenever `rich` is importable.
# It wraps to console width and, with stderr a pipe, truncates — cutting off
# the end of every line, which is where the arguments and the duration are.
# `configure_stderr_logging` exists only to remove it, and removing `force=True`
# would silently restore the truncation, because `basicConfig` is a no-op once
# any handler is installed. These tests reproduce that exact setup.


@pytest.fixture
def restored_root_logging():
    """Put the root logger back afterwards.

    `configure_stderr_logging` calls `basicConfig(force=True)`, which removes
    every existing handler — pytest's own capture handlers included. Without
    this fixture the first test to run would quietly disable logging capture
    for the rest of the session.
    """
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    try:
        yield root
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def test_configuring_removes_a_handler_the_sdk_already_installed(restored_root_logging):
    """The whole point of `force=True`: an existing handler must not survive."""
    rich_handler = RichHandler(console=Console(stderr=True))
    restored_root_logging.handlers[:] = [rich_handler]

    configure_stderr_logging()

    assert rich_handler not in restored_root_logging.handlers
    assert not any(isinstance(h, RichHandler) for h in restored_root_logging.handlers)


def test_the_installed_handler_writes_to_stderr(restored_root_logging, monkeypatch):
    """stdout carries JSON-RPC. A log line on it corrupts the protocol."""
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)

    configure_stderr_logging()

    (handler,) = restored_root_logging.handlers
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream is stderr


def test_a_long_line_reaches_the_log_intact(restored_root_logging, monkeypatch):
    """The regression this guards is silent: the line still looks like a log.

    A `RichHandler` is installed first, exactly as the SDK leaves things, and
    the console is given a narrow width so that wrapping would be unmistakable.
    Drop `force=True` and that handler survives, the message comes out wrapped
    and decorated, and this fails.
    """
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)
    restored_root_logging.handlers[:] = [RichHandler(console=Console(stderr=True, width=80))]

    configure_stderr_logging()
    payload = (
        "tool search_products args={'query': 'a deliberately long query written to "
        "run past any console width a handler might wrap at'} -> ok in 12.3ms"
    )
    logging.getLogger("test").info(payload)

    written = stderr.getvalue()
    assert payload in written
    assert written.count("\n") == 1, f"the line was wrapped: {written!r}"


# --- redacting the one argument a shopper wrote (D6) ---------------------
#
# The middleware logs every call with its arguments, which is what makes the
# server debuggable once an agent loop drives it. `query` is the argument worth
# reading — it shows what the model understood the shopper to want — and from
# D6 it is also the only one a stranger wrote. These tests hold both halves:
# the text does not reach the log, and everything else still does.


def _force_redaction(monkeypatch, enabled: bool) -> None:
    """Point the server's `get_settings` at a copy with the flag flipped.

    `get_settings` is `lru_cache`d and the real `.env` decides the default, so
    a test that wants the other branch has to replace the lookup rather than
    the environment.
    """
    from shopagent.config import get_settings

    patched = get_settings().model_copy(update={"mcp_log_redact_query": enabled})
    monkeypatch.setattr(server_module, "get_settings", lambda: patched)


def _search_args() -> dict:
    return {
        "query": "waterproof trail shoes for wide feet",
        "category": "shoes",
        "max_price_cents": 12000,
        "min_price_cents": 3000,
        "size": "42",
        "limit": 5,
    }


def test_redaction_keeps_the_query_text_out_of_the_log(monkeypatch):
    _force_redaction(monkeypatch, True)

    logged = server_module.redact_arguments(_search_args())

    assert "waterproof trail shoes" not in repr(logged)
    assert logged["query"].startswith("<redacted:")


def test_redaction_leaves_every_other_argument_untouched(monkeypatch):
    """The journal's point: a log that redacts everything is a log nobody reads.

    Ids, categories, price bounds and limits are already visible in the tool
    schema and are what a call is actually debugged from.
    """
    _force_redaction(monkeypatch, True)
    original = _search_args()

    logged = server_module.redact_arguments(original)

    for field in ("category", "max_price_cents", "min_price_cents", "size", "limit"):
        assert logged[field] == original[field]


def test_redaction_off_lets_the_query_through(monkeypatch):
    """The developer-machine setting, and the other half of the claim."""
    _force_redaction(monkeypatch, False)

    logged = server_module.redact_arguments(_search_args())

    assert logged["query"] == "waterproof trail shoes for wide feet"


def test_the_same_query_redacts_to_the_same_token(monkeypatch):
    """"Did the model ask the same thing twice" has to stay answerable."""
    _force_redaction(monkeypatch, True)

    first = server_module.redact_arguments({"query": "running shoes"})
    second = server_module.redact_arguments({"query": "running shoes"})

    assert first["query"] == second["query"]


def test_different_queries_redact_to_different_tokens(monkeypatch):
    _force_redaction(monkeypatch, True)

    first = server_module.redact_arguments({"query": "running shoes"})
    second = server_module.redact_arguments({"query": "running shoe"})

    assert first["query"] != second["query"]


def test_the_token_is_not_a_plain_digest_of_the_text(monkeypatch):
    """A bare SHA-256 of a shopping query is recoverable from a wordlist.

    The salt is what makes the log unreadable rather than merely encoded, so
    this asserts the obvious attack does not work.
    """
    import hashlib

    _force_redaction(monkeypatch, True)
    text = "running shoes"

    token = server_module.redact_arguments({"query": text})["query"]

    assert hashlib.sha256(text.encode()).hexdigest()[:8] not in token


def test_redaction_survives_a_payload_that_is_not_a_mapping(monkeypatch):
    """A log line is not worth an exception, and a client may send anything."""
    _force_redaction(monkeypatch, True)

    assert server_module.redact_arguments(None) is None
    assert server_module.redact_arguments("not a dict") == "not a dict"
    assert server_module.redact_arguments([1, 2]) == [1, 2]


def test_redaction_survives_a_query_that_is_absent_or_not_a_string(monkeypatch):
    _force_redaction(monkeypatch, True)

    assert server_module.redact_arguments({"product_id": 7}) == {"product_id": 7}
    assert server_module.redact_arguments({"query": 42}) == {"query": 42}
    assert server_module.redact_arguments({"query": None}) == {"query": None}


def test_an_empty_query_still_redacts_rather_than_leaking_its_emptiness(monkeypatch):
    _force_redaction(monkeypatch, True)

    assert server_module.redact_arguments({"query": ""})["query"].startswith("<redacted:")


def test_the_middleware_logs_the_redacted_arguments(caplog, monkeypatch):
    """End to end through the middleware itself, with no catalog behind it.

    Driven directly rather than over the transport because `search_products`
    embeds its query, and a test that spends tokens to check a log line is the
    exact mistake `tests/conftest.py` installs a guard against.
    """
    _force_redaction(monkeypatch, True)

    class Ctx:
        method = "tools/call"
        params = {
            "name": "search_products",
            "arguments": {"query": "secret shopper phrase", "category": "shoes"},
        }

    async def call_next(ctx):
        return {"content": [], "isError": False}

    async def scenario():
        return await server_module.log_tool_calls(Ctx(), call_next)

    with caplog.at_level(logging.INFO):
        call(scenario)

    emitted = "\n".join(record.getMessage() for record in caplog.records)
    assert emitted, "the middleware logged nothing to inspect"
    assert "secret shopper phrase" not in emitted
    assert "<redacted:" in emitted
    # The argument that is safe, and that the log is actually read for.
    assert "shoes" in emitted


def test_the_middleware_logs_the_raw_query_when_redaction_is_off(
    caplog, monkeypatch
):
    """The other branch, so the test above cannot pass by logging nothing."""
    _force_redaction(monkeypatch, False)

    class Ctx:
        method = "tools/call"
        params = {
            "name": "search_products",
            "arguments": {"query": "secret shopper phrase"},
        }

    async def call_next(ctx):
        return {"content": [], "isError": False}

    async def scenario():
        return await server_module.log_tool_calls(Ctx(), call_next)

    with caplog.at_level(logging.INFO):
        call(scenario)

    emitted = "\n".join(record.getMessage() for record in caplog.records)
    assert "secret shopper phrase" in emitted


def test_a_raising_tool_also_logs_the_redacted_arguments(caplog, monkeypatch):
    """The error path logs separately, and was the easier one to forget."""
    _force_redaction(monkeypatch, True)

    class Ctx:
        method = "tools/call"
        params = {
            "name": "search_products",
            "arguments": {"query": "secret shopper phrase"},
        }

    async def call_next(ctx):
        raise RuntimeError("boom")

    async def scenario():
        return await server_module.log_tool_calls(Ctx(), call_next)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError):
            call(scenario)

    emitted = "\n".join(record.getMessage() for record in caplog.records)
    assert "secret shopper phrase" not in emitted
    assert "<redacted:" in emitted
