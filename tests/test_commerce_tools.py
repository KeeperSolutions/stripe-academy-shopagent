"""The commerce tools the agent calls over HTTP (D9, step 1).

Nothing here reaches the network. `httpx.MockTransport` answers every request
inside the process, which is what lets the failure paths — a refused
connection, a timeout, a 401 — be exercised at all: none of them is reachable
against a healthy local API, and those are exactly the paths whose text the
model reads.
"""

from __future__ import annotations

import json

import httpx
import pytest

from shopagent.agent.memory import ConversationMemory
from shopagent.tools.commerce import register_commerce_tools
from shopagent.tools.http import CommerceAPI
from shopagent.tools.registry import ToolRegistry

TOOL_NAMES = [
    "add_to_cart",
    "view_cart",
    "remove_from_cart",
    "create_checkout",
    "check_order_status",
]

CART_ID = "11111111-1111-1111-1111-111111111111"
ITEM_ID = "22222222-2222-2222-2222-222222222222"
ORDER_ID = "33333333-3333-3333-3333-333333333333"


def build(handler, session=None):
    """A registry holding the five tools, wired to a fake commerce API."""
    api = CommerceAPI(
        base_url="http://commerce.test",
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )
    registry = ToolRegistry()
    session = session or ConversationMemory()
    register_commerce_tools(registry, api, session)
    return registry, session


def cart_body(items=()):
    return {
        "cart_id": CART_ID,
        "status": "open",
        "currency": "eur",
        "items": list(items),
        "total_cents": sum(item["line_total_cents"] for item in items),
    }


def one_line(variant_id=21, quantity=1, unit=9499):
    return {
        "item_id": ITEM_ID,
        "variant_id": variant_id,
        "sku": "FF-TRLGTX-42-BLK",
        "product_name": "Trail Runner GTX",
        "variant_label": "42 / black",
        "quantity": quantity,
        "unit_price_cents": unit,
        "line_total_cents": unit * quantity,
    }


def test_the_five_tools_are_registered():
    registry, _ = build(lambda request: httpx.Response(200, json=cart_body()))

    assert registry.names() == TOOL_NAMES


def test_no_tool_shows_the_model_a_cart_id():
    """The one identifier the tool layer keeps to itself.

    An id the model has to carry through a conversation is an id it will lose
    or invent, and the whole class of hallucination disappears if it never
    receives one. This is the assertion that keeps that true as tools are
    added.
    """
    registry, _ = build(lambda request: httpx.Response(200, json=cart_body()))

    for spec in registry.specs():
        schema = json.dumps(spec.to_openai_schema())
        assert "cart_id" not in schema, f"{spec.name} exposes cart_id"


def test_every_tool_validates_its_arguments_here():
    """Local tools carry an args_model, never a published schema (CLAUDE.md)."""
    registry, _ = build(lambda request: httpx.Response(200, json=cart_body()))

    for spec in registry.specs():
        assert spec.validates_locally, f"{spec.name} does not validate locally"


# --- the cart the model never sees ---------------------------------------


class Recorder:
    """A fake commerce API that remembers what was asked of it.

    The assertions that matter here are about *which* calls were made and in
    what order — the same reason D6 records the SQL `render_order` issues and
    D7 records the outgoing checkout payload. A tool that returned the right
    cart by fetching it three times would pass every assertion about its
    result.
    """

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(f"{request.method} {request.url.path}")
        for method, path, response in self.responses:
            if request.method == method and request.url.path == path:
                if isinstance(response, Exception):
                    raise response
                return response
        return httpx.Response(404, json={"detail": f"no fake for {request.url.path}"})


def created_cart():
    return ("POST", "/cart", httpx.Response(201, json=cart_body()))


def test_the_first_add_creates_a_cart_and_the_second_does_not():
    """A cart row per conversation, not per call."""
    recorder = Recorder([
        created_cart(),
        ("POST", f"/cart/{CART_ID}/items", httpx.Response(200, json=cart_body([one_line()]))),
    ])
    registry, session = build(recorder)

    registry.dispatch("add_to_cart", {"variant_id": 21})
    registry.dispatch("add_to_cart", {"variant_id": 21})

    assert recorder.calls.count("POST /cart") == 1
    assert session.cart_id == CART_ID


def test_view_cart_with_no_cart_says_it_is_empty_rather_than_failing():
    recorder = Recorder([])
    registry, _ = build(recorder)

    result = registry.dispatch("view_cart", {})

    assert result.ok
    assert "empty" in result.content
    assert recorder.calls == [], "an empty cart should not need the API at all"


def test_no_result_the_model_reads_carries_the_cart_id():
    """The schema half of this is not enough — a leak in a *result* is a leak.

    The model builds its next call from what came back, so a cart id in a
    result is a cart id it can quote, invent a variation of, or send to a tool
    that never asked for one.
    """
    recorder = Recorder([
        created_cart(),
        ("POST", f"/cart/{CART_ID}/items", httpx.Response(200, json=cart_body([one_line()]))),
        ("GET", f"/cart/{CART_ID}", httpx.Response(200, json=cart_body([one_line()]))),
    ])
    registry, _ = build(recorder)

    for name in ("add_to_cart", "view_cart"):
        arguments = {"variant_id": 21} if name == "add_to_cart" else {}
        content = registry.dispatch(name, arguments).content
        assert CART_ID not in content, f"{name} leaked the cart id"
        assert ITEM_ID not in content, f"{name} leaked a cart item id"
        assert "cart_id" not in content, f"{name} leaked the cart_id field"


def test_the_money_fields_keep_the_names_the_api_gave_them():
    """No third vocabulary for one amount (CLAUDE.md)."""
    recorder = Recorder([
        created_cart(),
        ("POST", f"/cart/{CART_ID}/items", httpx.Response(200, json=cart_body([one_line()]))),
    ])
    registry, _ = build(recorder)

    content = registry.dispatch("add_to_cart", {"variant_id": 21}).content

    assert '"unit_price_cents": 9499' in content
    assert '"line_total_cents": 9499' in content
    assert '"total_cents": 9499' in content
    assert "amount_cents" not in content


# --- removing a line -----------------------------------------------------


def test_remove_from_cart_deletes_the_line_holding_that_variant():
    """The model names a variant; the item id stays on this side."""
    recorder = Recorder([
        ("GET", f"/cart/{CART_ID}", httpx.Response(200, json=cart_body([one_line()]))),
        ("DELETE", f"/cart/{CART_ID}/items/{ITEM_ID}", httpx.Response(204)),
    ])
    registry, _ = build(recorder, ConversationMemory(cart_id=CART_ID))

    result = registry.dispatch("remove_from_cart", {"variant_id": 21})

    assert result.ok
    assert f"DELETE /cart/{CART_ID}/items/{ITEM_ID}" in recorder.calls


def test_remove_from_cart_refuses_a_variant_the_cart_does_not_hold():
    recorder = Recorder([
        ("GET", f"/cart/{CART_ID}", httpx.Response(200, json=cart_body([one_line(variant_id=21)]))),
    ])
    registry, _ = build(recorder, ConversationMemory(cart_id=CART_ID))

    result = registry.dispatch("remove_from_cart", {"variant_id": 99})

    assert not result.ok
    assert "99" in result.content and "21" in result.content
    assert not any(call.startswith("DELETE") for call in recorder.calls)


# --- checkout ------------------------------------------------------------


def checkout_ok():
    return ("POST", f"/orders/{ORDER_ID}/checkout", httpx.Response(
        201,
        json={
            "order_id": ORDER_ID,
            "checkout_session_id": "cs_test_123",
            "checkout_url": "https://checkout.stripe.com/c/pay/cs_test_123",
        },
    ))


def order_body(status="pending"):
    return {
        "order_id": ORDER_ID,
        "cart_id": CART_ID,
        "status": status,
        "currency": "eur",
        "items": [{k: v for k, v in one_line().items() if k != "item_id"}],
        "total_cents": 9499,
        "created_at": "2026-08-28T12:00:00Z",
        "customer_email": None,
    }


def test_create_checkout_places_the_order_then_asks_for_the_payment_link():
    recorder = Recorder([
        ("POST", "/orders", httpx.Response(201, json=order_body())),
        checkout_ok(),
    ])
    registry, session = build(recorder, ConversationMemory(cart_id=CART_ID))

    result = registry.dispatch("create_checkout", {})

    assert result.ok
    assert recorder.calls == ["POST /orders", f"POST /orders/{ORDER_ID}/checkout"]
    assert "https://checkout.stripe.com/c/pay/cs_test_123" in result.content
    assert session.order_id == ORDER_ID
    assert session.cart_id is None, "the ordered cart must not take another line"


def test_create_checkout_with_nothing_in_the_cart_never_reaches_the_api():
    recorder = Recorder([])
    registry, _ = build(recorder)

    result = registry.dispatch("create_checkout", {})

    assert not result.ok
    assert "empty" in result.content
    assert recorder.calls == []


def test_the_order_is_remembered_even_when_the_payment_link_fails():
    """A checkout that dies at Stripe still placed an order, and it is findable.

    503 is what the API answers when Stripe is unconfigured. The order exists,
    its stock is reserved, and a tool that could not then report on it would
    leave the model insisting nothing had happened.
    """
    recorder = Recorder([
        ("POST", "/orders", httpx.Response(201, json=order_body())),
        ("POST", f"/orders/{ORDER_ID}/checkout", httpx.Response(503, json={"detail": "STRIPE_SECRET_KEY is not set"})),
        ("GET", f"/orders/{ORDER_ID}", httpx.Response(200, json=order_body())),
    ])
    registry, session = build(recorder, ConversationMemory(cart_id=CART_ID))

    failed = registry.dispatch("create_checkout", {})
    status = registry.dispatch("check_order_status", {})

    assert not failed.ok
    assert session.order_id == ORDER_ID
    assert status.ok and '"status": "pending"' in status.content


def test_a_checkout_that_failed_at_stripe_can_be_retried_for_the_same_order():
    """The gap the test above stops one step short of.

    An order placed with no payment link left the conversation in a state
    neither the customer nor the model could leave: the cart id is gone, so a
    second `create_checkout` read an empty cart and said to add something,
    while the order sat pending and holding stock with no way to pay it and no
    way to cancel it. `POST /orders/{id}/checkout` is idempotent by lookup, so
    the honest answer is to call it again. Raised in review on PR #9.

    A hand-written handler rather than `Recorder`, which matches on method and
    path and would answer both checkout attempts with whichever was scripted
    first — and the whole point here is that the two answer differently.
    """
    calls = []

    def handler(request):
        calls.append(f"{request.method} {request.url.path}")
        if request.url.path == "/orders":
            return httpx.Response(201, json=order_body())
        if request.url.path == f"/orders/{ORDER_ID}":
            return httpx.Response(200, json=order_body())
        if request.url.path == f"/orders/{ORDER_ID}/checkout":
            # Stripe unconfigured on the first attempt, working on the second.
            if calls.count(f"POST /orders/{ORDER_ID}/checkout") == 1:
                return httpx.Response(503, json={"detail": "STRIPE_SECRET_KEY is not set"})
            return httpx.Response(201, json={
                "order_id": ORDER_ID,
                "checkout_session_id": "cs_test_123",
                "checkout_url": "https://checkout.stripe.com/c/pay/cs_test_123",
            })
        raise AssertionError(f"unexpected call: {request.url.path}")

    registry, session = build(handler, ConversationMemory(cart_id=CART_ID))

    failed = registry.dispatch("create_checkout", {})
    resumed = registry.dispatch("create_checkout", {})

    assert not failed.ok
    assert resumed.ok, "the second attempt must not be refused for an empty cart"
    assert "https://checkout.stripe.com/c/pay/cs_test_123" in resumed.content
    assert "empty" not in resumed.content
    assert session.order_id == ORDER_ID
    # No second order: the existing one is resumed, never replaced. A retry
    # that placed another would reserve the stock a second time.
    assert calls.count("POST /orders") == 1


def test_resuming_an_order_that_can_no_longer_be_paid_says_so_in_the_api_s_words():
    """A paid or cancelled order is refused by the API, not by a guess here.

    409 is the answer, and its sentence is the one the model should repeat.
    Checking that here is what keeps the resume path from having to know which
    statuses are payable — a second place that would drift from the lifecycle.
    """
    recorder = Recorder([
        ("GET", f"/orders/{ORDER_ID}", httpx.Response(200, json=order_body("paid"))),
        ("POST", f"/orders/{ORDER_ID}/checkout", httpx.Response(409, json={
            "detail": f"order {ORDER_ID} is paid and cannot be paid. Only a pending order can start a checkout."
        })),
    ])
    registry, _ = build(recorder, ConversationMemory(order_id=ORDER_ID))

    result = registry.dispatch("create_checkout", {})

    assert not result.ok
    assert "is paid and cannot be paid" in result.content


def test_check_order_status_reads_the_session_order_not_one_the_model_names():
    recorder = Recorder([("GET", f"/orders/{ORDER_ID}", httpx.Response(200, json=order_body("paid")))])
    registry, _ = build(recorder, ConversationMemory(order_id=ORDER_ID))

    result = registry.dispatch("check_order_status", {"order_id": "an-id-the-model-made-up"})

    assert result.ok
    assert recorder.calls == [f"GET /orders/{ORDER_ID}"]


def test_check_order_status_before_any_order_says_so():
    registry, _ = build(Recorder([]))

    result = registry.dispatch("check_order_status", {})

    assert not result.ok
    assert "no order" in result.content.lower()


# --- what the model is told when HTTP goes wrong -------------------------
#
# The five failure shapes below are the reason this file uses a fake transport
# rather than the FastAPI TestClient: none of them is reachable against a
# healthy local API, and the text the model reads in each is the whole point of
# the tool layer.


def failing(exc_or_response):
    """A registry whose every call fails the same way."""
    def handler(request):
        if isinstance(exc_or_response, Exception):
            raise exc_or_response
        return exc_or_response

    return build(handler, ConversationMemory(cart_id=CART_ID, order_id=ORDER_ID))[0]


def test_a_dead_api_reaches_the_model_as_advice_not_as_a_traceback():
    """uvicorn is not running — the failure this whole layer exists to dress.

    A `ConnectError` allowed through `dispatch`'s generic handler arrives as
    "the tool 'add_to_cart' failed while running: ConnectError: [Errno 61]
    Connection refused", which the model has been observed to read back to a
    customer. It must say what happened in the shop's terms and that repeating
    the call is pointless.
    """
    registry = failing(httpx.ConnectError("[Errno 61] Connection refused"))

    result = registry.dispatch("add_to_cart", {"variant_id": 21})

    assert not result.ok
    assert "ConnectError" not in result.content
    assert "Errno" not in result.content
    assert "temporarily unavailable" in result.content
    assert "Do not call this tool again" in result.content


def test_a_timeout_on_a_write_refuses_to_advise_a_retry():
    """After a timeout on a write, nothing here knows whether it landed."""
    registry = failing(httpx.ReadTimeout("timed out"))

    result = registry.dispatch("add_to_cart", {"variant_id": 21})

    assert not result.ok
    assert "not known whether this took effect" in result.content
    assert "Do NOT repeat it" in result.content
    assert "view_cart" in result.content


def test_a_timeout_on_a_read_allows_one_more_attempt():
    """The same failure, and the honest advice is the opposite one."""
    registry = failing(httpx.ReadTimeout("timed out"))

    result = registry.dispatch("check_order_status", {})

    assert not result.ok
    assert "Nothing was changed" in result.content
    assert "one more time" in result.content


# --- a socket that broke after the request went out -----------------------
#
# Raised in review on PR #9. Every non-timeout transport failure used to be
# mapped to "unreachable", whose message tells the model **"Nothing was
# charged"** — a definite claim. It is only definite for a connection that was
# never established. A read error or a protocol violation happens with the
# socket already open, so the request may well have arrived and been committed,
# and on `create_checkout` that is precisely an order placed behind a lost
# answer.


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadError("connection reset"),
        httpx.WriteError("broken pipe"),
        httpx.RemoteProtocolError("server disconnected"),
    ],
)
def test_a_broken_exchange_on_a_write_never_claims_nothing_was_charged(exc):
    registry = failing(exc)

    result = registry.dispatch("create_checkout", {})

    assert not result.ok
    assert "Nothing was charged" not in result.content
    assert "not known whether this took effect" in result.content
    assert "Do NOT repeat it" in result.content
    assert "check_order_status" in result.content


def test_a_broken_exchange_does_not_borrow_the_timeout_s_sentence():
    """It is a different cause, and the timeout's message names seconds.

    Telling a customer the shop "did not answer within 10 seconds" about a
    connection that was reset immediately is a sentence the model repeats and
    nobody can act on.
    """
    result = failing(httpx.ReadError("connection reset")).dispatch("create_checkout", {})

    assert "seconds" not in result.content
    assert "connection" in result.content.lower()


def test_a_broken_exchange_on_a_read_still_allows_one_more_attempt():
    """A read changed nothing whatever happened to the socket."""
    result = failing(httpx.ReadError("connection reset")).dispatch("check_order_status", {})

    assert not result.ok
    assert "Nothing was changed" in result.content
    assert "one more time" in result.content


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("[Errno 61] Connection refused"),
        httpx.ConnectTimeout("timed out"),
        httpx.PoolTimeout("no connection available"),
    ],
)
def test_only_a_connection_that_was_never_made_may_say_nothing_went_through(exc):
    """The other half of the split, so it cannot be widened back by accident."""
    result = failing(exc).dispatch("create_checkout", {})

    assert not result.ok
    assert "Nothing was charged" in result.content


def test_a_409_reaches_the_model_in_the_api_s_own_words():
    """The API already wrote these for a reader; rewriting them adds an author."""
    detail = "only 2 units of FF-TRLGTX-42-BLK are available; 5 were requested"
    registry = failing(httpx.Response(409, json={"detail": detail}))

    result = registry.dispatch("add_to_cart", {"variant_id": 21, "quantity": 5})

    assert not result.ok
    assert detail in result.content


def test_a_401_is_reported_as_our_fault_and_never_as_the_customer_s():
    """A configuration fault must not be dressed up as a shopping problem.

    The dangerous version of this message is one that reads like a refusal —
    the model would then tell the customer their order was declined, which is
    false, and might ask them for a credential to fix it, which is worse.
    """
    registry = failing(httpx.Response(401, json={"detail": "a valid X-API-Key header is required"}))

    result = registry.dispatch("view_cart", {})

    assert not result.ok
    assert "configuration fault on our side" in result.content
    assert "do not ask them for any password, key or account detail" in result.content.lower()
    assert "X-API-Key" not in result.content


def test_a_500_tells_the_model_nothing_about_what_broke():
    registry = failing(httpx.Response(500, json={"detail": "psycopg.OperationalError: connection refused on 5432"}))

    result = registry.dispatch("view_cart", {})

    assert not result.ok
    assert "psycopg" not in result.content
    assert "5432" not in result.content
    assert "temporarily unavailable" in result.content


def test_a_422_names_the_field_the_api_rejected():
    """Unreachable through the args model, and answered usefully anyway."""
    registry = failing(httpx.Response(422, json={
        "detail": [{"loc": ["body", "quantity"], "msg": "Input should be greater than 0"}]
    }))

    result = registry.dispatch("add_to_cart", {"variant_id": 21})

    assert not result.ok
    assert "quantity: Input should be greater than 0" in result.content


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("refused"),
        httpx.ReadTimeout("timed out"),
        httpx.Response(401, json={"detail": "nope"}),
        httpx.Response(409, json={"detail": "the cart has already been ordered"}),
        httpx.Response(500, text="<html>500</html>"),
        httpx.Response(200, text="not json at all"),
    ],
    ids=["refused", "timeout", "401", "409", "500", "not-json"],
)
@pytest.mark.parametrize("name", TOOL_NAMES)
def test_dispatch_never_raises_whatever_the_api_does(name, failure):
    """The registry's promise, held against a second process that can fail."""
    registry = failing(failure)
    arguments = {"variant_id": 21} if "cart" in name and name != "view_cart" else {}

    result = registry.dispatch(name, arguments)

    assert not result.ok
    assert result.content.startswith("Error:")
    assert "Traceback" not in result.content


def test_bad_arguments_still_come_back_from_the_registry_not_from_here():
    """Input validation is the registry's job and stays there (CLAUDE.md)."""
    registry, _ = build(lambda request: httpx.Response(200, json=cart_body()))

    result = registry.dispatch("add_to_cart", {"variant_id": -1})

    assert not result.ok
    assert "variant_id" in result.content
    assert "greater than 0" in result.content
