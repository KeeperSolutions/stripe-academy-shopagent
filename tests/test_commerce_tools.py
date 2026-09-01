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

from shopagent.agent.memory import ConversationMemory, RememberingRegistry
from shopagent.tools.commerce import register_commerce_tools
from shopagent.tools.http import CommerceAPI
from shopagent.tools.registry import ToolRegistry

TOOL_NAMES = [
    "add_to_cart",
    "view_cart",
    "remove_from_cart",
    "create_checkout",
    "check_order_status",
    "request_refund",
]

CART_ID = "11111111-1111-1111-1111-111111111111"
ITEM_ID = "22222222-2222-2222-2222-222222222222"
ORDER_ID = "33333333-3333-3333-3333-333333333333"


def build(handler, session=None):
    """A registry holding the six tools, wired to a fake commerce API."""
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


def test_the_six_tools_are_registered():
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


def test_no_result_the_model_reads_carries_the_payment_link():
    """The same rule as the cart id, and it was written for a measured failure.

    A Checkout Session URL is 475 opaque characters. Asked twice for the same
    session in one conversation, the model reproduced it correctly once and
    changed a single character the second time — `TlZQ` to `TlVQ`, at position
    329 — and Stripe answers 401 for the result. The customer gets a dead
    payment page. Found in the end-to-end run for PR #9.

    So the link is not in the result at all: it goes to the conversation's
    state and whatever is presenting the conversation prints the bytes the shop
    issued. A model cannot mistype a string it was never given.
    """
    recorder = Recorder([
        ("POST", "/orders", httpx.Response(201, json=order_body())),
        checkout_ok(),
    ])
    registry, session = build(recorder, ConversationMemory(cart_id=CART_ID))

    content = registry.dispatch("create_checkout", {}).content

    assert "checkout.stripe.com" not in content, "the payment link leaked to the model"
    assert "cs_test_123" not in content, "the session id leaked to the model"
    assert "checkout_url" not in content, "the field name leaked to the model"
    assert session.checkout_url, "and it still has to reach the layer that prints it"


def test_the_note_tells_the_model_not_to_write_a_link_it_does_not_have():
    """Without this the model invents one, which is the failure one step worse.

    A tool result that silently drops a field the model was expecting leaves it
    filling the gap from memory. The note has to say the link exists, that the
    customer already has it, and that writing one is wrong.
    """
    recorder = Recorder([
        ("POST", "/orders", httpx.Response(201, json=order_body())),
        checkout_ok(),
    ])
    registry, _ = build(recorder, ConversationMemory(cart_id=CART_ID))

    note = json.loads(registry.dispatch("create_checkout", {}).content)["note"].lower()

    assert "do not write a link" in note
    assert "customer" in note
    assert "pending" in note


def test_the_link_is_taken_once_so_it_is_not_reprinted_under_every_later_answer():
    """`take_checkout_url` answers "did a link arrive?", not "is there one?".

    The second question would print the payment page under every later answer
    in the conversation, and a payment page shown again beneath "your order is
    paid" is one somebody clicks.
    """
    url = "https://checkout.stripe.com/c/pay/cs_test_123"
    memory = ConversationMemory(checkout_url=url)

    assert memory.take_checkout_url() == url
    assert memory.take_checkout_url() is None


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
    assert session.order_id == ORDER_ID
    assert session.cart_id is None, "the ordered cart must not take another line"
    # The link reaches the conversation's state, which is what the CLI prints.
    assert session.checkout_url == "https://checkout.stripe.com/c/pay/cs_test_123"


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
    assert "empty" not in resumed.content
    assert session.order_id == ORDER_ID
    # The same link as the first checkout, and the customer gets it again: a
    # resume is what a shopper who lost the page asks for.
    assert session.checkout_url == "https://checkout.stripe.com/c/pay/cs_test_123"
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


# --- the gate and the resume, against the real API (D10, step 1) ---------
#
# Everything above is offline against `MockTransport`, which is what makes the
# failure paths reachable. This one is not, and it is the only test here that
# needs a database, because the claim is about rows: a `create_checkout`
# confirmed twice in one conversation must place *one* order.
#
# The claim is not obvious and it is not the gate's doing. `create_checkout`
# writes `order_id` and clears `cart_id` before it calls Stripe, so a second
# call takes the resume branch — `GET /orders/{id}` and the idempotent
# `POST /orders/{id}/checkout` D7 built, which hands back the session already
# stored rather than opening another. D10 put a second approval in front of
# that path; what this asserts is that the approval buys a payment link and not
# a second order.


class FakeCheckoutSession:
    """What Stripe returns, reduced to the two fields the router reads."""

    def __init__(self, identifier="cs_test_resume"):
        self.id = identifier
        self.url = f"https://checkout.stripe.com/c/pay/{identifier}"


@pytest.fixture
def commerce_through_the_api(authed_client, monkeypatch):
    """A `CommerceAPI` whose requests are served by the in-process FastAPI app.

    `authed_client` already routes handlers into the test's own transaction and
    carries the API key, so forwarding to it puts the agent's tools through the
    same door as any other client — which is the reason `tools/commerce.py`
    speaks HTTP in the first place.
    """
    from shopagent.api.routers import orders as orders_router

    sessions = []

    def fake_create(session, order_id):
        # Idempotent by lookup, the way D7's real one is: one session per order,
        # returned again on every later call.
        if not sessions:
            sessions.append(FakeCheckoutSession())
        return sessions[0]

    monkeypatch.setattr(
        orders_router.checkout_service, "create_checkout_session", fake_create
    )

    def handler(request: httpx.Request) -> httpx.Response:
        forwarded = authed_client.request(
            request.method,
            str(request.url),
            content=request.content or None,
            headers={
                name: value
                for name, value in request.headers.items()
                if name.lower() not in {"host", "content-length"}
            },
        )
        return httpx.Response(
            forwarded.status_code,
            content=forwarded.content,
            headers={"content-type": forwarded.headers.get("content-type", "application/json")},
        )

    # The real key rather than a placeholder: the forwarded request carries
    # the header the agent sent, so a wrong one here would be a 401 from the
    # app's own `require_api_key` — which is the point of routing through it.
    from shopagent.config import get_settings

    return CommerceAPI(
        base_url="http://commerce.test",
        api_key=get_settings().shopagent_api_key,
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.db
def test_a_second_confirmed_checkout_resumes_and_places_no_second_order(
    commerce_through_the_api, session
):
    from sqlalchemy import func, select

    from shopagent.agent import confirmation
    from shopagent.agent.guardrails import GuardedRegistry
    from shopagent.agent.memory import ConversationMemory
    from shopagent.api.models import Order
    from shopagent.catalog.models import Inventory, Price, Product, Variant
    from shopagent.config import get_settings
    from shopagent.tools.commerce import build_commerce_tools

    product = Product(
        name="Resume Fixture",
        description="A product that exists only for this test.",
        category="shoes",
        brand="Fleetfoot",
        variants=[
            Variant(
                size="42",
                color="blue",
                sku="FF-RESUME-42-BLU",
                prices=[
                    Price(currency=get_settings().currency, amount_cents=9499, active=True)
                ],
                inventory=Inventory(quantity=10, reserved=0),
            )
        ],
    )
    session.add(product)
    session.commit()
    variant_id = product.variants[0].id

    # A delta rather than an absolute count. `orders` holds what real people
    # did, so a row left behind by a manual run is not a reason for this to
    # fail — CLAUDE.md says so about the D6 suite and it is true here too.
    orders_before = session.execute(select(func.count()).select_from(Order)).scalar_one()

    memory = ConversationMemory()
    registry = GuardedRegistry(memory, can_confirm=True)
    for spec in build_commerce_tools(commerce_through_the_api, memory):
        registry.register(spec)

    def one_confirmed_checkout():
        """A whole turn's worth of the protocol, as a caller drives it."""
        memory.begin_turn(from_customer=True)
        asked = registry.dispatch("create_checkout", {})
        confirmation.resolve_pending(memory, confirmation.ScriptedConfirmer(answer=True))
        memory.begin_turn(from_customer=False)
        return asked, registry.dispatch("create_checkout", {})

    # The variant guard refuses an id the model was never shown, and in a real
    # conversation a search is what shows it. This test is about the checkout,
    # so the showing is recorded directly rather than standing a catalog up.
    memory.observe("search_products", {}, json.dumps({"results": [{"variant_id": variant_id}]}))
    assert registry.dispatch("add_to_cart", {"variant_id": variant_id}).ok

    _, first = one_confirmed_checkout()
    assert first.ok, first.content
    first_order_id = memory.order_id
    first_url = memory.checkout_url

    _, second = one_confirmed_checkout()

    assert second.ok, second.content
    assert memory.order_id == first_order_id, "the resume placed a second order"
    assert memory.checkout_url == first_url, "and it must be the same payment page"

    orders = session.execute(
        select(func.count()).select_from(Order).where(Order.id == first_order_id)
    ).scalar_one()
    assert orders == 1

    orders_after = session.execute(select(func.count()).select_from(Order)).scalar_one()
    assert orders_after == orders_before + 1, (
        f"{orders_after - orders_before} orders were placed by two confirmed checkouts"
    )


@pytest.mark.db
def test_the_second_approval_is_asked_about_the_order_and_not_an_empty_cart(
    commerce_through_the_api, session
):
    """The resume summary is what PR #9 found: a cart cleared by the first
    checkout must not be summarised as "Total: €0.00" for a real purchase."""
    from shopagent.agent import confirmation
    from shopagent.agent.guardrails import RESUMING, GuardedRegistry
    from shopagent.agent.memory import ConversationMemory
    from shopagent.catalog.models import Inventory, Price, Product, Variant
    from shopagent.config import get_settings
    from shopagent.tools.commerce import build_commerce_tools

    product = Product(
        name="Resume Summary Fixture",
        description="A product that exists only for this test.",
        category="shoes",
        brand="Fleetfoot",
        variants=[
            Variant(
                size="43",
                color="red",
                sku="FF-RESUME-43-RED",
                prices=[
                    Price(currency=get_settings().currency, amount_cents=9499, active=True)
                ],
                inventory=Inventory(quantity=10, reserved=0),
            )
        ],
    )
    session.add(product)
    session.commit()

    memory = ConversationMemory()
    registry = GuardedRegistry(memory, can_confirm=True)
    for spec in build_commerce_tools(commerce_through_the_api, memory):
        registry.register(spec)
    memory.observe(
        "search_products",
        {},
        json.dumps({"results": [{"variant_id": product.variants[0].id}]}),
    )
    registry.dispatch("add_to_cart", {"variant_id": product.variants[0].id})

    confirmer = confirmation.ScriptedConfirmer(answer=True)
    for _ in range(2):
        memory.begin_turn(from_customer=True)
        registry.dispatch("create_checkout", {})
        confirmation.resolve_pending(memory, confirmer)
        memory.begin_turn(from_customer=False)
        registry.dispatch("create_checkout", {})

    first_summary, second_summary = confirmer.asked
    assert "€94.99" in first_summary
    assert RESUMING in second_summary
    assert "€94.99" in second_summary
    assert "€0.00" not in second_summary


# --- asking for a refund, which is not the same as getting one -----------


def refund_body(order_status="paid", amount=9499):
    """What `POST /orders/{id}/refund` really answers.

    Built from `api/schemas.py::RefundResponse`, every field present including
    the two this tool deliberately drops. A fixture holding only what the
    assertions read could not show that `refund_status` is omitted on purpose,
    which is the decision most worth pinning here.
    """
    return {
        "order_id": ORDER_ID,
        "refund_id": "re_3UAncwRnt986EK7P1abcdefg",
        "refund_status": "succeeded",
        "amount_cents": amount,
        "currency": "eur",
        "order_status": order_status,
    }


def test_request_refund_asks_the_api_for_this_conversations_order():
    recorder = Recorder([("POST", f"/orders/{ORDER_ID}/refund", httpx.Response(202, json=refund_body()))])
    registry, _ = build(recorder, ConversationMemory(order_id=ORDER_ID))

    result = registry.dispatch("request_refund", {})

    assert result.ok
    assert recorder.calls == [f"POST /orders/{ORDER_ID}/refund"]


def test_request_refund_reports_a_request_and_never_a_completed_refund():
    """The whole of what this tool is careful about.

    `POST /orders/{id}/refund` answers 202: Stripe accepted, the money has not
    moved, and the order is still `paid` until `charge.refunded` lands. A
    result shaped like a finished action produces "your refund is complete",
    and the customer then reads `paid` when they ask again.
    """
    recorder = Recorder([("POST", f"/orders/{ORDER_ID}/refund", httpx.Response(202, json=refund_body()))])
    registry, _ = build(recorder, ConversationMemory(order_id=ORDER_ID))

    payload = json.loads(registry.dispatch("request_refund", {}).content)

    assert payload["refund_requested"] is True
    assert "refunded" not in payload, "a key called refunded reads as a finished one"
    # The status is included *because* it still says paid. It is the field that
    # would otherwise be assumed, which is the same argument `api/schemas.py`
    # makes for putting it in the response.
    assert payload["order_status"] == "paid"
    assert "not completed" in payload["note"]
    assert "check_order_status" in payload["note"]


def test_the_refund_result_hides_stripes_own_status():
    """The interesting omission, and the reason it is not a gap.

    Stripe reports `succeeded` immediately for a card. A model holding two
    statuses called "succeeded" and "paid" collapses them into one sentence,
    and the sentence it picks is the wrong one. The API returns it because an
    HTTP client can hold both; this layer does not, because this reader will
    not — and the note says what happened, so there is nothing for the model to
    fill in.
    """
    recorder = Recorder([("POST", f"/orders/{ORDER_ID}/refund", httpx.Response(202, json=refund_body()))])
    registry, _ = build(recorder, ConversationMemory(order_id=ORDER_ID))

    content = registry.dispatch("request_refund", {}).content

    assert "succeeded" not in content
    assert "refund_status" not in content


def test_the_refund_result_hides_the_refund_id():
    """The `cart_id` rule reaching its next piece of state.

    An opaque string the model has to carry is one it will eventually get
    wrong — measured on a Stripe URL in PR #9 — and no tool accepts a refund id
    back, so there is no argument for it to get wrong either. The order id is
    the reference a customer needs and it is returned.
    """
    recorder = Recorder([("POST", f"/orders/{ORDER_ID}/refund", httpx.Response(202, json=refund_body()))])
    registry, _ = build(recorder, ConversationMemory(order_id=ORDER_ID))

    content = registry.dispatch("request_refund", {}).content

    assert "re_3UAncwRnt986EK7P1abcdefg" not in content
    assert ORDER_ID in content


def test_the_refunded_amount_is_recorded_so_the_model_may_quote_it():
    """`agent/memory.py` collects keys ending in `_cents`, and this is one.

    The amount guardrail refuses a figure that never appeared in a tool result.
    Naming the field `amount_cents` rather than `amount` is what lets the model
    say "€94.99 is on its way back" without being corrected — and it is the
    naming rule in CLAUDE.md doing real work rather than being tidy.
    """
    recorder = Recorder([("POST", f"/orders/{ORDER_ID}/refund", httpx.Response(202, json=refund_body()))])
    memory = ConversationMemory(order_id=ORDER_ID)
    registry = ToolRegistry()
    api = CommerceAPI(
        base_url="http://commerce.test", api_key="k", transport=httpx.MockTransport(recorder)
    )
    register_commerce_tools(registry, api, memory)
    remembering = RememberingRegistry(memory)
    for spec in registry.specs():
        remembering.register(spec)

    remembering.dispatch("request_refund", {})

    assert 9499 in memory.seen_amount_cents


def test_request_refund_before_any_order_says_so_without_calling_the_api():
    def refuse(request):  # pragma: no cover - reaching it is the failure
        raise AssertionError(f"the API was called: {request.method} {request.url.path}")

    registry, _ = build(refuse)

    result = registry.dispatch("request_refund", {})

    assert not result.ok
    assert "nothing to refund" in result.content
    # The limit is named rather than left for the model to discover, because
    # the customer's next sentence is about an order this assistant cannot see.
    assert "earlier conversation" in result.content


def test_request_refund_takes_no_order_id_from_the_model():
    """The rule the whole module lives under, on the newest tool.

    An id the model carries is one it invents. `request_refund` refunds this
    conversation's order and nothing else, so an `order_id` in the arguments is
    ignored rather than honoured.
    """
    recorder = Recorder([("POST", f"/orders/{ORDER_ID}/refund", httpx.Response(202, json=refund_body()))])
    registry, _ = build(recorder, ConversationMemory(order_id=ORDER_ID))

    registry.dispatch("request_refund", {"order_id": "an-order-the-model-made-up"})

    assert recorder.calls == [f"POST /orders/{ORDER_ID}/refund"]


def test_request_refund_offers_the_model_no_amount_to_set():
    """There is nowhere for a partial refund to live, so there is no argument.

    `stripe_svc.create_refund` takes no amount and `orders.status` has no
    "partly refunded", so an `amount` here would be the model offering a
    customer something this shop cannot do — and `handle_charge_refunded`
    would log the result at ERROR and change nothing.
    """
    registry, _ = build(lambda request: httpx.Response(200, json=cart_body()))

    schema = next(
        spec.to_openai_schema()
        for spec in registry.specs()
        if spec.name == "request_refund"
    )

    assert schema["function"]["parameters"].get("properties", {}) == {}


def test_an_order_the_api_refuses_to_refund_reaches_the_model_in_its_own_words():
    """A 409 is written for whoever reads it, and rewriting it here would give
    one contract two authors. `refund_order` refuses a pending order, a
    zero-total one and one already refunded, each with its own sentence."""
    recorder = Recorder([(
        "POST",
        f"/orders/{ORDER_ID}/refund",
        httpx.Response(409, json={"detail": "order is pending, so it cannot be refunded"}),
    )])
    registry, _ = build(recorder, ConversationMemory(order_id=ORDER_ID))

    result = registry.dispatch("request_refund", {})

    assert not result.ok
    assert "order is pending, so it cannot be refunded" in result.content


@pytest.mark.parametrize("status", ["failed", "canceled"])
def test_a_refund_the_provider_already_rejected_is_not_reported_as_on_its_way(status):
    """Withholding `refund_status` is not the same as ignoring it. PR #11.

    Stripe can come back terminal on this very call. The success-shaped payload
    would then tell the customer their money is on its way when it is not — and
    nothing corrects it, because the order stays `paid` and
    `check_order_status` goes on saying so for ever.
    """
    recorder = Recorder([(
        "POST",
        f"/orders/{ORDER_ID}/refund",
        httpx.Response(202, json=refund_body(order_status="paid") | {"refund_status": status}),
    )])
    registry, _ = build(recorder, ConversationMemory(order_id=ORDER_ID))

    result = registry.dispatch("request_refund", {})

    assert not result.ok
    assert "did not go through" in result.content
    assert "on its way" not in result.content


@pytest.mark.parametrize("status", ["succeeded", "pending", "requires_action", None])
def test_every_non_terminal_refund_status_is_still_an_accepted_request(status):
    """The three Stripe values that mean "not over yet" all take the same path.

    `None` is in the list deliberately: it means the provider said nothing
    about the outcome, which is the ordinary accepted case and not a failure.
    """
    recorder = Recorder([(
        "POST",
        f"/orders/{ORDER_ID}/refund",
        httpx.Response(202, json=refund_body() | {"refund_status": status}),
    )])
    registry, _ = build(recorder, ConversationMemory(order_id=ORDER_ID))

    result = registry.dispatch("request_refund", {})

    assert result.ok
    assert json.loads(result.content)["refund_requested"] is True
