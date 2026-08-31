"""Cart and checkout, as tools the model can call (D9, step 1).

Five tools over the D6 commerce API: `add_to_cart`, `view_cart`,
`remove_from_cart`, `create_checkout`, `check_order_status`. They go over
HTTP rather than importing `api/services/` directly, which is a deliberate
cost: the agent then enters through the same door as any other client, carries
the same `X-API-Key`, and gets the same refusals in the same words. A shortcut
past the router would make the agent the one caller for whom the API's rules
are optional.

**The model never sees a cart id.** It appears in no schema and in no result.
An identifier the model has to carry across turns is one it will eventually
lose or invent, and the entire class of failure disappears if it is never
handed one — so the tool layer holds it and the tools take the arguments a
shopper would actually say: a variant, a quantity. `variant_id` is the one id
the model does handle, because the catalog tools put it there in the same
conversation and every commerce tool takes it in the same meaning.

An order id is different and is returned rather than hidden: it is the
reference a customer needs for support and the dashboard, and no tool accepts
one back, so there is no argument for the model to get wrong.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, Field

from shopagent.tools.http import (
    READ_TIMEOUT_SECONDS,
    CommerceAPI,
    CommerceAPIBroken,
    CommerceAPIInterrupted,
    CommerceAPIRefused,
    CommerceAPITimeout,
    CommerceAPIUnauthorized,
    CommerceAPIUnreachable,
)
from shopagent.tools.registry import ToolRegistry, ToolResult, ToolSpec


class HoldsCartState(Protocol):
    """The two attributes these tools need from a conversation's memory.

    A protocol rather than an import of `agent.memory.ConversationMemory`,
    which is the same choice `api/lifecycle.py` makes when it takes anything
    with a `status` attribute rather than an ORM row: `tools/` is reached by
    the agent, not the other way round, and a tool module that imported the
    agent's memory would invert that for the sake of two fields.

    Step 1 filled this with a small `CommerceSession` dataclass, declared
    temporary in its own docstring; step 3's `ConversationMemory` is what took
    it over. Everything below still only knows that something holds two ids.
    """

    cart_id: str | None
    order_id: str | None


# --- argument models -----------------------------------------------------
#
# Local tools, so `args_model` and never `parameters_schema` — the schema the
# model reads is generated from these, and `dispatch` validates every call
# against the same object, so the two cannot drift. Three of the five take no
# arguments at all, which is the shape the hidden cart id buys: "what is in my
# cart" needs nothing said about *which* cart.


class AddToCartArgs(BaseModel):
    variant_id: int = Field(
        gt=0,
        description=(
            "The variant_id of the exact variant to add — one size and colour "
            "of a product — as returned by search_products, "
            "get_product_details or check_stock. Not the product_id and not "
            "the sku."
        ),
    )
    quantity: int = Field(
        default=1,
        gt=0,
        description="How many units to add. Added to any line already in the cart.",
    )


class ViewCartArgs(BaseModel):
    """No arguments: the cart belongs to this conversation."""


class RemoveFromCartArgs(BaseModel):
    variant_id: int = Field(
        gt=0,
        description=(
            "The variant_id of the line to remove entirely, as shown by "
            "view_cart. Removes the whole line, not one unit of it."
        ),
    )


class CreateCheckoutArgs(BaseModel):
    """No arguments: it checks out the cart of this conversation."""


class CheckOrderStatusArgs(BaseModel):
    """No arguments: it reports on the order placed in this conversation."""


# --- failure text --------------------------------------------------------
#
# Written for the model, which is their only reader. Each says what happened,
# what to tell the customer, and whether calling again is worth a turn — the
# last part matters most, because the alternative to being told is the model
# spending its remaining rounds re-running a call that cannot succeed.
#
# The split between a read and a write is the only thing the wording varies on,
# and it is the one distinction that changes what is safe to do next: after a
# timeout on a write, nothing here knows whether the write landed.

_UNREACHABLE = (
    "Error: the shop's ordering system is not answering, so this did not go "
    "through. Nothing was charged. Tell the customer that ordering is "
    "temporarily unavailable and that browsing products still works. Do not "
    "call this tool again in this conversation turn — it will fail the same way."
)

_UNAUTHORIZED = (
    "Error: the ordering system rejected this assistant's credentials. That is "
    "a configuration fault on our side, not anything the customer did and not "
    "anything they can fix. Tell the customer that ordering is temporarily "
    "unavailable, do not ask them for any password, key or account detail, and "
    "do not call this tool again."
)

_BROKEN = (
    "Error: the ordering system failed while handling this request. Nothing "
    "was charged. Tell the customer that ordering is temporarily unavailable "
    "and suggest trying again later. Do not call this tool again in this turn."
)

_TIMEOUT_READ = (
    f"Error: the ordering system did not answer within {READ_TIMEOUT_SECONDS:.0f} "
    "seconds. Nothing was changed. Tell the customer the shop is slow right "
    "now; you may try this one more time, and if it fails again, say so plainly."
)

_TIMEOUT_WRITE = (
    f"Error: the ordering system did not answer within {READ_TIMEOUT_SECONDS:.0f} "
    "seconds, so it is not known whether this took effect. Do NOT repeat it — "
    "repeating could do it twice. Call view_cart (or check_order_status, if an "
    "order was being placed) to see what the true state is, and tell the "
    "customer what you find."
)

# The same unknown outcome, reached a different way: the connection broke after
# the request went out. It gets its own pair rather than reusing the timeout's
# because the timeout's names a number of seconds that did not elapse here, and
# a message that misdescribes what happened is one the model repeats to a
# customer. Raised in review on PR #9.
_INTERRUPTED_READ = (
    "Error: the connection to the ordering system broke before the answer came "
    "back. Nothing was changed. You may try this one more time, and if it fails "
    "again, tell the customer the shop is having trouble right now."
)

_INTERRUPTED_WRITE = (
    "Error: the connection to the ordering system broke after the request was "
    "sent, so it is not known whether this took effect. Do NOT repeat it — "
    "repeating could do it twice. Call view_cart (or check_order_status, if an "
    "order was being placed) to see what the true state is, and tell the "
    "customer what you find."
)


def _reports_failures(*, changes_state: bool) -> Callable[[Callable], Callable]:
    """Turn a transport failure into a `ToolResult` the model can act on.

    A `ToolResult` rather than a raised exception, because `dispatch` passes
    one through untouched while it wraps an exception in its own sentence —
    "the tool 'add_to_cart' failed while running: ConnectError: [Errno 61]
    Connection refused" is a traceback with a full stop after it, and the model
    is the last reader who should be given one. This is the same seam D5 uses
    to keep an MCP server's `isError` text intact.

    Bad *input* still raises, and is still formatted by the registry: that path
    names the field and is already written for the model. What is caught here
    is everything that goes wrong between two processes, which the registry has
    no way to describe.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(**kwargs: Any) -> Any:
            try:
                return fn(**kwargs)
            except CommerceAPIRefused as exc:
                # The API's own sentence, unchanged. Every 404 and 409 in
                # `api/routers/` was written for whoever reads it — "only 2
                # units of FF-TRLGTX-42-BLK are available" tells the model
                # exactly what to say and what to try instead. Rewriting it
                # here would give one contract two authors.
                return ToolResult(ok=False, content=f"Error: {exc.detail}", error=exc.detail)
            except CommerceAPITimeout:
                content = _TIMEOUT_WRITE if changes_state else _TIMEOUT_READ
                return ToolResult(ok=False, content=content, error="the ordering system timed out")
            except CommerceAPIInterrupted:
                content = _INTERRUPTED_WRITE if changes_state else _INTERRUPTED_READ
                return ToolResult(
                    ok=False,
                    content=content,
                    error="the connection to the ordering system broke mid-request",
                )
            except CommerceAPIUnreachable:
                return ToolResult(
                    ok=False, content=_UNREACHABLE, error="the ordering system is not answering"
                )
            except CommerceAPIUnauthorized:
                return ToolResult(
                    ok=False,
                    content=_UNAUTHORIZED,
                    error="the ordering system rejected this assistant's credentials",
                )
            except CommerceAPIBroken:
                # Deliberately says nothing about *what* broke. A 500 body is
                # written for whoever runs this server, and forwarding it hands
                # the model internals to repeat back to a customer.
                return ToolResult(
                    ok=False, content=_BROKEN, error="the ordering system failed"
                )

        return wrapper

    return decorator


def _refuse(message: str) -> ToolResult:
    """A refusal this layer knows without asking the API.

    An empty cart is a state rather than bad input, so it is not raised: a
    `ValueError` would reach the model wrapped in the registry's "failed while
    running" sentence, which reads like a crash and is not one.
    """
    return ToolResult(ok=False, content=f"Error: {message}", error=message)


# --- shaping what the model reads ---------------------------------------


def _line(item: dict[str, Any]) -> dict[str, Any]:
    """One cart or order line, with the ids the model has no use for removed.

    `item_id` goes: `remove_from_cart` takes a `variant_id`, so an item id in
    the result would be an identifier offered and never accepted — an invitation
    to send it back to a tool that does not want it.

    The money fields are passed through under the names `api/schemas.py` gave
    them. `unit_price_cents` and `line_total_cents` are already the flattened,
    resolved numbers this side is meant to read; renaming them again here would
    invent a third vocabulary for one amount, which is the drift CLAUDE.md's
    `amount_cents`/`price_cents` rule exists to stop.
    """
    return {
        "variant_id": item["variant_id"],
        "sku": item["sku"],
        "product_name": item["product_name"],
        "variant_label": item["variant_label"],
        "quantity": item["quantity"],
        "unit_price_cents": item["unit_price_cents"],
        "line_total_cents": item["line_total_cents"],
    }


def _cart_view(body: dict[str, Any]) -> dict[str, Any]:
    """A cart as the model reads it: no cart id, and the counts done here.

    `line_count` and `unit_count` are computed rather than left to the model
    for a measured reason — D5 recorded it summarising four variants across
    three products as "all three are available". Both numbers are arithmetic
    over a list it is holding, and arithmetic is the thing a tool result is for.
    """
    items = [_line(item) for item in body["items"]]
    return {
        "currency": body["currency"],
        "line_count": len(items),
        "unit_count": sum(item["quantity"] for item in items),
        "items": items,
        "total_cents": body["total_cents"],
    }


def _order_view(body: dict[str, Any]) -> dict[str, Any]:
    items = [_line(item) for item in body["items"]]
    return {
        "order_id": body["order_id"],
        "status": body["status"],
        "currency": body["currency"],
        "line_count": len(items),
        "items": items,
        "total_cents": body["total_cents"],
    }


# --- the tools -----------------------------------------------------------


def build_commerce_tools(api: CommerceAPI, state: HoldsCartState) -> list[ToolSpec]:
    """The five tools, closed over one conversation's API client and memory.

    A factory rather than a module-level registry with decorators, which is
    what `tools/basic.py` uses: those two tools are stateless, these five share
    a cart that belongs to one conversation. Closing over the state keeps the
    functions plain — each is still an ordinary callable a test can invoke
    directly — while making it impossible for two conversations in one process
    to end up sharing a basket.
    """

    def _ensure_cart() -> str:
        """The cart of this conversation, created on first use.

        Lazy on purpose: a cart row per conversation that only ever browses is
        a row nobody asked for. The id is stored the moment `POST /cart`
        answers, so a failure in the call that follows leaves one empty cart
        rather than a new one on every attempt.
        """
        if state.cart_id is None:
            body = api.request("POST", "/cart")
            state.cart_id = str(body["cart_id"])
        return state.cart_id

    @_reports_failures(changes_state=True)
    def add_to_cart(variant_id: int, quantity: int = 1) -> Any:
        cart_id = _ensure_cart()
        body = api.request(
            "POST",
            f"/cart/{cart_id}/items",
            json={"variant_id": variant_id, "quantity": quantity},
        )
        return _cart_view(body)

    @_reports_failures(changes_state=False)
    def view_cart() -> Any:
        if state.cart_id is None:
            # Not an error: a shopper who has added nothing has an empty cart,
            # and answering with a failure would have the model apologise for
            # a system that is working.
            return {
                "line_count": 0,
                "unit_count": 0,
                "items": [],
                "total_cents": 0,
                "note": "The cart is empty — nothing has been added in this conversation yet.",
            }
        return _cart_view(api.request("GET", f"/cart/{state.cart_id}"))

    @_reports_failures(changes_state=True)
    def remove_from_cart(variant_id: int) -> Any:
        if state.cart_id is None:
            return _refuse(
                "the cart is empty, so there is nothing to remove. Tell the "
                "customer their cart is already empty."
            )

        cart = api.request("GET", f"/cart/{state.cart_id}")
        # The lookup that keeps the item id on this side. `view_cart` shows the
        # model a variant, so a variant is what it can name; translating that
        # into the line's own id is this layer's job, and doing it against a
        # freshly read cart means the id cannot be stale.
        line = next(
            (item for item in cart["items"] if item["variant_id"] == variant_id), None
        )
        if line is None:
            held = ", ".join(str(item["variant_id"]) for item in cart["items"])
            return _refuse(
                f"the cart has no line for variant_id {variant_id}. It currently "
                f"holds: {held or 'nothing'}. Call view_cart and remove one of those."
            )

        api.request("DELETE", f"/cart/{state.cart_id}/items/{line['item_id']}")
        # Read back rather than subtracting the line here: the total is the
        # server's to compute, and D6 recomputes it from the database on every
        # read precisely so nobody else has to.
        return _cart_view(api.request("GET", f"/cart/{state.cart_id}"))

    def _payment_link(order: dict[str, Any]) -> dict[str, Any]:
        """Start or resume the Stripe checkout for an order that already exists.

        `POST /orders/{id}/checkout` is idempotent by lookup rather than by
        idempotency key — D7 stores `stripe_checkout_session_id` on the order
        and returns the session that is already open — so calling it a second
        time hands back the same payment page rather than making another. That
        is what lets the refusal below be a resumption instead of a dead end.
        """
        checkout = api.request("POST", f"/orders/{order['order_id']}/checkout")
        return {
            "order_id": order["order_id"],
            "status": order["status"],
            "currency": order["currency"],
            "total_cents": order["total_cents"],
            "checkout_url": checkout["checkout_url"],
            "note": (
                "Give the customer this checkout_url and ask them to pay there. "
                "The order is not paid until they do — say it is pending, never "
                "that it is complete."
            ),
        }

    @_reports_failures(changes_state=True)
    def create_checkout() -> Any:
        # An order already placed in this conversation is resumed, not refused.
        #
        # The two writes below leave a window: the order exists, the cart id is
        # gone, and the Stripe call can still fail — a 503 when no Stripe key is
        # configured, which this project treats as a normal state rather than a
        # startup error, or a connection that broke after the request went out.
        # The order is then pending and holding stock, with no payment page and
        # nothing pointing at it: a second `create_checkout` used to read an
        # empty cart and tell the customer to add something, so neither paying
        # nor cancelling was reachable through the agent at all.
        #
        # It also covers the ordinary case of a customer who lost the link.
        # An order that can no longer be paid is refused by the API in its own
        # words — `paid` and `cancelled` both answer 409 — which is the sentence
        # the model should be repeating anyway. Raised in review on PR #9.
        if state.cart_id is None and state.order_id is not None:
            return _payment_link(api.request("GET", f"/orders/{state.order_id}"))

        if state.cart_id is None:
            return _refuse(
                "there is nothing to check out — the cart is empty. Add at "
                "least one item with add_to_cart first."
            )

        order = api.request("POST", "/orders", json={"cart_id": state.cart_id})
        # Both writes happen before the Stripe call, and in this order. The
        # order id is stored first so `check_order_status` can find the order
        # even if the checkout below fails; the cart is released second because
        # `place_order` has already flipped it to `ordered`, and a cart id kept
        # here would send the next `add_to_cart` into a guaranteed 409.
        state.order_id = str(order["order_id"])
        state.cart_id = None

        return _payment_link(order)

    @_reports_failures(changes_state=False)
    def check_order_status() -> Any:
        if state.order_id is None:
            return _refuse(
                "no order has been placed in this conversation, so there is no "
                "status to report. Use create_checkout to place one."
            )
        return _order_view(api.request("GET", f"/orders/{state.order_id}"))

    return [
        ToolSpec(
            name="add_to_cart",
            description=(
                "Add units of one product variant to the customer's shopping "
                "cart. The cart belongs to this conversation and is created "
                "automatically on the first call — never ask the customer for "
                "a cart or basket number, and never mention one. Returns the "
                "whole cart after the change, with its total in cents. Check "
                "stock first if availability has not been established: this "
                "call is refused when too few units are free."
            ),
            args_model=AddToCartArgs,
            fn=add_to_cart,
        ),
        ToolSpec(
            name="view_cart",
            description=(
                "Show what is currently in the customer's cart: every line "
                "with its variant, quantity and price in cents, and the cart "
                "total. Takes no arguments. Call this before checking out, "
                "before answering any question about what the customer has "
                "chosen, and after any error that leaves the cart's state in "
                "doubt. Never state a cart total from memory."
            ),
            args_model=ViewCartArgs,
            fn=view_cart,
        ),
        ToolSpec(
            name="remove_from_cart",
            description=(
                "Remove one whole line from the cart, named by its variant_id "
                "as shown by view_cart. Removes every unit of that variant, "
                "not one of them. Returns the cart as it stands afterwards."
            ),
            args_model=RemoveFromCartArgs,
            fn=remove_from_cart,
        ),
        ToolSpec(
            name="create_checkout",
            description=(
                "Turn the current cart into an order and return a payment link "
                "for it. Takes no arguments. This is the point of no return: "
                "the cart is closed and the stock is reserved. Before it runs, "
                "the shop shows the customer what they are buying and asks them "
                "to confirm; if they decline, this call comes back as an error "
                "saying so and nothing is ordered. Read the result rather than "
                "assuming it worked. The order is pending, not paid, until the "
                "customer completes the payment page. If an order has already "
                "been placed in this conversation and is still pending, call "
                "this again to get its payment link back — it returns the same "
                "page rather than ordering anything a second time. That is how "
                "a customer who lost the link is helped."
            ),
            args_model=CreateCheckoutArgs,
            fn=create_checkout,
        ),
        ToolSpec(
            name="check_order_status",
            description=(
                "Report the order placed in this conversation: its status "
                "(pending, paid, cancelled, refunded), its lines and its "
                "total. Takes no arguments — it always refers to this "
                "conversation's own order. Use it after the customer says they "
                "have paid; payment is confirmed by the shop, not by the "
                "customer saying so."
            ),
            args_model=CheckOrderStatusArgs,
            fn=check_order_status,
        ),
    ]


def register_commerce_tools(
    registry: ToolRegistry, api: CommerceAPI, state: HoldsCartState
) -> HoldsCartState:
    """Put the five tools into a registry, the way D5 puts the MCP ones in.

    Returns the session so a caller that let this build one can still reach it.
    """
    for spec in build_commerce_tools(api, state):
        registry.register(spec)
    return state
