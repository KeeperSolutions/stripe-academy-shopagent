"""HTTP for orders (D6, step 4).

Parses, calls one service function, maps a domain exception to a status code.
Every rule about what may become an order lives in
`api/services/orders.py`, which imports no FastAPI and is therefore callable
from D8's webhook.

Authentication is not mentioned here: the router is mounted in `api/main.py`
with `dependencies=[Depends(require_api_key)]`.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from shopagent.api.db import get_session
from shopagent.api.lifecycle import IllegalTransition
from shopagent.api.schemas import (
    CheckoutSessionResponse,
    CreateOrderRequest,
    OrderItemResponse,
    OrderResponse,
)
from shopagent.payments import checkout as checkout_service
from shopagent.payments.checkout import (
    CheckoutTotalMismatch,
    OrderNotPayable,
    PaymentAlreadyInProgress,
)
from shopagent.payments.checkout import OrderNotFound as CheckoutOrderNotFound
from shopagent.payments.stripe_svc import MissingStripeKey
from shopagent.api.services import orders as order_service
from shopagent.payments import customers as customer_service
from shopagent.api.services.cart import CartNotFound
from shopagent.api.services.orders import (
    CartAlreadyOrdered,
    CartEmpty,
    LineNotPriceable,
    OrderNotFound,
    RenderedOrder,
    StockUnavailable,
)

router = APIRouter(prefix="/orders", tags=["orders"])


def _to_response(rendered: RenderedOrder) -> OrderResponse:
    return OrderResponse(
        order_id=rendered.order_id,
        cart_id=rendered.cart_id,
        status=rendered.status,
        currency=rendered.currency,
        items=[
            OrderItemResponse(
                variant_id=line.variant_id,
                sku=line.sku,
                product_name=line.product_name,
                variant_label=line.variant_label,
                quantity=line.quantity,
                unit_price_cents=line.unit_price_cents,
                line_total_cents=line.line_total_cents,
            )
            for line in rendered.lines
        ],
        total_cents=rendered.total_cents,
        created_at=rendered.created_at,
        customer_email=rendered.customer_email,
    )


@router.post("", status_code=status.HTTP_201_CREATED, response_model=OrderResponse)
def create_order(
    payload: CreateOrderRequest, session: Session = Depends(get_session)
) -> OrderResponse:
    """Turn a cart into a pending order.

    Every 409 below describes a cart that cannot be bought as it stands, which
    is a state the caller can act on: reduce a quantity, remove a line, start a
    new cart. That is why they are not 400s — nothing about the request is
    malformed.
    """
    try:
        order = order_service.place_order(session, payload.cart_id)
        # After `place_order`, never inside it: creating a Stripe Customer is a
        # network round trip, and doing it inside the transaction that holds
        # `SELECT ... FOR UPDATE` on inventory would keep those rows locked for
        # as long as Stripe takes to answer.
        customer_service.attach_customer(session, order, payload.customer_email)
    except CartNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except (
        CartAlreadyOrdered,
        CartEmpty,
        StockUnavailable,
        LineNotPriceable,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc

    return _to_response(order_service.render_order(session, order.id))


@router.get("/{order_id}", response_model=OrderResponse)
def read_order(
    order_id: uuid.UUID, session: Session = Depends(get_session)
) -> OrderResponse:
    """The order as it was placed, not as the catalog reads today."""
    try:
        return _to_response(order_service.render_order(session, order_id))
    except OrderNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc


@router.post(
    "/{order_id}/checkout",
    status_code=status.HTTP_201_CREATED,
    response_model=CheckoutSessionResponse,
)
def create_checkout(
    order_id: uuid.UUID, session: Session = Depends(get_session)
) -> CheckoutSessionResponse:
    """Start a Stripe Checkout for this order, or return the one in flight.

    A repeat call is not an error and does not create a second session: the
    order remembers the session it started, and an open one is handed back.
    That is what lets a shopper who closed the tab return to the same payment.

    503 rather than 500 when Stripe is unconfigured. The distinction is worth
    the extra line: 500 says the server broke, while 503 says this particular
    capability is not available right now — which is exactly true, and is a
    thing an operator can act on. The cart and order API is unaffected.
    """
    try:
        checkout_session = checkout_service.create_checkout_session(session, order_id)
    except CheckoutOrderNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except (OrderNotPayable, PaymentAlreadyInProgress) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except MissingStripeKey as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except CheckoutTotalMismatch as exc:
        # Not the caller's fault and not something a different request would
        # fix: the order's own lines disagree with its total, which is a bug on
        # this side. 500, but with a body that says what was wrong rather than
        # an unhandled traceback — and nothing was charged.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc

    if not checkout_session.url:
        # `Session.url` is Optional in the SDK: it is populated while a session
        # is open and empty once it is not. Reaching here means the session is
        # not payable despite having passed the checks above, and returning
        # `null` as a checkout URL would push that surprise onto the caller.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"the Checkout Session for order {order_id} has no URL, which "
                "means it is no longer open. Retry to start a new one."
            ),
        )

    return CheckoutSessionResponse(
        order_id=order_id,
        checkout_session_id=checkout_session.id,
        checkout_url=checkout_session.url,
    )


@router.post("/{order_id}/cancel", response_model=OrderResponse)
def cancel_order(
    order_id: uuid.UUID, session: Session = Depends(get_session)
) -> OrderResponse:
    """Cancel a pending order, release its stock, and close its payment page.

    409 on any order that is not pending, which includes a second cancellation:
    `cancelled` is terminal, so the lifecycle refuses it and the reservation
    cannot be released twice. That refusal is the mechanism — releasing twice
    would drive `reserved` below what the order ever held.

    `paid` orders are refused too, and deliberately. D6's transition table has
    no `paid -> cancelled`: once a charge settles the only way back is a
    refund, which is a movement of money rather than a change of mind.
    """
    try:
        order_service.cancel_order(session, order_id)
    except OrderNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except IllegalTransition as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc

    return _to_response(order_service.render_order(session, order_id))
