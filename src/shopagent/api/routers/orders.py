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
from shopagent.api.schemas import (
    CreateOrderRequest,
    OrderItemResponse,
    OrderResponse,
)
from shopagent.api.services import orders as order_service
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
