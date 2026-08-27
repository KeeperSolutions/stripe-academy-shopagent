"""HTTP for carts (D6).

This module parses, calls one service function, and turns a domain exception
into a status code. It holds no rule about what a cart may contain — that is
`api/services/cart.py`, which knows nothing about HTTP and can therefore be
called by D9's agent tools directly.

Authentication is not mentioned anywhere in this file. The router is mounted in
`api/main.py` with `dependencies=[Depends(require_api_key)]`, so every route
below is protected by where it lives rather than by whoever wrote it
remembering a decorator.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from shopagent.api.db import get_session
from shopagent.api.schemas import AddItemRequest, CartItemResponse, CartResponse
from shopagent.api.services import cart as cart_service
from shopagent.api.services.cart import (
    CartItemNotFound,
    CartLocked,
    CartNotFound,
    InsufficientStock,
    RenderedCart,
    VariantNotFound,
    VariantNotSellable,
)

router = APIRouter(prefix="/cart", tags=["cart"])


def _to_response(rendered: RenderedCart) -> CartResponse:
    return CartResponse(
        cart_id=rendered.cart_id,
        status=rendered.status,
        currency=rendered.currency,
        items=[
            CartItemResponse(
                item_id=line.item_id,
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
    )


def _not_found(exc: Exception) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


def _conflict(exc: Exception) -> HTTPException:
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@router.post("", status_code=status.HTTP_201_CREATED, response_model=CartResponse)
def create_cart(session: Session = Depends(get_session)) -> CartResponse:
    """Open an empty cart.

    Returns the whole cart rather than only its id. It costs one more read and
    it means every cart endpoint answers with the same shape, so a caller — the
    model on D9 included — never has to remember which one is the odd shape.
    """
    cart = cart_service.create_cart(session)
    return _to_response(cart_service.render_cart(session, cart.id))


@router.post("/{cart_id}/items", response_model=CartResponse)
def add_item(
    cart_id: uuid.UUID,
    payload: AddItemRequest,
    response: Response,
    session: Session = Depends(get_session),
) -> CartResponse:
    """Add a variant, or add to the line that is already there.

    201 when a line was created, 200 when an existing line was incremented.
    The distinction is real — an upsert that merges creates nothing — and it is
    the one bit of information a caller cannot otherwise get from the response
    without having remembered the cart's previous contents. A single fixed 201
    would be claiming a resource was created on a request that only changed a
    number.
    """
    try:
        _, created = cart_service.add_item(
            session, cart_id, payload.variant_id, payload.quantity
        )
    except CartNotFound as exc:
        raise _not_found(exc) from exc
    except VariantNotFound as exc:
        raise _not_found(exc) from exc
    except (CartLocked, InsufficientStock, VariantNotSellable) as exc:
        raise _conflict(exc) from exc

    response.status_code = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    return _to_response(cart_service.render_cart(session, cart_id))


@router.get("/{cart_id}", response_model=CartResponse)
def read_cart(
    cart_id: uuid.UUID, session: Session = Depends(get_session)
) -> CartResponse:
    """The cart, with its total recomputed from the database.

    Works on a cart that has become an order: an order has to stay readable
    after it is placed, so the lock applies to writes only.
    """
    try:
        return _to_response(cart_service.render_cart(session, cart_id))
    except CartNotFound as exc:
        raise _not_found(exc) from exc


@router.delete("/{cart_id}/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_item(
    cart_id: uuid.UUID,
    item_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> None:
    """Drop a line.

    404 when the item belongs to a different cart, not 403. A 403 would confirm
    that the id exists, which is the one fact an id that is not yours should
    not be able to establish.
    """
    try:
        cart_service.remove_item(session, cart_id, item_id)
    except (CartNotFound, CartItemNotFound) as exc:
        raise _not_found(exc) from exc
    except CartLocked as exc:
        raise _conflict(exc) from exc
