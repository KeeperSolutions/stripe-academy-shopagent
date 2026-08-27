"""Request and response bodies for the commerce API (D6, D8).

One file rather than one per resource. An order response embeds lines that are
shaped almost exactly like cart lines, and splitting by resource would mean
either a cross-import between two schema modules or two definitions of the same
thing drifting apart. They are one contract surface, so they live in one place.

**The names change here, and that is the point.** A database column is
`amount_cents` — `prices.amount_cents`, `order_items.unit_amount_cents`. What
comes out of this file is `unit_price_cents` and `total_cents`, because this is
the flattened view a reader gets: one number, already resolved to the active
price in the session currency, with no row and no `active` flag behind it. That
is the boundary CLAUDE.md draws between the two names, and this module is where
it is crossed. Never copy a column name through to a response just because it
was convenient.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from shopagent.api.lifecycle import OrderStatus
from shopagent.api.models import CartStatus


class AddItemRequest(BaseModel):
    """What `POST /cart/{cart_id}/items` accepts."""

    model_config = ConfigDict(extra="forbid")

    variant_id: int = Field(
        gt=0, description="The variant to add, as returned by the catalog search."
    )
    # gt=0 rather than a check inside the handler, so a zero or negative
    # quantity is a 422 with the field named, produced before any code runs.
    # The service refuses it a second time for callers that do not come
    # through HTTP.
    quantity: int = Field(
        default=1, gt=0, description="How many units to add. Added to any existing line."
    )


class CartItemResponse(BaseModel):
    """One line of a cart, priced."""

    item_id: uuid.UUID
    variant_id: int
    sku: str
    product_name: str
    variant_label: str | None
    quantity: int
    # Null when the variant has no active price in the session currency — a
    # state reachable only by deactivating a price after the line was added,
    # since adding an unpriced variant is refused. Showing the line without a
    # price beats dropping it silently.
    unit_price_cents: int | None
    line_total_cents: int | None


class CartResponse(BaseModel):
    """A cart, its lines, and the total computed on the server."""

    cart_id: uuid.UUID
    status: CartStatus
    currency: str
    items: list[CartItemResponse]
    # Sums only the lines that could be priced. Never read from a request.
    total_cents: int


class CreateOrderRequest(BaseModel):
    """What `POST /orders` accepts.

    A cart id and nothing else. No totals, no prices, no line list — every one
    of those is read from the database, because a client that can name its own
    total is a client that can name a lower one.
    """

    model_config = ConfigDict(extra="forbid")

    cart_id: uuid.UUID


class OrderItemResponse(BaseModel):
    """One snapshotted order line.

    Carries `variant_id` but no `item_id`. `variant_id` earns its place: D8's
    webhook and D9's agent tools both need to say *which product* an order line
    refers to, and the sku alone would make them look it up. An `item_id` would
    not — order lines are never addressed individually the way cart lines are,
    because nothing deletes or updates one. An id nobody can use is an id that
    invites somebody to try.

    `unit_price_cents`, not the column's `unit_amount_cents`. Same layer
    boundary as the cart response, crossed in the same place.
    """

    variant_id: int
    sku: str
    product_name: str
    variant_label: str | None
    quantity: int
    unit_price_cents: int
    line_total_cents: int


class OrderResponse(BaseModel):
    """An order, rendered from its own rows."""

    order_id: uuid.UUID
    cart_id: uuid.UUID
    status: OrderStatus
    currency: str
    items: list[OrderItemResponse]
    # The total stored at order time, never recomputed from the catalog.
    total_cents: int
    created_at: datetime
