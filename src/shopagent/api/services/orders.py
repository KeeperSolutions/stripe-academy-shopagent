"""Turning a cart into an order (D6, step 4).

No FastAPI, for the reason `api/lifecycle.py` gives: D8's Stripe webhook calls
into this module outside any HTTP request.

`place_order` is one transaction and one unit of work. Every check it makes is
authoritative — unlike the advisory stock read in `services/cart.py`, which has
no lock and promises nothing. The difference is the whole point of this step:
a cart is a statement of intent and may be optimistic, an order is a promise
and may not.

The sequence matters and is not rearrangeable:

  1. lock the cart row              `SELECT ... FOR UPDATE` on `carts`
  2. refuse unless it is `open`
  3. refuse an empty cart
  4. lock the inventory rows        one statement, `ORDER BY variant_id`
  5. check availability             authoritative, under those locks
  6. refuse a line with no price
  7. reserve                        `inventory.reserved += quantity`
  8. snapshot the lines             everything a receipt needs, copied
  9. total from the snapshot        not from a second read of the catalog
 10. lock the cart                  `carts.status = 'ordered'`

Step 1 is a lock on `carts`, not on `cart_items`. Locking the items would leave
the cart row itself readable, and two concurrent requests would both see `open`
and both proceed — which is exactly the race the UNIQUE on `orders.cart_id`
exists to stop from becoming two orders. The lock is what turns that constraint
from the thing that catches the bug into the thing that never has to.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from shopagent.api.lifecycle import OrderStatus
from shopagent.api.models import Cart, CartItem, CartStatus, Order, OrderItem
from shopagent.api.services.cart import CartNotFound, variant_label
from shopagent.catalog.models import Inventory, Price, Product, Variant
from shopagent.config import get_settings


class OrderError(Exception):
    """Base for everything this module raises."""


class OrderNotFound(OrderError):
    """No order with that id."""


class CartAlreadyOrdered(OrderError):
    """The cart has already been turned into an order."""


class CartEmpty(OrderError):
    """Nothing to buy.

    An order with no lines is a total of zero, which D7 would hand to Stripe as
    a charge for nothing.
    """


class LineNotPriceable(OrderError):
    """A cart line has no active price in the shop's currency.

    `services/cart.py` deliberately renders such a line with an empty price
    rather than dropping it, because a cart that quietly loses an item is
    worse than one that shows an item it cannot price. Here the same state has
    to be a hard error: a line missing from an order is goods that ship and are
    never charged for.
    """


class StockUnavailable(OrderError):
    """One or more lines ask for more units than exist.

    Carries every shortfall rather than the first, because a caller — the model
    on D9 especially — can fix one problem per turn only if it is told about
    all of them.
    """

    def __init__(self, shortfalls: list[tuple[str, int, int]]) -> None:
        self.shortfalls = shortfalls
        detail = "; ".join(
            f"{sku} needs {requested} but {available} are available"
            for sku, requested, available in shortfalls
        )
        super().__init__(
            f"the cart cannot be ordered as it stands: {detail}. "
            "Reduce those quantities or remove those lines."
        )


@dataclass(frozen=True)
class OrderLine:
    """One snapshotted order line, rendered.

    `unit_price_cents` rather than the column's `unit_amount_cents`: this is
    the flattened view a reader gets, and CLAUDE.md puts the rename exactly at
    that boundary.
    """

    variant_id: int
    sku: str
    product_name: str
    variant_label: str | None
    quantity: int
    unit_price_cents: int

    @property
    def line_total_cents(self) -> int:
        return self.unit_price_cents * self.quantity


@dataclass(frozen=True)
class RenderedOrder:
    """An order and everything needed to show it, with no catalog behind it."""

    order_id: uuid.UUID
    cart_id: uuid.UUID
    status: OrderStatus
    currency: str
    lines: list[OrderLine]
    total_cents: int
    created_at: datetime


def _lock_cart(session: Session, cart_id: uuid.UUID) -> Cart:
    """Take a row lock on the cart before reading its status.

    `with_for_update()` rather than `session.get()`: the status is about to be
    used to decide whether to write, and a read without a lock is a decision
    made on a value another transaction is free to change before the write
    lands. Both requests would see `open`.
    """
    cart = session.scalar(
        select(Cart).where(Cart.id == cart_id).with_for_update()
    )
    if cart is None:
        raise CartNotFound(f"no cart with id {cart_id}")
    return cart


def _lock_inventory(session: Session, variant_ids: list[int]) -> dict[int, Inventory]:
    """Lock every inventory row this order touches, in one statement.

    `ORDER BY variant_id` is not cosmetic and must not be dropped. Two orders
    covering the same two variants in opposite order would each hold the row
    the other is waiting for, and Postgres would resolve it by killing one with
    a deadlock error. Locking in a globally consistent order means the second
    transaction simply waits for the first — which is the behaviour wanted, and
    it costs one clause.

    One statement rather than a loop for the same reason: a loop acquires locks
    in the order Python iterates, which is a different order per request unless
    the caller sorted first.
    """
    rows = session.scalars(
        select(Inventory)
        .where(Inventory.variant_id.in_(variant_ids))
        .order_by(Inventory.variant_id.asc())
        .with_for_update()
    ).all()
    return {row.variant_id: row for row in rows}


def _active_prices(session: Session, variant_ids: list[int]) -> dict[int, int]:
    """The active price per variant in the shop's currency.

    At most one row per variant per currency, which D3's partial unique index
    enforces — so this is safe to fold into a dict without picking a winner.
    """
    currency = get_settings().currency
    rows = session.execute(
        select(Price.variant_id, Price.amount_cents).where(
            Price.variant_id.in_(variant_ids),
            Price.active.is_(True),
            Price.currency == currency,
        )
    ).all()
    return {variant_id: int(amount) for variant_id, amount in rows}


def _cart_lines(session: Session, cart_id: uuid.UUID) -> list[tuple]:
    """Cart lines joined to what they are called, ordered by sku.

    This is the last time the catalog is read for this order. Everything after
    it works from the values returned here, which is what makes the snapshot a
    snapshot rather than a cache that can be refreshed.
    """
    return session.execute(
        select(
            CartItem.variant_id,
            CartItem.quantity,
            Variant.sku,
            Variant.size,
            Variant.color,
            Product.name,
        )
        .join(Variant, Variant.id == CartItem.variant_id)
        .join(Product, Product.id == Variant.product_id)
        .where(CartItem.cart_id == cart_id)
        .order_by(Variant.sku.asc())
    ).all()


def _reserve(
    inventory: dict[int, Inventory], wanted: dict[int, int]
) -> None:
    """Hold the units this order needs.

    `reserved` goes up; `quantity` is deliberately untouched. Units leave
    `quantity` when they physically ship, and this project has no fulfilment
    flow — `fulfilled` is a status nothing transitions into automatically, so
    there is no moment at which decrementing would be correct. Available stock
    is `quantity - reserved` everywhere, so a reservation already makes the
    units unsellable. This is not an oversight.
    """
    for variant_id, quantity in wanted.items():
        inventory[variant_id].reserved += quantity


def _snapshot_lines(
    order_id: uuid.UUID, rows: list[tuple], prices: dict[int, int]
) -> list[OrderItem]:
    """Copy everything a receipt needs onto the order itself.

    The point of every column here is that rendering the order later joins
    nothing. Prices change, products get renamed, a variant can be deleted from
    the catalog outright — none of that may change what an order says was
    bought, because an order is a record of an event rather than a view over
    current data.
    """
    currency = get_settings().currency
    return [
        OrderItem(
            order_id=order_id,
            variant_id=variant_id,
            sku=sku,
            product_name=product_name,
            variant_label=variant_label(size, color),
            unit_amount_cents=prices[variant_id],
            currency=currency,
            quantity=quantity,
        )
        for variant_id, quantity, sku, size, color, product_name in rows
    ]


def place_order(session: Session, cart_id: uuid.UUID) -> Order:
    """Convert a cart into a pending order, or change nothing at all.

    One transaction. Every failure path rolls back explicitly rather than
    leaving the decision to whoever catches the exception: the router turns a
    domain error into a 409 and returns normally, so without the rollback here
    the reservations written before the failure would still be pending in the
    session and would be flushed by the next request to use it.
    """
    try:
        cart = _lock_cart(session, cart_id)

        if cart.status is not CartStatus.OPEN:
            raise CartAlreadyOrdered(
                f"cart {cart_id} is {cart.status.value} and cannot be ordered "
                "again. Start a new cart to buy something else."
            )

        rows = _cart_lines(session, cart_id)
        if not rows:
            raise CartEmpty(f"cart {cart_id} is empty and cannot be ordered")

        variant_ids = sorted({row[0] for row in rows})
        inventory = _lock_inventory(session, variant_ids)
        wanted = {variant_id: quantity for variant_id, quantity, *_ in rows}

        shortfalls: list[tuple[str, int, int]] = []
        for variant_id, quantity, sku, *_ in rows:
            stock = inventory.get(variant_id)
            available = 0 if stock is None else stock.quantity - stock.reserved
            if quantity > available:
                shortfalls.append((sku, quantity, available))
        if shortfalls:
            raise StockUnavailable(shortfalls)

        prices = _active_prices(session, variant_ids)
        unpriced = [sku for variant_id, _, sku, *_ in rows if variant_id not in prices]
        if unpriced:
            raise LineNotPriceable(
                f"no active price in {get_settings().currency} for "
                f"{', '.join(sorted(unpriced))}. These lines cannot be ordered; "
                "remove them from the cart or ask for the price to be restored."
            )

        _reserve(inventory, wanted)

        order = Order(
            cart_id=cart_id,
            status=OrderStatus.PENDING,
            currency=get_settings().currency,
            # Placeholder: replaced below from the snapshot. The column is NOT
            # NULL, and computing it before the lines exist would mean reading
            # the catalog a second time for numbers already in hand.
            total_amount_cents=0,
        )
        session.add(order)
        session.flush()

        lines = _snapshot_lines(order.id, rows, prices)
        session.add_all(lines)

        order.total_amount_cents = sum(
            line.unit_amount_cents * line.quantity for line in lines
        )

        cart.status = CartStatus.ORDERED

        session.commit()
        return order
    except Exception:
        session.rollback()
        raise


def render_order(session: Session, order_id: uuid.UUID) -> RenderedOrder:
    """Read an order back from its own rows and nothing else.

    Touches `orders` and `order_items` only. No join to `products`, `variants`
    or `prices` — that is the property the snapshot columns exist to provide,
    and `tests/test_api_orders.py` asserts it by recording the SQL this
    function actually issues.
    """
    order = session.get(Order, order_id)
    if order is None:
        raise OrderNotFound(f"no order with id {order_id}")

    rows = session.scalars(
        select(OrderItem)
        .where(OrderItem.order_id == order_id)
        .order_by(OrderItem.sku.asc())
    ).all()

    return RenderedOrder(
        order_id=order.id,
        cart_id=order.cart_id,
        status=OrderStatus(order.status),
        currency=order.currency,
        lines=[
            OrderLine(
                variant_id=row.variant_id,
                sku=row.sku,
                product_name=row.product_name,
                variant_label=row.variant_label,
                quantity=row.quantity,
                unit_price_cents=row.unit_amount_cents,
            )
            for row in rows
        ],
        # The stored total, not a recomputation. It was written from the
        # snapshot at order time and must not drift from it.
        total_cents=order.total_amount_cents,
        created_at=order.created_at,
    )
