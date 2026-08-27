"""Cart operations — create, add, read, remove (D6).

No FastAPI. Every failure leaves here as a domain exception and the router
turns it into a status code; see `api/routers/cart.py`. That is what lets D9's
`add_to_cart` tool call `add_item()` directly instead of making an HTTP request
to its own process.

Two things this module is careful about.

**The total is computed here, from the database, every time.** Nothing a client
sends contributes to it, and it is not stored on the cart — a cached total is a
number that is right until a price changes. The price used is the active one in
`settings.currency`, which D3's partial unique index already guarantees is at
most one row per variant per currency.

**The stock check is advisory and says so.** See `_check_stock_advisory`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from shopagent.api.models import Cart, CartItem, CartStatus
from shopagent.catalog.models import Inventory, Price, Product, Variant
from shopagent.config import get_settings


class CartError(Exception):
    """Base for everything this module raises, so a router can catch one type."""


class CartNotFound(CartError):
    """No cart with that id."""


class CartItemNotFound(CartError):
    """No such line in *this* cart.

    Also raised when the line exists in a different cart. The router answers
    404 either way, on purpose: a 403 would confirm that the id is real, which
    is exactly the fact an id that is not yours should not tell you.
    """


class VariantNotFound(CartError):
    """No variant with that id."""


class VariantNotSellable(CartError):
    """The variant exists but carries no active price in the shop's currency.

    Distinct from `VariantNotFound` because the row is real and the fix is
    different: a missing price is a catalog problem, not a bad id.
    """


class CartLocked(CartError):
    """The cart has become an order and can no longer be changed."""


class InsufficientStock(CartError):
    """Fewer units are available than the cart line would need."""

    def __init__(self, *, sku: str, requested: int, available: int) -> None:
        self.sku = sku
        self.requested = requested
        self.available = available
        super().__init__(
            f"{sku} has {available} available and the cart line would need "
            f"{requested}. Reduce the quantity or choose another variant."
        )


@dataclass(frozen=True)
class CartLine:
    """One rendered cart line: the row, joined to what it costs and is called.

    `unit_price_cents` is `price_cents` in the sense CLAUDE.md means — a
    resolved number for a reader, not the `amount_cents` column it came from.
    The rename happens here because here is where the layer changes.

    It is `None` when the variant has no active price in the shop's currency.
    That state is reachable: `add_item` refuses to create such a line, but a
    price can be deactivated afterwards. Reporting the line with an empty price
    is the honest answer — dropping it would make a cart quietly lose an item,
    and inventing a price would be worse.
    """

    item_id: uuid.UUID
    variant_id: int
    sku: str
    product_name: str
    variant_label: str | None
    quantity: int
    unit_price_cents: int | None

    @property
    def line_total_cents(self) -> int | None:
        if self.unit_price_cents is None:
            return None
        return self.unit_price_cents * self.quantity


@dataclass(frozen=True)
class RenderedCart:
    """A cart and everything a caller needs to show it."""

    cart_id: uuid.UUID
    status: CartStatus
    currency: str
    lines: list[CartLine]

    @property
    def total_cents(self) -> int:
        """The sum of the lines that have a price.

        An unpriced line contributes nothing rather than raising, so a cart
        holding one can still be read. `POST /orders` on D8 is where that has
        to become an error, because charging a total that silently omits a line
        is the failure this shape is designed to make visible.
        """
        return sum(line.line_total_cents or 0 for line in self.lines)


def variant_label(size: str | None, color: str | None) -> str | None:
    """"42 / blue", or whatever the variant actually has.

    Takes the two columns rather than a `Variant`, because `render_cart` reads
    them straight off a joined row and never builds the object. Shared with the
    order snapshot in step 4, which is why it is a function at all rather than
    an f-string in two places that drift.
    """
    parts = [part for part in (size, color) if part]
    return " / ".join(parts) if parts else None


def _active_price_cents(session: Session, variant_id: int) -> int | None:
    """The active price for this variant in the shop's currency, if there is one."""
    return session.scalar(
        select(Price.amount_cents).where(
            Price.variant_id == variant_id,
            Price.active.is_(True),
            Price.currency == get_settings().currency,
        )
    )


def _check_stock_advisory(session: Session, variant: Variant, wanted: int) -> None:
    """Refuse an obviously impossible quantity. Guarantee nothing.

    This is a read of `quantity - reserved` with no lock and no write. Between
    this check and any later one, another request can add the same variant to
    its own cart, and both will have been told there was room. That is
    acceptable here and only here: a cart is a statement of intent, and the
    cost of being wrong is a message at checkout rather than an oversold unit.

    The authoritative check is D8's, inside `POST /orders`, under `SELECT ...
    FOR UPDATE` on the inventory row and in the same transaction that writes
    `reserved`. Do not move stock reservation into this module, and do not read
    this function as a promise that the units are there — it is a courtesy that
    catches the common mistake early, in the place where a shopper can still
    do something about it.
    """
    stock = session.get(Inventory, variant.id)
    available = 0 if stock is None else stock.quantity - stock.reserved

    if wanted > available:
        raise InsufficientStock(
            sku=variant.sku, requested=wanted, available=available
        )


def _load_cart(session: Session, cart_id: uuid.UUID, *, lock: bool = False) -> Cart:
    """Fetch a cart, optionally taking a row lock on it first.

    Every path that goes on to *write* passes `lock=True`, and the reason is a
    race this module cannot otherwise win. `place_order` locks the same row,
    snapshots the items it finds and flips the status to `ordered`. An unlocked
    add reads `open`, is descheduled, and commits its line after that snapshot
    has already been taken — leaving an ordered cart holding an item that is on
    no order, which is goods a shopper believes they bought and nobody was
    charged for. The status check has to happen *under* the lock that
    `place_order` respects, or it is a decision made on a value another
    transaction is free to change before the write lands.

    Locking here also serialises two concurrent adds of the same variant.
    Without it both read "no existing line", both insert, and the UNIQUE on
    `(cart_id, variant_id)` turns the loser into an `IntegrityError` — a 500
    where the shopper should simply have got a merged line.

    `render_cart` passes `lock=False` on purpose. A read has no business taking
    a write lock: it would make every `GET /cart` queue behind an in-flight
    checkout for no benefit, since nothing it returns is used to decide a write.
    """
    statement = select(Cart).where(Cart.id == cart_id)
    if lock:
        statement = statement.with_for_update()

    cart = session.scalar(statement)
    if cart is None:
        raise CartNotFound(f"no cart with id {cart_id}")
    return cart


def _require_open(cart: Cart) -> None:
    if cart.status is not CartStatus.OPEN:
        raise CartLocked(
            f"cart {cart.id} is {cart.status.value} and can no longer be "
            "changed. Its contents were snapshotted onto an order; start a new "
            "cart to buy something else."
        )


# --- operations ----------------------------------------------------------


def create_cart(session: Session) -> Cart:
    """Open an empty cart."""
    cart = Cart(status=CartStatus.OPEN)
    session.add(cart)
    session.commit()
    return cart


def add_item(
    session: Session, cart_id: uuid.UUID, variant_id: int, quantity: int
) -> tuple[CartItem, bool]:
    """Add `quantity` of a variant, returning the line and whether it is new.

    An upsert, written as one: a variant already in the cart has its quantity
    incremented rather than gaining a second row. The UNIQUE on
    `(cart_id, variant_id)` would catch an append, but catching an
    `IntegrityError` and turning it into an update would make the constraint
    the mechanism instead of the net — and a net that is load-bearing is not a
    net. It stays underneath, for the concurrent case this select cannot see.

    The stock check runs against the *resulting* quantity, not the increment.
    Adding two to a line of three asks the catalog for five.
    """
    if quantity <= 0:
        # Pydantic refuses this at the edge, so reaching here means a
        # non-HTTP caller. Still refused, because a service that trusts its
        # callers is a service with one validation layer that can be skipped.
        raise ValueError("quantity must be greater than zero")

    cart = _load_cart(session, cart_id, lock=True)
    _require_open(cart)

    variant = session.get(Variant, variant_id)
    if variant is None:
        raise VariantNotFound(f"no variant with id {variant_id}")

    if _active_price_cents(session, variant_id) is None:
        raise VariantNotSellable(
            f"variant {variant.sku} has no active price in "
            f"{get_settings().currency} and cannot be added to a cart"
        )

    existing = session.scalar(
        select(CartItem).where(
            CartItem.cart_id == cart_id, CartItem.variant_id == variant_id
        )
    )

    resulting = quantity if existing is None else existing.quantity + quantity
    _check_stock_advisory(session, variant, resulting)

    if existing is None:
        line = CartItem(cart_id=cart_id, variant_id=variant_id, quantity=quantity)
        session.add(line)
        created = True
    else:
        existing.quantity = resulting
        line = existing
        created = False

    session.commit()
    return line, created


def remove_item(
    session: Session, cart_id: uuid.UUID, item_id: uuid.UUID
) -> None:
    """Drop one line from the cart.

    The line is looked up by both ids together. Fetching by `item_id` alone and
    then comparing carts would work, but this way a line belonging to someone
    else is indistinguishable from one that does not exist — which is the
    answer the router is going to give anyway.
    """
    cart = _load_cart(session, cart_id, lock=True)
    _require_open(cart)

    line = session.scalar(
        select(CartItem).where(CartItem.id == item_id, CartItem.cart_id == cart_id)
    )
    if line is None:
        raise CartItemNotFound(f"no item {item_id} in cart {cart_id}")

    session.delete(line)
    session.commit()


def render_cart(session: Session, cart_id: uuid.UUID) -> RenderedCart:
    """Read the cart back, priced, with its total computed here and now.

    One query, left-joined to `prices`: an outer join because a variant whose
    price was deactivated still has a cart line, and an inner join would drop
    it from both the list and the total — a cart quietly losing an item is a
    worse failure than one showing an item it cannot price.

    Works on a locked cart. An order has to remain readable after it is placed.
    """
    cart = _load_cart(session, cart_id)
    currency = get_settings().currency

    rows = session.execute(
        select(
            CartItem.id,
            CartItem.variant_id,
            CartItem.quantity,
            Variant.sku,
            Variant.size,
            Variant.color,
            Product.name,
            Price.amount_cents,
        )
        .join(Variant, Variant.id == CartItem.variant_id)
        .join(Product, Product.id == Variant.product_id)
        .outerjoin(
            Price,
            (Price.variant_id == Variant.id)
            & Price.active.is_(True)
            & (Price.currency == currency),
        )
        .where(CartItem.cart_id == cart_id)
        # Stable order, so two reads of an unchanged cart look identical and a
        # test can index into the list.
        .order_by(Variant.sku.asc())
    ).all()

    lines = [
        CartLine(
            item_id=item_id,
            variant_id=variant_id,
            sku=sku,
            product_name=product_name,
            variant_label=variant_label(size, color),
            quantity=quantity,
            unit_price_cents=None if amount_cents is None else int(amount_cents),
        )
        for (
            item_id,
            variant_id,
            quantity,
            sku,
            size,
            color,
            product_name,
            amount_cents,
        ) in rows
    ]

    return RenderedCart(
        cart_id=cart.id, status=cart.status, currency=currency, lines=lines
    )
