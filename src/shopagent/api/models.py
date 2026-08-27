"""Commerce schema — carts, cart items, orders, order items (D6).

Registered on the same `Base` as the catalog, imported from
`catalog.models`. A second declarative base would give `create_all` two
separate metadata objects, and the script that calls it would build whichever
half it happened to import — a schema that looks complete until the first
foreign key across the gap.

These four tables are not the catalog, and the rule the catalog lives under
does not extend to them. `products` and its children can be dropped and
reseeded because `catalog/seed.py`, not Postgres, is their source of truth. A
cart is what somebody put in it and an order is what somebody bought; nothing
regenerates those. From the first real order, changing this schema is a
migration.

Money stays `int` minor units, and the column suffix stays `amount_cents` —
the same name `prices.amount_cents` uses, because these are the same kind of
thing: a number in a database column. `price_cents` is reserved for the
flattened field the model reads, and the rename still happens exactly at that
boundary. `unit_amount_cents` also lands where D7 needs it: Stripe's line item
field is `unit_amount`, in minor units, and the column goes across untouched.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
    func,
    select,
)
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship

from shopagent.api.lifecycle import OrderStatus
from shopagent.catalog.models import Base


class CartStatus(StrEnum):
    """Whether a cart can still be changed.

    A cart locks when it becomes an order. `order_items` snapshots the prices
    and names it held at that moment, so a cart edited afterwards would leave
    the two disagreeing with no way to tell which was the purchase.
    """

    OPEN = "open"
    ORDERED = "ordered"


def _status_column(enum_type: type[StrEnum], name: str):
    """A status column stored as VARCHAR with a CHECK, never a native enum.

    `native_enum=False` on purpose. A Postgres enum type is changed with
    `ALTER TYPE`, which `create_all` never issues and which cannot be undone
    without recreating the type and every column using it. D8 adding a status
    is a realistic week, not a hypothetical, and it should cost a CHECK
    constraint rather than a migration over a table holding real orders.

    `create_constraint=True` is spelled out because SQLAlchemy 2.0 defaults it
    to False: without it the column is a bare VARCHAR and any string at all is
    accepted, which is the failure that looks like it works.

    `values_callable` stores the enum's *values* rather than its member names,
    so the column holds `"pending"` and not `"PENDING"`. The default is the
    name, and the mismatch would only surface as a row that fails to load.
    """
    return Enum(
        enum_type,
        name=name,
        native_enum=False,
        create_constraint=True,
        validate_strings=True,
        values_callable=lambda enum: [member.value for member in enum],
    )


class Cart(Base):
    """A basket, before it is bought."""

    __tablename__ = "carts"

    # UUID rather than a serial integer, and the same goes for the three
    # tables below. These identifiers leave the process: they reach the model
    # in conversation, and on D7 they travel to Stripe as `metadata.order_id`.
    # A guessable, enumerable integer in that position invites a shopper to
    # try the neighbouring number; a UUID also lets the id exist before the
    # row is flushed, which is what makes it safe to put in a response.
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    status: Mapped[CartStatus] = mapped_column(
        _status_column(CartStatus, "cart_status"), default=CartStatus.OPEN
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    items: Mapped[list[CartItem]] = relationship(
        back_populates="cart",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class CartItem(Base):
    """One variant, and how many of it, in one cart."""

    __tablename__ = "cart_items"
    __table_args__ = (
        CheckConstraint("quantity > 0", name="ck_cart_items_quantity_positive"),
        # Adding a variant a cart already holds is an increment, never a second
        # row. Two rows for one sku would show the shopper the same product
        # twice and, worse, would let one delete remove half a quantity they
        # think of as a single line.
        UniqueConstraint("cart_id", "variant_id", name="uq_cart_items_cart_variant"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    cart_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("carts.id", ondelete="CASCADE"), index=True
    )
    # CASCADE, unlike `order_items.variant_id` below. A cart is as disposable
    # as the catalog it points into: if a product is reseeded out of
    # existence, the right outcome is that it leaves the baskets it was in,
    # because it can no longer be bought. An order is the opposite case.
    variant_id: Mapped[int] = mapped_column(
        ForeignKey("variants.id", ondelete="CASCADE"), index=True
    )
    quantity: Mapped[int] = mapped_column(Integer)

    cart: Mapped[Cart] = relationship(back_populates="items")



class Order(Base):
    """A cart at the moment it was bought, plus where its payment stands."""

    __tablename__ = "orders"
    __table_args__ = (
        CheckConstraint(
            "total_amount_cents >= 0", name="ck_orders_total_amount_cents_positive"
        ),
        # One order per cart, enforced rather than remembered. `ordered` is a
        # terminal cart status with no way back to `open`, so this forbids
        # nothing the design allows — what it closes is a race the status check
        # cannot: two concurrent `POST /orders` for one cart both read `open`
        # before either writes, and both would otherwise succeed. Same argument
        # as the RESTRICT on `order_items.variant_id`: the database refuses,
        # instead of the application remembering to lock.
        UniqueConstraint("cart_id", name="uq_orders_cart"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # No ON DELETE clause: an order records which cart it came from, and a
    # cart that an order points at is not something to delete quietly.
    cart_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("carts.id"), index=True)
    status: Mapped[OrderStatus] = mapped_column(
        _status_column(OrderStatus, "order_status"), default=OrderStatus.PENDING
    )
    # Written once, at creation, from the prices in the database — never from
    # anything a client sent. After that it is history and nothing recomputes
    # it, which is the whole reason it is a column and not a SUM over the
    # items.
    total_amount_cents: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(3))
    # Both null until D7, and both here now on purpose. Adding a column to a
    # table that already holds orders is the migration this project has no
    # tool for; two nullable columns cost nothing today and remove that from
    # D7's path entirely.
    # Who is buying, to the extent this project has a notion of that (D7).
    #
    # Deliberately two nullable columns on `orders` rather than a `customers`
    # table. D6 has no concept of a user at all, and D9 introduces long-term
    # memory — a name, preferences, past orders — which is the requirement that
    # would actually shape such a table. Building it now means guessing that
    # shape a week early and then living with the guess, so an order carries
    # the little it knows and nothing claims to be a customer record.
    #
    # `stripe_customer_id` is set only when a Customer object was created;
    # `customer_email` can stand alone. Stripe refuses a Checkout Session that
    # carries both — see `payments/checkout.py`.
    customer_email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    stripe_customer_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    stripe_checkout_session_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    stripe_payment_intent_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    cart: Mapped[Cart] = relationship()
    items: Mapped[list[OrderItem]] = relationship(
        back_populates="order",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class OrderItem(Base):
    """One line of an order, frozen as it was when the order was placed.

    Every field a receipt needs is copied here rather than joined for: the sku,
    the product's name, the variant's label, the price charged. Prices change
    and products are renamed; an order is a record of what happened and must
    still render correctly years later, without reading the catalog at all.
    """

    __tablename__ = "order_items"
    __table_args__ = (
        CheckConstraint("quantity > 0", name="ck_order_items_quantity_positive"),
        CheckConstraint(
            "unit_amount_cents >= 0", name="ck_order_items_unit_amount_cents_positive"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    order_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("orders.id", ondelete="CASCADE"), index=True
    )
    # RESTRICT, and this is the load-bearing one. `scripts/seed_catalog.py
    # --reset` issues `DELETE FROM products`, which cascades to variants and
    # would carry the order lines with it. Postgres refuses instead. The guard
    # in `assert_no_orders` below exists so a developer meets a sentence
    # rather than an IntegrityError, but the guard is a courtesy and this is
    # the enforcement: it holds against any client, including psql.
    variant_id: Mapped[int] = mapped_column(
        ForeignKey("variants.id", ondelete="RESTRICT"), index=True
    )

    # --- snapshot: none of these is read back from the catalog -------------
    sku: Mapped[str] = mapped_column(String(64))
    product_name: Mapped[str] = mapped_column(String(200))
    # "42 / blue", or whatever the variant was; nullable because a variant with
    # neither size nor colour has nothing to label it with.
    variant_label: Mapped[str | None] = mapped_column(String(80), nullable=True)
    unit_amount_cents: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(3))
    quantity: Mapped[int] = mapped_column(Integer)

    order: Mapped[Order] = relationship(back_populates="items")


class OrdersExist(Exception):
    """Raised when an operation that would destroy order history is attempted."""


def count_orders(session: Session) -> int:
    """How many orders the database holds."""
    return session.scalar(select(func.count()).select_from(Order)) or 0


def assert_no_orders(session: Session, *, operation: str) -> None:
    """Refuse `operation` if any order exists.

    The catalog is disposable and the seeder rebuilds it; order history is
    neither. `order_items.variant_id` is ON DELETE RESTRICT, so the database
    would refuse the delete anyway — but it refuses with an IntegrityError
    naming a constraint, and this refuses with a sentence saying what is in
    the way and what to do instead.

    Lives here, next to the table it counts, rather than in the script that
    calls it: `scripts/` is not importable from the tests, and a guard nobody
    can test is a guard nobody can trust.
    """
    existing = count_orders(session)
    if existing:
        raise OrdersExist(
            f"refusing to {operation}: the database holds {existing} order(s), "
            "and the catalog cascade would take their line items with it. "
            "Orders are not seed data — drop the database volume if you really "
            "mean to start over."
        )
