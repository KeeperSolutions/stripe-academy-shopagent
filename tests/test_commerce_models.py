"""Tests for shopagent.api.models (D6).

A schema is a claim about what Postgres will accept, and only Postgres can
settle it — the same argument `tests/test_models.py` makes for the catalog.
Three of the claims here are load-bearing enough to be worth the database:

  * `create_all` builds the commerce tables alongside the catalog ones, which
    is only true while both model modules are imported. Miss that and the
    schema looks complete right up to the first missing relation.
  * `(cart_id, variant_id)` is unique, which is what makes "add this variant
    again" an increment rather than a second line.
  * `order_items.variant_id` is ON DELETE RESTRICT, which is what stops
    `scripts/seed_catalog.py --reset` from taking order history down with the
    catalog. Asserting it against SQLAlchemy's metadata would only assert that
    the source file says what it says.

The `engine` and `session` fixtures live in `tests/conftest.py`: every test
runs inside a transaction that is rolled back, so nothing written here
survives.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import delete, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from shopagent.api.lifecycle import OrderStatus
from shopagent.api.models import (
    Cart,
    CartItem,
    CartStatus,
    Order,
    OrderItem,
    OrdersExist,
    assert_no_orders,
    count_orders,
)
from shopagent.catalog.models import Inventory, Price, Product, Variant

pytestmark = pytest.mark.db

COMMERCE_TABLES = {"carts", "cart_items", "orders", "order_items"}


def make_variant(session: Session, *, sku: str = "SKU-D6-TEST-1") -> Variant:
    """A product with one variant, priced and in stock."""
    product = Product(
        name=f"Commerce Fixture {sku}",
        description="A product that exists only for a D6 test.",
        category="shoes",
        brand="Fleetfoot",
        variants=[
            Variant(
                size="42",
                color="blue",
                sku=sku,
                prices=[Price(currency="usd", amount_cents=8999)],
                inventory=Inventory(quantity=10, reserved=0),
            )
        ],
    )
    session.add(product)
    session.commit()
    return product.variants[0]


def make_order(session: Session, variant: Variant, *, quantity: int = 1) -> Order:
    """A cart turned into a pending order with one snapshotted line."""
    cart = Cart(status=CartStatus.ORDERED)
    session.add(cart)
    session.flush()

    order = Order(
        cart_id=cart.id,
        status=OrderStatus.PENDING,
        total_amount_cents=8999 * quantity,
        currency="usd",
        items=[
            OrderItem(
                variant_id=variant.id,
                sku=variant.sku,
                product_name=variant.product.name,
                variant_label=f"{variant.size} / {variant.color}",
                unit_amount_cents=8999,
                currency="usd",
                quantity=quantity,
            )
        ],
    )
    session.add(order)
    session.commit()
    return order


# --- the schema exists ---------------------------------------------------


def test_create_all_builds_the_four_commerce_tables(engine):
    present = set(inspect(engine).get_table_names())
    assert COMMERCE_TABLES <= present


def test_the_commerce_tables_share_the_catalog_metadata():
    # The reason the fixture above works at all: one `Base`, one `create_all`.
    # A second declarative base would build whichever half the caller imported.
    from shopagent.catalog.models import Base

    assert COMMERCE_TABLES <= set(Base.metadata.tables)
    assert {"products", "variants", "prices", "inventory"} <= set(Base.metadata.tables)


def test_ids_are_uuids_that_exist_before_the_row_is_written(session):
    cart = Cart()
    # Assigned client-side by the column default, not by the database, which
    # is what lets a response carry the id of something just created.
    session.add(cart)
    session.flush()
    assert isinstance(cart.id, uuid.UUID)


def test_a_cart_starts_open_and_an_order_starts_pending(session):
    variant = make_variant(session, sku="SKU-D6-DEFAULTS")
    cart = Cart()
    session.add(cart)
    session.commit()
    assert cart.status is CartStatus.OPEN

    order = make_order(session, variant)
    assert order.status is OrderStatus.PENDING


def test_statuses_are_stored_as_their_values_not_their_member_names(session):
    variant = make_variant(session, sku="SKU-D6-STORED")
    order = make_order(session, variant)

    stored = session.execute(
        text("SELECT status FROM orders WHERE id = :id"), {"id": order.id}
    ).scalar_one()
    # "pending", never "PENDING". The default for a SQLAlchemy Enum is the
    # member name, and the mismatch surfaces only as a row that fails to load.
    assert stored == "pending"


# --- one line per variant ------------------------------------------------


def test_the_same_variant_cannot_be_added_to_one_cart_twice(session):
    """The constraint behind the upsert. Adding again increments a quantity."""
    variant = make_variant(session, sku="SKU-D6-DUPE")
    cart = Cart()
    cart.items.append(CartItem(variant_id=variant.id, quantity=1))
    session.add(cart)
    session.commit()

    cart.items.append(CartItem(variant_id=variant.id, quantity=1))
    with pytest.raises(IntegrityError) as excinfo:
        session.commit()

    assert "uq_cart_items_cart_variant" in str(excinfo.value)


def test_the_same_variant_in_two_different_carts_is_fine(session):
    variant = make_variant(session, sku="SKU-D6-TWO-CARTS")
    for _ in range(2):
        cart = Cart()
        cart.items.append(CartItem(variant_id=variant.id, quantity=1))
        session.add(cart)
    session.commit()

    count = session.scalar(
        select(CartItem).where(CartItem.variant_id == variant.id).exists().select()
    )
    assert count is True


def test_a_cart_line_of_zero_is_rejected(session):
    variant = make_variant(session, sku="SKU-D6-ZERO")
    cart = Cart()
    cart.items.append(CartItem(variant_id=variant.id, quantity=0))
    session.add(cart)

    with pytest.raises(IntegrityError):
        session.commit()


# --- what a catalog reset may and may not take with it -------------------


def test_deleting_a_product_is_refused_while_an_order_line_points_at_it(session):
    """RESTRICT, and the reason `--reset` needs a guard in front of it.

    `scripts/seed_catalog.py --reset` issues exactly this DELETE. The cascade
    would reach `variants` and then `order_items`; Postgres refuses instead,
    which is the enforcement that holds against any client, psql included.
    """
    variant = make_variant(session, sku="SKU-D6-RESTRICT")
    make_order(session, variant)

    with pytest.raises(IntegrityError) as excinfo:
        session.execute(delete(Product).where(Product.id == variant.product_id))
        session.commit()

    message = str(excinfo.value)
    assert "order_items" in message


def test_deleting_a_product_does_take_its_cart_lines(session):
    """The other half: carts are as disposable as the catalog they point into.

    A product that no longer exists cannot be bought, so leaving it in a basket
    would only produce a line that fails at checkout.
    """
    variant = make_variant(session, sku="SKU-D6-CASCADE")
    cart = Cart()
    cart.items.append(CartItem(variant_id=variant.id, quantity=2))
    session.add(cart)
    session.commit()
    cart_id = cart.id

    session.execute(delete(Product).where(Product.id == variant.product_id))
    session.commit()

    remaining = session.scalars(
        select(CartItem).where(CartItem.cart_id == cart_id)
    ).all()
    assert remaining == []
    # The cart itself survives; only what it pointed at is gone.
    assert session.get(Cart, cart_id) is not None


# --- the guard in front of the script ------------------------------------


def test_the_guard_allows_a_reset_while_no_order_exists(session):
    assert count_orders(session) == 0
    assert_no_orders(session, operation="reset the catalog")


def test_the_guard_refuses_a_reset_once_an_order_exists(session):
    variant = make_variant(session, sku="SKU-D6-GUARD")
    make_order(session, variant)

    with pytest.raises(OrdersExist) as excinfo:
        assert_no_orders(session, operation="reset the catalog")

    message = str(excinfo.value)
    # Written for whoever typed the command: what was refused, and why.
    assert "reset the catalog" in message
    assert "1 order" in message


def test_the_guard_counts_orders_rather_than_order_lines(session):
    variant = make_variant(session, sku="SKU-D6-COUNT")
    make_order(session, variant, quantity=3)

    assert count_orders(session) == 1


# --- the order is a snapshot ---------------------------------------------


def test_an_order_line_renders_without_reading_the_catalog(session):
    """Every field a receipt needs is on the row itself.

    Prices change and products are renamed. An order is a record of what
    happened, and it has to still read correctly when the catalog no longer
    agrees with it.
    """
    variant = make_variant(session, sku="SKU-D6-SNAPSHOT")
    order = make_order(session, variant, quantity=2)

    # Move the catalog underneath it: a new active price, a renamed product.
    session.execute(delete(Price).where(Price.variant_id == variant.id))
    session.add(Price(variant_id=variant.id, currency="usd", amount_cents=4999))
    variant.product.name = "Renamed After The Order"
    session.commit()

    line = order.items[0]
    assert line.unit_amount_cents == 8999
    assert line.product_name == "Commerce Fixture SKU-D6-SNAPSHOT"
    assert line.sku == "SKU-D6-SNAPSHOT"
    assert line.variant_label == "42 / blue"
    assert order.total_amount_cents == 8999 * 2


def test_deleting_an_order_takes_its_lines(session):
    variant = make_variant(session, sku="SKU-D6-ORDER-CASCADE")
    order = make_order(session, variant)
    order_id = order.id

    session.delete(order)
    session.commit()

    remaining = session.scalars(
        select(OrderItem).where(OrderItem.order_id == order_id)
    ).all()
    assert remaining == []


def test_stripe_columns_exist_and_start_empty(session):
    """Present before D7 needs them, because adding a column to a table that
    already holds orders is the migration this project has no tool for."""
    variant = make_variant(session, sku="SKU-D6-STRIPE")
    order = make_order(session, variant)

    assert order.stripe_checkout_session_id is None
    assert order.stripe_payment_intent_id is None
