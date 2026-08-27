"""Tests for the order endpoints (D6, step 4).

Three claims are worth more than the status codes here, and each is checked by
observation rather than by trusting the code:

  * **the snapshot is a snapshot** — the catalog is edited underneath a placed
    order and the order is unmoved;
  * **rendering an order reads no catalog table** — proved by recording the SQL
    the request actually issues, not by checking that the fields came back
    populated. A join could populate them perfectly and still be the bug;
  * **the whole thing is one transaction** — a failure injected between
    reserving stock and writing the order must leave no order, no reservation
    and an open cart.

Every test runs inside the transaction `authed_client` rolls back, so the
inventory numbers these tests move are restored afterwards.
"""

from __future__ import annotations

import contextlib
import uuid

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.engine import Engine

from shopagent.api.models import Cart, CartStatus, Order, OrderItem
from shopagent.api.services import orders as order_service
from shopagent.catalog.models import Inventory, Price, Product, Variant

pytestmark = pytest.mark.db

MISSING_UUID = uuid.UUID("00000000-0000-0000-0000-000000000000")
CATALOG_TABLES = ("products", "variants", "prices", "inventory")


def make_variant(
    session,
    *,
    sku: str,
    amount_cents: int = 1000,
    quantity: int = 10,
    reserved: int = 0,
    name: str | None = None,
) -> Variant:
    product = Product(
        name=name or f"Order Fixture {sku}",
        description="A product that exists only for an order test.",
        category="shoes",
        brand="Fleetfoot",
        variants=[
            Variant(
                size="42",
                color="blue",
                sku=sku,
                prices=[Price(currency="usd", amount_cents=amount_cents, active=True)],
                inventory=Inventory(quantity=quantity, reserved=reserved),
            )
        ],
    )
    session.add(product)
    session.commit()
    return product.variants[0]


def new_cart(client) -> str:
    return client.post("/cart").json()["cart_id"]


def add(client, cart_id: str, variant_id: int, quantity: int = 1):
    return client.post(
        f"/cart/{cart_id}/items",
        json={"variant_id": variant_id, "quantity": quantity},
    )


def order(client, cart_id: str):
    return client.post("/orders", json={"cart_id": cart_id})


def stock(session, variant_id: int) -> tuple[int, int]:
    """(quantity, reserved) read fresh, never from a held object."""
    row = session.execute(
        select(Inventory.quantity, Inventory.reserved).where(
            Inventory.variant_id == variant_id
        )
    ).one()
    return int(row[0]), int(row[1])


@contextlib.contextmanager
def recorded_sql():
    """Every SQL statement executed inside the block.

    Registered on the `Engine` class so it catches statements issued through
    any connection, including the one the request handler is using.
    """
    statements: list[str] = []

    def listener(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(Engine, "before_cursor_execute", listener)
    try:
        yield statements
    finally:
        event.remove(Engine, "before_cursor_execute", listener)


# --- the happy path ------------------------------------------------------


def test_placing_an_order_is_201_and_mirrors_the_cart(authed_client, session):
    first = make_variant(session, sku="ORD-A", amount_cents=2500)
    second = make_variant(session, sku="ORD-B", amount_cents=999)
    cart_id = new_cart(authed_client)
    add(authed_client, cart_id, first.id, 2)
    add(authed_client, cart_id, second.id, 3)

    cart_body = authed_client.get(f"/cart/{cart_id}").json()
    response = order(authed_client, cart_id)

    assert response.status_code == 201
    body = response.json()
    assert body["cart_id"] == cart_id
    assert body["status"] == "pending"
    assert body["currency"] == "usd"
    assert body["created_at"]
    assert uuid.UUID(body["order_id"])

    # The same lines and the same money as the cart it came from.
    assert body["total_cents"] == cart_body["total_cents"] == (2500 * 2) + (999 * 3)
    assert [line["sku"] for line in body["items"]] == ["ORD-A", "ORD-B"]
    for order_line, cart_line in zip(body["items"], cart_body["items"], strict=True):
        assert order_line["sku"] == cart_line["sku"]
        assert order_line["quantity"] == cart_line["quantity"]
        assert order_line["unit_price_cents"] == cart_line["unit_price_cents"]
        assert order_line["line_total_cents"] == cart_line["line_total_cents"]
        assert order_line["product_name"] == cart_line["product_name"]
        assert order_line["variant_label"] == cart_line["variant_label"]


def test_reading_an_order_is_200(authed_client, session):
    variant = make_variant(session, sku="ORD-READ", amount_cents=4200)
    cart_id = new_cart(authed_client)
    add(authed_client, cart_id, variant.id, 2)
    order_id = order(authed_client, cart_id).json()["order_id"]

    response = authed_client.get(f"/orders/{order_id}")

    assert response.status_code == 200
    assert response.json()["total_cents"] == 8400


def test_reading_an_order_that_does_not_exist_is_404(authed_client):
    assert authed_client.get(f"/orders/{MISSING_UUID}").status_code == 404


def test_ordering_a_cart_that_does_not_exist_is_404(authed_client):
    assert order(authed_client, str(MISSING_UUID)).status_code == 404


# --- refusals ------------------------------------------------------------


def test_ordering_an_empty_cart_is_409(authed_client):
    cart_id = new_cart(authed_client)

    response = order(authed_client, cart_id)

    assert response.status_code == 409
    assert "empty" in response.json()["detail"]


def test_ordering_the_same_cart_twice_is_409_and_creates_one_order(
    authed_client, session
):
    """The status code is the cheap half. The count is the real assertion."""
    variant = make_variant(session, sku="ORD-TWICE")
    cart_id = new_cart(authed_client)
    add(authed_client, cart_id, variant.id, 1)

    first = order(authed_client, cart_id)
    second = order(authed_client, cart_id)

    assert first.status_code == 201
    assert second.status_code == 409

    how_many = session.scalar(
        select(func.count())
        .select_from(Order)
        .where(Order.cart_id == uuid.UUID(cart_id))
    )
    assert how_many == 1


def test_a_line_with_no_active_price_is_409(authed_client, session):
    """The cart tolerates this; the order must not.

    `services/cart.py` renders such a line with an empty price rather than
    dropping it. Here it is a hard error, because a line missing from an order
    is goods that ship and are never charged for.
    """
    priced = make_variant(session, sku="ORD-PRICED", amount_cents=500)
    losing = make_variant(session, sku="ORD-UNPRICED", amount_cents=700)
    cart_id = new_cart(authed_client)
    add(authed_client, cart_id, priced.id, 1)
    add(authed_client, cart_id, losing.id, 1)

    session.execute(
        Price.__table__.update()
        .where(Price.variant_id == losing.id)
        .values(active=False)
    )
    session.commit()

    # The cart still renders it, with no price and outside the total.
    cart_body = authed_client.get(f"/cart/{cart_id}").json()
    assert any(line["unit_price_cents"] is None for line in cart_body["items"])

    response = order(authed_client, cart_id)

    assert response.status_code == 409
    assert "ORD-UNPRICED" in response.json()["detail"]


def test_ordering_without_the_key_is_401(api_client):
    assert api_client.post("/orders", json={"cart_id": str(MISSING_UUID)}).status_code == 401
    assert api_client.get(f"/orders/{MISSING_UUID}").status_code == 401


# --- the cart is locked afterwards --------------------------------------


def test_the_cart_is_ordered_and_frozen_after_a_real_order(authed_client, session):
    """The first time the lock is exercised through `POST /orders` itself.

    Step 3 set the status by hand for want of an endpoint. This is the same
    claim with the real thing behind it.
    """
    variant = make_variant(session, sku="ORD-LOCK", quantity=20)
    cart_id = new_cart(authed_client)
    item_id = add(authed_client, cart_id, variant.id, 1).json()["items"][0]["item_id"]

    assert order(authed_client, cart_id).status_code == 201

    cart = session.get(Cart, uuid.UUID(cart_id))
    session.refresh(cart)
    assert cart.status is CartStatus.ORDERED

    assert add(authed_client, cart_id, variant.id, 1).status_code == 409
    assert authed_client.delete(f"/cart/{cart_id}/items/{item_id}").status_code == 409
    # Reading still works: an order has to stay readable after it is placed.
    assert authed_client.get(f"/cart/{cart_id}").status_code == 200


# --- the snapshot --------------------------------------------------------


def test_the_order_keeps_the_name_and_price_it_was_placed_at(authed_client, session):
    """Move the catalog underneath a placed order; the order does not move."""
    variant = make_variant(
        session, sku="ORD-SNAP", amount_cents=8999, name="Original Name"
    )
    cart_id = new_cart(authed_client)
    add(authed_client, cart_id, variant.id, 2)
    order_id = order(authed_client, cart_id).json()["order_id"]

    session.execute(
        Product.__table__.update()
        .where(Product.id == variant.product_id)
        .values(name="Renamed After The Order")
    )
    session.execute(
        Price.__table__.update()
        .where(Price.variant_id == variant.id)
        .values(active=False)
    )
    session.add(
        Price(variant_id=variant.id, currency="usd", amount_cents=1, active=True)
    )
    session.commit()

    body = authed_client.get(f"/orders/{order_id}").json()

    line = body["items"][0]
    assert line["product_name"] == "Original Name"
    assert line["unit_price_cents"] == 8999
    assert line["line_total_cents"] == 8999 * 2
    assert body["total_cents"] == 8999 * 2


def test_rendering_an_order_reads_no_catalog_table(authed_client, session):
    """Measured, not inferred.

    Checking that the response fields are populated would pass just as happily
    against an implementation that joins the catalog on every read — and that
    implementation is broken the moment a variant is deleted. So the SQL is
    recorded and inspected: `orders` and `order_items` may appear, the four
    catalog tables may not.
    """
    variant = make_variant(session, sku="ORD-NOJOIN", amount_cents=1500)
    cart_id = new_cart(authed_client)
    add(authed_client, cart_id, variant.id, 2)
    order_id = order(authed_client, cart_id).json()["order_id"]

    with recorded_sql() as statements:
        response = authed_client.get(f"/orders/{order_id}")

    assert response.status_code == 200
    assert response.json()["items"][0]["product_name"] == "Order Fixture ORD-NOJOIN"

    executed = " ".join(statements).lower()
    assert "order_items" in executed, "the recorder captured nothing to inspect"
    for table in CATALOG_TABLES:
        assert table not in executed, (
            f"rendering an order touched {table!r}: the snapshot is not being "
            "used, and this order breaks the day that row is deleted"
        )


def test_an_order_survives_its_variant_losing_its_product_name(
    authed_client, session
):
    """A stronger version of the same claim: the catalog row is gone entirely.

    `order_items.variant_id` is ON DELETE RESTRICT so the variant itself
    cannot go, but the product can be renamed to anything and the price row
    removed outright. The order still renders in full.
    """
    variant = make_variant(session, sku="ORD-GONE", amount_cents=3300)
    cart_id = new_cart(authed_client)
    add(authed_client, cart_id, variant.id, 1)
    order_id = order(authed_client, cart_id).json()["order_id"]

    session.execute(Price.__table__.delete().where(Price.variant_id == variant.id))
    session.commit()

    body = authed_client.get(f"/orders/{order_id}").json()

    assert body["items"][0]["unit_price_cents"] == 3300
    assert body["items"][0]["variant_label"] == "42 / blue"
    assert body["total_cents"] == 3300


# --- reservation ---------------------------------------------------------


def test_ordering_reserves_exactly_the_ordered_quantity(authed_client, session):
    """Recorded before and after, never assumed from the seed."""
    variant = make_variant(session, sku="ORD-RESERVE", quantity=12, reserved=1)
    quantity_before, reserved_before = stock(session, variant.id)

    cart_id = new_cart(authed_client)
    add(authed_client, cart_id, variant.id, 4)
    assert order(authed_client, cart_id).status_code == 201

    quantity_after, reserved_after = stock(session, variant.id)

    assert reserved_after == reserved_before + 4
    # `quantity` is untouched on purpose: units leave it when they ship, and
    # this project has no fulfilment flow to ship them.
    assert quantity_after == quantity_before


def test_a_cart_asking_for_exactly_what_is_available_succeeds(authed_client, session):
    variant = make_variant(session, sku="ORD-EXACT", quantity=9, reserved=4)
    cart_id = new_cart(authed_client)

    # available is 9 - 4 = 5
    add(authed_client, cart_id, variant.id, 5)

    assert order(authed_client, cart_id).status_code == 201
    _, reserved_after = stock(session, variant.id)
    assert reserved_after == 9


def test_one_unit_more_than_available_is_409(authed_client, session):
    variant = make_variant(session, sku="ORD-ONE-OVER", quantity=9, reserved=4)
    cart_id = new_cart(authed_client)
    add(authed_client, cart_id, variant.id, 5)

    # Slip past the cart's advisory check by moving stock after the line exists.
    session.execute(
        Inventory.__table__.update()
        .where(Inventory.variant_id == variant.id)
        .values(reserved=5)
    )
    session.commit()

    response = order(authed_client, cart_id)

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "ORD-ONE-OVER" in detail
    assert "4 are available" in detail


def test_a_failed_order_reserves_nothing_at_all(authed_client, session):
    """Including the line that would have succeeded.

    This is the test that the ten steps are one transaction rather than ten.
    The first variant has plenty of stock and would reserve cleanly; the second
    cannot, and its failure has to undo the first.
    """
    fine = make_variant(session, sku="ORD-FINE", quantity=50, reserved=0)
    short = make_variant(session, sku="ORD-SHORT", quantity=50, reserved=0)
    cart_id = new_cart(authed_client)
    add(authed_client, cart_id, fine.id, 2)
    add(authed_client, cart_id, short.id, 2)

    session.execute(
        Inventory.__table__.update()
        .where(Inventory.variant_id == short.id)
        .values(reserved=49)
    )
    session.commit()

    fine_before = stock(session, fine.id)
    short_before = stock(session, short.id)

    response = order(authed_client, cart_id)

    assert response.status_code == 409
    assert stock(session, fine.id) == fine_before
    assert stock(session, short.id) == short_before
    assert session.scalar(select(func.count()).select_from(Order)) == 0
    assert session.get(Cart, uuid.UUID(cart_id)).status is CartStatus.OPEN


def test_the_stock_message_names_every_short_line_not_just_the_first(
    authed_client, session
):
    first = make_variant(session, sku="ORD-SHORT-1", quantity=10, reserved=9)
    second = make_variant(session, sku="ORD-SHORT-2", quantity=10, reserved=9)
    cart_id = new_cart(authed_client)
    add(authed_client, cart_id, first.id, 1)
    add(authed_client, cart_id, second.id, 1)

    session.execute(
        Inventory.__table__.update()
        .where(Inventory.variant_id.in_([first.id, second.id]))
        .values(reserved=10)
    )
    session.commit()

    detail = order(authed_client, cart_id).json()["detail"]

    assert "ORD-SHORT-1" in detail
    assert "ORD-SHORT-2" in detail


# --- atomicity under an injected failure --------------------------------


def test_a_crash_after_reserving_leaves_no_trace(authed_client, session, monkeypatch):
    """Fail between reserving stock and writing the order lines.

    `_snapshot_lines` is the first thing to run after `_reserve`, so replacing
    it puts the failure exactly in the window where a non-transactional
    implementation would already have moved `reserved` but not yet have an
    order to justify it — units held for a purchase that does not exist, and
    nothing to find them by.

    Called through the service rather than the client because the router does
    not catch this: an unexpected exception is not a 409, and TestClient would
    re-raise it here anyway.
    """
    variant = make_variant(session, sku="ORD-CRASH", quantity=30, reserved=3)
    cart_id = new_cart(authed_client)
    add(authed_client, cart_id, variant.id, 5)

    before = stock(session, variant.id)

    def explode(*args, **kwargs):
        raise RuntimeError("injected failure between reserving and snapshotting")

    monkeypatch.setattr(order_service, "_snapshot_lines", explode)

    with pytest.raises(RuntimeError, match="injected failure"):
        order_service.place_order(session, uuid.UUID(cart_id))

    assert stock(session, variant.id) == before
    assert session.scalar(select(func.count()).select_from(Order)) == 0
    assert session.scalar(select(func.count()).select_from(OrderItem)) == 0

    cart = session.get(Cart, uuid.UUID(cart_id))
    assert cart.status is CartStatus.OPEN
    # And the cart is still usable afterwards, which is what "rolled back"
    # has to mean in practice.
    assert add(authed_client, cart_id, variant.id, 1).status_code == 200


def test_the_cart_can_be_ordered_after_a_crash_is_repaired(
    authed_client, session, monkeypatch
):
    """The rollback leaves the cart genuinely orderable, not merely untouched."""
    variant = make_variant(session, sku="ORD-RETRY", quantity=30, reserved=0)
    cart_id = new_cart(authed_client)
    add(authed_client, cart_id, variant.id, 2)

    def explode(*args, **kwargs):
        raise RuntimeError("injected")

    monkeypatch.setattr(order_service, "_snapshot_lines", explode)
    with pytest.raises(RuntimeError):
        order_service.place_order(session, uuid.UUID(cart_id))

    monkeypatch.undo()

    assert order(authed_client, cart_id).status_code == 201
    _, reserved_after = stock(session, variant.id)
    assert reserved_after == 2


def test_the_locks_are_actually_issued(authed_client, session):
    """`FOR UPDATE` on `carts`, and on `inventory` ordered by variant_id.

    Both are invisible in behaviour under a single-threaded test — an
    implementation that dropped them would pass every other test in this file
    and deadlock or double-order only in production. So the statements are read
    directly.
    """
    first = make_variant(session, sku="ORD-LOCK-SQL-1")
    second = make_variant(session, sku="ORD-LOCK-SQL-2")
    cart_id = new_cart(authed_client)
    add(authed_client, cart_id, first.id, 1)
    add(authed_client, cart_id, second.id, 1)

    with recorded_sql() as statements:
        assert order(authed_client, cart_id).status_code == 201

    normalised = [" ".join(s.lower().split()) for s in statements]

    cart_lock = [s for s in normalised if "from carts" in s and "for update" in s]
    assert cart_lock, "the cart row was read without FOR UPDATE"

    inventory_lock = [
        s for s in normalised if "from inventory" in s and "for update" in s
    ]
    assert inventory_lock, "the inventory rows were read without FOR UPDATE"
    assert any("order by inventory.variant_id" in s for s in inventory_lock), (
        "the inventory lock has no ORDER BY: two orders covering the same "
        "variants in opposite order will deadlock"
    )
