"""Tests for the cart endpoints (D6, step 3).

Every status code in the step's table is pinned here, because a status code is
a contract with the model on D9 as much as with a browser: a 409 tells it to
change the request and a 404 tells it the thing is gone, and the difference is
the difference between a useful next turn and a loop.

All of it runs through `authed_client`, which is `api_client` with the key
attached — so the handler writes into the test's own transaction and the whole
file leaves nothing behind. See `tests/conftest.py`.
"""

from __future__ import annotations

import contextlib
import uuid

import pytest
from sqlalchemy import event, select
from sqlalchemy.engine import Engine

from shopagent.api.models import Cart, CartItem, CartStatus
from shopagent.catalog.models import Inventory, Price, Product, Variant

pytestmark = pytest.mark.db

MISSING_UUID = uuid.UUID("00000000-0000-0000-0000-000000000000")


def make_variant(
    session,
    *,
    sku: str,
    amount_cents: int = 1000,
    quantity: int = 10,
    reserved: int = 0,
    name: str | None = None,
    active: bool = True,
) -> Variant:
    """One product, one variant, one price, one stock row."""
    product = Product(
        name=name or f"Cart Fixture {sku}",
        description="A product that exists only for a cart test.",
        category="shoes",
        brand="Fleetfoot",
        variants=[
            Variant(
                size="42",
                color="blue",
                sku=sku,
                prices=[Price(currency="usd", amount_cents=amount_cents, active=active)],
                inventory=Inventory(quantity=quantity, reserved=reserved),
            )
        ],
    )
    session.add(product)
    session.commit()
    return product.variants[0]


def new_cart(client) -> str:
    response = client.post("/cart")
    assert response.status_code == 201
    return response.json()["cart_id"]


def add(client, cart_id: str, variant_id: int, quantity: int = 1):
    return client.post(
        f"/cart/{cart_id}/items",
        json={"variant_id": variant_id, "quantity": quantity},
    )


# --- creating ------------------------------------------------------------


def test_creating_a_cart_is_201_and_returns_an_empty_priced_cart(authed_client):
    response = authed_client.post("/cart")

    assert response.status_code == 201
    body = response.json()
    assert uuid.UUID(body["cart_id"])
    assert body["status"] == "open"
    assert body["currency"] == "usd"
    assert body["items"] == []
    assert body["total_cents"] == 0


# --- adding --------------------------------------------------------------


def test_adding_a_new_variant_is_201(authed_client, session):
    variant = make_variant(session, sku="CART-ADD-1", amount_cents=8999)
    cart_id = new_cart(authed_client)

    response = add(authed_client, cart_id, variant.id, 2)

    assert response.status_code == 201
    body = response.json()
    assert len(body["items"]) == 1
    line = body["items"][0]
    assert line["variant_id"] == variant.id
    assert line["sku"] == "CART-ADD-1"
    assert line["product_name"] == "Cart Fixture CART-ADD-1"
    assert line["variant_label"] == "42 / blue"
    assert line["quantity"] == 2
    assert line["unit_price_cents"] == 8999
    assert line["line_total_cents"] == 8999 * 2
    assert body["total_cents"] == 8999 * 2


def test_adding_the_same_variant_again_is_200_and_upserts(authed_client, session):
    """One row, summed quantity — not a second line.

    The 200 rather than 201 is the point: nothing was created. A caller that
    sees 200 knows it merged into a line that was already there, which is the
    one fact the response body cannot tell it without remembering the cart's
    previous contents.
    """
    variant = make_variant(session, sku="CART-UPSERT", amount_cents=500)
    cart_id = new_cart(authed_client)

    first = add(authed_client, cart_id, variant.id, 2)
    second = add(authed_client, cart_id, variant.id, 3)

    assert first.status_code == 201
    assert second.status_code == 200

    body = second.json()
    assert len(body["items"]) == 1
    assert body["items"][0]["quantity"] == 5
    assert body["total_cents"] == 2500

    rows = session.scalars(
        select(CartItem).where(CartItem.cart_id == uuid.UUID(cart_id))
    ).all()
    assert len(rows) == 1


def test_a_quantity_of_zero_or_less_is_422(authed_client, session):
    variant = make_variant(session, sku="CART-QTY")
    cart_id = new_cart(authed_client)

    for quantity in (0, -1):
        response = add(authed_client, cart_id, variant.id, quantity)
        assert response.status_code == 422


def test_adding_to_a_cart_that_does_not_exist_is_404(authed_client, session):
    variant = make_variant(session, sku="CART-NO-CART")

    response = add(authed_client, str(MISSING_UUID), variant.id)

    assert response.status_code == 404


def test_adding_a_variant_that_does_not_exist_is_404(authed_client):
    cart_id = new_cart(authed_client)

    response = add(authed_client, cart_id, 987654321)

    assert response.status_code == 404


# --- stock ---------------------------------------------------------------


def test_more_than_available_is_409(authed_client, session):
    variant = make_variant(session, sku="CART-STOCK", quantity=5, reserved=2)
    cart_id = new_cart(authed_client)

    ok = add(authed_client, cart_id, variant.id, 3)
    too_many = add(authed_client, cart_id, variant.id, 1)

    # quantity - reserved is 3, so three fit and a fourth does not.
    assert ok.status_code == 201
    assert too_many.status_code == 409
    assert "available" in too_many.json()["detail"]


def test_the_stock_check_counts_the_resulting_quantity_not_the_increment(
    authed_client, session
):
    """Adding two to a line of three asks the catalog for five, not two."""
    variant = make_variant(session, sku="CART-STOCK-SUM", quantity=4, reserved=0)
    cart_id = new_cart(authed_client)

    assert add(authed_client, cart_id, variant.id, 3).status_code == 201
    assert add(authed_client, cart_id, variant.id, 2).status_code == 409


def test_a_refused_add_leaves_no_reservation(authed_client, session):
    """This step reads inventory and never writes it. Reserving is step 4."""
    variant = make_variant(session, sku="CART-NO-RESERVE", quantity=5)
    cart_id = new_cart(authed_client)
    add(authed_client, cart_id, variant.id, 3)

    stock = session.get(Inventory, variant.id)
    session.refresh(stock)
    assert stock.reserved == 0
    assert stock.quantity == 5


# --- reading and the total ----------------------------------------------


def test_reading_a_cart_is_200(authed_client, session):
    variant = make_variant(session, sku="CART-READ", amount_cents=1234)
    cart_id = new_cart(authed_client)
    add(authed_client, cart_id, variant.id, 2)

    response = authed_client.get(f"/cart/{cart_id}")

    assert response.status_code == 200
    assert response.json()["total_cents"] == 2468


def test_reading_a_cart_that_does_not_exist_is_404(authed_client):
    assert authed_client.get(f"/cart/{MISSING_UUID}").status_code == 404


def test_the_total_sums_different_variants_at_different_prices(authed_client, session):
    cheap = make_variant(session, sku="CART-CHEAP", amount_cents=250)
    dear = make_variant(session, sku="CART-DEAR", amount_cents=9950)
    cart_id = new_cart(authed_client)

    add(authed_client, cart_id, cheap.id, 4)
    add(authed_client, cart_id, dear.id, 2)

    body = authed_client.get(f"/cart/{cart_id}").json()

    assert len(body["items"]) == 2
    assert body["total_cents"] == (250 * 4) + (9950 * 2)


def test_an_inactive_price_is_not_counted(authed_client, session):
    """`active` is what makes a superseded price readable without being charged.

    A second, inactive row at a different amount must change nothing. The
    partial unique index from D3 permits it precisely so price history
    survives, and a total that picked it up would charge last month's price.
    """
    variant = make_variant(session, sku="CART-ACTIVE", amount_cents=1000)
    session.add(
        Price(variant_id=variant.id, currency="usd", amount_cents=9999, active=False)
    )
    session.commit()

    cart_id = new_cart(authed_client)
    add(authed_client, cart_id, variant.id, 2)

    body = authed_client.get(f"/cart/{cart_id}").json()

    assert body["items"][0]["unit_price_cents"] == 1000
    assert body["total_cents"] == 2000


def test_a_variant_with_no_active_price_cannot_be_added(authed_client, session):
    variant = make_variant(session, sku="CART-UNPRICED", active=False)
    cart_id = new_cart(authed_client)

    response = add(authed_client, cart_id, variant.id)

    assert response.status_code == 409
    assert "no active price" in response.json()["detail"]


def test_a_line_whose_price_was_deactivated_is_shown_without_one(
    authed_client, session
):
    """The cart still lists it, and the total leaves it out.

    Reachable only by deactivating a price after the line was added, since
    adding an unpriced variant is refused. Dropping the line would make a cart
    quietly lose an item; inventing a price would be worse. Turning this into
    an error is `POST /orders`' job in step 4, where a total that omits a line
    would actually be charged.
    """
    priced = make_variant(session, sku="CART-STILL-PRICED", amount_cents=700)
    losing = make_variant(session, sku="CART-LOSES-PRICE", amount_cents=300)
    cart_id = new_cart(authed_client)
    add(authed_client, cart_id, priced.id, 1)
    add(authed_client, cart_id, losing.id, 5)

    session.execute(
        Price.__table__.update()
        .where(Price.variant_id == losing.id)
        .values(active=False)
    )
    session.commit()

    body = authed_client.get(f"/cart/{cart_id}").json()

    lines = {line["sku"]: line for line in body["items"]}
    assert len(lines) == 2
    assert lines["CART-LOSES-PRICE"]["unit_price_cents"] is None
    assert lines["CART-LOSES-PRICE"]["line_total_cents"] is None
    assert lines["CART-LOSES-PRICE"]["quantity"] == 5
    assert body["total_cents"] == 700


# --- removing ------------------------------------------------------------


def test_removing_an_item_is_204_and_updates_the_total(authed_client, session):
    keep = make_variant(session, sku="CART-KEEP", amount_cents=1000)
    drop = make_variant(session, sku="CART-DROP", amount_cents=2500)
    cart_id = new_cart(authed_client)
    add(authed_client, cart_id, keep.id, 1)
    drop_body = add(authed_client, cart_id, drop.id, 2).json()

    item_id = next(
        line["item_id"] for line in drop_body["items"] if line["sku"] == "CART-DROP"
    )
    before = authed_client.get(f"/cart/{cart_id}").json()["total_cents"]
    assert before == 1000 + 5000

    response = authed_client.delete(f"/cart/{cart_id}/items/{item_id}")

    assert response.status_code == 204
    assert response.content == b""

    after = authed_client.get(f"/cart/{cart_id}").json()
    assert after["total_cents"] == 1000
    assert [line["sku"] for line in after["items"]] == ["CART-KEEP"]


def test_removing_an_item_that_does_not_exist_is_404(authed_client):
    cart_id = new_cart(authed_client)

    response = authed_client.delete(f"/cart/{cart_id}/items/{MISSING_UUID}")

    assert response.status_code == 404


def test_removing_an_item_from_a_cart_that_does_not_exist_is_404(authed_client):
    response = authed_client.delete(f"/cart/{MISSING_UUID}/items/{MISSING_UUID}")

    assert response.status_code == 404


def test_an_item_belonging_to_another_cart_is_404_not_403(authed_client, session):
    """404 on purpose: 403 would confirm the id is real.

    An id that is not yours should not be able to establish that it exists,
    and the two carts here are indistinguishable from the caller's side.
    """
    variant = make_variant(session, sku="CART-OTHER")
    mine = new_cart(authed_client)
    theirs = new_cart(authed_client)

    body = add(authed_client, theirs, variant.id, 1).json()
    their_item = body["items"][0]["item_id"]

    response = authed_client.delete(f"/cart/{mine}/items/{their_item}")

    assert response.status_code == 404

    # And the line is untouched in the cart it does belong to.
    still_there = authed_client.get(f"/cart/{theirs}").json()
    assert len(still_there["items"]) == 1


# --- a cart that has become an order ------------------------------------


def lock(session, cart_id: str) -> None:
    """What `POST /orders` will do in step 4, done directly for now."""
    cart = session.get(Cart, uuid.UUID(cart_id))
    cart.status = CartStatus.ORDERED
    session.commit()


def test_adding_to_an_ordered_cart_is_409(authed_client, session):
    variant = make_variant(session, sku="CART-LOCKED-ADD")
    cart_id = new_cart(authed_client)
    add(authed_client, cart_id, variant.id, 1)
    lock(session, cart_id)

    response = add(authed_client, cart_id, variant.id, 1)

    assert response.status_code == 409
    assert "ordered" in response.json()["detail"]


def test_removing_from_an_ordered_cart_is_409(authed_client, session):
    variant = make_variant(session, sku="CART-LOCKED-DEL")
    cart_id = new_cart(authed_client)
    item_id = add(authed_client, cart_id, variant.id, 1).json()["items"][0]["item_id"]
    lock(session, cart_id)

    response = authed_client.delete(f"/cart/{cart_id}/items/{item_id}")

    assert response.status_code == 409


def test_an_ordered_cart_can_still_be_read(authed_client, session):
    """The lock is on writes. An order has to stay readable after it is placed."""
    variant = make_variant(session, sku="CART-LOCKED-GET", amount_cents=4200)
    cart_id = new_cart(authed_client)
    add(authed_client, cart_id, variant.id, 2)
    lock(session, cart_id)

    response = authed_client.get(f"/cart/{cart_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ordered"
    assert body["total_cents"] == 8400


# --- authentication ------------------------------------------------------


def test_every_cart_route_needs_the_key(api_client, session):
    """Explicit here as well as in the sweep, because this is the flow that
    would actually be attacked: an unauthenticated write."""
    variant = make_variant(session, sku="CART-NOAUTH")

    assert api_client.post("/cart").status_code == 401
    assert api_client.get(f"/cart/{MISSING_UUID}").status_code == 401
    assert add(api_client, str(MISSING_UUID), variant.id).status_code == 401
    assert (
        api_client.delete(f"/cart/{MISSING_UUID}/items/{MISSING_UUID}").status_code
        == 401
    )


# --- the cart row is locked on every write (review on PR #6) -------------
#
# `place_order` locks the cart, snapshots what it finds and flips the status to
# `ordered`. An unlocked add can read `open`, be descheduled, and commit its
# line after that snapshot was taken — an ordered cart holding an item that is
# on no order, which is goods a shopper believes they bought and nobody was
# charged for. The check has to happen under the same lock `place_order` takes.
#
# Like the locks in `test_api_orders.py`, this is invisible in single-threaded
# behaviour: an implementation without it passes every other test in this file.
# So the statements are read instead.


@contextlib.contextmanager
def recorded_sql():
    statements: list[str] = []

    def listener(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(Engine, "before_cursor_execute", listener)
    try:
        yield statements
    finally:
        event.remove(Engine, "before_cursor_execute", listener)


def _cart_selects(statements) -> list[str]:
    return [
        " ".join(s.lower().split())
        for s in statements
        if "from carts" in " ".join(s.lower().split())
    ]


def test_adding_an_item_locks_the_cart_row(authed_client, session):
    variant = make_variant(session, sku="CART-LOCK-ADD-SQL")
    cart_id = new_cart(authed_client)

    with recorded_sql() as statements:
        assert add(authed_client, cart_id, variant.id, 1).status_code == 201

    locked = [s for s in _cart_selects(statements) if "for update" in s]
    assert locked, (
        "the cart row was read without FOR UPDATE, so an add can commit onto a "
        "cart that place_order has already snapshotted and closed"
    )


def test_removing_an_item_locks_the_cart_row(authed_client, session):
    variant = make_variant(session, sku="CART-LOCK-DEL-SQL")
    cart_id = new_cart(authed_client)
    item_id = add(authed_client, cart_id, variant.id, 1).json()["items"][0]["item_id"]

    with recorded_sql() as statements:
        assert authed_client.delete(f"/cart/{cart_id}/items/{item_id}").status_code == 204

    locked = [s for s in _cart_selects(statements) if "for update" in s]
    assert locked, "the cart row was read without FOR UPDATE on the delete path"


def test_reading_a_cart_takes_no_write_lock(authed_client, session):
    """The other half, and it matters as much.

    A read that took `FOR UPDATE` would make every `GET /cart` queue behind an
    in-flight checkout for no benefit, since nothing it returns decides a write.
    """
    variant = make_variant(session, sku="CART-LOCK-GET-SQL")
    cart_id = new_cart(authed_client)
    add(authed_client, cart_id, variant.id, 1)

    with recorded_sql() as statements:
        assert authed_client.get(f"/cart/{cart_id}").status_code == 200

    selects = _cart_selects(statements)
    assert selects, "the recorder captured no read of the cart to inspect"
    assert not [s for s in selects if "for update" in s], (
        "GET /cart took a write lock on the cart row"
    )
