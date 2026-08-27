"""Tests for the Stripe catalog sync (D7, step 2).

Most of this is offline, and can be because `build_plan` reads Postgres and
touches no network — which is the same split that makes `--dry-run` provable
rather than asserted.

The `stripe`-marked tests create real objects in test mode and archive them
afterwards. Stripe has no delete for a Product that carries a Price, and none
at all for a Price, so archiving is the cleanup: `active=false` retires the
object while leaving it resolvable, which is what anything that was ever bought
requires.
"""

from __future__ import annotations

import uuid

import pytest

from shopagent.catalog.models import Inventory, Price, Product, Variant
from shopagent.payments import catalog_sync, stripe_svc
from shopagent.payments.catalog_sync import build_plan, run_sync


def make_product(
    session,
    *,
    sku: str,
    name: str = "Sync Fixture",
    amount_cents: int = 4200,
    active: bool = True,
    currency: str = "usd",
) -> Product:
    product = Product(
        name=f"{name} {sku}",
        description=f"A product that exists only for a sync test ({sku}).",
        category="shoes",
        brand="Fleetfoot",
        variants=[
            Variant(
                size="42",
                color="blue",
                sku=sku,
                prices=[Price(currency=currency, amount_cents=amount_cents, active=active)],
                inventory=Inventory(quantity=5, reserved=0),
            )
        ],
    )
    session.add(product)
    session.commit()
    return product


def plan_for(session, product: Product):
    """The planned product and prices belonging to one fixture row."""
    plan = build_plan(session)
    planned_product = next(p for p in plan.products if p.product_id == product.id)
    prices = [p for p in plan.prices if p.product_id == product.id]
    return plan, planned_product, prices


# --- the mapping, without a single call ----------------------------------


@pytest.mark.db
def test_a_local_product_maps_onto_the_stripe_fields(session):
    product = make_product(session, sku="SYNC-MAP", amount_cents=9950)

    _, planned, prices = plan_for(session, product)

    assert planned.name == product.name
    assert planned.description == product.description
    assert planned.sku_group == "SYNC-MAP"

    (price,) = prices
    # The whole point of storing minor units since D3: no conversion here.
    assert price.amount_cents == 9950
    assert price.currency == "usd"
    assert price.sku == "SYNC-MAP"


@pytest.mark.db
def test_an_inactive_price_is_not_what_gets_synced(session):
    """Same definition of "active" the cart and the order use."""
    product = make_product(session, sku="SYNC-ACTIVE", amount_cents=1000)
    session.add(
        Price(
            variant_id=product.variants[0].id,
            currency="usd",
            amount_cents=9999,
            active=False,
        )
    )
    session.commit()

    _, _, prices = plan_for(session, product)

    (price,) = prices
    assert price.amount_cents == 1000


@pytest.mark.db
def test_a_variant_with_no_active_price_is_skipped_and_reported(session):
    """Not an error, and not silent either.

    The same state D6 calls `VariantNotSellable`. A variant that quietly never
    reaches Stripe is indistinguishable from one that was never there, so it is
    named in the report.
    """
    product = make_product(session, sku="SYNC-UNPRICED", active=False)

    plan = build_plan(session)

    assert not [p for p in plan.prices if p.product_id == product.id]
    skipped = [s for s in plan.skipped if s.sku == "SYNC-UNPRICED"]
    assert len(skipped) == 1
    assert "no active price" in skipped[0].reason


@pytest.mark.db
def test_a_price_in_another_currency_does_not_count(session):
    product = make_product(session, sku="SYNC-EUR", currency="eur")

    plan = build_plan(session)

    assert not [p for p in plan.prices if p.product_id == product.id]
    assert any(s.sku == "SYNC-EUR" for s in plan.skipped)


@pytest.mark.db
def test_an_already_synced_row_is_not_planned_again(session):
    product = make_product(session, sku="SYNC-DONE")
    session.execute(
        Product.__table__.update()
        .where(Product.id == product.id)
        .values(stripe_product_id="prod_already_there")
    )
    session.execute(
        Variant.__table__.update()
        .where(Variant.id == product.variants[0].id)
        .values(stripe_price_id="price_already_there")
    )
    session.commit()

    _, planned, prices = plan_for(session, product)

    assert planned.needs_creating is False
    assert prices[0].needs_creating is False


# --- idempotency keys are derived, not random ----------------------------


@pytest.mark.db
def test_the_idempotency_key_is_stable_across_runs(session):
    """A random key would make every re-run a fresh request to Stripe."""
    product = make_product(session, sku="SYNC-KEY")

    _, first, first_prices = plan_for(session, product)
    _, second, second_prices = plan_for(session, product)

    assert first.idempotency_key == second.idempotency_key
    assert first_prices[0].idempotency_key == second_prices[0].idempotency_key


@pytest.mark.db
def test_a_changed_amount_changes_the_price_idempotency_key(session):
    """A Stripe Price is immutable, so a different amount is a different object.

    Reusing the key would have Stripe replay the old Price and report success
    for a create that never happened.
    """
    product = make_product(session, sku="SYNC-KEY-AMOUNT", amount_cents=1000)
    _, _, before = plan_for(session, product)

    session.execute(
        Price.__table__.update()
        .where(Price.variant_id == product.variants[0].id)
        .values(amount_cents=2000)
    )
    session.commit()

    _, _, after = plan_for(session, product)

    assert before[0].idempotency_key != after[0].idempotency_key


# --- a dry run writes nothing, and that is demonstrated ------------------


@pytest.mark.db
def test_a_dry_run_calls_no_write_api(session, monkeypatch):
    """Proved by breaking every write, not by trusting the flag.

    If `run_sync(dry_run=True)` reached any of these the test would fail with
    the message below rather than a passing assertion. The autouse guard in
    `conftest.py` is a second net underneath: it blocks the SDK's request
    funnel for any unmarked test, so even a call that bypassed these functions
    could not leave the process.
    """
    make_product(session, sku="SYNC-DRY")

    def explode(*args, **kwargs):
        raise AssertionError("a dry run called a Stripe write API")

    monkeypatch.setattr(stripe_svc, "create_product", explode)
    monkeypatch.setattr(stripe_svc, "create_price", explode)
    monkeypatch.setattr(stripe_svc, "list_prices", explode)

    summary = run_sync(session, dry_run=True)

    assert summary.dry_run is True
    assert summary.products_created > 0, "the dry run planned nothing to check"


@pytest.mark.db
def test_a_dry_run_writes_no_ids_to_the_database(session, monkeypatch):
    product = make_product(session, sku="SYNC-DRY-DB")
    monkeypatch.setattr(stripe_svc, "create_product", lambda **kw: None)
    monkeypatch.setattr(stripe_svc, "create_price", lambda **kw: None)
    monkeypatch.setattr(stripe_svc, "list_prices", lambda *a, **kw: [])

    run_sync(session, dry_run=True)

    session.refresh(product)
    assert product.stripe_product_id is None
    assert product.variants[0].stripe_price_id is None


# --- drift is reported, never repaired -----------------------------------


@pytest.mark.db
def test_a_changed_local_price_is_reported_as_drift(session, monkeypatch):
    """Stripe Prices are immutable, so the sync names the mismatch and stops.

    Repairing would mean creating a replacement Price and archiving the old
    one. Nothing is charged from these objects, so drift costs a stale
    dashboard rather than a wrong charge — and a script that silently retires
    objects in somebody's Stripe account is one nobody can reason about.
    """
    product = make_product(session, sku="SYNC-DRIFT", amount_cents=1000)
    session.execute(
        Variant.__table__.update()
        .where(Variant.id == product.variants[0].id)
        .values(stripe_price_id="price_drifted")
    )
    session.commit()

    class FakePrice:
        id = "price_drifted"
        unit_amount = 9999

    monkeypatch.setattr(stripe_svc, "list_prices", lambda *a, **kw: [FakePrice()])

    drifted = catalog_sync.detect_price_drift(build_plan(session))

    (drift,) = [d for d in drifted if d.sku == "SYNC-DRIFT"]
    assert drift.local_amount_cents == 1000
    assert drift.stripe_amount_cents == 9999


# --- the real thing ------------------------------------------------------


@pytest.mark.stripe
def test_creating_a_product_and_price_in_test_mode_and_cleaning_up():
    """One Product, one Price, archived afterwards.

    Deliberately not driven through `run_sync`: that would write ids into the
    real catalog rows and leave the database holding pointers to objects this
    test is about to archive.
    """
    marker = uuid.uuid4().hex[:12]
    product = None
    price = None

    try:
        product = stripe_svc.create_product(
            name=f"ShopAgent test fixture {marker}",
            description="Created by an automated test; safe to archive.",
            sku_group=f"TEST-{marker}",
            idempotency_key=f"shopagent-test-product-{marker}",
        )
        assert product.id.startswith("prod_")
        assert product.livemode is False

        price = stripe_svc.create_price(
            product_id=product.id,
            unit_amount_cents=4242,
            currency="usd",
            sku=f"TEST-{marker}",
            idempotency_key=f"shopagent-test-price-{marker}",
        )
        assert price.id.startswith("price_")
        assert price.unit_amount == 4242
        assert price.currency == "usd"
        assert price.product == product.id
        assert price.livemode is False

        # The second half of the idempotency story: the same key returns the
        # same object rather than a second one.
        again = stripe_svc.create_product(
            name=f"ShopAgent test fixture {marker}",
            description="Created by an automated test; safe to archive.",
            sku_group=f"TEST-{marker}",
            idempotency_key=f"shopagent-test-product-{marker}",
        )
        assert again.id == product.id, "the idempotency key did not prevent a duplicate"

    finally:
        # Archive, not delete. Stripe refuses to delete a Product that has a
        # Price, and offers no delete for a Price at all.
        if price is not None:
            stripe_svc.archive_price(price.id)
        if product is not None:
            stripe_svc.archive_product(product.id)


@pytest.mark.stripe
def test_an_archived_fixture_is_inactive_not_gone():
    """What cleanup can and cannot mean, asserted rather than assumed."""
    marker = uuid.uuid4().hex[:12]
    product = stripe_svc.create_product(
        name=f"ShopAgent archive check {marker}",
        description="Created by an automated test; safe to archive.",
        sku_group=f"TEST-{marker}",
        idempotency_key=f"shopagent-archive-{marker}",
    )
    try:
        archived = stripe_svc.archive_product(product.id)
        assert archived.active is False
        # Still resolvable: anything ever bought has to stay readable.
        assert stripe_svc.get_client().v1.products.retrieve(product.id).id == product.id
    finally:
        stripe_svc.archive_product(product.id)
