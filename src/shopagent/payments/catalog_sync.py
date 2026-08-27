"""Mirror the local catalog into Stripe Products and Prices (D7, step 2).

An isolated deliverable. Nothing in the checkout path reads what this writes —
see the note on `Variant.stripe_price_id` — so a stale or missing Stripe object
cannot produce a wrong charge. It exists so the catalog is visible in the
dashboard and so the Products/Prices API is exercised against real objects.

The module reads and plans; the script in `scripts/sync_stripe_catalog.py` is a
CLI around it. Splitting them is what makes `--dry-run` provable rather than
asserted: the plan is built by a function that touches no network at all, and a
dry run is that function and nothing else.

**Idempotency is two mechanisms, deliberately not one.**

*Skipping by local id* is the durable half. A product whose
`stripe_product_id` is already set is not re-created, ever. This costs no API
call and survives forever, which is what makes a second run genuinely free.

*Stripe idempotency keys* are the crash half, and they cover the window the
first mechanism cannot: the object was created in Stripe but the process died
before its id reached Postgres. On the next run the local row still looks
unsynced, so it would be created again — except the key is derived from the
local row, so Stripe replays the original response instead. That protection
lasts 24 hours, which is Stripe's retention for keys and therefore the real
lifetime of this half. It is a safety net for a bad afternoon, not a
substitute for storing the id.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from shopagent.catalog.models import Price, Product, Variant
from shopagent.config import get_settings
from shopagent.payments import stripe_svc


@dataclass(frozen=True)
class PlannedProduct:
    """A local product and what, if anything, has to happen to it."""

    product_id: int
    name: str
    description: str
    sku_group: str
    stripe_product_id: str | None

    @property
    def needs_creating(self) -> bool:
        return self.stripe_product_id is None

    @property
    def idempotency_key(self) -> str:
        # Derived from the local row, never random: a random key would make
        # every re-run a fresh request and defeat the whole point.
        return f"shopagent-product-v1-{self.product_id}"


@dataclass(frozen=True)
class PlannedPrice:
    """A local variant's active price and what has to happen to it."""

    variant_id: int
    product_id: int
    sku: str
    amount_cents: int
    currency: str
    stripe_price_id: str | None

    @property
    def needs_creating(self) -> bool:
        return self.stripe_price_id is None

    @property
    def idempotency_key(self) -> str:
        # The amount is part of the key. A Stripe Price is immutable, so a
        # different amount is a different object and must not be answered with
        # a replay of the old one.
        return (
            f"shopagent-price-v1-{self.variant_id}-{self.currency}-{self.amount_cents}"
        )


@dataclass(frozen=True)
class SkippedVariant:
    """A variant the sync could not handle, and why."""

    sku: str
    reason: str


@dataclass(frozen=True)
class DriftedPrice:
    """A synced Price whose Stripe amount no longer matches the local one."""

    sku: str
    stripe_price_id: str
    local_amount_cents: int
    stripe_amount_cents: int


@dataclass
class SyncPlan:
    """Everything the sync intends to do, computed without touching Stripe."""

    products: list[PlannedProduct] = field(default_factory=list)
    prices: list[PlannedPrice] = field(default_factory=list)
    skipped: list[SkippedVariant] = field(default_factory=list)

    @property
    def products_to_create(self) -> list[PlannedProduct]:
        return [p for p in self.products if p.needs_creating]

    @property
    def prices_to_create(self) -> list[PlannedPrice]:
        return [p for p in self.prices if p.needs_creating]


@dataclass
class SyncSummary:
    """What a run actually did, for the report at the end."""

    products_created: int = 0
    products_skipped: int = 0
    prices_created: int = 0
    prices_skipped: int = 0
    skipped: list[SkippedVariant] = field(default_factory=list)
    drifted: list[DriftedPrice] = field(default_factory=list)
    dry_run: bool = False

    def as_lines(self) -> list[str]:
        verb = "would create" if self.dry_run else "created"
        return [
            f"  products {verb:<12} {self.products_created}",
            f"  products skipped     {self.products_skipped}",
            f"  prices   {verb:<12} {self.prices_created}",
            f"  prices   skipped     {self.prices_skipped}",
        ]


def build_plan(session: Session) -> SyncPlan:
    """Work out what the sync would do. Reads Postgres, touches no network.

    A variant with no active price in the shop's currency is not an error and
    does not stop the run — it is simply not sellable, which is the same state
    D6's `VariantNotSellable` describes. It is collected into `skipped` so the
    report can name it, because a variant that quietly never reaches Stripe is
    indistinguishable from one that was never there.
    """
    currency = get_settings().currency
    plan = SyncPlan()

    products = session.scalars(select(Product).order_by(Product.id.asc())).all()

    for product in products:
        variants = sorted(product.variants, key=lambda v: v.id)
        plan.products.append(
            PlannedProduct(
                product_id=product.id,
                name=product.name,
                description=product.description,
                # The product has no sku of its own; its first variant's is the
                # closest stable handle for tracing a dashboard row back here.
                sku_group=variants[0].sku if variants else f"product-{product.id}",
                stripe_product_id=product.stripe_product_id,
            )
        )

        for variant in variants:
            active = [
                price
                for price in variant.prices
                if price.active and price.currency == currency
            ]
            if not active:
                plan.skipped.append(
                    SkippedVariant(
                        sku=variant.sku,
                        reason=f"no active price in {currency}",
                    )
                )
                continue

            # D3's partial unique index guarantees at most one, so there is no
            # winner to pick.
            price = active[0]
            plan.prices.append(
                PlannedPrice(
                    variant_id=variant.id,
                    product_id=product.id,
                    sku=variant.sku,
                    amount_cents=price.amount_cents,
                    currency=price.currency,
                    stripe_price_id=variant.stripe_price_id,
                )
            )

    return plan


def detect_price_drift(plan: SyncPlan) -> list[DriftedPrice]:
    """Find synced Prices whose Stripe amount no longer matches the local one.

    Reported, never repaired. A Stripe Price is immutable — changing an amount
    means creating a new Price and archiving the old one — and this script
    deliberately does not do that on its own. Two reasons. Nothing charges from
    these objects, so drift costs an out-of-date dashboard rather than a wrong
    charge; and a script that silently retires objects in somebody's Stripe
    account is one nobody can reason about afterwards. Naming the drift and
    leaving the decision to a person is the honest trade for a sync that is not
    on the billing path.

    One `prices.list` call rather than one retrieve per variant: the accounts
    this runs against hold tens of prices, and sixty round trips to answer a
    question about staleness would make the common case — no drift at all —
    the slowest part of the script.
    """
    synced = {p.stripe_price_id: p for p in plan.prices if p.stripe_price_id}
    if not synced:
        return []

    drifted: list[DriftedPrice] = []
    for stripe_price in stripe_svc.list_prices():
        local = synced.get(stripe_price.id)
        if local is None:
            continue
        if stripe_price.unit_amount != local.amount_cents:
            drifted.append(
                DriftedPrice(
                    sku=local.sku,
                    stripe_price_id=stripe_price.id,
                    local_amount_cents=local.amount_cents,
                    stripe_amount_cents=stripe_price.unit_amount,
                )
            )
    return drifted


def run_sync(session: Session, *, dry_run: bool = False) -> SyncSummary:
    """Create whatever is missing in Stripe and write the ids back.

    Each id is committed as soon as it is known rather than in one commit at
    the end. A crash halfway then leaves the ids that were already earned in
    the database, so the next run skips them locally — the cheap mechanism —
    instead of relying on the 24-hour idempotency key to cover everything that
    came before.
    """
    plan = build_plan(session)
    summary = SyncSummary(skipped=plan.skipped, dry_run=dry_run)

    stripe_ids: dict[int, str] = {
        p.product_id: p.stripe_product_id
        for p in plan.products
        if p.stripe_product_id
    }

    for planned in plan.products:
        if not planned.needs_creating:
            summary.products_skipped += 1
            continue

        summary.products_created += 1
        if dry_run:
            continue

        created = stripe_svc.create_product(
            name=planned.name,
            description=planned.description,
            sku_group=planned.sku_group,
            idempotency_key=planned.idempotency_key,
        )
        stripe_ids[planned.product_id] = created.id
        session.execute(
            Product.__table__.update()
            .where(Product.id == planned.product_id)
            .values(stripe_product_id=created.id)
        )
        session.commit()

    for planned in plan.prices:
        if not planned.needs_creating:
            summary.prices_skipped += 1
            continue

        summary.prices_created += 1
        if dry_run:
            continue

        product_id = stripe_ids.get(planned.product_id)
        if product_id is None:
            # Its product failed to sync, so there is nothing to hang the
            # price on. Report rather than raise: one bad product should not
            # cost the other twenty-nine.
            summary.prices_created -= 1
            summary.skipped.append(
                SkippedVariant(
                    sku=planned.sku,
                    reason="its product has no Stripe id",
                )
            )
            continue

        created = stripe_svc.create_price(
            product_id=product_id,
            unit_amount_cents=planned.amount_cents,
            currency=planned.currency,
            sku=planned.sku,
            idempotency_key=planned.idempotency_key,
        )
        session.execute(
            Variant.__table__.update()
            .where(Variant.id == planned.variant_id)
            .values(stripe_price_id=created.id)
        )
        session.commit()

    if not dry_run:
        summary.drifted = detect_price_drift(build_plan(session))

    return summary
