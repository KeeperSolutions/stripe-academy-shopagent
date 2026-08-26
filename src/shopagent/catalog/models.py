"""Catalog schema — products, variants, prices, inventory (D3).

Four tables rather than one wide product row, because the demo scenario needs
them apart: "add the second one in size 42" picks a *variant*, the price shown
belongs to that variant, and stock is counted per variant too. Collapsing them
would make D9's cart tools guess which row they are talking about.

Money is `amount_cents: int` and never anything else. Stripe charges in the
smallest currency unit, so an integer here goes to the API untouched; a float
or a Numeric would need a conversion at the boundary, and that conversion is
where rounding bugs at checkout come from.

The HNSW index on `embedding` is not declared here. It belongs to step 4 of
D3, when there are vectors to index — building it over an empty, all-NULL
column would only teach the syntax in the wrong place.
"""

from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Dimension of text-embedding-3-small, the model in `settings.embedding_model`.
# Hard-coded rather than read from settings: the number is baked into the
# column type at CREATE TABLE time, so changing the setting later cannot change
# an existing column. Swapping embedding models is a migration, not a config
# edit, and this constant is where that becomes visible.
EMBEDDING_DIM = 1536


class Base(DeclarativeBase):
    """Declarative base for every catalog table.

    D6 adds cart and order tables to this same metadata, so `create_all` keeps
    building the whole schema in one call.
    """


class Product(Base):
    """A sellable item, independent of size or colour."""

    __tablename__ = "products"
    __table_args__ = (
        # Products have no natural key of their own, which let the seeder
        # create a second "Runner's Cap" when the first one's only variant had
        # been deleted: identity was inferred from a surviving sku, and there
        # was none. Name and brand together are the identity the seed data
        # already assumes, so the database now enforces it and `seed.py` can
        # look a product up by it.
        UniqueConstraint("name", "brand", name="uq_products_name_brand"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text)
    # The index on this column is declared below as a functional index over
    # `lower(category)`, because that is the expression the search compares
    # against. A plain index would sit unused: `WHERE lower(category) = 'shoes'`
    # cannot read one built on the raw column.
    category: Mapped[str] = mapped_column(String(60))
    brand: Mapped[str] = mapped_column(String(80))
    # Nullable on purpose. Rows arrive from the seed (step 2) with no vector
    # and are filled in by the embedding pass (step 4); NOT NULL would make
    # those two steps a single, unsplittable one.
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )

    variants: Mapped[list[Variant]] = relationship(
        back_populates="product",
        # Two layers, doing different jobs. `delete-orphan` is the ORM one: it
        # handles objects already loaded in the session. `passive_deletes`
        # tells SQLAlchemy not to load the children just to delete them
        # one-by-one, and to trust the ON DELETE CASCADE on the foreign key
        # instead — which also covers a DELETE that never goes through the ORM.
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Variant(Base):
    """One buyable configuration of a product: this size, this colour."""

    __tablename__ = "variants"

    id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.id", ondelete="CASCADE"), index=True
    )
    # Nullable: not everything has a size or a colour (a water bottle has
    # neither), and an empty string would be a worse way of saying so.
    size: Mapped[str | None] = mapped_column(String(20))
    color: Mapped[str | None] = mapped_column(String(40))
    # The stable public identifier. Unique because the cart (D6) and the
    # Stripe line items (D7) both key on it; a duplicate would silently move
    # stock between two different products.
    sku: Mapped[str] = mapped_column(String(64), unique=True)

    product: Mapped[Product] = relationship(back_populates="variants")
    prices: Mapped[list[Price]] = relationship(
        back_populates="variant",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    inventory: Mapped[Inventory | None] = relationship(
        back_populates="variant",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )


class Price(Base):
    """What a variant costs, in one currency.

    A row per currency, and `active` rather than a delete, because a price that
    was charged has to stay readable after it stops being offered — an order
    from last week refers to it.
    """

    __tablename__ = "prices"
    __table_args__ = (
        # A negative price is not a discount, it is a bug. Cheap to forbid in
        # the schema; expensive to notice at checkout.
        CheckConstraint("amount_cents >= 0", name="ck_prices_amount_cents_positive"),
        # At most one *active* price per variant per currency. Superseded rows
        # stay, which is the point of the `active` flag, so the uniqueness is
        # partial. Without it the search join returns a variant once per active
        # row: the same sku reaching the model twice, with two different
        # `price_cents`, and nothing to say which one it would be charged.
        Index(
            "uq_prices_one_active_per_variant_currency",
            "variant_id",
            "currency",
            unique=True,
            postgresql_where=text("active"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    variant_id: Mapped[int] = mapped_column(
        ForeignKey("variants.id", ondelete="CASCADE"), index=True
    )
    # ISO-4217, lowercase, matching what Stripe sends and expects ("usd").
    currency: Mapped[str] = mapped_column(String(3), default="usd")
    # INTEGER, spelled out rather than inferred, because this is the one column
    # in the schema where the type is the whole point. Minor units: $89.99 is
    # 8999. See the module docstring.
    amount_cents: Mapped[int] = mapped_column(Integer)
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    variant: Mapped[Variant] = relationship(back_populates="prices")


class Inventory(Base):
    """Stock for one variant.

    `variant_id` is both primary and foreign key: one stock row per variant, a
    constraint the database enforces rather than the application remembering.
    `reserved` is what a pending checkout holds — available stock is
    `quantity - reserved`, which is what D9's `check_stock` reports.
    """

    __tablename__ = "inventory"
    __table_args__ = (
        CheckConstraint("quantity >= 0", name="ck_inventory_quantity_positive"),
        CheckConstraint("reserved >= 0", name="ck_inventory_reserved_positive"),
        # Reserved units are a subset of the units on hand, so `quantity -
        # reserved` is what is left to sell. Allowing reserved past quantity
        # made that difference negative, and a negative `available` travels
        # straight through `check_stock` into the guardrail D9 builds on top
        # of it.
        CheckConstraint(
            "reserved <= quantity", name="ck_inventory_reserved_within_quantity"
        ),
    )

    variant_id: Mapped[int] = mapped_column(
        ForeignKey("variants.id", ondelete="CASCADE"), primary_key=True
    )
    quantity: Mapped[int] = mapped_column(Integer, default=0)
    reserved: Mapped[int] = mapped_column(Integer, default=0)

    variant: Mapped[Variant] = relationship(back_populates="inventory")


# Declared out here rather than in `__table_args__` because it indexes an
# expression rather than a column, and the expression needs the mapped
# attribute to build. `search.py` compares `lower(products.category)`; an index
# on the raw column cannot serve that, which EXPLAIN confirms by falling back
# to a sequential scan.
Index("ix_products_category_lower", func.lower(Product.category))
