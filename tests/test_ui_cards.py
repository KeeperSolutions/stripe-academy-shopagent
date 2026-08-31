"""How a search result is laid out (D11, step 4).

Offline, and about arithmetic on a list rather than about pixels: which rows a
product collapses to, and which results go behind the fold.
"""

from __future__ import annotations

from shopagent.ui.cards import CARDS_SHOWN, group_variants, split_for_display
from shopagent.ui.session import ProductCard, VariantCard


def variant(variant_id: int, color: str | None, size: str | None, price: str = "€94.99",
            available: int = 3) -> VariantCard:
    return VariantCard(
        variant_id=variant_id,
        sku=f"SKU-{variant_id}",
        size=size,
        color=color,
        price_cents=9499,
        price=price,
        available=available,
    )


def card(product_id: int, *variants: VariantCard) -> ProductCard:
    return ProductCard(
        product_id=product_id,
        name=f"Product {product_id}",
        brand="Fleetfoot",
        category="shoes",
        description="a shoe",
        variants=variants,
    )


def test_one_colour_in_three_sizes_becomes_one_row():
    """The whole point: three lines of the same shoe became one."""
    groups = group_variants(
        (variant(1, "black", "41"), variant(2, "black", "42"), variant(3, "black", "43"))
    )

    assert len(groups) == 1
    assert groups[0].color == "black"
    assert groups[0].label == "41, 42, 43"
    assert groups[0].price == "€94.99"


def test_two_colours_stay_two_rows():
    groups = group_variants(
        (variant(1, "black", "41"), variant(2, "black", "42"), variant(3, "olive", "42"))
    )

    assert [(g.color, g.label) for g in groups] == [("black", "41, 42"), ("olive", "42")]


def test_one_colour_at_two_prices_stays_two_rows():
    """A colour with two prices is two facts.

    Merging them would put one number against stock it does not cover — the
    same reason `prices` holds one active row per currency per variant rather
    than one per product.
    """
    groups = group_variants(
        (variant(1, "black", "41"), variant(2, "black", "42", price="€109.99"))
    )

    assert [(g.price, g.label) for g in groups] == [("€94.99", "41"), ("€109.99", "42")]


def test_a_sold_out_size_stays_visible_and_is_marked():
    """Dropping it would answer "do you have 42?" by omission.

    "There is no 42" and "the 42 is gone" are different sentences, and the
    second is the one a shopper needs.
    """
    groups = group_variants(
        (
            variant(1, "navy", "41"),
            variant(2, "navy", "42", available=0),
            variant(3, "navy", "43"),
        )
    )

    (group,) = groups
    assert group.label == "41, 43"
    assert group.sold_out == ("42",)
    assert group.all_sold_out is False


def test_a_colour_with_nothing_left_says_so():
    groups = group_variants((variant(1, "navy", "43", available=0),))

    (group,) = groups
    assert group.sizes == ()
    assert group.sold_out == ("43",)
    assert group.all_sold_out is True


def test_a_variant_with_no_size_still_draws_its_colour_and_price():
    """Two rows in the seed have neither size nor colour — a water bottle."""
    groups = group_variants((variant(1, "silver", None),))

    (group,) = groups
    assert group.label == ""
    assert group.color == "silver"
    assert group.all_sold_out is False


def test_a_product_with_no_colour_at_all_is_one_group():
    groups = group_variants((variant(1, None, "41"), variant(2, None, "42")))

    (group,) = groups
    assert group.color is None
    assert group.label == "41, 42"


def test_the_order_the_catalog_returned_is_kept():
    """`catalog/search.py` decides order. A second opinion here is how two
    orderings start disagreeing."""
    groups = group_variants(
        (variant(1, "olive", "43"), variant(2, "black", "41"), variant(3, "olive", "41"))
    )

    assert [g.color for g in groups] == ["olive", "black"]
    assert groups[0].label == "43, 41"


def test_nothing_is_dropped_behind_the_fold():
    cards = tuple(card(n, variant(n, "black", "42")) for n in range(5))

    shown, folded = split_for_display(cards)

    assert len(shown) == CARDS_SHOWN
    assert shown + folded == cards


def test_a_short_result_has_no_fold():
    cards = (card(1, variant(1, "black", "42")),)

    shown, folded = split_for_display(cards)

    assert shown == cards
    assert folded == ()
