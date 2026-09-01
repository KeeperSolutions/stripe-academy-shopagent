"""The colour map covers what the shop actually sells (D11, step 2).

Two of these run offline against `catalog/seed.py`, which CLAUDE.md names as
the source of truth for every product row. One runs against the database,
because a seed and a database are two different claims: a catalog seeded
before a colour was added holds rows the seed file no longer describes, and
only the database can say so.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from shopagent.catalog.models import Variant
from shopagent.catalog.seed import CATALOG
from shopagent.ui import colors

SEED_COLORS = {
    colors.normalise(variant.color)
    for product in CATALOG
    for variant in product.variants
    if colors.normalise(variant.color) is not None
}


def test_the_map_is_exactly_the_colours_the_catalog_sells():
    """Both directions, and the second half is the one worth explaining.

    Coverage stops an unswatched colour reaching a browser as a grey nothing.
    The reverse stops this file growing into a speculative list of thirty
    names nobody checked — untested branches for colours no product has, while
    the one that actually arrived is still missing. Same argument D9 made for
    pinning the whole tool list rather than the tools a test cared about.
    """
    assert set(colors.SWATCHES) == SEED_COLORS


def test_no_variant_in_the_seed_falls_back():
    for product in CATALOG:
        for variant in product.variants:
            if variant.color is None:
                assert colors.swatch(variant.color) is None
                continue
            assert colors.swatch(variant.color) != colors.FALLBACK, variant.sku


def test_a_known_colour_maps_to_its_hex():
    assert colors.swatch("black") == colors.SWATCHES["black"]
    assert colors.swatch("BLACK") == colors.SWATCHES["black"]
    assert colors.swatch("  olive  ") == colors.SWATCHES["olive"]


def test_an_unknown_colour_falls_back_rather_than_failing():
    """A card prints the name beside the square, so a fallback is readable
    rather than mysterious — which is what makes it an acceptable answer."""
    assert colors.swatch("fuchsia") == colors.FALLBACK
    assert colors.swatch("Heather Marl") == colors.FALLBACK


def test_a_variant_with_no_colour_gets_no_swatch():
    """Two rows in the seed are like that on purpose: a water bottle has no
    colour and no size, and both columns are nullable because of it."""
    assert colors.swatch(None) is None
    assert colors.swatch("") is None
    assert colors.swatch("   ") is None


def test_every_value_is_a_plain_hex():
    """The one thing that reaches a browser as markup.

    The card renderer interpolates a swatch into HTML with
    `unsafe_allow_html`, and this is why that is safe: the only value that
    goes in comes from here. A product name or a colour name never does.
    """
    for name, value in {**colors.SWATCHES, "fallback": colors.FALLBACK}.items():
        assert colors.HEX.match(value), f"{name} is not a plain hex: {value!r}"


def test_the_pale_colours_are_the_ones_that_get_an_outline():
    """A border rather than a darker fill: changing the fill would answer
    "what colour is this shoe" with a lie about the shoe."""
    assert colors.needs_outline("white") is True
    assert colors.needs_outline("WHITE") is True
    assert colors.needs_outline("black") is False
    assert colors.needs_outline(None) is False
    assert colors.NEEDS_OUTLINE <= set(colors.SWATCHES)


@pytest.mark.db
def test_no_colour_in_the_database_falls_back(session):
    """The seed is the source of truth for what *should* be there; this asks
    the database what *is*. They come apart whenever a catalog was seeded
    before a colour was added, which is the case a browser would meet first.
    """
    stored = {
        colors.normalise(color)
        for (color,) in session.execute(select(Variant.color).distinct())
    }
    stored.discard(None)

    unmapped = sorted(name for name in stored if name not in colors.SWATCHES)

    assert not unmapped, (
        f"the catalog holds colours with no swatch: {unmapped}. Add them to "
        f"shopagent.ui.colors.SWATCHES, or reseed if they are stale rows."
    )
