"""A variant's colour name, as something a browser can draw (D11, step 2).

`Variant.color` is a free string — `catalog/models.py` declares it
`String(40)`, nullable, and `catalog/seed.py` is the only writer. There is no
hex anywhere in the schema and no image column either, so a product card in the
browser is text plus one drawn square, and this is where that square gets its
colour.

**The map covers exactly what the catalog sells, and a test holds it in both
directions.** Coverage stops an unswatched colour reaching the browser as a
grey nothing. The reverse half — no name here that the catalog does not carry —
exists because a colour map is the easiest file in a project to grow into a
speculative list of thirty names nobody checked, at which point it is thirty
untested branches and the one colour that actually arrived is still missing.
The same argument D9 made for pinning the whole tool list rather than the tools
a test happened to care about.

**A swatch is decoration over a name that is always shown.** The renderer
prints `variant.color` as text beside the square, so an unknown colour is
readable rather than mysterious — which is what makes a neutral fallback an
acceptable answer instead of a silent wrong one.

**Only a hex from this file is ever interpolated into markup.** The card
renderer builds its swatch with `unsafe_allow_html`, and the one value that
goes in comes from `SWATCHES` — never a product name, a colour name, or
anything else the catalog holds. That is the whole reason this returns a hex
rather than a fragment of HTML: a function that returned markup would be a
place for catalog text to end up inside it. `test_ui_colors.py` asserts every
value is a six-digit hex, so the claim is mechanical rather than a habit.
"""

from __future__ import annotations

import re

# Six digits, lowercase, always with the leading `#`. Checked by test rather
# than by convention, because these are the only strings in this project that
# reach a browser as markup.
HEX = re.compile(r"^#[0-9a-f]{6}$")

# What the fifteen colours in `catalog/seed.py` look like. Chosen to read as
# the material rather than as the pure name — a running shoe's "olive" is a
# muted drab, not `#808000` — and kept dark enough to sit on a light page and
# light enough to sit on a dark one, since a Streamlit theme can be either.
SWATCHES: dict[str, str] = {
    "black": "#1c1c1e",
    "blue": "#2f6fd0",
    "brown": "#6b4a2f",
    "charcoal": "#40444a",
    "green": "#2f7d4f",
    "grey": "#8b9096",
    "mustard": "#c8971f",
    "navy": "#1f2f57",
    "olive": "#6b6c3a",
    "orange": "#d8712a",
    "red": "#c0392b",
    "sand": "#cbb08a",
    "silver": "#b9bec4",
    "white": "#f2f2f0",
    "yellow": "#e3c033",
}

# What an unnamed colour gets. Deliberately a neutral that belongs to no
# product: a fallback that looked like one of the fifteen would say something
# false about the item rather than saying nothing about it.
FALLBACK = "#9aa0a6"

# Colours light enough that a square of them is invisible on a light page.
# They get a border rather than a different fill, because changing the fill
# would be answering "what colour is this shoe" with a lie about the shoe.
NEEDS_OUTLINE = frozenset({"white", "silver", "sand", "yellow"})


def normalise(color: str | None) -> str | None:
    """A colour name in the one form this module compares.

    `catalog/search.py` already lowercases and strips when it filters on
    colour, so this is the same reading of the same free string rather than a
    second opinion about it.
    """
    if color is None:
        return None
    cleaned = color.strip().lower()
    return cleaned or None


def swatch(color: str | None) -> str | None:
    """The hex to draw for one variant's colour.

    Three answers, and the difference between the first two matters:

    - `None` for a variant that *has* no colour. Two rows in the seed are like
      that on purpose — a water bottle has no colour and no size — and the card
      should draw nothing rather than draw a question mark.
    - `FALLBACK` for a colour this map has not been taught. Something is drawn,
      because the name is printed beside it either way.
    - The mapped hex otherwise.
    """
    cleaned = normalise(color)
    if cleaned is None:
        return None
    return SWATCHES.get(cleaned, FALLBACK)


def needs_outline(color: str | None) -> bool:
    """Whether this swatch would disappear into a light page without a border."""
    cleaned = normalise(color)
    # An unknown colour is drawn in the fallback grey, which needs no help.
    return cleaned in NEEDS_OUTLINE
