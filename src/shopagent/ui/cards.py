"""How a search result is laid out on the page (D11, step 4).

**The cards are the raw search result and are never filtered against what the
model wrote.** The temptation is real — the step 2 screenshot shows five
products with three or four variants each while the model's prose named two —
and it is refused, for the reason D9 refused an instruction in place of a
guardrail: matching a product name as a substring of the model's answer makes
the page's content depend on whether the model happened to spell it the same
way. A shop that shows what it found, and a model that recommends two of them,
are two honest things. A page that silently hides what the model did not
mention is one of them lying.

So the problem is density, not content, and this file is the answer to
density. Two rules:

1. **Variants collapse by colour.** Three sizes of one black shoe at one price
   is one line, not three: `black · 41, 42, 43 — €94.99`. What the reader is
   scanning for is which colours exist and roughly what it costs, and the sizes
   are the detail inside that.
2. **A size that cannot be bought stays visible and is marked.** Dropping it
   would answer "do you have 42?" by omission, which reads as "there is no 42"
   rather than "the 42 is gone" — and the second is the sentence a shopper
   needs. It is struck through with the reason spelled out.

A colour with two prices is two rows, not one. That happens whenever a
variant is priced differently from its siblings, and merging them would put one
number against stock it does not cover — the same reason `prices` has one
active row per currency per variant rather than one per product.
"""

from __future__ import annotations

from dataclasses import dataclass

from shopagent.ui.session import ProductCard, VariantCard

# How many results are drawn open before the rest go behind a fold. Two,
# because two is what the model itself typically recommends out of a search of
# five, and because a conversation whose last message is four screens tall is
# one where the next question is off the bottom of the page. The rest are one
# click away and nothing is hidden.
CARDS_SHOWN = 2


@dataclass(frozen=True)
class VariantGroup:
    """Every size of one product in one colour at one price."""

    color: str | None
    price: str
    # In the order the catalog returned them, which is the order the search
    # ranked them. Not sorted here: `catalog/search.py` decides order, and a
    # second opinion about it in a renderer is how two orderings start
    # disagreeing.
    sizes: tuple[str, ...]
    sold_out: tuple[str, ...]

    @property
    def all_sold_out(self) -> bool:
        """Whether nothing in this colour can be bought at all."""
        return not self.sizes and bool(self.sold_out)

    @property
    def label(self) -> str:
        """The sizes, as one readable run. Empty when the variant has none."""
        return ", ".join(self.sizes)


def group_variants(variants: tuple[VariantCard, ...]) -> tuple[VariantGroup, ...]:
    """Collapse a product's variants into one row per colour and price.

    Keyed on both, because a colour with two prices is two facts. Keyed in
    first-seen order rather than sorted, so the row order still reflects what
    the catalog returned.

    A variant with no size — a water bottle, and two rows in the seed are like
    that — contributes an empty `sizes` and is still a group, so the colour and
    the price are drawn. A product with no colours at all collapses to a single
    group whose `color` is `None`, which is what a renderer needs to know to
    draw no swatch.
    """
    grouped: dict[tuple[str | None, str], tuple[list[str], list[str]]] = {}
    for variant in variants:
        key = (variant.color, variant.price)
        in_stock, sold_out = grouped.setdefault(key, ([], []))
        if variant.size is None:
            continue
        (in_stock if variant.in_stock else sold_out).append(variant.size)

    return tuple(
        VariantGroup(
            color=color,
            price=price,
            sizes=tuple(available),
            sold_out=tuple(gone),
        )
        for (color, price), (available, gone) in grouped.items()
    )


def split_for_display(
    cards: tuple[ProductCard, ...],
) -> tuple[tuple[ProductCard, ...], tuple[ProductCard, ...]]:
    """The results drawn open, and the ones behind a fold.

    Nothing is dropped: the second tuple is what a renderer puts in an
    expander, and its length is what the expander's label has to say out loud.
    A fold that did not name its own size would read as the end of the list.
    """
    return cards[:CARDS_SHOWN], cards[CARDS_SHOWN:]
