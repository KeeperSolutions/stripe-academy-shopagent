"""Seed data for the catalog (D3, step 2).

Thirty products, written by hand and fully deterministic: no random, no
generator, no faker. Steps 3 and 4 assert against these exact rows, and a
catalog that differs between runs would turn every one of those assertions into
a coin toss.

The descriptions are the part that matters most. Step 4 embeds
`name + brand + category + description`, so this prose *is* the search index —
lorem ipsum would produce vectors that cluster on nothing. They read like shop
copy because that is the register the embedding model has seen most of.

One word is deliberately absent from this entire file, comments included, and a
test in `tests/test_seed.py` enforces its absence: the letters r-a-i-n. The D3
definition of done is that "something to run in when it's wet through" finds the
weatherproof shoes without the query word appearing in any description. If the
copy said so literally, a `LIKE` would find them too and the semantic proof
would be worth nothing. So the shoes talk about membranes, spray, puddles and
grip on slick stone, and let the embedding do the joining.

The test checks for those letters as a substring, which also rules out a set of
otherwise innocent words: the usual term for what you do at the gym, the usual
word for rough ground underfoot, and what a sink does with water. Every one of
them contains the forbidden four. If a description here reads as though it went
the long way round to avoid a word, that is why.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from shopagent.catalog.models import Inventory, Price, Product, Variant
from shopagent.config import get_settings


@dataclass(frozen=True)
class VariantSpec:
    """One buyable configuration, with its price and its stock."""

    sku: str
    amount_cents: int
    quantity: int
    size: str | None = None
    color: str | None = None
    reserved: int = 0
    # An earlier price, kept as an inactive row rather than overwritten. Only
    # one product carries one; it exists so `active` has something to filter.
    previous_amount_cents: int | None = None


@dataclass(frozen=True)
class ProductSpec:
    """A product and every variant of it."""

    name: str
    brand: str
    category: str
    description: str
    variants: tuple[VariantSpec, ...]


# Categories are lowercase and single-word wherever possible: D9 fills
# `ProductQuery.category` from a model's structured output, and every extra
# spelling is another value the model can plausibly invent.
CATALOG: tuple[ProductSpec, ...] = (
    # --- shoes ---------------------------------------------------------
    ProductSpec(
        name="Trail Runner GTX",
        brand="Fleetfoot",
        category="shoes",
        description=(
            "A GORE-TEX lined running shoe for the grey months. The membrane "
            "keeps spray out of the sock liner while the lugged outsole bites "
            "into mud and slick stone."
        ),
        variants=(
            VariantSpec("FF-TRLGTX-41-BLK", 9499, 8, "41", "black"),
            VariantSpec("FF-TRLGTX-42-BLK", 9499, 12, "42", "black", reserved=2),
            VariantSpec("FF-TRLGTX-43-BLK", 9499, 5, "43", "black"),
            VariantSpec("FF-TRLGTX-42-OLV", 9499, 3, "42", "olive"),
        ),
    ),
    ProductSpec(
        name="Storm Pace 4",
        brand="Fleetfoot",
        category="shoes",
        description=(
            "A road running shoe built for wet mornings and standing water. The "
            "coated knit upper sheds splashes instead of soaking them up, and the "
            "sticky outsole holds its line through puddles and on greasy pavement."
        ),
        variants=(
            VariantSpec("FF-STRMP4-41-NVY", 8999, 6, "41", "navy"),
            VariantSpec("FF-STRMP4-42-NVY", 8999, 9, "42", "navy"),
            # Out of stock: D9 has to refuse a checkout on this one.
            VariantSpec("FF-STRMP4-43-NVY", 8999, 0, "43", "navy"),
        ),
    ),
    ProductSpec(
        name="Cloud Sprint 2",
        brand="Aerostep",
        category="shoes",
        description=(
            "A light, fast road running shoe for dry days and easy miles. "
            "Breathable mesh keeps the weight down, which is exactly why it "
            "offers no weather protection at all."
        ),
        variants=(
            VariantSpec("AE-CLDSP2-41-WHT", 7499, 11, "41", "white"),
            # The one variant with a price history: it used to cost 8999.
            VariantSpec(
                "AE-CLDSP2-42-WHT", 7499, 14, "42", "white", previous_amount_cents=8999
            ),
            VariantSpec("AE-CLDSP2-43-WHT", 7499, 7, "43", "white"),
        ),
    ),
    ProductSpec(
        name="Summit Peak Pro",
        brand="Northridge",
        category="shoes",
        description=(
            "Our flagship long-distance running shoe. A carbon plate under a "
            "weather-sealed upper, built for hard efforts in conditions that "
            "would ruin a mesh racer."
        ),
        variants=(
            VariantSpec("NR-SMTPRO-42-CHR", 14999, 4, "42", "charcoal"),
            VariantSpec("NR-SMTPRO-43-CHR", 14999, 2, "43", "charcoal"),
        ),
    ),
    ProductSpec(
        name="Ridge Hiker Mid",
        brand="Northridge",
        category="shoes",
        description=(
            "A mid-cut hiking boot with a waterproof membrane and a deep lug "
            "pattern. Made for churned-up paths, wet rock and stream crossings "
            "that come up over the ankle."
        ),
        variants=(
            VariantSpec("NR-RDGHKM-41-BRN", 12999, 5, "41", "brown"),
            VariantSpec("NR-RDGHKM-42-BRN", 12999, 0, "42", "brown"),
            VariantSpec("NR-RDGHKM-43-BRN", 12999, 6, "43", "brown"),
        ),
    ),
    ProductSpec(
        name="City Walker",
        brand="Ambleside",
        category="shoes",
        description=(
            "A clean leather sneaker for the commute and the weekend. "
            "Cushioned insole, understated stitching, wipes clean with a damp "
            "cloth."
        ),
        variants=(
            VariantSpec("AM-CTYWLK-42-WHT", 6999, 10, "42", "white"),
            VariantSpec("AM-CTYWLK-43-WHT", 6999, 8, "43", "white"),
        ),
    ),
    ProductSpec(
        name="Studio Flex",
        brand="Aerostep",
        category="shoes",
        description=(
            "A flat, flexible shoe for lifting and floor work indoors. Wide "
            "toe box, grippy sole, none of the stack height that makes a road "
            "shoe wobble under load."
        ),
        variants=(
            VariantSpec("AE-STDFLX-42-GRY", 6499, 9, "42", "grey"),
            VariantSpec("AE-STDFLX-44-GRY", 6499, 4, "44", "grey"),
        ),
    ),
    ProductSpec(
        name="Harbor Slip-On",
        brand="Ambleside",
        category="shoes",
        description=(
            "A canvas slip-on for warm, dry days. Elastic side panels, a "
            "washable footbed, and it packs flat into the side of a bag."
        ),
        variants=(
            VariantSpec("AM-HRBSLP-42-SND", 4999, 12, "42", "sand"),
            VariantSpec("AM-HRBSLP-43-SND", 4999, 5, "43", "sand"),
        ),
    ),
    # --- jackets -------------------------------------------------------
    ProductSpec(
        name="Harbor Windshell",
        brand="Northridge",
        category="jackets",
        description=(
            "A light windshell with a water-repellent finish that turns away a "
            "passing shower. Packs into its own chest pocket and weighs less "
            "than the phone in the other one."
        ),
        variants=(
            VariantSpec("NR-HRBWND-S-BLU", 7499, 7, "S", "blue"),
            VariantSpec("NR-HRBWND-M-BLU", 7499, 9, "M", "blue"),
            VariantSpec("NR-HRBWND-L-BLU", 7499, 0, "L", "blue"),
            VariantSpec("NR-HRBWND-M-BLK", 7499, 6, "M", "black"),
        ),
    ),
    ProductSpec(
        name="Storm Guard Shell",
        brand="Northridge",
        category="jackets",
        description=(
            "A three-layer GORE-TEX hardshell for a full day of foul weather. "
            "Taped seams throughout, underarm zips to dump heat on the climb, "
            "and a stiffened hood brim that keeps water off your face."
        ),
        variants=(
            VariantSpec("NR-STMGRD-M-RED", 19999, 4, "M", "red"),
            VariantSpec("NR-STMGRD-L-RED", 19999, 3, "L", "red"),
        ),
    ),
    ProductSpec(
        name="Alpine Down Parka",
        brand="Northridge",
        category="jackets",
        description=(
            "800-fill down under a weather-resistant shell, cut long for "
            "standing still in the cold. The hood fits over a helmet."
        ),
        variants=(
            VariantSpec("NR-ALPDWN-M-BLK", 24999, 3, "M", "black"),
            VariantSpec("NR-ALPDWN-L-BLK", 24999, 2, "L", "black"),
        ),
    ),
    ProductSpec(
        name="Grid Fleece Midlayer",
        brand="Ambleside",
        category="jackets",
        description=(
            "A grid fleece that holds warmth without bulk. Wears alone on a "
            "cool morning and slides under a shell when the sky turns."
        ),
        variants=(
            VariantSpec("AM-GRDFLC-M-GRY", 5999, 11, "M", "grey"),
            VariantSpec("AM-GRDFLC-L-GRY", 5999, 8, "L", "grey"),
        ),
    ),
    ProductSpec(
        name="Commuter Softshell",
        brand="Aerostep",
        category="jackets",
        description=(
            "A stretch softshell with a durable water-repellent face fabric: "
            "enough cover for a damp ride into the office, breathable enough "
            "that you are not soaked from the inside on the way home."
        ),
        variants=(
            VariantSpec("AE-CMTSFT-M-OLV", 8999, 6, "M", "olive"),
            VariantSpec("AE-CMTSFT-L-OLV", 8999, 5, "L", "olive"),
        ),
    ),
    ProductSpec(
        name="Packable Wind Jacket",
        brand="Fleetfoot",
        category="jackets",
        description=(
            "Forty grams of ripstop that stops the wind and shrugs off "
            "drizzle. Stuffs into a fist-sized pouch you can belt to your "
            "waist."
        ),
        variants=(
            VariantSpec("FF-PCKWND-S-BLU", 4999, 9, "S", "blue"),
            VariantSpec("FF-PCKWND-M-BLU", 4999, 12, "M", "blue"),
        ),
    ),
    # --- bags ----------------------------------------------------------
    ProductSpec(
        name="Daypack 22L",
        brand="Northridge",
        category="bags",
        description=(
            "A 22-litre pack for a day on the hill, with a floating lid, side "
            "compression straps and a hip pocket sized for a flask."
        ),
        variants=(
            VariantSpec("NR-DYPK22-NA-GRN", 6999, 8, None, "green"),
            VariantSpec("NR-DYPK22-NA-BLK", 6999, 10, None, "black"),
        ),
    ),
    ProductSpec(
        name="Commuter Backpack 18L",
        brand="Aerostep",
        category="bags",
        description=(
            "An 18-litre pack with a padded 15-inch laptop sleeve and a coated "
            "base, so setting it down on a wet bench costs you nothing."
        ),
        variants=(VariantSpec("AE-CMTBP18-NA-BLK", 8499, 14, None, "black"),),
    ),
    ProductSpec(
        name="Dry Duffel 40L",
        brand="Northridge",
        category="bags",
        description=(
            "A welded 40-litre duffel with a roll-top closure. Everything "
            "inside stays dry through a crossing on an open deck or a night "
            "strapped to a roof rack."
        ),
        variants=(VariantSpec("NR-DRYDF40-NA-YLW", 9999, 0, None, "yellow"),),
    ),
    ProductSpec(
        name="Hip Pack 3L",
        brand="Fleetfoot",
        category="bags",
        description=(
            "A three-litre hip pack that sits still while you move. Holds a "
            "phone, keys, a few gels and a soft flask without swinging."
        ),
        variants=(
            VariantSpec("FF-HIPPK3-NA-BLK", 3499, 15, None, "black"),
            VariantSpec("FF-HIPPK3-NA-BLU", 3499, 7, None, "blue"),
        ),
    ),
    ProductSpec(
        name="Weekender Holdall 45L",
        brand="Ambleside",
        category="bags",
        description=(
            "A 45-litre waxed canvas holdall with leather grab handles and a "
            "vented shoe compartment at one end. Sized to go in the overhead "
            "locker."
        ),
        variants=(VariantSpec("AM-WKNDHL45-NA-BRN", 12999, 4, None, "brown"),),
    ),
    # --- accessories ---------------------------------------------------
    ProductSpec(
        name="Merino Beanie",
        brand="Ambleside",
        category="accessories",
        description=(
            "A fine-knit merino beanie that fits under a hood without bunching "
            "and stays warm when it is damp."
        ),
        variants=(
            VariantSpec("AM-MRNBNE-NA-CHR", 2499, 18, None, "charcoal"),
            VariantSpec("AM-MRNBNE-NA-MUS", 2499, 9, None, "mustard"),
        ),
    ),
    ProductSpec(
        name="Runner's Cap",
        brand="Fleetfoot",
        category="accessories",
        description=(
            "A five-panel cap with a water-repellent brim and a mesh back. "
            "Keeps spray and low winter sun out of your eyes."
        ),
        variants=(VariantSpec("FF-RNRCAP-NA-BLK", 2999, 22, None, "black"),),
    ),
    ProductSpec(
        name="Merino Crew Socks",
        brand="Cobbleway",
        category="accessories",
        description=(
            "Merino crew socks with a cushioned heel and a flat toe seam. Warm "
            "even when they are wet, which wool manages and cotton does not."
        ),
        variants=(
            VariantSpec("CW-MRNCRW-S-GRY", 1899, 20, "S", "grey"),
            VariantSpec("CW-MRNCRW-M-GRY", 1899, 25, "M", "grey"),
            VariantSpec("CW-MRNCRW-L-GRY", 1899, 16, "L", "grey"),
        ),
    ),
    ProductSpec(
        name="Reflective Armband",
        brand="Fleetfoot",
        category="accessories",
        description=(
            "A snap-on reflective band for dark mornings and darker evenings. "
            "Catches headlights from a couple of hundred metres back."
        ),
        # No size, no colour: both columns are nullable and this is why.
        variants=(VariantSpec("FF-RFLARM-NA-NA", 1299, 30),),
    ),
    ProductSpec(
        name="Insulated Water Bottle 750ml",
        brand="Cobbleway",
        category="accessories",
        description=(
            "A double-walled steel bottle that holds heat for twelve hours and "
            "cold for twenty-four. The lid seals tight enough to go in a bag "
            "lid-down."
        ),
        variants=(VariantSpec("CW-INSBTL750-NA-NA", 3299, 26),),
    ),
    ProductSpec(
        name="Touchscreen Gloves",
        brand="Aerostep",
        category="accessories",
        description=(
            "Thin liner gloves with conductive fingertips, so the phone still "
            "works without taking them off. Silicone dots on the palm hold a "
            "grip on damp handlebars."
        ),
        variants=(
            VariantSpec("AE-TCHGLV-M-BLK", 2299, 13, "M", "black"),
            VariantSpec("AE-TCHGLV-L-BLK", 2299, 10, "L", "black"),
        ),
    ),
    # --- equipment -----------------------------------------------------
    ProductSpec(
        name="Trekking Poles",
        brand="Northridge",
        category="equipment",
        description=(
            "A pair of three-section aluminium poles with cork grips and "
            "flick locks. They take the load off your knees on a long descent."
        ),
        variants=(VariantSpec("NR-TRKPOL-NA-SLV", 7999, 9, None, "silver"),),
    ),
    ProductSpec(
        name="Headlamp 400lm",
        brand="Cobbleway",
        category="equipment",
        description=(
            "A 400-lumen headlamp with a sealed housing and a red night mode. "
            "Rechargeable over USB-C, and it survives a downpour."
        ),
        variants=(VariantSpec("CW-HDLMP400-NA-BLK", 4499, 17, None, "black"),),
    ),
    ProductSpec(
        name="Hydration Vest 5L",
        brand="Fleetfoot",
        category="equipment",
        description=(
            "A five-litre vest that carries two soft flasks up front and a "
            "shell in the back pocket. Sits close enough not to bounce over "
            "hours of running."
        ),
        variants=(
            VariantSpec("FF-HYDVST5-S-BLK", 11999, 5, "S", "black"),
            VariantSpec("FF-HYDVST5-M-BLK", 11999, 6, "M", "black"),
            VariantSpec("FF-HYDVST5-L-BLK", 11999, 0, "L", "black"),
        ),
    ),
    ProductSpec(
        name="Foam Roller",
        brand="Cobbleway",
        category="equipment",
        description=(
            "A firm 45 cm foam roller with a moulded surface. For calves and "
            "quads on the days after a long effort."
        ),
        variants=(VariantSpec("CW-FMROLL45-NA-BLU", 2799, 12, None, "blue"),),
    ),
    ProductSpec(
        name="Compact Dry Bag 10L",
        brand="Northridge",
        category="equipment",
        description=(
            "A 10-litre roll-top sack that keeps a spare layer dry inside a "
            "pack. Weighs almost nothing and doubles as a pillow stuff sack."
        ),
        variants=(
            VariantSpec("NR-DRYBG10-NA-ORG", 1999, 21, None, "orange"),
            VariantSpec("NR-DRYBG10-NA-GRN", 1999, 14, None, "green"),
        ),
    ),
)


@dataclass
class SeedSummary:
    """What one seed run actually wrote."""

    products_created: int = 0
    products_skipped: int = 0
    variants_created: int = 0
    prices_created: int = 0
    inventory_created: int = 0

    def as_lines(self) -> list[str]:
        return [
            f"  products created   {self.products_created}",
            f"  products skipped   {self.products_skipped}",
            f"  variants created   {self.variants_created}",
            f"  prices created     {self.prices_created}",
            f"  inventory rows     {self.inventory_created}",
        ]


def _product_for(session: Session, spec: ProductSpec) -> Product | None:
    """Find the stored product for a spec, through any variant it already has.

    Products have no natural key of their own — two shops can sell a shoe with
    the same name. The sku is unique, so it is the identity used here, and a
    product is "already seeded" when one of its variants is present.
    """
    return session.scalars(
        select(Product)
        .join(Variant)
        .where(Variant.sku.in_([variant.sku for variant in spec.variants]))
        .limit(1)
    ).first()


def seed_catalog(
    session: Session, catalog: tuple[ProductSpec, ...] = CATALOG
) -> SeedSummary:
    """Write every missing row from `catalog` and leave the present ones alone.

    Idempotent through the sku and nothing else: a second run finds every sku
    already stored and writes nothing. Deliberately not "delete everything and
    re-insert" — that would change every primary key on each run, and by D6 a
    cart would be pointing at a variant id that no longer means what it did.

    Rows already in the database are never updated. Editing a description here
    and re-running will not change what is stored; use `--reset`, which is the
    honest way to say "the catalog is disposable".
    """
    currency = get_settings().currency
    stored_skus = set(session.scalars(select(Variant.sku)))
    summary = SeedSummary()

    for spec in catalog:
        missing = tuple(v for v in spec.variants if v.sku not in stored_skus)
        if not missing:
            summary.products_skipped += 1
            continue

        product = _product_for(session, spec)
        if product is None:
            product = Product(
                name=spec.name,
                description=spec.description,
                category=spec.category,
                brand=spec.brand,
            )
            session.add(product)
            summary.products_created += 1

        for spec_variant in missing:
            variant = Variant(
                product=product,
                size=spec_variant.size,
                color=spec_variant.color,
                sku=spec_variant.sku,
            )
            variant.prices.append(
                Price(
                    currency=currency,
                    amount_cents=spec_variant.amount_cents,
                    active=True,
                )
            )
            summary.prices_created += 1
            if spec_variant.previous_amount_cents is not None:
                variant.prices.append(
                    Price(
                        currency=currency,
                        amount_cents=spec_variant.previous_amount_cents,
                        active=False,
                    )
                )
                summary.prices_created += 1
            variant.inventory = Inventory(
                quantity=spec_variant.quantity, reserved=spec_variant.reserved
            )
            summary.variants_created += 1
            summary.inventory_created += 1
            session.add(variant)

    session.commit()
    return summary


def reset_catalog(session: Session) -> int:
    """Delete every product, and with it every variant, price and stock row.

    One statement: the ON DELETE CASCADE on the foreign keys clears the three
    child tables, which is the half of the cascade that D3 step 1 tested.
    """
    deleted = session.execute(delete(Product)).rowcount
    session.commit()
    return deleted
