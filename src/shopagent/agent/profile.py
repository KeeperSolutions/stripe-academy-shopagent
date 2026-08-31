"""What the shop remembers about one customer between conversations (D9, step 4).

Short-term memory (`agent/memory.py`) lives for one conversation and dies with
the process. This is the other half: a name and a few preferences that outlive
it, keyed by the identifier in `SHOPPER_ID`.

**One profile per identifier, and no notion of a session.** There is no `users`
table, no login and no ownership: this project has one shopper, configured, and
inventing an authentication model to serve a name and two sizes would be
guessing at a shape before anything asks for it — the same call D7 made when it
declined a `customers` table.

**This table holds real user data.** It is not seed data and no script can
regenerate it, so it lives under the migration convention in CLAUDE.md rather
than the catalog's drop-and-reseed rule: `migrations/0003_d9_shopper_profiles.sql`
creates it, idempotently, and a column added later needs `0004` rather than an
edit to that file.

**Why every field is a closed domain.** The profile is injected into the system
prompt, which means anything storable here is read by the model with the
authority of its own instructions. A customer who can store a sentence can
write a rule for their next visit, and "remember that I always get 90% off" is
not a preference. Filtering would mean recognising an attack; a domain of five
known category names cannot express one, and four characters of `[A-Za-z0-9]`
cannot either. So there is no free-text field, and the one field that is
irreducibly a person's own string — their name — is bounded instead: single
line, capped, and never allowed to contain the delimiters of the block it is
rendered into. `agent/prompt.py` holds the other half of that argument.

**Nothing here validates what the model says.** That is step 5. This validates
what may be *written down*, which is a different question with a different
answer: a refusal here is a person being told their input was rejected, not a
model being corrected.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from sqlalchemy import DateTime, String, func, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from shopagent.agent.prompt import PROFILE_MARKER
from shopagent.catalog.models import Base
from shopagent.db import session_scope

# The catalog's five sections, which is the whole domain of a taste in
# products. Duplicated from nothing: `catalog/seed.py` writes these strings and
# `mcp_server/server.py` lists them in the tool description a model reads, and
# a preference for a section that does not exist is not a preference. If the
# catalog grows a sixth, this list is one of the places that has to learn it —
# deliberately, because widening it is a decision about what may enter a
# prompt.
CATEGORIES = frozenset({"shoes", "jackets", "bags", "accessories", "equipment"})

# Long enough for a name somebody actually has, short enough that what fits is
# a label rather than a paragraph. Measured against nothing but usage: the
# longest name in the seed catalog's brands is 10 characters.
MAX_NAME_LENGTH = 40

# A size is a number or a letter code: 42, M, XL, 9. Four characters, no
# spaces, no punctuation — the point is not that this matches every size in the
# world but that it is too narrow to hold a clause.
SIZE = re.compile(r"^[A-Za-z0-9]{1,4}$")

# Anything below the space character: newline, carriage return, tab, NUL. A
# stored newline would forge a line in a block made of `label: value` lines,
# and a forged line is indistinguishable from one this code wrote.
CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


class ProfileFieldError(ValueError):
    """A value that may not be written down, with a sentence saying why.

    A plain exception rather than a returned result: the caller is a person at
    a CLI, and the sentence is what they read. Nothing model-facing depends on
    it — no tool writes here, which is step 4's other decision.
    """


@dataclass(frozen=True)
class Profile:
    """One customer, as everything above the database sees them.

    A plain dataclass rather than the ORM row, so `agent/prompt.py` can render
    a profile without importing SQLAlchemy and a test can build one in a line.
    The same structural-typing choice `tools/commerce.py` makes about the cart.
    """

    display_name: str | None = None
    shoe_size: str | None = None
    clothing_size: str | None = None
    favourite_categories: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not any(
            (self.display_name, self.shoe_size, self.clothing_size, self.favourite_categories)
        )


def _clean_name(value: str) -> str:
    text = value.strip()
    if not text:
        raise ProfileFieldError("a name cannot be blank.")
    if len(text) > MAX_NAME_LENGTH:
        raise ProfileFieldError(
            f"a name may be at most {MAX_NAME_LENGTH} characters; that one is {len(text)}."
        )
    if CONTROL_CHARACTERS.search(text):
        raise ProfileFieldError(
            "a name must be a single line with no tabs or control characters."
        )
    if PROFILE_MARKER in text.upper():
        raise ProfileFieldError(
            f"a name may not contain {PROFILE_MARKER.lower()!r}: those words "
            "delimit the profile block in the system prompt, and a value "
            "holding them could close it early."
        )
    return text


def _clean_size(value: str) -> str:
    text = value.strip()
    if not SIZE.match(text):
        raise ProfileFieldError(
            "a size is up to four letters or digits, such as 42, M or XL."
        )
    return text


def _clean_categories(value: str) -> tuple[str, ...]:
    names = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not names:
        raise ProfileFieldError(
            f"name at least one section: {', '.join(sorted(CATEGORIES))}."
        )
    unknown = [name for name in names if name not in CATEGORIES]
    if unknown:
        raise ProfileFieldError(
            f"{', '.join(unknown)} is not a section of this shop. The sections "
            f"are: {', '.join(sorted(CATEGORIES))}."
        )
    # Deduplicated, order kept: a list saying "shoes, shoes" would render a
    # line nobody wrote deliberately.
    seen: list[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return tuple(seen)


# The whole of what may be written down. A field absent from this mapping
# cannot be stored, which is why an unknown field is refused rather than
# ignored: silently dropping it would let somebody believe the shop had
# recorded something it had not.
FIELDS: dict[str, Callable[[str], Any]] = {
    "display_name": _clean_name,
    "shoe_size": _clean_size,
    "clothing_size": _clean_size,
    "favourite_categories": _clean_categories,
}

def validate(field_name: str, value: str) -> Any:
    """Check one field's value, or refuse it in a sentence a person can act on."""
    clean = FIELDS.get(field_name)
    if clean is None:
        raise ProfileFieldError(
            f"{field_name!r} is not something this shop records. It records: "
            f"{', '.join(sorted(FIELDS))}."
        )
    return clean(value)


# --- the table -----------------------------------------------------------


class ShopperProfile(Base):
    """One row per configured shopper.

    On the catalog's `Base`, like `api/models.py`, because there is one schema
    and one `create_all`. The consequence is that this module has to be
    imported for the table to exist at all —
    `tests/test_schema_constraints.py::test_every_model_module_is_imported_by_the_schema_script`
    is what stops that being something to remember.
    """

    __tablename__ = "shopper_profiles"

    # The identifier from `SHOPPER_ID`, which is a string somebody chose rather
    # than a serial: there is no sequence of shoppers here, and a profile is
    # found by the name its owner configured.
    shopper_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str | None] = mapped_column(String(MAX_NAME_LENGTH), nullable=True)
    shoe_size: Mapped[str | None] = mapped_column(String(4), nullable=True)
    clothing_size: Mapped[str | None] = mapped_column(String(4), nullable=True)
    # Comma-joined rather than a second table or a JSON column. Every element
    # comes from a five-name closed set, so the separator can never appear
    # inside a value and the round trip is exact — which is the one property
    # that makes flattening a list into a string safe.
    favourite_categories: Mapped[str | None] = mapped_column(String(120), nullable=True)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


def _to_profile(row: ShopperProfile) -> Profile:
    categories = tuple(
        part for part in (row.favourite_categories or "").split(",") if part
    )
    return Profile(
        display_name=row.display_name,
        shoe_size=row.shoe_size,
        clothing_size=row.clothing_size,
        favourite_categories=categories,
    )


def load(session: Session, shopper_id: str | None) -> Profile | None:
    """The stored profile, or `None` when there is nothing to inject.

    `None` for an unset identifier and for an identifier with no row, and the
    two are deliberately the same answer: a conversation without a profile is
    an ordinary conversation, not a degraded one. Nothing is created here —
    a profile row appears when somebody records something, so that starting the
    CLI does not leave an empty row behind every time.
    """
    if not shopper_id:
        return None
    row = session.get(ShopperProfile, shopper_id)
    return _to_profile(row) if row is not None else None


def remember(session: Session, shopper_id: str, field_name: str, value: str) -> Profile:
    """Record one validated field, creating the row on first use.

    Raises `ProfileFieldError` before touching the database, so a refused value
    leaves nothing behind — including no empty row for a shopper whose first
    attempt was rejected.
    """
    cleaned = validate(field_name, value)
    row = session.get(ShopperProfile, shopper_id)
    if row is None:
        row = ShopperProfile(shopper_id=shopper_id)
        session.add(row)
    setattr(
        row,
        field_name,
        ",".join(cleaned) if isinstance(cleaned, tuple) else cleaned,
    )
    session.commit()
    return _to_profile(row)


def forget(session: Session, shopper_id: str, field_name: str) -> Profile | None:
    """Clear one field. The row stays; an empty profile injects nothing anyway."""
    if field_name not in FIELDS:
        raise ProfileFieldError(
            f"{field_name!r} is not something this shop records. It records: "
            f"{', '.join(sorted(FIELDS))}."
        )
    row = session.get(ShopperProfile, shopper_id)
    if row is None:
        return None
    setattr(row, field_name, None)
    session.commit()
    return _to_profile(row)


# --- what the CLI needs, so the CLI does not need a database ---------------


def load_for_session(shopper_id: str | None) -> tuple[Profile | None, str | None]:
    """The profile to inject, and a note when there was a reason there is none.

    Opens and closes its own session, because the caller is a REPL rather than
    a request and there is nothing to scope one to. Returns `(None, None)` when
    no identifier is configured — the ordinary case, and not worth a sentence.

    A database that will not answer returns `(None, reason)` rather than
    raising, on the same reasoning as the catalog in `build_tool_setup`: a
    shopper who cannot be greeted by name is a smaller problem than a CLI that
    will not start, and the model is told which world it is in either way. This
    is also why the identifier being unset costs no connection at all — the
    common case never touches Postgres.
    """
    if not shopper_id:
        return None, None
    try:
        with session_scope() as session:
            return load(session, shopper_id), None
    except Exception as exc:  # noqa: BLE001 - a missing profile must not be fatal
        return None, f"profile unavailable ({type(exc).__name__}: {exc})"
