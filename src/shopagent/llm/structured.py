"""Turning free text into a validated catalogue query (D2, used from D3).

The user says "running shoes under €100 in size 42"; `catalog/search.py` on D3
wants `keywords`, `category`, `max_price_cents` and `size` as typed fields.
This module is the seam between the two, and it is a real parser rather than a
demonstration — D3 and D9 call it.

Two rules it exists to enforce:

**Money is an integer number of cents, converted exactly once.** "under €100"
becomes `10000`, never `100` and never `100.0`. The rule is stated in the
system prompt below and enforced by the model's own strict field type, so no
other module has to know that dollars were ever involved. A float here is a
rounding bug at checkout on D7.

**A field that cannot be extracted is None, never invented.** `None` means the
user did not say; `0` would mean free, and a guessed category silently narrows
a search the user meant to be broad.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from shopagent.config import get_settings
from shopagent.llm.client import LLMClient, Message
from shopagent.money import format_amount

# Worked examples generated from the shop's currency rather than typed against
# it. These sentences say what a price bound means, and they said "dollars" for
# a week after the shop moved to EUR — a unit the model is taught wrongly is a
# search it runs wrongly, silently. Same idiom as `agent/prompt.py` and
# `mcp_server/server.py`. Raised in review on PR #9.
_CURRENCY = get_settings().currency
_HUNDRED = format_amount(10000, _CURRENCY)
_FORTY_NINE_NINETY_NINE = format_amount(4999, _CURRENCY)
_TWENTY = format_amount(2000, _CURRENCY)

# Only two of these are actually enforced by the API today — a schema missing
# `additionalProperties: false` is rejected with "Invalid schema for
# response_format: In context=(), 'additionalProperties' is required to be
# supplied and to be false", and every property must appear in `required`.
# `title` and `default` were accepted when tried on 2026-08-17, but neither is
# in the documented strict subset, so both are stripped rather than relied on.
_STRIPPED_KEYWORDS = ("title", "default")


class StructuredOutputError(ValueError):
    """The model's answer could not be turned into the requested model."""


class ProductQuery(BaseModel):
    """A catalogue search, extracted from what the user said.

    Every field except `keywords` is optional and means "the user did not say".
    """

    keywords: list[str] = Field(
        description=(
            "The words to search the catalogue with. The search matches on this "
            "list, so an empty list finds nothing. Include the product noun "
            "itself, even when that same word also fills the category field: "
            "'blue jacket under €80' is ['jacket'], not []. When the text "
            "describes what is wanted without naming a product, use the "
            "describing words: 'something warm for winter' is "
            "['warm', 'winter']. Leave out prices, sizes and colours, which "
            "have their own fields. Empty only when the text is not about a "
            "product at all, such as a greeting."
        ),
    )
    category: str | None = Field(
        default=None,
        description=(
            "Product category if the user named one, lowercase and plural, "
            "e.g. 'shoes', 'jackets'. Null if they did not."
        ),
    )
    max_price_cents: int | None = Field(
        default=None,
        strict=True,
        ge=0,
        description=(
            "Upper price bound as a whole number of CENTS, never whole "
            f"{_CURRENCY.upper()} and never a decimal: '{_HUNDRED}' is 10000, "
            f"'{_FORTY_NINE_NINETY_NINE}' is 4999. Null if the user set no "
            "upper bound."
        ),
    )
    min_price_cents: int | None = Field(
        default=None,
        strict=True,
        ge=0,
        description=(
            "Lower price bound as a whole number of CENTS, same rule as "
            "max_price_cents. Null if the user set no lower bound."
        ),
    )
    size: str | None = Field(
        default=None,
        description="Size as the user gave it, e.g. '42', 'M'. Null if unstated.",
    )
    color: str | None = Field(
        default=None,
        description="Colour in English, lowercase, e.g. 'blue'. Null if unstated.",
    )


def _strictify(node: Any) -> Any:
    """Recursively rewrite one JSON Schema node into strict form."""
    if isinstance(node, list):
        return [_strictify(item) for item in node]
    if not isinstance(node, dict):
        return node

    out = {
        key: _strictify(value)
        for key, value in node.items()
        if key not in _STRIPPED_KEYWORDS
    }
    if out.get("type") == "object" and "properties" in out:
        out["additionalProperties"] = False
        # Every property, including the ones Pydantic left out because they
        # have defaults. Under strict there is no such thing as an omitted
        # field; optionality is expressed as a union with null instead, which
        # is what Pydantic already emits for `X | None`.
        out["required"] = list(out["properties"])
    return out


def strict_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """The strict-mode JSON Schema for `model`.

    This is the transform deliberately left out of `tools/registry.py`: tool
    schemas stay non-strict for D2, and doing it in one place for one caller
    beats doing it everywhere for no reason. `$defs` and `$ref` are left alone
    — strict mode supports both, so nested models need no flattening.

    `_strictify` builds new dicts rather than editing in place, so Pydantic's
    own output is left untouched and needs no defensive copy.
    """
    return _strictify(model.model_json_schema())


PRODUCT_QUERY_SCHEMA = strict_schema_for(ProductQuery)

SYSTEM_PROMPT = (
    "You extract a product search from what a shopper says. "
    "Fill only what the text actually states. Any field the text does not "
    "state is null — never guess a category, colour or size, and never use 0 "
    "for a price that was not mentioned, because 0 means free. "
    f"Prices are whole numbers of CENTS, never whole {_CURRENCY.upper()} and "
    f"never decimals: 'under {_HUNDRED}' is max_price_cents 10000, "
    f"'{_FORTY_NINE_NINETY_NINE}' is 4999, "
    f"'at least {_TWENTY}' is min_price_cents 2000."
)


def _messages(text: str) -> list[Message]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def parse_product_query(text: str, client: LLMClient | None = None) -> ProductQuery:
    """Extract a `ProductQuery` from free text.

    Raises StructuredOutputError if the model refuses, or answers with
    something that is not JSON, or with JSON that does not satisfy the model. Strict mode makes both unlikely, not
    impossible: a refusal, a truncated answer or a schema change all land here,
    and a bare JSONDecodeError three frames down says nothing useful about why.
    """
    client = client if client is not None else LLMClient()
    try:
        content, _usage = client.chat_structured(
            _messages(text), PRODUCT_QUERY_SCHEMA, "product_query"
        )
    except ValueError as exc:
        # A refusal surfaces from the client as a plain ValueError. It is one
        # of the failures this function documents, so it leaves as the same
        # type as the others — a caller should not need two except branches
        # for "no usable answer". Anything that is not a ValueError (a
        # transport error, say) is a different problem and travels untouched.
        raise StructuredOutputError(str(exc)) from exc

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(
            f"the model did not return JSON ({exc}). It returned: {content!r}"
        ) from exc

    if not isinstance(payload, dict):
        raise StructuredOutputError(
            f"the model returned a {type(payload).__name__}, not a JSON object. "
            f"It returned: {content!r}"
        )

    try:
        return ProductQuery(**payload)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in err['loc']) or '(root)'}: {err['msg']}"
            for err in exc.errors()
        )
        raise StructuredOutputError(
            f"the model's answer does not fit ProductQuery ({problems}). "
            f"It returned: {content!r}"
        ) from exc
