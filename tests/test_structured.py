"""Tests for shopagent.llm.structured.

No network and no SDK call — the client is a fake that returns whatever JSON
the scenario needs, including JSON the model has no business returning.

Most of this file is about `max_price_cents`. It is the field D3 and D9 will
filter on, and the one place a dollar figure becomes an integer number of
cents. A float that slips through here is a rounding bug at checkout on D7.
"""

import json

import pytest
from pydantic import BaseModel, Field

from shopagent.llm.structured import (
    PRODUCT_QUERY_SCHEMA,
    ProductQuery,
    StructuredOutputError,
    parse_product_query,
    strict_schema_for,
)
from shopagent.llm.usage import CallUsage

FULL = {
    "keywords": ["running", "shoes"],
    "category": "shoes",
    "max_price_cents": 10000,
    "min_price_cents": None,
    "size": "42",
    "color": None,
}


class FakeClient:
    """Returns a prepared response body; records what it was sent."""

    def __init__(self, content):
        self.content = content
        self.seen_messages = None
        self.seen_schema = None
        self.seen_name = None

    def chat_structured(self, messages, schema, schema_name):
        self.seen_messages = list(messages)
        self.seen_schema = schema
        self.seen_name = schema_name
        return self.content, CallUsage("fake-model", 100, 20)


def parse(payload, text="running shoes under $100"):
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return parse_product_query(text, client=FakeClient(body))


# --- a well-formed answer ----------------------------------------------


def test_a_valid_response_fills_the_model():
    query = parse(FULL)

    assert query.keywords == ["running", "shoes"]
    assert query.category == "shoes"
    assert query.max_price_cents == 10000
    assert query.size == "42"


def test_fields_the_text_does_not_mention_come_back_as_none():
    """None means "not stated". Zero would mean "free", which is a lie."""
    query = parse(FULL)

    assert query.min_price_cents is None
    assert query.color is None


def test_a_query_with_no_price_leaves_both_bounds_none():
    query = parse({**FULL, "max_price_cents": None, "size": None})

    assert query.max_price_cents is None
    assert query.min_price_cents is None


def test_the_user_text_is_what_gets_sent():
    client = FakeClient(json.dumps(FULL))

    parse_product_query("blue jacket under 80 dollars", client=client)

    assert any(
        "blue jacket under 80 dollars" in m["content"] for m in client.seen_messages
    )


def test_the_instruction_states_the_cents_rule():
    """The dollar-to-cent conversion lives in one place: this prompt."""
    client = FakeClient(json.dumps(FULL))

    parse_product_query("anything", client=client)

    system = client.seen_messages[0]["content"]
    assert "cent" in system.lower()
    assert "10000" in system, "an example beats a rule the model has to infer"


# --- money is an integer number of cents --------------------------------


def test_a_fractional_price_is_refused():
    with pytest.raises(StructuredOutputError):
        parse({**FULL, "max_price_cents": 99.5})


def test_a_whole_number_float_is_refused_too():
    """100.0 is the dangerous one — lax coercion would accept it silently."""
    with pytest.raises(StructuredOutputError):
        parse({**FULL, "max_price_cents": 100.0})


def test_a_price_as_a_string_is_refused():
    with pytest.raises(StructuredOutputError):
        parse({**FULL, "max_price_cents": "10000"})


def test_a_negative_price_is_refused():
    with pytest.raises(StructuredOutputError):
        parse({**FULL, "min_price_cents": -1})


def test_dollars_are_not_silently_accepted_as_cents():
    """Nothing can catch this at runtime, so pin the type at least."""
    query = parse({**FULL, "max_price_cents": 10000})

    assert isinstance(query.max_price_cents, int)
    assert not isinstance(query.max_price_cents, float)


# --- malformed answers --------------------------------------------------


def test_a_response_that_is_not_json_raises_a_clear_error():
    with pytest.raises(StructuredOutputError, match="JSON"):
        parse("Sorry, I could not parse that request.")


def test_an_empty_response_raises_a_clear_error():
    with pytest.raises(StructuredOutputError):
        parse("")


def test_json_that_is_not_an_object_raises_a_clear_error():
    with pytest.raises(StructuredOutputError):
        parse("[1, 2, 3]")


def test_a_response_failing_validation_names_the_field():
    with pytest.raises(StructuredOutputError, match="keywords"):
        parse({**FULL, "keywords": "running shoes"})


def test_a_missing_required_field_names_it():
    payload = {k: v for k, v in FULL.items() if k != "keywords"}

    with pytest.raises(StructuredOutputError, match="keywords"):
        parse(payload)


def test_the_error_carries_what_the_model_actually_returned():
    """Without it, debugging a bad parse means reproducing the call."""
    with pytest.raises(StructuredOutputError, match="not json"):
        parse("not json")


# --- the schema sent to the model ---------------------------------------


def test_the_schema_has_every_field():
    assert set(PRODUCT_QUERY_SCHEMA["properties"]) == {
        "keywords", "category", "max_price_cents",
        "min_price_cents", "size", "color",
    }


def test_the_schema_has_the_right_types():
    props = PRODUCT_QUERY_SCHEMA["properties"]

    assert props["keywords"]["type"] == "array"
    assert props["keywords"]["items"]["type"] == "string"

    # Optional fields are a union with null — the documented way to express
    # "optional" when strict mode requires every field to be in `required`.
    def branch_types(field):
        return {b.get("type") for b in props[field]["anyOf"]}

    assert branch_types("max_price_cents") == {"integer", "null"}
    assert branch_types("min_price_cents") == {"integer", "null"}
    assert branch_types("category") == {"string", "null"}
    assert branch_types("size") == {"string", "null"}
    assert branch_types("color") == {"string", "null"}


def test_the_price_bound_survives_the_transform():
    """`minimum` is in the supported strict subset; stripping it would be a loss."""
    integer_branch = next(
        b for b in PRODUCT_QUERY_SCHEMA["properties"]["max_price_cents"]["anyOf"]
        if b.get("type") == "integer"
    )

    assert integer_branch["minimum"] == 0


def test_no_price_field_is_a_number():
    """`"type": "number"` here is how a float reaches the checkout."""
    dumped = json.dumps(PRODUCT_QUERY_SCHEMA)

    assert '"number"' not in dumped


def test_the_field_descriptions_survive_the_transform():
    """They are the model's only guidance on what each field means."""
    props = PRODUCT_QUERY_SCHEMA["properties"]

    assert "cent" in props["max_price_cents"]["description"].lower()


# --- the strict transform -----------------------------------------------
#
# Confirmed against the live API on 2026-08-17: without these two changes the
# request is rejected with "Invalid schema for response_format: In context=(),
# 'additionalProperties' is required to be supplied and to be false."


def objects_in(node):
    """Every object schema anywhere in the tree, root included."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            yield node
        for value in node.values():
            yield from objects_in(value)
    elif isinstance(node, list):
        for item in node:
            yield from objects_in(item)


class Inner(BaseModel):
    tag: str
    note: str | None = None


class Outer(BaseModel):
    name: str = Field(description="A name.")
    count: int = 3
    inner: Inner | None = None


def test_every_object_forbids_additional_properties():
    schema = strict_schema_for(Outer)

    found = list(objects_in(schema))
    assert len(found) >= 2, "the nested model must be reached too"
    for obj in found:
        assert obj["additionalProperties"] is False


def test_every_property_is_required_even_the_defaulted_ones():
    schema = strict_schema_for(Outer)

    for obj in objects_in(schema):
        assert sorted(obj["required"]) == sorted(obj["properties"])


def test_defaults_are_stripped():
    """Pydantic emits `default`; it is not in the documented strict subset."""
    assert '"default"' not in json.dumps(strict_schema_for(Outer))


def test_titles_are_stripped():
    assert '"title"' not in json.dumps(strict_schema_for(Outer))


def test_the_transform_does_not_mutate_pydantic_output():
    before = json.dumps(Outer.model_json_schema(), sort_keys=True)

    strict_schema_for(Outer)

    assert json.dumps(Outer.model_json_schema(), sort_keys=True) == before


def test_refs_and_defs_survive():
    """$defs and $ref are supported under strict; flattening them is not needed."""
    schema = strict_schema_for(Outer)

    assert "$defs" in schema
    assert "Inner" in schema["$defs"]


def test_product_query_schema_is_the_strict_form():
    assert PRODUCT_QUERY_SCHEMA["additionalProperties"] is False
    assert sorted(PRODUCT_QUERY_SCHEMA["required"]) == sorted(
        PRODUCT_QUERY_SCHEMA["properties"]
    )


def test_the_schema_travels_as_json():
    json.dumps(PRODUCT_QUERY_SCHEMA)


# --- what the client is asked for ---------------------------------------


def test_the_strict_schema_is_what_gets_sent():
    client = FakeClient(json.dumps(FULL))

    parse_product_query("anything", client=client)

    assert client.seen_schema == PRODUCT_QUERY_SCHEMA
    assert client.seen_name


def test_product_query_can_be_built_directly():
    """It is an ordinary Pydantic model; D3 constructs it without an LLM."""
    query = ProductQuery(keywords=["boots"], max_price_cents=7999)

    assert query.max_price_cents == 7999
    assert query.color is None
