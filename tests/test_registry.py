"""Tests for shopagent.tools.registry.

No network, no OpenAI SDK. The tool under test is defined here rather than
imported from tools/basic.py on purpose: the registry must work with *any*
tool, and a test that leans on the real tools would start failing for reasons
that have nothing to do with the registry.

The bulk of these tests is about failure. The model will send bad arguments,
and every one of those paths has to come back as a ToolResult the model can
read and correct itself from — never as an exception.
"""

import json

import pytest
from pydantic import BaseModel, Field

from shopagent.tools.registry import ToolRegistry, ToolResult, ToolSpec


class EchoArgs(BaseModel):
    """Arguments for the echo tool."""

    message: str = Field(description="Text to repeat back.")
    times: int = Field(
        default=1, ge=1, le=5, description="How many times to repeat it, 1-5."
    )


def echo(message: str, times: int = 1) -> str:
    return " ".join([message] * times)


def explode(**_kwargs) -> str:
    raise RuntimeError("tool blew up")


ECHO_SPEC = ToolSpec(
    name="echo",
    description="Repeat a message back to the caller.",
    args_model=EchoArgs,
    fn=echo,
)


class BoomArgs(BaseModel):
    pass


BOOM_SPEC = ToolSpec(
    name="boom",
    description="Always raises.",
    args_model=BoomArgs,
    fn=explode,
)


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ECHO_SPEC)
    return reg


# --- schema generation -------------------------------------------------


def test_to_openai_schema_has_the_chat_completions_shape():
    """Chat Completions nests the definition under "function".

    The Responses API uses a flat shape; llm/client.py talks to Chat
    Completions, so the nested one is the correct target here.
    """
    schema = ECHO_SPEC.to_openai_schema()

    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "echo"
    assert fn["description"] == "Repeat a message back to the caller."
    assert fn["parameters"]["type"] == "object"


def test_schema_parameters_contain_both_fields_with_their_descriptions():
    props = ECHO_SPEC.to_openai_schema()["function"]["parameters"]["properties"]

    assert set(props) == {"message", "times"}
    assert props["message"]["type"] == "string"
    assert props["times"]["type"] == "integer"
    # The descriptions are the only thing the model has to go on.
    assert "repeat" in props["message"]["description"].lower()
    assert "1-5" in props["times"]["description"]


def test_schema_marks_only_the_field_without_a_default_as_required():
    params = ECHO_SPEC.to_openai_schema()["function"]["parameters"]

    assert params["required"] == ["message"]


def test_schema_is_json_serialisable():
    """It travels to the API as JSON; a stray Python object would 500 late."""
    json.dumps(ECHO_SPEC.to_openai_schema())


def test_openai_schemas_returns_every_registered_tool(registry):
    registry.register(BOOM_SPEC)

    schemas = registry.openai_schemas()

    assert [s["function"]["name"] for s in schemas] == ["echo", "boom"]


def test_openai_schemas_is_empty_for_an_empty_registry():
    assert ToolRegistry().openai_schemas() == []


# --- lookup ------------------------------------------------------------


def test_get_returns_the_spec(registry):
    assert registry.get("echo") is ECHO_SPEC


def test_get_returns_none_for_an_unknown_name(registry):
    assert registry.get("nope") is None


def test_specs_returns_every_registered_spec(registry):
    """Callers that need the specs should not go through get() and a None check."""
    registry.register(BOOM_SPEC)

    assert registry.specs() == [ECHO_SPEC, BOOM_SPEC]


def test_specs_is_empty_for_an_empty_registry():
    assert ToolRegistry().specs() == []


def test_registering_the_same_name_twice_raises():
    """A collision is a bug in our code, not model input — fail loudly."""
    reg = ToolRegistry()
    reg.register(ECHO_SPEC)

    with pytest.raises(ValueError, match="echo"):
        reg.register(ECHO_SPEC)


def test_tool_decorator_registers_and_returns_the_function():
    reg = ToolRegistry()

    @reg.tool(name="echo", description="Repeat a message.", args_model=EchoArgs)
    def decorated(message: str, times: int = 1) -> str:
        return message * times

    assert decorated("hi") == "hi", "the decorator must not swallow the function"
    assert reg.get("echo") is not None
    assert reg.dispatch("echo", '{"message": "hi"}').content == "hi"


# --- successful dispatch -----------------------------------------------


def test_dispatch_runs_the_tool_and_returns_its_output(registry):
    result = registry.dispatch("echo", '{"message": "hi", "times": 3}')

    assert result.ok is True
    assert result.content == "hi hi hi"
    assert result.error is None


def test_dispatch_applies_pydantic_defaults(registry):
    result = registry.dispatch("echo", '{"message": "hi"}')

    assert result.ok is True
    assert result.content == "hi"


def test_dispatch_accepts_an_already_parsed_dict(registry):
    """MCP (D5) hands over parsed arguments; the API hands over a JSON string."""
    result = registry.dispatch("echo", {"message": "hi", "times": 2})

    assert result.ok is True
    assert result.content == "hi hi"


def test_dispatch_treats_empty_arguments_as_no_arguments():
    """A no-arg tool call arrives as "" or "{}" depending on the model."""
    reg = ToolRegistry()

    @reg.tool(name="ping", description="Ping.", args_model=BoomArgs)
    def ping() -> str:
        return "pong"

    assert reg.dispatch("ping", "").content == "pong"
    assert reg.dispatch("ping", "{}").content == "pong"


def test_a_result_that_cannot_be_turned_into_text_is_reported_not_raised():
    """Rendering the result is as failure-prone as producing it.

    CPython caps int->str at 4300 digits (the CVE-2020-10735 mitigation), so
    `str()` on a big enough integer raises. Anything that happens after the
    tool has run still has to reach the model as a `tool` message.
    """
    reg = ToolRegistry()

    @reg.tool(name="huge", description="Returns a huge number.", args_model=BoomArgs)
    def huge() -> int:
        return 10**5000

    result = reg.dispatch("huge", "{}")

    assert result.ok is False
    assert result.content


def test_a_result_that_is_not_json_serialisable_is_reported_not_raised():
    reg = ToolRegistry()

    @reg.tool(name="cyclic", description="Returns a cycle.", args_model=BoomArgs)
    def cyclic() -> dict:
        d: dict = {}
        d["self"] = d
        return d

    result = reg.dispatch("cyclic", "{}")

    assert result.ok is False
    assert result.content


def test_non_string_return_values_are_serialised(registry):
    """A `tool` message must be a string; dicts and numbers are not."""
    reg = ToolRegistry()

    @reg.tool(name="stock", description="Stock level.", args_model=BoomArgs)
    def stock() -> dict:
        return {"variant_id": 7, "quantity": 3}

    result = reg.dispatch("stock", "{}")

    assert result.ok is True
    assert isinstance(result.content, str)
    assert json.loads(result.content) == {"variant_id": 7, "quantity": 3}


# --- failure: unknown tool ---------------------------------------------


def test_unknown_tool_does_not_raise(registry):
    result = registry.dispatch("teleport", "{}")

    assert result.ok is False
    assert result.error is not None


def test_unknown_tool_message_lists_the_tools_that_do_exist(registry):
    registry.register(BOOM_SPEC)

    result = registry.dispatch("teleport", "{}")

    assert "teleport" in result.content
    assert "echo" in result.content and "boom" in result.content


def test_unknown_tool_on_an_empty_registry_says_so():
    result = ToolRegistry().dispatch("teleport", "{}")

    assert result.ok is False
    assert "no tools" in result.content.lower()


# --- failure: malformed JSON -------------------------------------------


def test_malformed_json_does_not_raise(registry):
    result = registry.dispatch("echo", '{"message": "hi",}')

    assert result.ok is False
    assert result.error is not None


def test_malformed_json_message_tells_the_model_what_went_wrong(registry):
    result = registry.dispatch("echo", "not json at all")

    assert "JSON" in result.content


def test_json_that_is_not_an_object_is_rejected(registry):
    """`[1, 2]` and `5` are valid JSON but cannot be keyword arguments."""
    for raw in ("[1, 2]", "5", '"hi"', "null"):
        result = registry.dispatch("echo", raw)

        assert result.ok is False, f"{raw!r} should not be accepted"
        assert "object" in result.content.lower()


# --- failure: validation -----------------------------------------------


def test_value_out_of_range_does_not_raise(registry):
    result = registry.dispatch("echo", '{"message": "hi", "times": 99}')

    assert result.ok is False
    assert result.error is not None


def test_validation_message_names_the_offending_field(registry):
    result = registry.dispatch("echo", '{"message": "hi", "times": 99}')

    assert "times" in result.content, "the model cannot fix what it cannot locate"


def test_validation_message_explains_the_constraint(registry):
    """Naming the field is not enough — the model needs the expectation too."""
    result = registry.dispatch("echo", '{"message": "hi", "times": 99}')

    assert "5" in result.content


def test_missing_required_field_names_it(registry):
    result = registry.dispatch("echo", '{"times": 2}')

    assert result.ok is False
    assert "message" in result.content


def test_wrong_type_names_the_field_and_the_expected_type(registry):
    result = registry.dispatch("echo", '{"message": "hi", "times": "many"}')

    assert result.ok is False
    assert "times" in result.content
    assert "integer" in result.content.lower()


def test_several_bad_fields_are_all_reported(registry):
    """One round-trip per error would waste the model's turns."""
    result = registry.dispatch("echo", '{"times": 99}')

    assert "message" in result.content
    assert "times" in result.content


# --- failure: the tool itself raises ------------------------------------


def test_exception_inside_the_tool_does_not_escape_dispatch(registry):
    registry.register(BOOM_SPEC)

    result = registry.dispatch("boom", "{}")

    assert result.ok is False
    assert result.error is not None


def test_exception_message_reaches_the_model(registry):
    registry.register(BOOM_SPEC)

    result = registry.dispatch("boom", "{}")

    assert "boom" in result.content
    assert "tool blew up" in result.content


# --- the contract that holds all of this together -----------------------


def test_dispatch_never_raises_whatever_the_model_sends(registry):
    """The one invariant the agent loop depends on."""
    registry.register(BOOM_SPEC)
    garbage = [
        "",
        "{}",
        "null",
        "[]",
        "{'message': 'single quotes'}",
        '{"message": null}',
        '{"message": "hi", "times": -4}',
        '{"message": {"nested": "object"}}',
        '{"unexpected": "field"}',
        "   ",           # whitespace only
        "{" * 500,
    ]

    for name in ("echo", "boom", "does_not_exist"):
        for raw in garbage:
            result = registry.dispatch(name, raw)

            assert isinstance(result.content, str) and result.content, (
                f"empty content for {name}({raw!r}) — the model gets nothing"
            )


def test_a_reason_that_already_ends_in_a_period_does_not_get_a_second_one(registry):
    """Tools end their messages with a full stop; _failure adds one too."""
    reg = ToolRegistry()

    @reg.tool(name="picky", description="Raises a punctuated error.", args_model=BoomArgs)
    def picky() -> str:
        raise ValueError("that will not do.")

    assert ".." not in reg.dispatch("picky", "{}").content


def test_a_failed_result_always_carries_content_for_the_tool_message(registry):
    """`content` is what goes back as the `tool` message; it is never empty."""
    result = registry.dispatch("echo", "garbage")

    assert result.content
    assert result.error in result.content


# --- a tool that brings its own schema (D5, step 2) ----------------------
#
# A remote tool publishes JSON Schema, not a Pydantic model, and validates its
# own arguments. `ToolSpec` therefore takes either `args_model` or
# `parameters_schema` — never both, never neither. Everything above this line
# is the `args_model` half and is unchanged, which is the point: widening the
# spec must not have moved anything a local tool relies on.


REMOTE_SCHEMA = {
    "type": "object",
    "properties": {"product_id": {"type": "integer", "title": "Product Id"}},
    "required": ["product_id"],
    "title": "remoteArguments",
}


def remote_fn(**arguments):
    """Stand-in for the call that would cross a pipe."""
    return {"received": arguments}


def test_a_schema_backed_spec_publishes_the_schema_it_was_given():
    """Passed through as published, not regenerated — see ToolSpec."""
    spec = ToolSpec(
        name="remote", description="A remote tool.", fn=remote_fn,
        parameters_schema=REMOTE_SCHEMA,
    )

    schema = spec.to_openai_schema()

    assert schema == {
        "type": "function",
        "function": {
            "name": "remote",
            "description": "A remote tool.",
            "parameters": REMOTE_SCHEMA,
        },
    }
    assert schema["function"]["parameters"] is REMOTE_SCHEMA


def test_a_model_backed_spec_is_unchanged_by_the_widening():
    """The D2 shape, asserted again now that a second shape exists."""
    schema = ECHO_SPEC.to_openai_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo"
    assert schema["function"]["parameters"] == EchoArgs.model_json_schema()


def test_validates_locally_distinguishes_the_two_kinds():
    remote = ToolSpec(
        name="r", description="d", fn=remote_fn, parameters_schema=REMOTE_SCHEMA
    )

    assert ECHO_SPEC.validates_locally is True
    assert remote.validates_locally is False


def test_a_spec_describing_its_arguments_in_neither_way_is_refused():
    """Our bug, not the model's — so it fails at construction, loudly."""
    with pytest.raises(ValueError, match="neither args_model nor parameters_schema"):
        ToolSpec(name="broken", description="d", fn=remote_fn)


def test_a_spec_describing_its_arguments_in_both_ways_is_refused():
    """Two descriptions of one contract are two things free to disagree."""
    with pytest.raises(ValueError, match="both args_model and parameters_schema"):
        ToolSpec(
            name="broken", description="d", fn=remote_fn,
            args_model=EchoArgs, parameters_schema=REMOTE_SCHEMA,
        )


def test_a_schema_backed_tool_receives_the_arguments_unvalidated():
    """No Pydantic step: the dict reaches the tool as the model sent it."""
    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="remote", description="d", fn=remote_fn, parameters_schema=REMOTE_SCHEMA)
    )

    result = registry.dispatch("remote", {"product_id": 7})

    assert result.ok is True
    assert json.loads(result.content) == {"received": {"product_id": 7}}


@pytest.mark.parametrize(
    "raw_args",
    [
        {},                                   # required field missing
        {"product_id": "not an integer"},     # wrong type
        {"product_id": 1, "extra": True},     # a field the schema does not have
        '{"product_id": 3}',                  # a JSON string, decoded before the call
    ],
)
def test_dispatch_never_raises_on_a_schema_backed_tool(raw_args):
    """The D2 invariant holds for the new shape too.

    None of these is rejected here — the server owns that judgement — so they
    reach the tool. What matters is that `dispatch` comes back with a
    `ToolResult` in every case instead of letting anything escape.
    """
    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="remote", description="d", fn=remote_fn, parameters_schema=REMOTE_SCHEMA)
    )

    result = registry.dispatch("remote", raw_args)

    assert isinstance(result, ToolResult)
    assert result.content


def test_malformed_json_is_still_caught_before_a_schema_backed_tool_runs():
    """The steps before validation are shared, and still apply."""
    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="remote", description="d", fn=remote_fn, parameters_schema=REMOTE_SCHEMA)
    )

    result = registry.dispatch("remote", "{not json")

    assert result.ok is False
    assert "not valid JSON" in result.content


def test_a_tool_returning_a_tool_result_has_it_passed_through():
    """How a remote tool reports a failure the registry cannot detect.

    Without this the `ok=False` would be flattened into a success and the
    model would read the failure text as an answer.
    """
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="remote",
            description="d",
            fn=lambda **kw: ToolResult(ok=False, content="the server said no", error="nope"),
            parameters_schema=REMOTE_SCHEMA,
        )
    )

    result = registry.dispatch("remote", {"product_id": 1})

    assert result.ok is False
    assert result.content == "the server said no"
    assert result.error == "nope"
