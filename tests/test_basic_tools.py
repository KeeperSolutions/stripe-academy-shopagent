"""Tests for shopagent.tools.basic.

No network. `get_time` reads the clock, so the assertions are about the shape
and the offset of what comes back, never about a fixed instant.

The calculator tests are the important half. It is the one place in the
project where a string produced by an LLM gets interpreted, so most of what
follows is about what it must refuse to do.
"""

import time
from datetime import datetime, timedelta

import pytest

from shopagent.tools.basic import MAX_EXPONENT, REGISTRY, calculator, get_time

# Asia/Tokyo has no DST — its offset is +09:00 all year, so the assertion
# cannot start failing in March.
TOKYO_OFFSET = timedelta(hours=9)


# --- get_time ----------------------------------------------------------


def test_get_time_returns_a_parsable_iso_string_with_the_right_offset():
    result = REGISTRY.dispatch("get_time", '{"timezone": "Asia/Tokyo"}')

    assert result.ok is True
    parsed = datetime.fromisoformat(result.content)
    assert parsed.utcoffset() == TOKYO_OFFSET


def test_get_time_defaults_to_utc():
    result = REGISTRY.dispatch("get_time", "{}")

    assert result.ok is True
    parsed = datetime.fromisoformat(result.content)
    assert parsed.utcoffset() == timedelta(0)
    assert result.content.endswith("+00:00")


def test_get_time_handles_a_dst_zone():
    """Europe/Zagreb is +01:00 in winter and +02:00 in summer; both are valid."""
    result = REGISTRY.dispatch("get_time", '{"timezone": "Europe/Zagreb"}')

    assert result.ok is True
    offset = datetime.fromisoformat(result.content).utcoffset()
    assert offset in (timedelta(hours=1), timedelta(hours=2))


def test_unknown_timezone_is_reported_to_the_model_not_raised():
    result = REGISTRY.dispatch("get_time", '{"timezone": "Mars/Olympus_Mons"}')

    assert result.ok is False
    assert "IANA" in result.content


def test_unknown_timezone_message_shows_an_example():
    """"Use an IANA name" is useless to a model that just produced one it thought was IANA."""
    result = REGISTRY.dispatch("get_time", '{"timezone": "PST"}')

    assert result.ok is False
    assert "/" in result.content, "the message must contain a concrete Area/City example"


def test_a_timezone_that_is_an_invalid_key_does_not_escape():
    """ZoneInfo raises ValueError, not ZoneInfoNotFoundError, for these."""
    for bad in ("../../etc/passwd", "/absolute/path", ""):
        result = REGISTRY.dispatch("get_time", {"timezone": bad})

        assert result.ok is False, f"{bad!r} should be rejected"


def test_get_time_is_callable_as_a_plain_function():
    assert datetime.fromisoformat(get_time("Asia/Tokyo")).utcoffset() == TOKYO_OFFSET
    assert get_time.__doc__, "the docstring is what a developer reads here"


# --- calculator: arithmetic --------------------------------------------


def test_operator_precedence():
    assert REGISTRY.dispatch("calculator", '{"expression": "2+3*4"}').content == "14"


def test_parentheses_override_precedence():
    assert REGISTRY.dispatch("calculator", '{"expression": "(2+3)*4"}').content == "20"


def test_negative_numbers():
    assert REGISTRY.dispatch("calculator", '{"expression": "-5 + 2"}').content == "-3"
    assert REGISTRY.dispatch("calculator", '{"expression": "3 * -4"}').content == "-12"


def test_decimals():
    result = REGISTRY.dispatch("calculator", '{"expression": "(14.99 + 3.50) * 2"}')

    assert result.ok is True
    assert float(result.content) == pytest.approx(36.98)


def test_the_remaining_allowed_operators():
    cases = {
        "10 / 4": 2.5,
        "10 // 4": 2,
        "10 % 4": 2,
        "2 ** 8": 256,
        "+7": 7,
    }
    for expression, expected in cases.items():
        result = REGISTRY.dispatch("calculator", {"expression": expression})

        assert result.ok is True, f"{expression} -> {result.content}"
        assert float(result.content) == expected


def test_calculator_is_callable_as_a_plain_function():
    assert calculator("2+3*4") == 14
    assert calculator.__doc__


# --- calculator: arithmetic failures -----------------------------------


def test_division_by_zero_is_an_error_not_an_exception():
    result = REGISTRY.dispatch("calculator", '{"expression": "1/0"}')

    assert result.ok is False
    # Asserting on "zero" alone would pass even with the handler deleted:
    # dispatch catches everything, and ZeroDivisionError's own text says
    # "division by zero". "divisor" appears only in our message.
    assert "divisor" in result.content.lower()


def test_floor_division_and_modulo_by_zero_too():
    for expression in ("1 // 0", "1 % 0", "0 ** -1"):
        result = REGISTRY.dispatch("calculator", {"expression": expression})

        assert result.ok is False, f"{expression} should be rejected"


def test_syntax_error_is_reported_to_the_model():
    result = REGISTRY.dispatch("calculator", '{"expression": "2 +* 3"}')

    assert result.ok is False
    assert result.content


def test_empty_expression_is_rejected():
    for expression in ("", "   "):
        assert REGISTRY.dispatch("calculator", {"expression": expression}).ok is False


# --- calculator: resource exhaustion -----------------------------------


def test_huge_exponent_is_refused_quickly():
    """2**10000000 would pin a core for minutes; it must never be computed."""
    started = time.perf_counter()

    result = REGISTRY.dispatch("calculator", '{"expression": "2**10000000"}')

    elapsed = time.perf_counter() - started
    assert result.ok is False
    assert elapsed < 1.0, f"took {elapsed:.2f}s — the guard ran too late or not at all"


def test_the_exponent_ceiling_reports_the_exponent_itself():
    """The size ceiling would also catch this, with a vaguer message.

    "the exponent 10000000 is too large, the limit is 100" tells the model what
    to change; "the result is too large" leaves it guessing between the base
    and the exponent. Pinning the message keeps the cheaper, clearer check from
    quietly becoming dead code behind the size ceiling.
    """
    content = REGISTRY.dispatch("calculator", '{"expression": "2**10000000"}').content

    assert "exponent" in content.lower()
    assert str(MAX_EXPONENT) in content


def test_stacked_exponents_are_refused_quickly():
    """The exponents are each small; the result is not."""
    started = time.perf_counter()

    result = REGISTRY.dispatch("calculator", '{"expression": "(9**99)**99"}')

    elapsed = time.perf_counter() - started
    assert result.ok is False
    assert elapsed < 1.0, f"took {elapsed:.2f}s"


def test_a_result_too_big_to_print_does_not_escape_dispatch():
    """Every factor clears both power guards; their product still cannot be printed.

    base 999999999999 has a 40-bit length, so 40 * 100 stays under
    MAX_RESULT_BITS and the exponent is exactly MAX_EXPONENT — yet ten such
    factors multiply out to some 12,000 digits, past CPython's 4300-digit
    int-to-string limit. Multiplication has no size guard of its own, which is
    why the registry has to survive a result it cannot render.
    """
    expression = "*".join(["(999999999999**100)"] * 10)
    assert len(expression) <= 200, "the expression must clear the length limit"

    result = REGISTRY.dispatch("calculator", {"expression": expression})

    assert result.ok is False
    assert result.content


def test_an_over_long_expression_is_rejected_before_parsing():
    result = REGISTRY.dispatch("calculator", {"expression": "1+" * 5000 + "1"})

    assert result.ok is False
    assert "long" in result.content.lower()


# --- calculator: attack vectors ----------------------------------------
#
# The reason this tool is not built on eval(). Each of these is a string an
# LLM can be talked into producing.


def test_import_and_shell_execution_is_refused():
    result = REGISTRY.dispatch(
        "calculator", {"expression": "__import__('os').system('ls')"}
    )

    assert result.ok is False
    assert "ls" not in result.content or "not allowed" in result.content.lower()


def test_bare_name_access_is_refused():
    result = REGISTRY.dispatch("calculator", '{"expression": "x+1"}')

    assert result.ok is False


def test_attribute_access_is_refused():
    """The gateway to every sandbox escape: (1).__class__.__bases__[0]."""
    result = REGISTRY.dispatch("calculator", {"expression": "(1).__class__"})

    assert result.ok is False


def test_a_range_of_hostile_expressions_are_all_refused():
    hostile = [
        "__import__('os').system('ls')",
        "__import__('subprocess').run(['id'])",
        "open('/etc/passwd').read()",
        "().__class__.__bases__[0].__subclasses__()",
        "(1).__class__.__mro__",
        "eval('1+1')",
        "exec('x=1')",
        "globals()",
        "[i for i in range(10**9)]",
        "lambda: 1",
        "[1, 2, 3][0]",
        "{'a': 1}['a']",
        "'a' * 10**9",  # a memory blow-up with no function call in sight
        "f'{1}'",
        "1 if 1 else 2",
        "(y := 5)",
        "not 1",
        "1 < 2",
        "True and False",
        "print(1)",
    ]

    for expression in hostile:
        result = REGISTRY.dispatch("calculator", {"expression": expression})

        assert result.ok is False, f"{expression!r} was NOT refused: {result.content}"
        assert result.content, "a refusal still has to tell the model something"


def test_string_constants_are_refused_so_multiplication_cannot_allocate():
    """`'a' * 10**9` has no call and no name — only a non-numeric constant."""
    result = REGISTRY.dispatch("calculator", {"expression": "'aaaa'"})

    assert result.ok is False


def test_booleans_are_refused_even_though_bool_is_an_int():
    result = REGISTRY.dispatch("calculator", {"expression": "True + 1"})

    assert result.ok is False


def test_refusals_explain_what_is_allowed_instead():
    """A refusal the model cannot learn from produces the same call again."""
    result = REGISTRY.dispatch("calculator", {"expression": "sqrt(16)"})

    assert result.ok is False
    assert "+" in result.content and "*" in result.content


# --- registration ------------------------------------------------------


def test_both_tools_are_registered():
    assert [s["function"]["name"] for s in REGISTRY.openai_schemas()] == [
        "get_time",
        "calculator",
    ]


def test_schemas_describe_their_arguments_for_the_model():
    schemas = {s["function"]["name"]: s["function"] for s in REGISTRY.openai_schemas()}

    time_props = schemas["get_time"]["parameters"]["properties"]
    assert "IANA" in time_props["timezone"]["description"]

    calc_props = schemas["calculator"]["parameters"]["properties"]
    assert calc_props["expression"]["type"] == "string"
    assert calc_props["expression"]["description"]


def test_timezone_is_optional_and_expression_is_not():
    schemas = {s["function"]["name"]: s["function"] for s in REGISTRY.openai_schemas()}

    assert schemas["get_time"]["parameters"].get("required", []) == []
    assert schemas["calculator"]["parameters"]["required"] == ["expression"]
