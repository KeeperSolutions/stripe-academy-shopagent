"""The first two tools the agent can call (D2).

Both are ordinary Python functions with ordinary signatures. They are callable
and testable without the registry; the registry only wraps them. Each raises on
bad input, and `ToolRegistry.dispatch` turns that into a `tool` message the
model can read and correct itself from.

The descriptions and the argument docstrings in this module are read by the
model, not by a developer. They are written accordingly: what the tool does,
when to reach for it, what it expects, and in what format it answers.

`calculator` deliberately does not use `eval()`, not even with emptied
globals. Every published sandbox built that way has been escaped through
attribute chains such as `().__class__.__bases__[0].__subclasses__()`, and the
expression here comes from an LLM, which is to say from whoever is talking to
it. Instead the expression is parsed to an AST and walked against an allow-list
of node types: anything that is not a number or one of seven arithmetic
operators never gets evaluated at all.
"""

from __future__ import annotations

import ast
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from shopagent.tools.registry import ToolRegistry

REGISTRY = ToolRegistry()

# --- calculator limits -------------------------------------------------
# An allow-list stops code execution but not resource exhaustion: `2**10000000`
# contains nothing but a number and an operator, and would still pin a core for
# minutes. These bounds are checked before the arithmetic runs, never after.

MAX_EXPRESSION_LENGTH = 200
# Also caps nesting depth, which is what keeps ast.parse away from a
# RecursionError on input like "((((((...1...))))))".

MAX_EXPONENT = 100
# Rejects a huge exponent outright.

MAX_RESULT_BITS = 4096
# ~1200 digits. Catches the case each exponent is small but they stack, as in
# `(9**99)**99`, where the exponents pass the check above and the result does
# not fit in memory.

_ALLOWED_BINARY_OPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
)
_ALLOWED_UNARY_OPS = (ast.UAdd, ast.USub)

# Repeated in every refusal: a model that is only told "no" produces the same
# call again on the next turn.
_ALLOWED = (
    "Only numbers and the operators + - * / // % ** with parentheses are "
    "allowed — no variables, no function calls, no text."
)


class CalculatorError(ValueError):
    """Raised for any expression the calculator will not evaluate."""


class GetTimeArgs(BaseModel):
    """Arguments for get_time."""

    timezone: str = Field(
        default="UTC",
        description=(
            "IANA time zone name in Area/City form, for example 'Europe/Zagreb', "
            "'America/New_York' or 'Asia/Tokyo'. Abbreviations such as 'PST' or "
            "'CET' are not valid. Defaults to 'UTC'."
        ),
    )


class CalculatorArgs(BaseModel):
    """Arguments for calculator."""

    expression: str = Field(
        description=(
            "The arithmetic expression to evaluate, for example "
            "'(14.99 + 3.50) * 2'. Integers, decimals, the operators "
            "+ - * / // % ** and parentheses only. Do not include variables, "
            "function names, units or currency symbols."
        ),
    )


@REGISTRY.tool(
    name="get_time",
    description=(
        "Get the current date and time in a given time zone. Use this whenever "
        "the current time matters — never state the time from your own "
        "knowledge, because you do not have a clock. Returns an ISO 8601 "
        "timestamp including the UTC offset, for example "
        "'2026-08-17T14:30:00.123456+02:00'."
    ),
    args_model=GetTimeArgs,
)
def get_time(timezone: str = "UTC") -> str:
    """Return the current time in `timezone` as an ISO 8601 string.

    Raises ValueError if the zone is not a known IANA name.
    """
    try:
        zone = ZoneInfo(timezone)
    except Exception as exc:
        # ZoneInfo raises ZoneInfoNotFoundError for an unknown zone but
        # ValueError for a malformed key ("", "/absolute", ".."), and the
        # distinction means nothing to the model — both are "that is not a
        # zone name". The example in the message is the point: telling a model
        # to "use an IANA name" does not help when it believed 'PST' was one.
        raise ValueError(
            f"{timezone!r} is not a known time zone. Provide an IANA time zone "
            f"name in Area/City form, such as 'Europe/Zagreb', "
            f"'America/New_York' or 'UTC'"
        ) from exc
    return datetime.now(zone).isoformat()


@REGISTRY.tool(
    name="calculator",
    description=(
        "Evaluate an arithmetic expression and return the result. Use this for "
        "every calculation rather than working it out yourself, so the answer "
        "is checked rather than guessed. Accepts integers, decimals, the "
        "operators + - * / // % ** and parentheses. It cannot resolve "
        "variables, call functions, or read units and currency symbols — pass "
        "'(14.99 + 3.50) * 2', not '$14.99 plus $3.50, doubled'."
    ),
    args_model=CalculatorArgs,
)
def calculator(expression: str) -> int | float:
    """Evaluate an arithmetic `expression` and return its numeric result.

    Raises CalculatorError for anything that is not pure arithmetic, and for
    arithmetic whose result would be too large to compute.
    """
    if not expression or not expression.strip():
        raise CalculatorError(f"the expression is empty. {_ALLOWED}")

    if len(expression) > MAX_EXPRESSION_LENGTH:
        # Checked before parsing: ast.parse itself is what a deeply nested
        # expression would take down.
        raise CalculatorError(
            f"the expression is too long ({len(expression)} characters, the "
            f"limit is {MAX_EXPRESSION_LENGTH}). Split the calculation into "
            f"several smaller calls"
        )

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(
            f"the expression is not valid arithmetic ({exc.msg}). {_ALLOWED}"
        ) from exc

    return _evaluate(tree.body)


def _evaluate(node: ast.AST) -> int | float:
    """Walk the AST, refusing every node type that is not plain arithmetic.

    An allow-list, never a deny-list: an unrecognised node type is refused by
    default, so a Python release that adds new syntax cannot silently widen
    what this accepts.
    """
    if isinstance(node, ast.Constant):
        # bool is a subclass of int, so `True + 1` would otherwise sail
        # through as 2. Complex numbers and strings are refused here too —
        # `'a' * 10**9` allocates a gigabyte without a single function call.
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculatorError(
                f"{node.value!r} is not a number. {_ALLOWED}"
            )
        return node.value

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARY_OPS):
            raise CalculatorError(
                f"the unary operator {type(node.op).__name__} is not allowed. "
                f"{_ALLOWED}"
            )
        operand = _evaluate(node.operand)
        return +operand if isinstance(node.op, ast.UAdd) else -operand

    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINARY_OPS):
            raise CalculatorError(
                f"the operator {type(node.op).__name__} is not allowed. {_ALLOWED}"
            )
        # Both operands are resolved first, so a refusal deeper in the tree
        # surfaces before any arithmetic happens.
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        return _apply(node.op, left, right)

    raise CalculatorError(f"{_describe(node)} is not allowed. {_ALLOWED}")


def _describe(node: ast.AST) -> str:
    """Name the refused construct in terms the model can act on."""
    names = {
        ast.Name: "a variable name",
        ast.Call: "a function call",
        ast.Attribute: "attribute access",
        ast.Subscript: "indexing",
        ast.List: "a list",
        ast.Tuple: "a tuple",
        ast.Dict: "a dictionary",
        ast.Set: "a set",
        ast.ListComp: "a comprehension",
        ast.SetComp: "a comprehension",
        ast.DictComp: "a comprehension",
        ast.GeneratorExp: "a generator expression",
        ast.Lambda: "a lambda",
        ast.IfExp: "a conditional expression",
        ast.Compare: "a comparison",
        ast.BoolOp: "a boolean operator",
        ast.JoinedStr: "an f-string",
        ast.NamedExpr: "an assignment",
        ast.Starred: "unpacking",
    }
    return names.get(type(node), f"{type(node).__name__} syntax")


def _apply(op: ast.operator, left: int | float, right: int | float) -> int | float:
    """Run one checked arithmetic operation."""
    if isinstance(op, ast.Pow):
        _check_power(left, right)

    try:
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.FloorDiv):
            return left // right
        if isinstance(op, ast.Mod):
            return left % right
        return left**right
    except ZeroDivisionError as exc:
        raise CalculatorError(
            "division by zero is undefined. Check the divisor before dividing"
        ) from exc
    except OverflowError as exc:
        raise CalculatorError(
            "the result is too large to represent. Use smaller numbers"
        ) from exc


def _check_power(base: int | float, exponent: int | float) -> None:
    """Refuse a power whose result would be too large, before computing it.

    `left ** right` is not interruptible: once CPython starts on `2**10000000`
    nothing gets it back. The size therefore has to be estimated up front.
    """
    if abs(exponent) > MAX_EXPONENT:
        raise CalculatorError(
            f"the exponent {exponent} is too large (the limit is "
            f"{MAX_EXPONENT}). Such a result cannot be computed"
        )

    # An int result has roughly base.bit_length() * exponent bits. Each
    # exponent in `(9**99)**99` passes the check above while the result runs to
    # some 19,000 digits.
    if isinstance(base, int) and isinstance(exponent, int) and exponent > 0:
        if base.bit_length() * exponent > MAX_RESULT_BITS:
            raise CalculatorError(
                "the result of that power is too large to compute. Use a "
                "smaller base or exponent"
            )
