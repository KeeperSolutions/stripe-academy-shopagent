"""Single source of truth for the tools the agent can call (D2).

One ToolSpec holds everything about a tool: its name, the description the model
reads, the Pydantic model describing its arguments, and the Python function
that runs it. The JSON schema the model sees is *generated* from that Pydantic
model — it is never written by hand, so the schema and the validation can never
drift apart.

The registry is deliberately a replaceable source rather than a hard-coded
list. On D5 the tools come from an MCP server, and on D9 MCP and HTTP tools are
mixed; both cases build ToolSpecs and register them, and the agent loop does
not change.

The other half of this module is error handling. `dispatch` never raises. The
model *will* send malformed JSON, miss required fields and invent values out of
range, and every one of those has to come back as a `tool` message it can read
and correct itself from. That is why the error strings here are written as
prose aimed at an LLM, not as log lines aimed at a developer — the model is
their only reader.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

# Schema format: Chat Completions nests the definition under "function"
# ({"type": "function", "function": {...}}), while the Responses API takes the
# same fields flat. llm/client.py talks to Chat Completions, so the nested
# shape is the one built here. Verified against the openai 3.0.0 type stubs
# (ChatCompletionFunctionToolParam) on 2026-08-17.
#
# `strict: true` is available on this shape but is NOT set: Pydantic's
# model_json_schema() output is not accepted as-is under strict mode (it omits
# `additionalProperties: false`, lists only non-defaulted fields in `required`,
# and emits `default`/`title` keys). Turning strict on means transforming the
# schema first — a decision left open on purpose, and contained entirely within
# to_openai_schema() when it is made.


@dataclass(frozen=True)
class ToolResult:
    """The outcome of one tool call, in the form the agent loop needs.

    `content` is what gets sent back to the model as the `tool` message and is
    never empty — on failure it carries the explanation instead of the result.
    `error` is the short machine-readable cause for logs, and is always a
    substring of `content`.
    """

    ok: bool
    content: str
    error: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    """One tool: what the model sees, and what actually runs."""

    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[..., Any]

    def to_openai_schema(self) -> dict[str, Any]:
        """The tool definition as the `tools` parameter expects it.

        `parameters` is Pydantic's own output, passed through untouched. The
        `title` and `default` keys it adds are ignored by the API in
        non-strict mode; see the note at the top of this module before
        enabling strict.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


def _to_content(value: Any) -> str:
    """Render a tool's return value as the string a `tool` message requires."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        # default=str so a date or a Decimal from the catalog (D3) cannot turn
        # a successful call into a serialisation crash.
        return json.dumps(value, default=str)
    return str(value)


def _format_validation_error(exc: ValidationError) -> str:
    """Turn a ValidationError into instructions the model can act on.

    Every field is reported in one go: one error per round-trip would burn the
    model's turns on a call it could have fixed all at once. Each line names
    the field and states what was expected, because a message the model cannot
    locate is a message it cannot correct.
    """
    lines = []
    for err in exc.errors():
        field = ".".join(str(part) for part in err["loc"]) or "(root)"
        line = f"  - {field}: {err['msg']}"
        if "input" in err:
            line += f" (received {err['input']!r})"
        lines.append(line)
    return "\n".join(lines)


class ToolRegistry:
    """The tools available in one conversation."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    # --- registration ---------------------------------------------------

    def register(self, spec: ToolSpec) -> ToolSpec:
        """Add a tool. This is the entry point an MCP adapter (D5) will use."""
        if spec.name in self._specs:
            # A name collision is a bug in our code, not model input, so it
            # fails loudly here rather than silently shadowing a tool.
            raise ValueError(f"a tool named {spec.name!r} is already registered")
        self._specs[spec.name] = spec
        return spec

    def tool(
        self, name: str, description: str, args_model: type[BaseModel]
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register the decorated function as a tool, and return it unchanged.

        The function stays an ordinary Python function that can be called and
        tested directly — the registry wraps it, it does not replace it.
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(
                ToolSpec(
                    name=name,
                    description=description,
                    args_model=args_model,
                    fn=fn,
                )
            )
            return fn

        return decorator

    # --- lookup ---------------------------------------------------------

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return list(self._specs)

    def openai_schemas(self) -> list[dict[str, Any]]:
        """Every registered tool, ready for the `tools` parameter."""
        return [spec.to_openai_schema() for spec in self._specs.values()]

    # --- execution ------------------------------------------------------

    def _failure(self, reason: str, advice: str) -> ToolResult:
        # A tool's own message usually ends in a full stop, and so does the
        # one added here — "no text.. Do not repeat" reads as a typo to the
        # model as much as to a person.
        reason = reason.rstrip().removesuffix(".")
        return ToolResult(ok=False, content=f"Error: {reason}. {advice}", error=reason)

    def _available(self) -> str:
        if not self._specs:
            return "No tools are available in this conversation; answer directly."
        listed = ", ".join(sorted(self._specs))
        return f"Available tools: {listed}. Call one of those, or answer without a tool."

    def dispatch(self, name: str, raw_args: str | dict[str, Any] | None) -> ToolResult:
        """Run a tool call from the model and return what to send back.

        `raw_args` is whatever the model produced: the JSON string from
        `tool_call.function.arguments`, or an already-parsed dict when the
        arguments came from somewhere else (MCP, on D5).

        This never raises. Every failure — unknown tool, malformed JSON, failed
        validation, an exception inside the tool — comes back as a ToolResult
        whose `content` explains the problem to the model, so it gets a chance
        to fix the call on the next turn instead of taking the app down.
        """
        spec = self.get(name)
        if spec is None:
            return self._failure(f"there is no tool named {name!r}", self._available())

        try:
            arguments = self._parse(spec, raw_args)
        except _BadArguments as bad:
            return bad.result

        try:
            value = spec.fn(**arguments.model_dump())
        except Exception as exc:  # noqa: BLE001 - a tool may raise anything
            # Broad on purpose: a buggy tool must not end the conversation.
            # The exception type is included because it is often the only clue
            # the model gets about whether retrying is worth it.
            return self._failure(
                f"the tool {name!r} failed while running: "
                f"{type(exc).__name__}: {exc}",
                "Do not repeat the identical call. Either correct the arguments "
                "or tell the user this is currently unavailable.",
            )

        try:
            content = _to_content(value)
        except Exception as exc:  # noqa: BLE001 - see below
            # Rendering the result is a second, separate failure point, and it
            # sits after the tool has already succeeded. CPython refuses
            # str() on an integer over 4300 digits, and json.dumps refuses a
            # cyclic structure — both would otherwise escape dispatch and take
            # down the conversation over a call that actually worked.
            return self._failure(
                f"the tool {name!r} produced a result that cannot be sent back "
                f"as text ({type(exc).__name__}: {exc})",
                "Ask for a smaller or more specific result.",
            )

        return ToolResult(ok=True, content=content)

    def _parse(
        self, spec: ToolSpec, raw_args: str | dict[str, Any] | None
    ) -> BaseModel:
        """Decode and validate arguments, or raise _BadArguments carrying the reply."""
        if raw_args is None:
            raw_args = {}
        if isinstance(raw_args, str):
            # A tool that takes no arguments comes back as "" from some models
            # and "{}" from others; both mean the same thing.
            stripped = raw_args.strip()
            if not stripped:
                raw_args = {}
            else:
                try:
                    raw_args = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise _BadArguments(
                        self._failure(
                            f"the arguments for {spec.name!r} are not valid JSON "
                            f"({exc})",
                            "Send the arguments as a single JSON object, for "
                            'example {"field": "value"}.',
                        )
                    ) from exc

        if not isinstance(raw_args, dict):
            # `[1, 2]`, `5` and `null` are valid JSON but cannot become
            # keyword arguments, so they are caught here rather than by
            # Pydantic, which would report it far less clearly.
            raise _BadArguments(
                self._failure(
                    f"the arguments for {spec.name!r} must be a JSON object with "
                    f"named fields, but a {type(raw_args).__name__} was received",
                    'Send the arguments as a single JSON object, for example '
                    '{"field": "value"}.',
                )
            )

        try:
            return spec.args_model(**raw_args)
        except ValidationError as exc:
            raise _BadArguments(
                ToolResult(
                    ok=False,
                    content=(
                        f"Error: invalid arguments for {spec.name!r}:\n"
                        f"{_format_validation_error(exc)}\n"
                        f"Correct those fields and call {spec.name!r} again."
                    ),
                    error=f"invalid arguments for {spec.name!r}",
                )
            ) from exc


class _BadArguments(Exception):
    """Internal control flow only — carries the ToolResult back to dispatch.

    Never escapes this module: dispatch catches it, and the value it holds is
    what the model receives.
    """

    def __init__(self, result: ToolResult) -> None:
        super().__init__(result.error)
        self.result = result
