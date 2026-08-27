"""Turning MCP tools into the shapes the agent loop already speaks (D5, step 1).

Two translations, in opposite directions. On the way out, an MCP tool becomes a
Chat Completions tool definition — the same nested `{"type": "function", ...}`
object `ToolRegistry.to_openai_schema` produces, so the loop cannot tell where a
tool came from. On the way back, a `CallToolResult` becomes the string a `tool`
message carries.

**Nothing here names a tool.** Every function takes whatever `tools/list`
returned and translates it. A tool added to the server tomorrow appears in the
model's list without a change to this file, which is the property that makes D5
worth doing at all — and the one a hardcoded mapping would quietly destroy.

**The MCP input schema is passed through untouched, and that is a checked
claim, not an assumption.** Chat Completions rejects some JSON Schema
constructs, so the schemas the D4 server produces were inspected before this
was written: all four carry `type`, `properties`, an optional `required`, and a
`title` — no `$ref`, no `$defs`, no `allOf`. `title` is the same key Pydantic
already emits through the registry, and the API ignores it in non-strict mode.
So no rewriting happens here. That is worth stating because a schema doctored in
transit is a schema the Inspector and the model disagree about. If a future
server does emit `$defs` — a tool taking a nested model would — this is where
that shows up, and it should be handled deliberately rather than papered over.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any, Protocol

# The description sent when a tool has none. MCP allows it to be absent;
# a model given a name and no explanation guesses from the name alone, so the
# gap is made visible rather than sent as an empty string.
MISSING_DESCRIPTION = "(no description provided by the server)"

# What comes back when a tool result carries nothing renderable at all. It
# should be unreachable against the D4 server, whose search envelope guarantees
# one content block even for an empty result, but a `tool` message with empty
# content is a turn the model cannot interpret, so there is a floor.
EMPTY_RESULT = "The tool returned no content."


class SupportsToolSchema(Protocol):
    """The parts of an MCP `Tool` this module reads.

    A protocol rather than the SDK's class, so the adapter can be tested
    against fabricated tools — which is also the test that it is not keying off
    anything specific to our own server.
    """

    name: str
    description: str | None
    input_schema: dict[str, Any]


def to_openai_tool(tool: SupportsToolSchema) -> dict[str, Any]:
    """One MCP tool as one entry of the `tools` parameter.

    The shape matches `ToolSpec.to_openai_schema` exactly: a `function` object
    holding `name`, `description` and `parameters`. `parameters` is the MCP
    `inputSchema` verbatim — see the module docstring for why it is not
    rewritten.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or MISSING_DESCRIPTION,
            "parameters": tool.input_schema,
        },
    }


def to_openai_tools(tools: Iterable[SupportsToolSchema]) -> list[dict[str, Any]]:
    """Every tool the server offers, ready for the `tools` parameter."""
    return [to_openai_tool(tool) for tool in tools]


def _text_blocks(content: Sequence[Any]) -> list[str]:
    """The text out of a result's content blocks, ignoring the other kinds.

    MCP content can also be an image or an embedded resource. Neither belongs
    in a `tool` message, and neither is anything the catalog server produces,
    so they are skipped rather than stringified into noise.
    """
    return [
        block.text
        for block in content
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]


def result_to_content(result: Any) -> str:
    """Render a tool result as the text a `tool` message carries.

    Prefers the content blocks over `structuredContent`, which is the opposite
    of what convenience suggests and is deliberate. Content is the field the
    protocol guarantees for every result: an error carries its message there and
    nothing in `structuredContent` at all, and a tool without an output schema
    fills only content. Reading structured output first would mean two code
    paths and a `None` check that decides what the model sees on the turn it
    most needs to understand — the failing one.

    Falls back to `structuredContent` only when there are no text blocks, and to
    a fixed sentence when there is nothing at all, so the caller always has
    something to put in the message.
    """
    text = "\n".join(_text_blocks(getattr(result, "content", None) or []))
    if text:
        return text

    structured = getattr(result, "structured_content", None)
    if structured is not None:
        # default=str for the same reason the registry uses it: a value the
        # catalog produced must not turn a successful call into a crash.
        return json.dumps(structured, default=str)

    return EMPTY_RESULT


def is_error(result: Any) -> bool:
    """Whether the server reported this call as a failure.

    A thin reading of `isError`, given a name so callers do not have to know
    that the flag arrives snake-cased on the SDK object. Absent means success:
    the field is optional in the protocol and its absence means the call was
    fine.
    """
    return bool(getattr(result, "is_error", False))
