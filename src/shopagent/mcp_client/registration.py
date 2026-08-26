"""Putting MCP tools into the local registry (D5, step 2).

The agent loop from D2 takes a `ToolRegistry` and knows nothing else. This
module is what lets an MCP server fill one, so that a tool reached over a pipe
and a tool defined in `tools/basic.py` are the same kind of thing by the time
the loop sees them. Nothing here names a tool: whatever `tools/list` returned
is what gets registered, so a tool added to the server tomorrow appears without
a change on this side.

**`is_error` has to survive the trip, and it is the whole reason this module is
not three lines.** D4's finding on the server was that a returned string
describing a failure is indistinguishable from success. The mirror image is
true here: an MCP result carrying `isError` folded into `ToolResult(ok=True)`
hands the model an error message dressed as an answer, and the model reads it
as a product that exists. So the flag is mapped explicitly, and a test asserts
it in both directions.

**The messages are not rewritten.** D4 wrote every failure text for the model —
naming the field, saying what to do next — and measured them. Translating them
again here would replace prose aimed at a model with prose aimed at whoever
wrote this function.
"""

from __future__ import annotations

from typing import Any, Protocol

from shopagent.mcp_client.adapter import MISSING_DESCRIPTION, is_error, result_to_content
from shopagent.tools.registry import ToolRegistry, ToolResult, ToolSpec


class SupportsToolCalls(Protocol):
    """The part of `MCPToolClient` this module uses.

    A protocol so the registration path can be tested against a fake client,
    which is the only way to exercise the `is_error` mapping without a live
    server and a deliberately broken catalog.
    """

    def list_tools(self) -> list[Any]: ...

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any: ...


def _make_caller(client: SupportsToolCalls, tool_name: str):
    """Build the function the registry will run for one remote tool.

    Returns a `ToolResult` rather than a value, which `dispatch` passes straight
    through. That is what preserves `is_error`: a tool returning a plain value
    is always a success as far as the registry is concerned, and the failure
    would vanish exactly here.
    """

    def call(**arguments: Any) -> ToolResult:
        result = client.call_tool(tool_name, arguments)
        content = result_to_content(result)

        if is_error(result):
            return ToolResult(
                ok=False,
                content=content,
                error=f"the tool {tool_name!r} reported an error",
            )

        return ToolResult(ok=True, content=content)

    call.__name__ = f"mcp_{tool_name}"
    call.__doc__ = f"Call the MCP tool {tool_name!r} on the connected server."
    return call


def register_mcp_tools(registry: ToolRegistry, client: SupportsToolCalls) -> list[ToolSpec]:
    """Register every tool the client's server offers, and return the specs.

    Dynamic on purpose: the list comes from the server, and a name is never
    written down here.

    A name already in the registry is a hard failure, not something to work
    around. Renaming or prefixing the incoming tool would put a name in front of
    the model that the server does not answer to, and the model would call it
    and be told no such tool exists — a bug that surfaces one layer away from
    its cause. `ToolRegistry.register` already refuses a duplicate; this simply
    does not try to be cleverer than that.
    """
    registered: list[ToolSpec] = []

    for tool in client.list_tools():
        spec = ToolSpec(
            name=tool.name,
            description=tool.description or MISSING_DESCRIPTION,
            fn=_make_caller(client, tool.name),
            parameters_schema=tool.input_schema,
        )
        registered.append(registry.register(spec))

    return registered
