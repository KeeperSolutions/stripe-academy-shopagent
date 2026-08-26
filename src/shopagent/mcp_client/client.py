"""Talking to the catalog MCP server from synchronous code (D5, step 1).

The server from D4 is a separate process reached over stdio, and the SDK that
speaks to it is asynchronous. Everything on this side of the project is not:
`llm/loop.py`, `tools/registry.py` and `catalog/` are all ordinary blocking
functions. This module is where those two facts meet, and it is the only place
in the project that has to know the SDK is async at all.

**The bridge is a blocking portal, not `anyio.run` per call.** An MCP session is
stateful: the subprocess is spawned once, the handshake runs once, and every
later `tools/call` travels the same pipes. Wrapping each call in `anyio.run`
would open an event loop, spawn a Python interpreter, handshake, ask one
question and kill it again — a process per tool call, and the D4 measurement
that the server holds one pooled database connection would become meaningless.
So a background thread runs one event loop for the client's whole lifetime, and
`anyio.from_thread.start_blocking_portal` hands synchronous callers a way to
submit coroutines to it and block for the answer. `anyio` is already a
dependency of `mcp`, so this costs no new package, and asyncio stays confined to
this file rather than spreading through the project because one module needed
it.

**Teardown order is the thing to get right.** An `ExitStack` holds the portal
and the session, and unwinds last-in-first-out: the session closes, which closes
the transport, which closes the server's stdin and then escalates to killing the
process tree; only then does the portal's loop stop. Entering the portal first
is what guarantees the loop is still running while the session needs it. Both
paths are tested — a normal close and an exception mid-conversation — because a
leaked server process is invisible for an hour and then is not.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from contextlib import ExitStack, asynccontextmanager
from typing import Any

from anyio.from_thread import start_blocking_portal
from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp_types import CallToolResult, Tool

# How the catalog server is started. The module path rather than the script in
# `scripts/`, because that is the form CLAUDE.md documents and the one that does
# not depend on a working directory; `sys.executable` rather than "python", so
# the server runs in the same virtualenv as the client that spawned it.
DEFAULT_COMMAND = sys.executable
DEFAULT_ARGS = ("-m", "shopagent.mcp_server.server")


@asynccontextmanager
async def _open_session(parameters: StdioServerParameters):
    """Spawn the server, handshake, and yield a live session.

    One context manager for all three steps so the synchronous side has a
    single thing to enter and a single thing to unwind. `initialize` is part of
    it because a session that has not handshaked cannot answer anything, and a
    caller has no use for one.
    """
    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


class MCPToolClient:
    """A synchronous handle on the tools one MCP server offers.

    Deliberately says nothing about *which* server: the tools it exposes are
    whatever `tools/list` returns. Nothing here names `search_products`, so a
    tool added to the server tomorrow arrives without a change on this side —
    which is the property D5 exists to demonstrate.

    Use it as a context manager. The server process starts on entry and is gone
    by the time the block ends.
    """

    def __init__(
        self,
        command: str = DEFAULT_COMMAND,
        args: tuple[str, ...] = DEFAULT_ARGS,
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        self._parameters = StdioServerParameters(
            command=command,
            args=list(args),
            env=dict(env) if env is not None else None,
            cwd=cwd,
        )
        self._stack: ExitStack | None = None
        self._portal: Any = None
        self._session: ClientSession | None = None

    # --- lifecycle ------------------------------------------------------

    def __enter__(self) -> MCPToolClient:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def start(self) -> None:
        """Start the event loop thread, spawn the server, and handshake."""
        if self._session is not None:
            raise RuntimeError("this client is already started")

        stack = ExitStack()
        try:
            # Entered first, so it is torn down last: the session's own
            # shutdown needs a running loop to do the process cleanup on.
            self._portal = stack.enter_context(start_blocking_portal())
            self._session = stack.enter_context(
                self._portal.wrap_async_context_manager(_open_session(self._parameters))
            )
        except BaseException:
            stack.close()
            self._portal = None
            self._session = None
            raise

        self._stack = stack

    def close(self) -> None:
        """Shut the session down and make sure the server process is gone.

        Safe to call twice, and safe to call on a client that never started —
        `__exit__` runs whatever happened inside the block.
        """
        stack, self._stack = self._stack, None
        self._session = None
        self._portal = None
        if stack is not None:
            stack.close()

    @property
    def is_started(self) -> bool:
        return self._session is not None

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError(
                "this client is not started. Use it as a context manager, "
                "or call start() first."
            )
        return self._session

    # --- the protocol, synchronously ------------------------------------

    def list_tools(self) -> list[Tool]:
        """Every tool the server advertises, in the order it lists them.

        Returns the SDK's own `Tool` objects rather than a local wrapper: they
        already carry the name, the description and both schemas, and copying
        that into a parallel dataclass would be a second definition to keep in
        step for no gain. The adapter takes them from here.
        """
        session = self._require_session()
        return list(self._run(session.list_tools()).tools)

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> CallToolResult:
        """Call one tool by name and hand back the protocol result whole.

        The result is not unwrapped or judged here. Whether the call failed is
        `result.is_error`, and turning either outcome into text for the model is
        the adapter's job — this method's only responsibility is that the
        request reaches the server and the reply comes back intact.
        """
        session = self._require_session()
        return self._run(session.call_tool(name, dict(arguments or {})))

    def _run(self, coro: Any) -> Any:
        """Run one coroutine on the client's event loop and block for it."""
        return self._portal.call(lambda: coro)
