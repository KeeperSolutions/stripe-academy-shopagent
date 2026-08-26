"""The catalog MCP server — transport skeleton (D4, step 1).

This step proves the transport and nothing else. There is one tool, `ping`,
which returns a constant; the three real catalog tools arrive in step 2. The
module deliberately imports nothing from `catalog/`, so that a failure here can
only be a transport or lifecycle failure and never a search bug wearing a
transport costume.

**The SDK is not the one the plan describes.** `notes/plans` says `FastMCP`,
which was the v1 entry point and does not exist in `mcp==2.0.0`:
`mcp.server.fastmcp` was renamed to `mcp.server.mcpserver` and `FastMCP` to
`MCPServer`. The decorators and handler signatures survived the rename intact,
so this is an import change rather than a redesign, but a v1 tutorial will not
run against the pinned version.

**Nothing here may write to stdout.** stdio transport *is* stdout: a stray
`print` lands in the middle of a JSON-RPC frame and the client drops the
connection with a parse error that names neither the print nor the tool. The
logger below writes to stderr, which the client leaves alone, and it is the
only reporting channel this module gets.
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version

from mcp.server import MCPServer

logger = logging.getLogger(__name__)

# What the client sees in the server list. It names the surface rather than the
# project, because D5 adds a second source of tools (local HTTP commerce) and
# "shopagent" alone would not say which of the two answered.
SERVER_NAME = "shopagent-catalog"


def _package_version() -> str:
    """Report the installed package version, so it cannot drift from pyproject.

    Read at runtime rather than copied into a literal: the handshake then
    reports what is actually running. An uninstalled source tree has no
    metadata to read, which is not worth failing a server over.
    """
    try:
        return version("shopagent")
    except PackageNotFoundError:  # pragma: no cover - only outside an install
        return "0.0.0+unknown"


server = MCPServer(SERVER_NAME, version=_package_version())


@server.tool()
def ping() -> str:
    """Check that the catalog server is reachable.

    Diagnostic only, and deliberately so: it takes no arguments, touches no
    database and returns a fixed string. That is what makes it useful. When a
    catalog tool fails, `ping` separates the two explanations — if `ping`
    answers, the transport and the server process are healthy and the fault is
    in the catalog or the database behind it; if `ping` does not answer, no
    result from any other tool means anything.

    Returns the string "pong". It says nothing whatsoever about the catalog,
    and a successful call is not evidence that any product exists.
    """
    logger.info("ping")
    return "pong"


def main() -> None:
    """Serve over stdio until the client disconnects.

    `run` is synchronous and opens the event loop itself, so there is no
    `anyio.run` here. Logging is configured to stderr on the way in; without a
    handler the SDK's own warnings would be invisible, which is a bad trade in
    a process whose only other output channel is a protocol stream.
    """
    logging.basicConfig(level=logging.INFO)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
