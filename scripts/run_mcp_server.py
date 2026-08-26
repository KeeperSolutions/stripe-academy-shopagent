"""Serve the catalog over MCP on stdio (D4, step 1).

A launcher, not a second entry point: it calls the same `main()` that
`python -m shopagent.mcp_server.server` calls, and adds nothing.

    python scripts/run_mcp_server.py

It exists because the MCP Inspector parses `-m` as one of its own flags, so
`--cli ... python -m shopagent.mcp_server.server` never reaches the module and
times out on a Python that is reading its script from stdin. A path argument
has no such collision, which makes this the form to hand any external tool:

    npx @modelcontextprotocol/inspector --cli \\
        .venv/bin/python scripts/run_mcp_server.py --method tools/list

This script prints nothing. stdout belongs to the JSON-RPC stream, and a
progress line of the sort the other scripts in this directory end with would
corrupt the first frame.
"""

from __future__ import annotations

from shopagent.mcp_server.server import main

if __name__ == "__main__":
    main()
