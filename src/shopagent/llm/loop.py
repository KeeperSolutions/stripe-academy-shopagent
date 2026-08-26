"""CLI agent with tool calling and cost tracking (D1→D2).

    python -m shopagent.llm.loop

The conversation history is a plain list of dicts, deliberately kept visible in
`main()` rather than hidden behind a class — the `tool` messages that D2 adds
to it are most of what there is to learn here.

One user input can take several model calls: the model asks for a tool, reads
the result, then asks for another. `run_tool_loop` is therefore a `while`, not
an `if`. It is non-streaming, because reassembling tool calls out of streamed
deltas is bookkeeping that would obscure the chaining this file is meant to
show. `LLMClient.stream_chat` is unchanged and still works; it is simply not
what the CLI drives now.

D5 added the catalog, and the shape of that change is the point. `run_tool_loop`
below is byte-for-byte what D2 left: it takes a registry and a list of schemas
and does not care where either came from. Everything D5 needed happens before
the loop starts — assembling the registry and owning the subprocess — so the
tools reached over a pipe and the two defined in `tools/basic.py` are the same
kind of thing by the time the loop sees them. That was the bet D2 made when it
made the registry a parameter instead of a global, and this file is where it
either paid off or did not.
"""

from __future__ import annotations

import sys
from contextlib import ExitStack
from dataclasses import dataclass

from shopagent.config import get_settings
from shopagent.llm.client import LLMClient, Message, ToolCall
from shopagent.llm.usage import UsageTracker
from shopagent.mcp_client.client import MCPToolClient
from shopagent.mcp_client.registration import register_mcp_tools
from shopagent.tools.basic import REGISTRY
from shopagent.tools.registry import ToolRegistry, ToolResult

SYSTEM_PROMPT = (
    "You are ShopAgent, an online shopping assistant. "
    "Always reply in English, regardless of the language of the question. "
    "Keep answers short and concrete, with no preamble and no restating of "
    "the question. If you do not know something or lack the data, say so "
    "plainly — never guess. "
    "Use the tools for anything they cover: you have no clock, so never state "
    "a time from memory, and never do arithmetic in your head."
)

# What the catalog tools are for, not how they work. Each one already carries a
# description written for a model to read, and repeating that here would give
# the same contract two authors and let them drift. This says only when to
# reach for them, and what is never allowed to come from memory.
CATALOG_PROMPT = (
    " The product catalogue is available through tools. Every product name, "
    "price, size, colour and stock level you state must have come from a tool "
    "result in this conversation — never from memory, and never inferred from "
    "what a product sounds like. When the user asks about products, search "
    "first and answer from what comes back. If a search returns a count of 0, "
    "say plainly that nothing matched and suggest a broader search; do not "
    "offer a product that was not in a result."
)

# Said when the catalog server could not be reached, so the model does not
# apologise for its own memory when the real answer is that a tool is missing.
NO_CATALOG_PROMPT = (
    " The product catalogue is NOT available in this session: the tools that "
    "search it could not be loaded. If the user asks about products, prices or "
    "stock, say the catalogue is unavailable right now. Do not answer from "
    "memory and do not invent products."
)

# One user input may legitimately need several rounds: a tool call, a look at
# the result, another call. Eight leaves room for the D9 chain (search, check
# stock, add to cart, view cart, checkout — five) plus a couple of rounds spent
# correcting a rejected argument, while still capping what one input can spend.
# A model that has not finished by then is looping, not working.
MAX_TOOL_ITERATIONS = 8

# Tool output goes to the terminal as well as to the model. The model gets all
# of it; the terminal gets this much, so one long catalogue result cannot bury
# the conversation.
MAX_SHOWN_RESULT = 300

HELP = """\
Commands:
  /cost    cost and call count for this session
  /reset   clear the conversation history (cost is kept)
  /tools   the tools available to the model
  /help    this list
  /exit    quit"""


def _initial_messages(catalog_available: bool = True) -> list[Message]:
    extra = CATALOG_PROMPT if catalog_available else NO_CATALOG_PROMPT
    return [{"role": "system", "content": SYSTEM_PROMPT + extra}]


def _shorten(text: str, limit: int = MAX_SHOWN_RESULT) -> str:
    """Collapse a result to one readable line for the terminal."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return f"{flat[:limit]}… ({len(text):,} chars)"


def _show_tool_call(call: ToolCall, index: int, total: int) -> None:
    print(f"  [tool {index}/{total}] {call.name}({_shorten(call.arguments, 160)})")


def _show_tool_result(result: ToolResult) -> None:
    print(f"  [{'    ok    ' if result.ok else '   error  '}] {_shorten(result.content)}")


def run_tool_loop(
    client: LLMClient,
    registry: ToolRegistry,
    messages: list[Message],
    tools: list[dict],
) -> None:
    """Drive one user input to a final answer, running tools along the way.

    `messages` is appended to in place, and is left in a state the API will
    accept whichever branch is taken: every assistant turn that requested tools
    is followed by one `tool` message per call, with matching ids. Leaving a
    single call unanswered makes the next request a 400, which is why the tool
    messages are appended in the same pass that produced them.
    """
    for _ in range(MAX_TOOL_ITERATIONS):
        reply = client.chat_with_tools(messages, tools)
        messages.append(reply.to_message())

        if reply.content:
            # The model often narrates before calling a tool; that text is part
            # of the answer either way.
            print(f"\nshopagent> {reply.content}")

        if not reply.tool_calls:
            if not reply.content:
                print("\nshopagent> [empty answer]")
            return

        total = len(reply.tool_calls)
        for index, call in enumerate(reply.tool_calls, start=1):
            _show_tool_call(call, index, total)
            # dispatch never raises: a bad name, unparsable arguments, failed
            # validation and an exception inside the tool all come back as a
            # ToolResult, and the text goes to the model rather than to stdout
            # so it can correct itself on the next round.
            result = registry.dispatch(call.name, call.arguments)
            _show_tool_result(result)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result.content,
            })

    print(
        f"\n[stopped after {MAX_TOOL_ITERATIONS} tool rounds without a final "
        f"answer — the model kept asking for tools]"
    )
    # Told to the model too, as a system note rather than as something the
    # assistant said, so the next turn knows why its chain was cut short.
    messages.append({
        "role": "system",
        "content": (
            f"The tool call limit of {MAX_TOOL_ITERATIONS} rounds was reached "
            f"for that request, so it was stopped. Answer with what you have, "
            f"or tell the user what you could not determine."
        ),
    })


@dataclass(frozen=True)
class ToolSetup:
    """The tools one session runs with, and whether the catalog is among them."""

    registry: ToolRegistry
    catalog_available: bool
    note: str | None = None


def build_tool_setup(
    stack: ExitStack,
    *,
    catalog_enabled: bool | None = None,
    client_factory: type[MCPToolClient] = MCPToolClient,
) -> ToolSetup:
    """Assemble the registry this session will use.

    The local tools always go in. The catalog tools are added when the switch
    is on and the server actually starts, and their absence is reported rather
    than raised: a catalog that will not open is a smaller problem than a CLI
    that will not start, and the model is told about it either way.

    `stack` owns the client. Whatever ends the session — a clean `/exit`, a
    Ctrl+C at the prompt, or an exception on the way out — unwinds it, and the
    server subprocess goes with it.

    `client_factory` exists so a test can inject a client that fails to start.
    """
    registry = ToolRegistry()
    for spec in REGISTRY.specs():
        registry.register(spec)

    if catalog_enabled is None:
        catalog_enabled = get_settings().mcp_catalog_enabled

    if not catalog_enabled:
        return ToolSetup(
            registry=registry,
            catalog_available=False,
            note="catalog disabled (MCP_CATALOG_ENABLED=false)",
        )

    try:
        client = stack.enter_context(client_factory())
        register_mcp_tools(registry, client)
    except Exception as exc:  # noqa: BLE001 - a missing catalog must not be fatal
        # Broad on purpose, and the same reasoning as `dispatch`: the server is
        # a separate process over a pipe, and everything from a bad interpreter
        # path to a failed handshake to an unreachable database arrives here.
        # None of them is worth refusing to start over.
        return ToolSetup(
            registry=registry,
            catalog_available=False,
            note=f"catalog unavailable ({type(exc).__name__}: {exc})",
        )

    return ToolSetup(registry=registry, catalog_available=True)


def main() -> None:
    tracker = UsageTracker()
    try:
        client = LLMClient(tracker=tracker)
    except Exception as exc:  # invalid/missing key, broken config
        print(f"[error] could not create the client: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    with ExitStack() as stack:
        setup = build_tool_setup(stack)
        _run_session(client, tracker, setup)


def _run_session(client: LLMClient, tracker: UsageTracker, setup: ToolSetup) -> None:
    """The REPL itself, once the tools are decided.

    Split out from `main` so the catalog's lifetime is visibly the session's
    lifetime: everything below runs inside the `ExitStack` that owns the server.
    """
    registry = setup.registry
    messages = _initial_messages(setup.catalog_available)
    tools = registry.openai_schemas()

    print(
        f"ShopAgent · model {client.model} · "
        f"{len(tools)} tools ({', '.join(registry.names())}) · /help for commands"
    )
    if setup.note:
        print(f"[{setup.note}]")

    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+C or Ctrl+D at the prompt means a clean exit.
            print()
            break

        if not user_input:
            continue

        if user_input == "/exit":
            break
        if user_input == "/help":
            print(HELP)
            continue
        if user_input == "/cost":
            print(tracker.summary())
            continue
        if user_input == "/reset":
            messages = _initial_messages(setup.catalog_available)
            print("[conversation history cleared; session cost is kept]")
            continue
        if user_input == "/tools":
            for spec in registry.specs():
                print(f"  {spec.name}: {spec.description}")
            continue

        # Where to rewind to if this turn fails part-way. A turn can append
        # several messages, and an assistant turn whose tool calls never got
        # their `tool` messages would make every later request a 400 — so a
        # broken turn is removed whole rather than patched up.
        history_length = len(messages)
        messages.append({"role": "user", "content": user_input})
        calls_before = len(tracker.calls)

        try:
            run_tool_loop(client, registry, messages, tools)
        except KeyboardInterrupt:
            # Aborts this answer only — the application stays alive.
            print("\n[interrupted]")
            del messages[history_length:]
        except Exception as exc:
            print(f"\n[error] {type(exc).__name__}: {exc}")
            del messages[history_length:]

        _print_cost(tracker, calls_before)

    print()
    print(tracker.summary())


def _print_cost(tracker: UsageTracker, calls_before: int) -> None:
    """Print the cost of calls made in this turn, then the session total."""
    for call in tracker.calls[calls_before:]:
        print(
            f"[{call.total_tokens:,} tokens "
            f"(prompt {call.prompt_tokens:,}, answer {call.completion_tokens:,}"
            f"{f', cached {call.cached_tokens:,}' if call.cached_tokens else ''}) "
            f"· ${call.cost_usd:.6f} "
            f"· session ${tracker.total_cost_usd:.6f}]"
        )


if __name__ == "__main__":
    main()
