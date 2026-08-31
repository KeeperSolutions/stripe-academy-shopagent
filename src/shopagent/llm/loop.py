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

D9 took the system prompt out to `agent/prompt.py`. This file is a mechanism
— a `while`, a message list, a dispatch — and what it says to the model is
policy; keeping them apart is what lets a sentence about quoting prices be
edited without opening the loop that D2 and D5 both claimed had not changed.

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
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass

from shopagent.agent import profile as profiles
from shopagent.agent.guardrails import GuardedClient, GuardedRegistry
from shopagent.agent.memory import ConversationMemory
from shopagent.agent.prompt import PROFILE_LABELS, initial_messages
from shopagent.config import get_settings
from shopagent.db import session_scope
from shopagent.llm.client import LLMClient, Message, ToolCall
from shopagent.llm.usage import UsageTracker
from shopagent.mcp_client.client import MCPToolClient
from shopagent.mcp_client.registration import register_mcp_tools
from shopagent.tools.basic import REGISTRY
from shopagent.tools.commerce import register_commerce_tools
from shopagent.tools.http import CommerceAPI
from shopagent.tools.registry import ToolRegistry, ToolResult

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
  /cost              cost and call count for this session
  /reset             clear the conversation history (cost is kept)
  /tools             the tools available to the model
  /profile           what the shop remembers about you between conversations
  /remember k=v      record one field: display_name, shoe_size, clothing_size,
                     favourite_categories (from: shoes, jackets, bags,
                     accessories, equipment)
  /forget k          clear one field
  /help              this list
  /exit              quit"""


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
    # Everything this conversation holds outside its message list (D9, step
    # 3): the cart and order ids the model is never shown, the last search in
    # the order it came back, and every variant id that has appeared in a
    # result. Exposed here because the CLI and the tests are the only things
    # that can see it — nothing is put in front of the model.
    memory: ConversationMemory | None = None


def build_tool_setup(
    stack: ExitStack,
    *,
    catalog_enabled: bool | None = None,
    client_factory: type[MCPToolClient] = MCPToolClient,
    confirm: Callable[[str], bool] | None = None,
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
    # The registry is the one thing every tool call passes through, which is
    # what makes it the place a conversation's memory is filled: nothing in the
    # loop has to remember to record anything, and a tool added later is
    # remembered because of where it is registered rather than because whoever
    # added it knew to.
    memory = ConversationMemory()
    registry = GuardedRegistry(memory, confirm=confirm)
    for spec in REGISTRY.specs():
        registry.register(spec)

    # The commerce tools go in unconditionally, and before the catalog is even
    # considered. They reach a different service over a different protocol, so
    # `MCP_CATALOG_ENABLED` has no business deciding whether a cart exists —
    # and a session that can list a basket but not search is a comprehensible
    # state, where a switch that turned off both would make one failure look
    # like the other.
    #
    # Nothing here can fail the way the catalog can: building the client opens
    # no connection, so an API that is down is discovered at the first call and
    # answered by the tool itself, in words written for the model. The stack
    # owns the client for the same reason it owns the MCP subprocess — whatever
    # ends the session closes the sockets.
    register_commerce_tools(registry, stack.enter_context(CommerceAPI()), memory)

    if catalog_enabled is None:
        catalog_enabled = get_settings().mcp_catalog_enabled

    if not catalog_enabled:
        return ToolSetup(
            registry=registry,
            catalog_available=False,
            memory=memory,
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
            memory=memory,
            note=f"catalog unavailable ({type(exc).__name__}: {exc})",
        )

    return ToolSetup(registry=registry, catalog_available=True, memory=memory)


def main() -> None:
    tracker = UsageTracker()
    try:
        client = LLMClient(tracker=tracker)
    except Exception as exc:  # invalid/missing key, broken config
        print(f"[error] could not create the client: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    with ExitStack() as stack:
        setup = build_tool_setup(stack, confirm=_ask_to_confirm)
        _run_session(client, tracker, setup)


def _run_session(client: LLMClient, tracker: UsageTracker, setup: ToolSetup) -> None:
    """The REPL itself, once the tools are decided.

    Split out from `main` so the catalog's lifetime is visibly the session's
    lifetime: everything below runs inside the `ExitStack` that owns the server.
    """
    registry = setup.registry
    # The client is wrapped, not replaced: `run_tool_loop` below is the
    # unmodified D2 function and still takes a client, a registry, a message
    # list and a list of schemas. What changed is that one of those four checks
    # the answer before handing it back.
    client = GuardedClient(client, setup.memory)
    shopper_id = get_settings().shopper_id
    profile, profile_note = profiles.load_for_session(shopper_id)
    messages = initial_messages(setup.catalog_available, profile=profile)
    tools = registry.openai_schemas()

    print(
        f"ShopAgent · model {client.model} · "
        f"{len(tools)} tools ({', '.join(registry.names())}) · /help for commands"
    )
    if setup.note:
        print(f"[{setup.note}]")
    if profile_note:
        print(f"[{profile_note}]")
    if profile is not None:
        print("[profile loaded; /profile to see it]")

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
            # The profile is re-read rather than reused, so a `/remember` made
            # during this session takes effect here. It deliberately does not
            # take effect mid-conversation: rewriting a system message the
            # model has already been answering from would change the rules
            # under it without anything in the transcript saying so.
            profile, _ = profiles.load_for_session(shopper_id)
            messages = initial_messages(setup.catalog_available, profile=profile)
            print("[conversation history cleared; session cost is kept]")
            continue
        if user_input == "/profile":
            _show_profile(shopper_id, profile)
            continue
        if user_input.startswith("/remember"):
            _remember(shopper_id, user_input.removeprefix("/remember").strip())
            continue
        if user_input.startswith("/forget"):
            _forget(shopper_id, user_input.removeprefix("/forget").strip())
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



def _ask_to_confirm(summary: str) -> bool:
    """Show what is being bought and wait for a person to answer (D9, step 5).

    The summary comes from `agent/guardrails.py`, which built it from a real
    `view_cart` call — not from anything the model said. This function only
    prints it and reads a line.

    Anything that is not an explicit yes is a no, including end-of-input. A
    piped session, a closed terminal or a stray newline must not buy anything:
    the safe answer to "could not ask" is the same as the answer to "they said
    no", and it is the only one that cannot cost somebody money.
    """
    print("\n  About to place this order:")
    print(summary)
    try:
        answer = input("  Place the order? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  [not confirmed]")
        return False
    return answer in {"y", "yes"}


# --- the profile commands (D9, step 4) -----------------------------------
#
# A CLI command rather than a tool, and that is the security decision of this
# step rather than a convenience. A write *tool* would be the eleventh in a
# list already at ten, and it would be the model deciding what counts as a
# preference — turning a customer's sentence into stored text, which is
# precisely the free text this feature is built to not have. A command means a
# person names the field and the value, and every value is validated against a
# domain before it can reach a system prompt.


def _require_shopper(shopper_id: str | None) -> bool:
    if shopper_id:
        return True
    print(
        "[no SHOPPER_ID is configured, so there is nothing to remember against. "
        "Set it in .env — see .env.example.]"
    )
    return False


def _show_profile(shopper_id: str | None, injected) -> None:
    """What is stored, and whether the running conversation has it yet.

    Read from the database rather than from the copy loaded at startup: the
    question `/profile` asks is what the shop remembers, and a `/remember`
    made a minute ago has changed that. The copy in the prompt is a different
    thing and deliberately does not move mid-conversation, so the two are
    reported separately rather than one being passed off as the other.
    """
    stored, note = profiles.load_for_session(shopper_id)
    if note:
        print(f"[{note}]")
        return
    if stored is None:
        print("[nothing is remembered about you yet; /remember k=v to record something]")
        return

    for name, label in PROFILE_LABELS.items():
        value = getattr(stored, name, None)
        if value:
            text = ", ".join(value) if isinstance(value, tuple) else value
            print(f"  {label}: {text}")
    if stored != injected:
        print("[this conversation is still running on the profile it started with; "
              "/reset to pick up the change]")


def _remember(shopper_id: str | None, argument: str) -> None:
    if not _require_shopper(shopper_id):
        return
    field_name, separator, value = argument.partition("=")
    if not separator:
        print('[usage: /remember field=value, for example /remember shoe_size=42]')
        return
    try:
        with session_scope() as session:
            profiles.remember(session, shopper_id, field_name.strip(), value.strip())
    except profiles.ProfileFieldError as refused:
        # The refusal is read by a person, so it is printed as it was written
        # rather than turned into a stack trace.
        print(f"[not recorded: {refused}]")
    except Exception as exc:  # noqa: BLE001 - a failed write must not end the session
        print(f"[not recorded: {type(exc).__name__}: {exc}]")
    else:
        print(f"[recorded; it reaches the assistant on /reset or the next run]")


def _forget(shopper_id: str | None, field_name: str) -> None:
    if not _require_shopper(shopper_id):
        return
    try:
        with session_scope() as session:
            profiles.forget(session, shopper_id, field_name.strip())
    except profiles.ProfileFieldError as refused:
        print(f"[not cleared: {refused}]")
    except Exception as exc:  # noqa: BLE001
        print(f"[not cleared: {type(exc).__name__}: {exc}]")
    else:
        print("[cleared; it reaches the assistant on /reset or the next run]")

if __name__ == "__main__":
    main()
