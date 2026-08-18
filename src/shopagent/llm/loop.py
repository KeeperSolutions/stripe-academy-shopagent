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
"""

from __future__ import annotations

import sys

from shopagent.llm.client import LLMClient, Message, ToolCall
from shopagent.llm.usage import UsageTracker
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


def _initial_messages() -> list[Message]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


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


def main() -> None:
    tracker = UsageTracker()
    try:
        client = LLMClient(tracker=tracker)
    except Exception as exc:  # invalid/missing key, broken config
        print(f"[error] could not create the client: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    messages = _initial_messages()
    tools = REGISTRY.openai_schemas()

    print(
        f"ShopAgent · model {client.model} · "
        f"{len(tools)} tools ({', '.join(REGISTRY.names())}) · /help for commands"
    )

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
            messages = _initial_messages()
            print("[conversation history cleared; session cost is kept]")
            continue
        if user_input == "/tools":
            for spec in REGISTRY.specs():
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
            run_tool_loop(client, REGISTRY, messages, tools)
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
