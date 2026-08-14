"""Streaming CLI chatbot with cost tracking (D1 deliverable).

    python -m shopagent.llm.loop

The conversation history is a plain list of dicts, deliberately kept visible in
`main()` rather than hidden behind a class. On D2 the same list receives `tool`
messages — hiding it now would make the tool loop unreadable.
"""

from __future__ import annotations

import sys

from shopagent.llm.client import LLMClient
from shopagent.llm.usage import UsageTracker

SYSTEM_PROMPT = (
    "You are ShopAgent, an online shopping assistant. "
    "Always reply in English, regardless of the language of the question. "
    "Keep answers short and concrete, with no preamble and no restating of "
    "the question. If you do not know something or lack the data, say so "
    "plainly — never guess."
)

HELP = """\
Commands:
  /cost    cost and call count for this session
  /reset   clear the conversation history (cost is kept)
  /help    this list
  /exit    quit"""


def _initial_messages() -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def main() -> None:
    tracker = UsageTracker()
    try:
        client = LLMClient(tracker=tracker)
    except Exception as exc:  # invalid/missing key, broken config
        print(f"[error] could not create the client: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    messages = _initial_messages()

    print(f"ShopAgent · model {client.model} · /help for commands")

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

        messages.append({"role": "user", "content": user_input})
        calls_before = len(tracker.calls)
        parts: list[str] = []

        print("\nshopagent> ", end="", flush=True)
        try:
            for delta in client.stream_chat(messages):
                print(delta, end="", flush=True)
                parts.append(delta)
            print()
        except KeyboardInterrupt:
            # Aborts this answer only — the application stays alive.
            print("\n[interrupted]")
        except Exception as exc:
            print(f"\n[error] {type(exc).__name__}: {exc}")
            # Drop the user message: with no answer to it the history is
            # inconsistent, and the next call would start from a broken state.
            messages.pop()
            _print_cost(tracker, calls_before)
            continue

        answer = "".join(parts)
        if answer:
            # A partial answer (after an interrupt) knowingly stays in the
            # history — the model said it, so the rest of the conversation
            # has to know about it.
            messages.append({"role": "assistant", "content": answer})
        else:
            messages.pop()

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
