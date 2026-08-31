"""What must not leave this machine, and what replaces it (D10, step 2).

D6 settled a version of this question and the answer is on the shelf: the MCP
server logs every tool call with its arguments, `query` is the one argument
that is free text a shopper typed, so `redact_arguments()` replaces it with a
salted digest and `MCP_LOG_REDACT_QUERY` turns that off on a developer's own
machine. The reasoning is in CLAUDE.md and the closed gap is in JOURNAL.

**A trace is that question again, one step worse, and the repository was
contradicting itself.** The MCP log stays on this disk and still redacts. A
trace carries the same arguments *plus* the profile name, the amounts, the
order id and the whole conversation to a third party, over the network, and
would have redacted nothing. The stricter rule was on the weaker path.

**`query` cannot be decided on its own, which is the finding this module is
built around.** It is not the only free text in a trace and it is not even the
first: the customer's own message is what produced it, the model's answer
quotes it back, and the system prompt carries `display_name` from the profile —
measured, not feared, in the D10 step 1 live run, where the model opened its
answer with the customer's first name. Redacting `query` while sending the
sentence it was derived from is theatre. So there is one rule and one switch,
and they cover every field a person wrote:

    TRACE_REDACT_TEXT (default true)

**What is redacted**: the `query` argument, the content of user messages, the
content of assistant messages, and the system prompt. **What is not**: tool
names, every other tool argument (a variant id, a category from a closed set, a
price bound, a limit), tool results, amounts, order ids, product names, token
counts, cost, latency, and which guardrail refused what. None of those is
anybody's personal data; they are this shop's own data and this process's own
measurements, and they are the reason a trace is worth having.

**The cost, stated plainly**: with the switch on, a trace cannot answer "what
did the customer say" or "what did the model understand them to want". It can
answer what the plan actually asks of it — what the conversation cost, which
tools ran in what order, which guardrail fired, and where the time went. Turning
it off on your own machine gives the rest back, which is the same bargain
`MCP_LOG_REDACT_QUERY` offers and for the same reason: the safe setting must
not be the one somebody has to remember to type.

**A digest rather than a blank**, exactly as D6 argued. `<redacted>` everywhere
would lose the one question a redacted trace can still answer — whether the
same text appeared twice — and that is the question a repeated search or a
customer repeating themselves is read from. Keyed rather than plain, because
the space of things a shopper types is small enough that a wordlist recovers a
bare SHA-256 of it. No length beside it, because with the salt in place the
length is the only thing left that could narrow a guess.

**This is a sibling of `redact_arguments()` and deliberately not an import of
it.** Two reasons, and the second is the real one. The MCP server runs in a
separate process, so its per-process salt is a different secret from this one
and sharing the function would not share the salt anyway. And `mcp_server/` is
a thin protocol wrapper by rule — giving it a dependency on this project's
observability so that `obs/` could borrow eight lines would invert the one
direction that module is allowed to point in.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any

from shopagent.config import get_settings

# The tool argument that is free text. The same name D6 named, for the same
# reason, and named here rather than imported for the reason in the docstring.
SENSITIVE_ARGUMENT = "query"

# The roles whose `content` is something a person wrote or something written
# back to them. `tool` is absent on purpose: a tool message is a result this
# shop produced, which is the half of the conversation the trace is for.
#
# `system` is present because of what is *in* it rather than who wrote it. The
# prompt is this project's own text — except for the profile block, which
# carries `display_name`, a string the customer typed about themselves. A rule
# that let the system message through because "we wrote it" would be true of
# every byte but the one that matters.
REDACTED_ROLES = frozenset({"system", "user", "assistant"})

# A per-process salt, generated at import and never logged, never sent. It is
# what makes these digests correlatable within one conversation without being
# readable by anybody — including by whoever is reading the trace.
#
# Per-process rather than configured, the same choice D6 made: the question a
# digest answers is "did this text appear twice in this conversation", and a
# conversation lives inside one process. A stable salt would be a long-lived
# secret to store, rotate and eventually leak, bought for a correlation nobody
# has asked for.
_SALT = os.urandom(32)


def digest(text: str) -> str:
    """Eight hex characters of a keyed digest — an equality token, not a hash.

    Truncated for the reason D6 truncates: what is needed is something that
    compares equal to itself, and sixty-four characters in every span's input
    would push everything worth reading off the side of the screen.
    """
    return hmac.new(_SALT, text.encode("utf-8"), hashlib.sha256).hexdigest()[:8]


def redacting() -> bool:
    """Whether free text is being replaced before it leaves the process."""
    return get_settings().trace_redact_text


def redact_text(text: Any) -> Any:
    """One free-text value, as it should appear in a trace.

    Anything that is not a string comes back untouched: a trace is not worth an
    exception, and a caller is free to hand this whatever it has.
    """
    if not redacting() or not isinstance(text, str):
        return text
    if not text:
        # An empty string is not text somebody wrote, and a digest of it would
        # be the same eight characters in every trace this project ever
        # produces — a constant that reads like a value.
        return text
    return f"<redacted:{digest(text)}>"


def redact_arguments(arguments: Any) -> Any:
    """A tool call's arguments, as they should appear in a trace.

    Replaces `query` and leaves everything else alone. Everything else is an
    id, a category from a closed set, a price bound or a limit — values already
    visible in the tool schema the model reads, and the reason a tool span is
    worth looking at.

    A *string* is redacted whole. That is the payload `ToolRegistry.dispatch`
    could not decode either, so nothing here knows which part of it is free
    text — and the rule this module enforces is about what a person wrote, not
    about what happens to parse. Debugging a malformed call from the digest is
    worse than debugging it from the text, and it is the trade the default
    already makes everywhere else.
    """
    if not redacting():
        return arguments
    if isinstance(arguments, str):
        return redact_text(arguments)
    if not isinstance(arguments, dict):
        return arguments
    value = arguments.get(SENSITIVE_ARGUMENT)
    if not isinstance(value, str) or not value:
        return arguments
    return {**arguments, SENSITIVE_ARGUMENT: redact_text(value)}


def redact_tool_calls(tool_calls: Any) -> Any:
    """The tool calls on an assistant message, as they should appear.

    **This is the leak the first version of this module shipped**, and the
    comment that caused it said the arguments were "handled where the tool
    itself is traced". They are — and the message list replays them *again*,
    verbatim, into the input of every later generation. Measured on a real
    trace: `"query":"trail running shoes"` appeared eighteen times in one
    conversation, in plaintext, while the `search_products` span beside it
    showed a digest.

    The reasoning was wrong in a specific way worth keeping: a field is not
    redacted because some code path redacts it, but because every path that
    carries it does. That is the same shape as D8's live-mode guard — coverage
    is a property of the paths, not of the sentence describing them.

    The test that missed it built messages with `content` and no `tool_calls`,
    which is a fixture omitting a field the real object always has: a blind
    spot with the shape of coverage, exactly as D8's `refunded_event` was.
    """
    if not redacting() or not isinstance(tool_calls, list):
        return tool_calls

    redacted = []
    for call in tool_calls:
        function = call.get("function") if isinstance(call, dict) else None
        arguments = function.get("arguments") if isinstance(function, dict) else None
        if not isinstance(arguments, str) or not arguments:
            redacted.append(call)
            continue
        redacted.append(
            {**call, "function": {**function, "arguments": _redact_argument_json(arguments)}}
        )
    return redacted


def _redact_argument_json(arguments: str) -> str:
    """One `function.arguments` string, redacted in place.

    Kept as a JSON string of the same shape rather than replaced wholesale,
    because everything in it but `query` is what makes the trace readable — the
    variant id, the category, the limit. An argument string that does not
    decode is digested whole, for the reason `redact_arguments` gives.
    """
    try:
        parsed = json.loads(arguments)
    except ValueError:
        return redact_text(arguments)
    if not isinstance(parsed, dict):
        return redact_text(arguments)
    return json.dumps(redact_arguments(parsed))


def redact_messages(messages: Any) -> Any:
    """A conversation, as it should appear in a trace.

    Every message keeps its role and its shape, so the trace still shows how
    many turns there were, which of them asked for tools and which answered.
    What goes is the prose.

    `tool_calls` on an assistant message go through `redact_tool_calls`. An
    earlier version left them alone on the argument that the tool's own span
    already redacted them, and that shipped a leak: the message list replays
    the same arguments into every later generation. See `redact_tool_calls`.
    """
    if not redacting() or not isinstance(messages, list):
        return messages

    redacted = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in REDACTED_ROLES:
            redacted.append(message)
            continue
        replacement = dict(message)
        content = message.get("content")
        if isinstance(content, str) and content:
            replacement["content"] = redact_text(content)
        if "tool_calls" in message:
            replacement["tool_calls"] = redact_tool_calls(message["tool_calls"])
        redacted.append(replacement)
    return redacted


def redact_identifier(identifier: Any) -> Any:
    """A shopper id, as it should appear on a trace.

    `SHOPPER_ID` is this project's whole notion of identity — the primary key
    of `shopper_profiles` — and on one machine it is usually a person's own
    name or handle. Digested, it still groups every conversation that shopper
    had into one Langfuse user, which is what the field is for, without saying
    who they are.

    `None` stays `None`. A conversation without a profile is an ordinary
    conversation, and a digest of nothing would invent a user.
    """
    if identifier is None:
        return None
    return redact_text(identifier)
