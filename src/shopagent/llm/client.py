"""Thin wrapper around the OpenAI SDK (D1).

The only place in the project allowed to import `openai`. Everything else goes
through LLMClient, so swapping providers touches exactly one file.

Uses the Chat Completions API, not Responses. The goal of D1 is to see the
agent loop from the inside — Responses keeps conversation state on the server
and would hide precisely what needs to be learned here.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from shopagent.config import get_settings
from shopagent.llm.usage import CallUsage, UsageTracker

Message = dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """One tool call the model asked for."""

    id: str
    name: str
    # Exactly the string the model produced. It is NOT parsed here: the model
    # does not always emit valid JSON, and validating it is the registry's job.
    # Parsing it twice, in two places, is how the two drift apart.
    arguments: str


@dataclass(frozen=True)
class AssistantMessage:
    """One assistant turn, with the SDK left behind.

    `chat()` returns text because D1 needed nothing else. A tool loop needs the
    whole turn — the text may be absent, the tool calls may be several — and
    letting the SDK's own object out of this module would put knowledge of the
    provider into the agent loop, which is the one thing this file exists to
    prevent.
    """

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: CallUsage | None = None

    def to_message(self) -> Message:
        """This turn as the dict to append to the conversation history.

        The tool calls are replayed verbatim, ids included. The next request
        matches each `tool` message to them by id, so a regenerated or dropped
        id turns the following call into a 400.
        """
        message: Message = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            # Omitted entirely when empty: `"tool_calls": []` is rejected.
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        return message


def _cached_tokens(usage: Any) -> int:
    """Cached input tokens from a usage object; 0 if the SDK does not report them.

    On Chat Completions this is `usage.prompt_tokens_details.cached_tokens`.
    The field is optional and depends on the SDK version and the model, so a
    missing value means "no cache", never an error.
    """
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    return getattr(details, "cached_tokens", 0) or 0


class LLMClient:
    """Send messages, get a response. Usage is recorded along the way."""

    def __init__(self, tracker: UsageTracker | None = None) -> None:
        settings = get_settings()
        # api_key MUST be passed explicitly: the key deliberately never reaches
        # os.environ (config.py is the sole reader of the environment), so a
        # bare OpenAI() would always fail with "Missing credentials".
        self._client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.reasoning_effort = settings.openai_reasoning_effort
        self.tracker = tracker if tracker is not None else UsageTracker()

    def _record(self, usage: Any) -> CallUsage | None:
        if usage is None:
            return None
        return self.tracker.record(
            model=self.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached_tokens=_cached_tokens(usage),
        )

    def chat(
        self, messages: Sequence[Message], temperature: float | None = None
    ) -> tuple[str, CallUsage]:
        """Blocking call: return the response text and this call's usage.

        `temperature=None` means the parameter is not sent at all, which is not
        the same as sending its default value. `gpt-5.6-luna` accepts only its
        own default; any explicit value returns `400 unsupported_value`. The
        parameter exists for models that do support it (e.g. `gpt-4o-mini`).
        The caller decides — this client never guesses what a model supports.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            **({} if temperature is None else {"temperature": temperature}),
        )
        call = self._record(response.usage)
        if call is None:
            # Should not happen on non-streaming calls, but recording a zero
            # call beats letting the tracker silently skip one.
            call = self.tracker.record(self.model, 0, 0)
        return response.choices[0].message.content or "", call

    def chat_with_tools(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> AssistantMessage:
        """Send the conversation plus tool schemas; return the whole turn.

        Non-streaming, deliberately. Reconstructing tool calls from a stream
        means accumulating deltas per index — the name arrives in one chunk,
        the arguments in fragments across the next, the id only once — and D2
        is about chaining calls, not about that bookkeeping. `stream_chat`
        stays for text-only conversation.

        `tools` is the output of `ToolRegistry.openai_schemas()`. An empty list
        is not sent: the API rejects `tools: []`, which is not the same thing
        as "no tools".

        `reasoning_effort` rides along only when tools do. gpt-5.6-luna returns
        400 for function tools on Chat Completions unless it is 'none', and
        this project stays on Chat Completions deliberately; a model that does
        not know the parameter returns 400 if it is sent, so it comes from
        configuration rather than from a guess made here.
        """
        extra: dict[str, Any] = {}
        if tools:
            extra["tools"] = list(tools)
            if self.reasoning_effort:
                extra["reasoning_effort"] = self.reasoning_effort

        response = self._client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            **extra,
        )
        call = self._record(response.usage)
        if call is None:
            call = self.tracker.record(self.model, 0, 0)

        message = response.choices[0].message
        tool_calls = [
            ToolCall(
                id=raw.id,
                name=raw.function.name,
                arguments=raw.function.arguments or "",
            )
            # The response type is a union discriminated by `type`; custom tool
            # calls carry no `function` and nothing here can dispatch them.
            for raw in (message.tool_calls or [])
            if getattr(raw, "type", "function") == "function"
        ]
        return AssistantMessage(
            content=message.content, tool_calls=tool_calls, usage=call
        )

    def stream_chat(
        self, messages: Sequence[Message], temperature: float | None = None
    ) -> Iterator[str]:
        """Stream the response, yielding text deltas only.

        Usage is recorded once the stream reaches its end — a generator that is
        never exhausted writes nothing to the tracker unless it is closed.

        `temperature=None` means the parameter is not sent at all. `gpt-5.6-luna`
        accepts only its own default; any explicit value returns
        `400 unsupported_value`. The parameter exists for models that do
        support it (e.g. `gpt-4o-mini`).
        """
        # create() sits OUTSIDE the try block on purpose. A request that fails
        # here (404 unknown model, 400 bad parameter) never reaches the finally
        # branch and so leaves no tracker entry, while a stream that starts and
        # is then interrupted does get recorded. The distinction is intentional:
        # a rejected request is never billed, an interrupted stream is.
        # Please do not "tidy" this line into the try block.
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            **({} if temperature is None else {"temperature": temperature}),
            stream=True,
            # Without this, usage never arrives at all when streaming.
            stream_options={"include_usage": True},
        )
        recorded = False
        try:
            for chunk in stream:
                # Usage arrives in the final chunk, which carries an EMPTY
                # choices list. That is why usage is handled first and choices
                # is checked separately — chunk.choices[0] would raise
                # IndexError on that chunk.
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    self._record(usage)
                    recorded = True

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        finally:
            if not recorded:
                # Interrupted before the final chunk (Ctrl+C, break, exception):
                # usage never arrived, so there are no numbers. A zeroed entry
                # is recorded to keep the session's call count accurate — an
                # incomplete record beats no record, and numbers are never
                # invented.
                self.tracker.record(self.model, 0, 0)
