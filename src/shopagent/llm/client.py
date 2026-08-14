"""Thin wrapper around the OpenAI SDK (D1).

The only place in the project allowed to import `openai`. Everything else goes
through LLMClient, so swapping providers touches exactly one file.

Uses the Chat Completions API, not Responses. The goal of D1 is to see the
agent loop from the inside — Responses keeps conversation state on the server
and would hide precisely what needs to be learned here.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from openai import OpenAI

from shopagent.config import get_settings
from shopagent.llm.usage import CallUsage, UsageTracker

Message = dict[str, Any]


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
