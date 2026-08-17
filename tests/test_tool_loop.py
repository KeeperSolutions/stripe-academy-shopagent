"""Tests for the tool loop in shopagent.llm.loop.

No network. The client is replaced by a fake that hands back prepared
AssistantMessages, so each test describes one conversation the model could
drive and asserts what the loop did with it.

The invariant under most of these: every `tool_call` the model makes gets a
`tool` message with the same id, in the same turn. Miss one and the next
request is a 400 — which is the failure this file exists to prevent.
"""

import pytest
from pydantic import BaseModel, Field

from shopagent.llm.client import AssistantMessage, ToolCall
from shopagent.llm.loop import MAX_TOOL_ITERATIONS, run_tool_loop
from shopagent.llm.usage import UsageTracker
from shopagent.tools.registry import ToolRegistry


class NumberArgs(BaseModel):
    value: int = Field(ge=0, le=10, description="A whole number from 0 to 10.")


def make_registry(log: list) -> ToolRegistry:
    """A registry whose tools record every call, so the loop can be observed."""
    registry = ToolRegistry()

    @registry.tool(name="double", description="Double a number.", args_model=NumberArgs)
    def double(value: int) -> int:
        log.append(("double", value))
        return value * 2

    @registry.tool(name="negate", description="Negate a number.", args_model=NumberArgs)
    def negate(value: int) -> int:
        log.append(("negate", value))
        return -value

    return registry


class FakeClient:
    """Replays prepared replies and remembers what it was asked.

    `endless` is used once the scripted replies run out, which is how the
    runaway-model scenario is expressed without an infinite list.
    """

    def __init__(self, replies, endless=None):
        self._replies = list(replies)
        self._endless = endless
        self.model = "fake-model"
        self.tracker = UsageTracker()
        self.seen_messages: list[list[dict]] = []
        self.seen_tools: list = []

    def chat_with_tools(self, messages, tools=None):
        # Copied, because the loop keeps mutating the same list afterwards.
        self.seen_messages.append([dict(m) for m in messages])
        self.seen_tools.append(tools)

        if self._replies:
            content, tool_calls = self._replies.pop(0)
        elif self._endless is not None:
            content, tool_calls = self._endless
        else:
            raise AssertionError(
                "the loop asked for more model calls than the scenario provides"
            )

        usage = self.tracker.record(self.model, 100, 20, cached_tokens=0)
        return AssistantMessage(
            content=content, tool_calls=list(tool_calls), usage=usage
        )

    @property
    def call_count(self) -> int:
        return len(self.seen_messages)


def start() -> list[dict]:
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "go"},
    ]


def assert_every_tool_call_was_answered(messages: list[dict]) -> None:
    """Each tool_call id must appear exactly once as a tool_call_id."""
    requested = [
        call["id"]
        for message in messages
        if message["role"] == "assistant"
        for call in message.get("tool_calls") or []
    ]
    answered = [m["tool_call_id"] for m in messages if m["role"] == "tool"]

    assert sorted(requested) == sorted(answered), (
        f"requested {requested}, answered {answered} — an unanswered tool_call "
        f"makes the next request a 400"
    )


# --- 1. chaining --------------------------------------------------------


def test_two_tools_chained_across_three_model_calls():
    log: list = []
    client = FakeClient([
        (None, [ToolCall("call_a", "double", '{"value": 4}')]),
        (None, [ToolCall("call_b", "negate", '{"value": 3}')]),
        ("8 and -3.", []),
    ])
    messages = start()

    run_tool_loop(client, make_registry(log), messages, tools=[])

    assert log == [("double", 4), ("negate", 3)]
    assert client.call_count == 3


def test_chaining_leaves_the_history_in_the_right_order():
    log: list = []
    client = FakeClient([
        (None, [ToolCall("call_a", "double", '{"value": 4}')]),
        (None, [ToolCall("call_b", "negate", '{"value": 3}')]),
        ("8 and -3.", []),
    ])
    messages = start()

    run_tool_loop(client, make_registry(log), messages, tools=[])

    assert [m["role"] for m in messages] == [
        "system", "user",
        "assistant", "tool",
        "assistant", "tool",
        "assistant",
    ]
    assert_every_tool_call_was_answered(messages)


def test_the_tool_results_reach_the_model_as_tool_messages():
    log: list = []
    client = FakeClient([
        (None, [ToolCall("call_a", "double", '{"value": 4}')]),
        ("Eight.", []),
    ])
    messages = start()

    run_tool_loop(client, make_registry(log), messages, tools=[])

    tool_message = next(m for m in messages if m["role"] == "tool")
    assert tool_message["content"] == "8"
    assert tool_message["tool_call_id"] == "call_a"


def test_the_second_model_call_sees_the_first_tool_result():
    """Chaining only works if the result is in the history before the next call."""
    log: list = []
    client = FakeClient([
        (None, [ToolCall("call_a", "double", '{"value": 4}')]),
        ("Eight.", []),
    ])

    run_tool_loop(client, make_registry(log), start(), tools=[])

    second_call = client.seen_messages[1]
    assert [m["role"] for m in second_call] == ["system", "user", "assistant", "tool"]
    assert second_call[-1]["content"] == "8"


# --- 2. parallel tool calls in one reply --------------------------------


def test_two_tool_calls_in_one_reply_both_get_answered():
    log: list = []
    client = FakeClient([
        (None, [
            ToolCall("call_a", "double", '{"value": 4}'),
            ToolCall("call_b", "negate", '{"value": 7}'),
        ]),
        ("8 and -7.", []),
    ])
    messages = start()

    run_tool_loop(client, make_registry(log), messages, tools=[])

    assert log == [("double", 4), ("negate", 7)]
    assert [m["role"] for m in messages] == [
        "system", "user", "assistant", "tool", "tool", "assistant"
    ]
    assert_every_tool_call_was_answered(messages)
    assert client.call_count == 2, "parallel calls are one round trip, not two"


def test_parallel_tool_messages_keep_their_own_ids():
    log: list = []
    client = FakeClient([
        (None, [
            ToolCall("call_a", "double", '{"value": 4}'),
            ToolCall("call_b", "negate", '{"value": 7}'),
        ]),
        ("done", []),
    ])
    messages = start()

    run_tool_loop(client, make_registry(log), messages, tools=[])

    answers = {m["tool_call_id"]: m["content"] for m in messages if m["role"] == "tool"}
    assert answers == {"call_a": "8", "call_b": "-7"}


# --- 3. the model corrects itself ---------------------------------------


def test_bad_arguments_come_back_as_an_error_and_the_retry_succeeds():
    """The point of the whole error-handling design: recoverable, not fatal."""
    log: list = []
    client = FakeClient([
        (None, [ToolCall("call_a", "double", '{"value": 99}')]),   # out of range
        (None, [ToolCall("call_b", "double", '{"value": 9}')]),    # corrected
        ("Eighteen.", []),
    ])
    messages = start()

    run_tool_loop(client, make_registry(log), messages, tools=[])

    assert log == [("double", 9)], "the invalid call must never reach the function"
    assert client.call_count == 3
    assert_every_tool_call_was_answered(messages)


def test_the_error_text_is_what_the_model_is_given_to_correct_from():
    log: list = []
    client = FakeClient([
        (None, [ToolCall("call_a", "double", '{"value": 99}')]),
        (None, [ToolCall("call_b", "double", '{"value": 9}')]),
        ("Eighteen.", []),
    ])
    messages = start()

    run_tool_loop(client, make_registry(log), messages, tools=[])

    failed, succeeded = [m for m in messages if m["role"] == "tool"]
    assert "value" in failed["content"], "the model cannot fix what it cannot locate"
    assert "10" in failed["content"], "nor without knowing the bound"
    assert succeeded["content"] == "18"


def test_malformed_json_arguments_are_survivable_too():
    log: list = []
    client = FakeClient([
        (None, [ToolCall("call_a", "double", "{not json at all")]),
        ("I will stop.", []),
    ])
    messages = start()

    run_tool_loop(client, make_registry(log), messages, tools=[])

    assert log == []
    assert "JSON" in next(m for m in messages if m["role"] == "tool")["content"]


def test_an_unknown_tool_name_does_not_end_the_turn():
    log: list = []
    client = FakeClient([
        (None, [ToolCall("call_a", "teleport", "{}")]),
        ("No such tool.", []),
    ])
    messages = start()

    run_tool_loop(client, make_registry(log), messages, tools=[])

    assert client.call_count == 2
    assert_every_tool_call_was_answered(messages)


def test_a_tool_that_raises_still_gets_a_tool_message():
    registry = ToolRegistry()

    @registry.tool(name="boom", description="Raises.", args_model=NumberArgs)
    def boom(value: int) -> int:
        raise RuntimeError("upstream is down")

    client = FakeClient([
        (None, [ToolCall("call_a", "boom", '{"value": 1}')]),
        ("It is down.", []),
    ])
    messages = start()

    run_tool_loop(client, registry, messages, tools=[])

    assert_every_tool_call_was_answered(messages)
    assert "upstream is down" in next(
        m for m in messages if m["role"] == "tool"
    )["content"]


# --- 4. a model that never stops ----------------------------------------


def test_a_runaway_model_is_cut_off_at_the_iteration_limit():
    log: list = []
    client = FakeClient(
        [], endless=(None, [ToolCall("call_x", "double", '{"value": 1}')])
    )
    messages = start()

    run_tool_loop(client, make_registry(log), messages, tools=[])

    assert client.call_count == MAX_TOOL_ITERATIONS
    assert len(log) == MAX_TOOL_ITERATIONS


def test_the_limit_leaves_a_valid_history_and_tells_the_model_why():
    log: list = []
    client = FakeClient(
        [], endless=(None, [ToolCall("call_x", "double", '{"value": 1}')])
    )
    messages = start()

    run_tool_loop(client, make_registry(log), messages, tools=[])

    assert_every_tool_call_was_answered(messages)
    assert "limit" in messages[-1]["content"].lower()
    assert messages[-1]["role"] != "assistant", (
        "the note is from the harness, not something the model said"
    )


def test_the_limit_is_reported_to_the_user(capsys):
    """Silence here looks like a finished answer that simply had no text."""
    log: list = []
    client = FakeClient(
        [], endless=(None, [ToolCall("call_x", "double", '{"value": 1}')])
    )

    run_tool_loop(client, make_registry(log), start(), tools=[])

    out = capsys.readouterr().out.lower()
    assert "stopped" in out
    assert str(MAX_TOOL_ITERATIONS) in out


def test_max_tool_iterations_leaves_room_for_a_real_chain():
    """Three calls is the D2 demo; D9 chains search, stock, cart and checkout."""
    assert MAX_TOOL_ITERATIONS >= 6


# --- 5. no tools needed -------------------------------------------------


def test_a_plain_answer_dispatches_nothing():
    log: list = []
    client = FakeClient([("Hello, how can I help?", [])])
    messages = start()

    run_tool_loop(client, make_registry(log), messages, tools=[])

    assert log == []
    assert client.call_count == 1
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]


def test_a_plain_answer_is_printed(capsys):
    client = FakeClient([("Hello, how can I help?", [])])

    run_tool_loop(client, make_registry([]), start(), tools=[])

    assert "Hello, how can I help?" in capsys.readouterr().out


# --- visibility ---------------------------------------------------------


def test_each_tool_call_is_shown_to_the_user(capsys):
    """Seeing the loop is the point of D2, not a nicety."""
    log: list = []
    client = FakeClient([
        (None, [ToolCall("call_a", "double", '{"value": 4}')]),
        ("Eight.", []),
    ])

    run_tool_loop(client, make_registry(log), start(), tools=[])

    out = capsys.readouterr().out
    assert "double" in out, "the tool name"
    assert '"value": 4' in out or "'value': 4" in out, "the arguments"
    assert "8" in out, "the result"


def test_a_failed_tool_call_is_shown_as_a_failure(capsys):
    log: list = []
    client = FakeClient([
        (None, [ToolCall("call_a", "double", '{"value": 99}')]),
        ("Sorry.", []),
    ])

    run_tool_loop(client, make_registry(log), start(), tools=[])

    out = capsys.readouterr().out.lower()
    assert "error" in out or "failed" in out


# --- cost accounting ----------------------------------------------------


def test_every_model_call_in_the_turn_is_recorded():
    """One user input, three model calls — the cost line must total all three."""
    log: list = []
    client = FakeClient([
        (None, [ToolCall("call_a", "double", '{"value": 4}')]),
        (None, [ToolCall("call_b", "negate", '{"value": 3}')]),
        ("8 and -3.", []),
    ])
    calls_before = len(client.tracker.calls)

    run_tool_loop(client, make_registry(log), start(), tools=[])

    assert len(client.tracker.calls[calls_before:]) == 3


# --- the tool schemas are actually sent ---------------------------------


def test_the_schemas_are_passed_on_every_call_not_just_the_first():
    log: list = []
    tools = [{"type": "function", "function": {"name": "double"}}]
    client = FakeClient([
        (None, [ToolCall("call_a", "double", '{"value": 4}')]),
        ("Eight.", []),
    ])

    run_tool_loop(client, make_registry(log), start(), tools=tools)

    assert client.seen_tools == [tools, tools]


@pytest.mark.parametrize("content", ["", None])
def test_an_empty_final_answer_does_not_crash(content):
    client = FakeClient([(content, [])])
    messages = start()

    run_tool_loop(client, make_registry([]), messages, tools=[])

    assert messages[-1]["role"] == "assistant"
