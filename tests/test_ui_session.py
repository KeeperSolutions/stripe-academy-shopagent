"""The browser's turn logic, driven with no browser (D11, step 1).

Every test here runs offline: no Streamlit, no MCP subprocess, no HTTP, no
model. That is the property `src/shopagent/ui/session.py` is shaped for, and
the first test is the one that keeps it true.
"""

from __future__ import annotations

import ast
import json
import threading
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from shopagent.agent.activity import ActivityLog, RecordingRegistry, ToolCallRecord
from shopagent.obs.tracing import Tracer
from shopagent.tools.registry import ToolRegistry, ToolResult, ToolSpec
from shopagent.ui import session as ui

SESSION_PATH = Path(ui.__file__)
SESSION_SOURCE = SESSION_PATH.read_text()
SESSION_TREE = ast.parse(SESSION_SOURCE)

ACTIVITY_SOURCE = Path(
    __import__("shopagent.agent.activity", fromlist=["x"]).__file__
).read_text()


def imported_names(tree: ast.AST) -> set[str]:
    """Every module name reached by an `import` in this file, dotted roots kept."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


def calls_in(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
        )
    ]


# --- the module is testable without a browser ----------------------------


def test_the_turn_module_does_not_import_streamlit():
    """The whole reason this module exists apart from `app.py`.

    An import here would make every test below need a Streamlit runtime, and
    would put a rerun-scoped framework underneath logic whose whole job is to
    survive a rerun. `app.py` renders; this decides.
    """
    assert "streamlit" not in imported_names(SESSION_TREE)
    # The word appears in the prose above, explaining what this file is set
    # against. What must not appear is a *use* of it: an import, or a bare `st`
    # the way every Streamlit script spells it.
    used = {
        node.id
        for node in ast.walk(SESSION_TREE)
        if isinstance(node, ast.Name) and node.id in {"st", "streamlit"}
    }
    assert not used, f"ui/session.py reaches for Streamlit: {used}"


def test_the_activity_log_does_not_import_streamlit_either():
    assert "streamlit" not in imported_names(ast.parse(ACTIVITY_SOURCE))


def test_the_turn_module_names_no_streamlit_cache_decorator():
    """`@st.cache_resource` belongs one file up, and step 2 is where it goes.

    The memoisation here has to hold for a test and a script as well, which is
    what makes it the wrong job for a decorator that only exists inside a
    Streamlit runtime.
    """
    decorated = {
        ast.unparse(decorator)
        for node in ast.walk(SESSION_TREE)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        for decorator in node.decorator_list
    }
    assert not any("cache" in name for name in decorated), decorated


# --- it drives the same loop the CLI and the runner drive ----------------


def test_the_session_builds_no_registry_of_its_own():
    """The claim, checked structurally rather than promised in a docstring.

    A UI that constructed its own `ToolRegistry` would answer every behavioural
    test correctly while presenting a shop with no gate, no memory and no
    catalog. Same guard, same wording and same reason as
    `tests/test_evals.py::test_the_runner_builds_no_registry_of_its_own`.
    """
    for forbidden in ("ToolRegistry", "GuardedRegistry", "RememberingRegistry"):
        assert not calls_in(SESSION_TREE, forbidden), (
            f"ui/session.py constructs a {forbidden} of its own; it must take "
            f"the one `build_tool_setup` returns, the way the CLI does"
        )


def test_the_session_calls_the_same_two_functions_the_cli_calls():
    assert calls_in(SESSION_TREE, "build_tool_setup"), "the UI must use the CLI's setup"
    assert calls_in(SESSION_TREE, "run_tool_loop"), "the UI must drive the CLI's loop"


def test_the_session_answers_a_confirmation_through_the_protocol():
    """Not with a copy of it, and not by writing to the memory itself."""
    assert calls_in(SESSION_TREE, "resolve_pending")
    assert calls_in(SESSION_TREE, "follow_up_note")

    reached = {
        node.attr
        for node in ast.walk(SESSION_TREE)
        if isinstance(node, ast.Attribute) and node.attr.startswith("_pending")
    }
    assert not reached, f"ui/session.py reaches into the memory's private state: {reached}"

    for forbidden in ("answer_confirmation", "park_confirmation", "take_confirmation"):
        called = [
            node
            for node in calls_in(SESSION_TREE, forbidden)
            if isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_memory"
        ]
        assert not called, (
            f"ui/session.py calls memory.{forbidden} directly; the confirmation "
            f"protocol is `resolve_pending` plus one follow-up turn"
        )


def test_run_tool_loop_is_still_the_function_d2_wrote():
    """D11 added a UI, a spend cap, an activity log and a card capture.

    None of them opened the loop. The hash is the same one D5, D9 and D10 each
    leaned on, and this is the day a browser was the thing that did not need it
    to change.
    """
    import hashlib
    import inspect

    from shopagent.llm.loop import run_tool_loop

    digest = hashlib.sha256(inspect.getsource(run_tool_loop).encode()).hexdigest()
    assert digest == (
        "161bdc1cac8d446b85b98ce1c2fcb269627d4305712348cde39f34fe52f49d00"
    ), "run_tool_loop changed"


# --- the profile is read and never written -------------------------------


def test_the_session_offers_no_way_to_write_a_profile():
    """`/remember` and `/forget` stay CLI commands.

    A profile is injected into the system prompt, so a write path is a write
    path onto the assistant's own instructions — D9's argument for a closed
    domain of five categories. A browser input is a new door onto that surface
    and this step does not open one.
    """
    assert calls_in(SESSION_TREE, "load_for_session"), "the profile must still be read"
    for forbidden in ("remember", "forget", "validate"):
        assert not calls_in(SESSION_TREE, forbidden), (
            f"ui/session.py calls profile.{forbidden}; profile writing stays in the CLI"
        )
    assert not hasattr(ui.BrowserSession, "remember")
    assert not hasattr(ui.BrowserSession, "forget")
    # A property with no setter: assigning to it raises rather than storing.
    assert isinstance(ui.BrowserSession.profile, property)
    assert ui.BrowserSession.profile.fset is None


# --- the activity log ----------------------------------------------------


def _registry_with(tool_name: str, answer: str, ok: bool = True) -> ToolRegistry:
    registry = ToolRegistry()

    def run(**kwargs):
        if not ok:
            raise ValueError(answer)
        return json.loads(answer)

    registry.register(
        ToolSpec(
            name=tool_name,
            description="a tool",
            fn=run,
            parameters_schema={"type": "object", "properties": {}},
        )
    )
    return registry


def test_the_activity_log_records_name_arguments_duration_and_outcome():
    log = ActivityLog()
    registry = RecordingRegistry(_registry_with("get_thing", '{"a": 1}'), log)

    registry.dispatch("get_thing", '{"query": "trail shoes"}')

    (call,) = log.calls
    assert call.name == "get_thing"
    assert call.ok is True
    assert call.error is None
    assert "trail shoes" in call.arguments
    assert call.duration_ms >= 0.0
    assert call.result_chars == len('{"a": 1}')


def test_the_activity_log_records_a_refusal_with_its_reason():
    """A gate refusal is a `ToolResult`, not an exception — so only an outer
    wrapper sees it, and a refused checkout is the row a panel most needs."""
    log = ActivityLog()
    registry = RecordingRegistry(ToolRegistry(), log)

    result = registry.dispatch("no_such_tool", "{}")

    assert result.ok is False
    (call,) = log.calls
    assert call.ok is False
    assert call.error


def test_a_turn_clears_the_previous_turns_activity():
    log = ActivityLog()
    registry = RecordingRegistry(_registry_with("get_thing", '{"a": 1}'), log)
    registry.dispatch("get_thing", "{}")
    assert len(log.calls) == 1

    log.begin_turn()
    registry.dispatch("get_thing", "{}")

    assert len(log.calls) == 1


def test_the_activity_log_does_not_keep_a_failed_calls_payload():
    """An error message is written for the model and reads oddly to a person;
    `error` already carries the short machine-facing half."""
    log = ActivityLog()
    registry = RecordingRegistry(ToolRegistry(), log)
    registry.dispatch("no_such_tool", "{}")
    assert log.calls[0].content == ""


def test_a_records_payload_is_not_in_its_repr():
    record = ToolCallRecord(
        name="search_products", arguments="{}", ok=True, duration_ms=1.0, content="x" * 5000
    )
    assert "xxxx" not in repr(record)


# --- driving a whole turn, offline ---------------------------------------
#
# The fakes below stand in for the three things a real session reaches: the
# model, the catalog server and the commerce API. None of them touches a
# network, a subprocess or a database, which is what makes every test in this
# section free and offline — the rule `pytest tests/` exists to keep.


@dataclass
class FakeToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class FakeUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 5
    cached_tokens: int = 0
    total_tokens: int = 15
    cost_usd: float = 0.01
    model: str = "fake"


@dataclass
class FakeReply:
    content: str | None = None
    tool_calls: list = field(default_factory=list)
    usage: FakeUsage = field(default_factory=FakeUsage)

    def to_message(self) -> dict:
        message: dict = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        return message


class FakeClient:
    """A model that answers from a script, and bills the tracker as it goes."""

    model = "fake-model"

    def __init__(self, tracker=None, replies=None) -> None:
        self._tracker = tracker
        self._replies = list(replies or [])
        self.seen: list[list] = []

    def chat_with_tools(self, messages, tools=None):
        self.seen.append(list(messages))
        reply = self._replies.pop(0) if self._replies else FakeReply(content="done")
        if self._tracker is not None and reply.usage is not None:
            self._tracker.calls.append(reply.usage)
        return reply


@dataclass
class FakeTool:
    name: str
    description: str
    input_schema: dict


@dataclass
class FakeToolResult:
    content: list
    is_error: bool = False


@dataclass
class FakeBlock:
    text: str
    type: str = "text"


class FakeCatalogClient:
    """An MCP client that publishes `search_products` and answers from a queue."""

    def __init__(self, answers=None) -> None:
        self.answers = list(answers or [])
        self.calls: list[tuple[str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def list_tools(self):
        return [
            FakeTool(
                name="search_products",
                description="find products",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ]

    def call_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        answer = self.answers.pop(0) if self.answers else "{}"
        return FakeToolResult(content=[FakeBlock(text=answer)])


class FakeCommerceAPI:
    """Stands in for `CommerceAPI`. Nothing in this file dispatches a cart tool."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def close(self) -> None:
        return None


def a_catalog_answer(product_id: int, name: str, variant_id: int, price_cents: int) -> str:
    return json.dumps(
        {
            "count": 1,
            "results": [
                {
                    "product_id": product_id,
                    "name": name,
                    "brand": "Fake",
                    "category": "shoes",
                    "description": "a shoe",
                    "variants": [
                        {
                            "variant_id": variant_id,
                            "sku": f"SKU-{variant_id}",
                            "size": "42",
                            "color": "black",
                            "price_cents": price_cents,
                            "available": 3,
                        }
                    ],
                }
            ],
        }
    )


def a_search_turn(query: str) -> list[FakeReply]:
    """The two replies one searching turn takes: ask for the tool, then answer."""
    return [
        FakeReply(tool_calls=[FakeToolCall(id="c1", name="search_products",
                                           arguments=json.dumps({"query": query}))]),
        FakeReply(content=f"Here is what I found for {query}."),
    ]


@pytest.fixture
def offline(monkeypatch):
    """A session factory that reaches nothing outside the process."""

    def build(replies=None, answers=None, tracer=None, **kwargs):
        catalog = FakeCatalogClient(answers=answers)
        resources = ui.SharedResources(
            tracer=tracer or Tracer(),
            catalog_factory=lambda: catalog,
            commerce_factory=FakeCommerceAPI,
        )
        client = FakeClient(replies=replies)
        monkeypatch.setattr(
            ui, "LLMClient", lambda tracker=None: setattr(client, "_tracker", tracker) or client
        )
        session = ui.BrowserSession(resources, catalog_enabled=True, **kwargs)
        session.fake_client = client
        session.fake_catalog = catalog
        return session

    return build


# --- the cards are captured onto the message that produced them ----------


def test_a_second_search_does_not_move_the_first_messages_cards(offline):
    """The defect this whole design is set against.

    Reading `ConversationMemory.last_search` when a bubble is drawn would put
    the newest results under every older answer, because D9 makes every search
    *replace* the previous one on purpose — "the second one" can only mean the
    second row of the list the customer is looking at now. So the rows are
    captured when they are produced.
    """
    session = offline(
        replies=a_search_turn("boots") + a_search_turn("sandals"),
        answers=[
            a_catalog_answer(1, "Trail Runner", 111, 9499),
            a_catalog_answer(2, "Summit Peak", 222, 18998),
        ],
    )

    first = session.send("find me boots")
    second = session.send("now find me sandals")

    assert [card.name for card in first.messages[1].cards] == ["Trail Runner"]
    assert [card.name for card in second.messages[1].cards] == ["Summit Peak"]
    # And the memory still holds only the newest, which is what it is for.
    assert session._memory.last_search.results[0]["name"] == "Summit Peak"
    # The transcript, re-read after both turns, still says the same thing.
    shop_bubbles = [m for m in session.transcript if m.role == ui.SHOP]
    assert [c.name for bubble in shop_bubbles for c in bubble.cards] == [
        "Trail Runner",
        "Summit Peak",
    ]


def test_a_card_carries_the_variant_colour_and_a_formatted_price(offline):
    """`catalog/search.py` returns `color` per variant, so no second call is
    needed to render one. It returns no image, and a card is therefore text."""
    session = offline(
        replies=a_search_turn("boots"), answers=[a_catalog_answer(1, "Trail Runner", 111, 9499)]
    )

    (card,) = session.send("find me boots").messages[1].cards
    (variant,) = card.variants

    assert variant.color == "black"
    assert variant.size == "42"
    assert variant.price_cents == 9499
    assert variant.price == "€94.99"
    assert variant.in_stock is True
    assert not hasattr(variant, "image_url")


def test_a_turn_with_no_search_carries_no_cards(offline):
    session = offline(replies=[FakeReply(content="Hello.")])
    assert session.send("hello").messages[1].cards == ()


# --- the activity strip --------------------------------------------------


def test_the_turn_reports_the_tool_calls_it_made(offline):
    session = offline(
        replies=a_search_turn("boots"), answers=[a_catalog_answer(1, "Trail Runner", 111, 9499)]
    )

    (call,) = session.send("find me boots").messages[1].activity

    assert call.name == "search_products"
    assert "boots" in call.arguments
    assert call.ok is True
    assert call.duration_ms >= 0.0


def test_a_turns_activity_does_not_include_the_previous_turns(offline):
    session = offline(
        replies=a_search_turn("boots") + [FakeReply(content="Hello.")],
        answers=[a_catalog_answer(1, "Trail Runner", 111, 9499)],
    )

    session.send("find me boots")
    result = session.send("hello")

    assert result.messages[1].activity == ()


# --- the resources this process must not build twice ---------------------


@pytest.fixture
def fresh_process(monkeypatch):
    """A process with nothing shared yet, restored afterwards.

    The memoisation in `ui.session` is process-wide by design, so a test that
    exercised it without resetting would be measuring whatever ran before it.
    """
    monkeypatch.setattr(ui, "_shared", None)
    monkeypatch.setattr(ui, "_process_stack", ExitStack())
    yield
    ui._shared = None


class CountingTracer(Tracer):
    built = 0

    def __init__(self) -> None:
        super().__init__(client=None)
        CountingTracer.built += 1


def test_the_tracer_is_built_once_per_process_not_once_per_call(fresh_process, monkeypatch):
    """D10 measured what a second one costs, and a browser reruns on every click.

    Langfuse keeps one resource manager per public key, process-wide. The second
    `shutdown()` enqueues a stop sentinel per consumer onto a queue whose
    consumers are already dead, and the next `flush()` — which is
    `queue.join()` — waits for a `task_done()` that cannot come. Two eval passes
    hung there. Streamlit would reach that state in a handful of clicks.
    """
    CountingTracer.built = 0
    monkeypatch.setattr(ui, "build_tracer", CountingTracer)

    first = ui.shared_resources()
    for _ in range(10):
        assert ui.shared_resources() is first

    assert CountingTracer.built == 1


def test_a_turn_builds_no_tracer_of_its_own(fresh_process, monkeypatch, offline):
    """The falsification target. A session that reached for `build_tracer` —
    or that a rerun rebuilt — would push this past one."""
    CountingTracer.built = 0
    monkeypatch.setattr(ui, "build_tracer", CountingTracer)
    tracer = ui.shared_resources().tracer

    session = offline(replies=[FakeReply(content="a"), FakeReply(content="b")], tracer=tracer)
    session.send("hello")
    session.send("again")

    assert CountingTracer.built == 1


def test_nothing_in_the_ui_layer_shuts_a_tracer_down_per_turn():
    """`shutdown()` appears exactly once, in the process-wide teardown.

    A `shutdown()` reachable from a turn or from a closing browser tab is the
    D10 hang with a UI in front of it. The one this process performs is
    `shutdown_shared_resources`, registered with `atexit`.
    """
    shutdowns = [
        node
        for node in ast.walk(SESSION_TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "shutdown"
    ]
    assert len(shutdowns) == 1, "a tracer is shut down somewhere other than process teardown"

    enclosing = [
        node.name
        for node in ast.walk(SESSION_TREE)
        if isinstance(node, ast.FunctionDef)
        and any(call in ast.walk(node) for call in shutdowns)
    ]
    assert enclosing == ["shutdown_shared_resources"], enclosing


class CountingResource:
    """A stand-in for the MCP subprocess: counts how often it is really built."""

    built = 0
    closed = 0

    def __init__(self) -> None:
        CountingResource.built += 1

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        CountingResource.closed += 1
        return None


def test_a_borrowed_resource_is_built_once_and_survives_the_borrowers_stack():
    """A session ending must not take the catalog out from under the others.

    `build_tool_setup` owns whatever it is given — it enters the factory into
    the caller's `ExitStack`. That is right for the CLI, where the stack's life
    is the process's, and wrong for a browser tab.
    """
    CountingResource.built = CountingResource.closed = 0
    process_stack = ExitStack()
    borrowed = ui._Borrowed(CountingResource, process_stack, threading.Lock())

    # Entered exactly as `build_tool_setup` enters it: the factory is called,
    # and the result is what goes into the session's stack. A fixture that
    # skipped the call would be testing a shape the real caller never produces.
    for _ in range(3):
        with ExitStack() as session_stack:
            assert isinstance(session_stack.enter_context(borrowed()), CountingResource)

    assert CountingResource.built == 1
    assert CountingResource.closed == 0

    process_stack.close()
    assert CountingResource.closed == 1


def test_two_tabs_arriving_together_do_not_spawn_two_subprocesses():
    """Streamlit runs each browser session's script on its own thread."""
    CountingResource.built = 0
    borrowed = ui._Borrowed(CountingResource, ExitStack(), threading.Lock())
    ready = threading.Barrier(8)

    def open_a_tab():
        ready.wait()
        with ExitStack() as stack:
            stack.enter_context(borrowed())

    threads = [threading.Thread(target=open_a_tab) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert CountingResource.built == 1


def test_two_browser_sessions_do_not_share_a_cart(fresh_process):
    """`@st.cache_resource` is shared across every browser session in the
    process, so a `ConversationMemory` cached up there would put one tab's
    basket in another's. `agent/memory.py` says never shared, never global."""
    resources = ui.SharedResources(
        tracer=Tracer(),
        catalog_factory=FakeCatalogClient,
        commerce_factory=FakeCommerceAPI,
    )
    one = ui.BrowserSession(resources, catalog_enabled=False)
    two = ui.BrowserSession(resources, catalog_enabled=False)

    assert one._memory is not two._memory
    one._memory.cart_id = "cart-one"
    assert two._memory.cart_id is None

    one.close()
    two.close()


def test_closing_a_session_does_not_close_the_shared_clients(fresh_process):
    CountingResource.built = CountingResource.closed = 0
    process_stack = ExitStack()
    lock = threading.Lock()
    resources = ui.SharedResources(
        tracer=Tracer(),
        catalog_factory=ui._Borrowed(CountingResource, process_stack, lock),
        commerce_factory=ui._Borrowed(CountingResource, process_stack, lock),
    )

    session = ui.BrowserSession(resources, catalog_enabled=False)
    session.close()

    assert CountingResource.closed == 0
    process_stack.close()


# --- the spend cap -------------------------------------------------------


def test_the_cap_stops_the_next_turn_and_says_so(offline):
    """A browser is clicked far faster than a terminal is typed into.

    The cost meter has existed since D1; what D11 adds is a threshold. It is
    checked at the *door* of a turn, so the conversation is never left holding
    an assistant turn whose tool calls have no answers — which is a 400 on
    every later request.
    """
    session = offline(
        replies=[FakeReply(content="one"), FakeReply(content="two")], spend_cap_usd=0.005
    )

    first = session.send("hello")
    # The turn that crosses the cap is a real answer; the next one is refused.
    assert first.refused is False
    assert first.cap_reached is True
    assert first.messages[1].text == "one"

    second = session.send("again")

    assert second.refused is True
    assert second.cap_reached is True
    assert second.messages[1].notice == ui.CAP_NOTICE
    assert second.messages[1].text == ""
    # Nothing was billed for the refused turn: the model was never called.
    assert second.session_cost_usd == first.session_cost_usd
    assert session.fake_client.seen and len(session.fake_client.seen) == 1


def test_the_conversation_stays_readable_after_the_cap(offline):
    """A cap that erased the transcript would be a worse failure than the bill."""
    session = offline(
        replies=[FakeReply(content="here is a shoe")], spend_cap_usd=0.005
    )
    session.send("hello")
    session.send("and another thing")

    said = [message.text for message in session.transcript]

    assert "hello" in said
    assert "here is a shoe" in said
    assert "and another thing" in said, "the refused message is still shown"
    assert session.transcript[-1].notice == ui.CAP_NOTICE


def test_the_refused_message_never_reaches_the_model(offline):
    """It is shown, not answered. A user message with no assistant turn after
    it is one the next request would answer out of order."""
    session = offline(replies=[FakeReply(content="one")], spend_cap_usd=0.005)
    session.send("hello")
    before = list(session._messages)

    session.send("this one is refused")

    assert session._messages == before


def test_the_cap_notice_names_no_dollar_figure():
    """The shop's costs are in USD and every price a shopper was quoted is in
    EUR. Two currencies, one of them not this customer's business."""
    assert "$" not in ui.CAP_NOTICE
    assert "USD" not in ui.CAP_NOTICE


def test_the_cap_comes_from_settings_by_default(fresh_process, monkeypatch):
    from shopagent.config import get_settings

    resources = ui.SharedResources(
        tracer=Tracer(), catalog_factory=FakeCatalogClient, commerce_factory=FakeCommerceAPI
    )
    session = ui.BrowserSession(resources, catalog_enabled=False)
    try:
        assert session.cap_usd == get_settings().ui_spend_cap_usd
        assert session.cap_usd > 0
    finally:
        session.close()


# --- the confirmation, answered in a later request -----------------------


def test_a_parked_confirmation_is_shown_and_nothing_is_bought(offline):
    """`send` never asks anybody. It returns with the question on it.

    That is the half of the D10 protocol built for this caller: the answer
    arrives in a later HTTP request, so there is no callable that could have
    blocked for it.
    """
    session = offline(replies=[FakeReply(content="one moment")])
    session._memory.park_confirmation("create_checkout", "Trail Runner — €94.99")

    assert session.pending == ui.PendingApproval(
        tool="create_checkout", summary="Trail Runner — €94.99"
    )
    assert session.fake_client.seen == []


def test_answering_a_confirmation_records_it_and_drives_one_follow_up_turn(offline):
    from shopagent.agent.confirmation import CONFIRMED_NOTE

    session = offline(replies=[FakeReply(content="ordered")])
    session._memory.begin_turn(from_customer=True)
    session._memory.park_confirmation("create_checkout", "Trail Runner — €94.99")

    result = session.answer_confirmation(True)

    assert result.messages[0].text == "ordered"
    # The note went to the model as a *system* message: the customer pressed a
    # button in the shop's own interface, and recording that as speech would put
    # words in the transcript they never typed.
    (sent,) = [m for m in session._messages if m.get("content") == CONFIRMED_NOTE]
    assert sent["role"] == "system"
    # And the approval is spendable on exactly the turn that was driven.
    spent = session._memory.take_confirmation("create_checkout")
    assert spent is not None and spent.answer is True


def test_a_declined_confirmation_carries_the_declined_note(offline):
    from shopagent.agent.confirmation import DECLINED_NOTE

    session = offline(replies=[FakeReply(content="nothing was ordered")])
    session._memory.begin_turn(from_customer=True)
    session._memory.park_confirmation("create_checkout", "Trail Runner — €94.99")

    session.answer_confirmation(False)

    assert any(m.get("content") == DECLINED_NOTE for m in session._messages)


def test_answering_when_nothing_is_parked_changes_nothing(offline):
    session = offline(replies=[FakeReply(content="unused")])

    result = session.answer_confirmation(True)

    assert result.messages == ()
    assert session.fake_client.seen == []


def test_a_new_customer_message_drops_an_unanswered_approval(offline):
    """D10's rule reaching this caller unchanged.

    An approval that outlives its turn is an answer sitting apart from the
    question it answered — the shape of defect that appears wherever a "yes" is
    recognised by its wording rather than by what it replies to.
    """
    session = offline(replies=[FakeReply(content="ok")])
    session._memory.park_confirmation("create_checkout", "Trail Runner — €94.99")
    assert session.pending is not None

    session.send("actually, show me jackets")

    assert session.pending is None
    assert session._memory.take_confirmation("create_checkout") is None


def test_the_session_never_confirms_by_itself(offline):
    """`can_confirm` is true because a person is reachable — through a button
    in a later request, never from inside a dispatch. The confirmer must not be
    invoked by driving a turn."""
    session = offline(replies=[FakeReply(content="ok")])
    session._memory.park_confirmation("create_checkout", "a summary")

    session.send("hello")

    assert session._confirmer.asked == []


def test_an_unset_confirmer_answers_no(offline):
    """Reaching it without an answer means a caller resolved a confirmation it
    never had one for. The safe answer to "could not ask" is "they said no"."""
    session = offline(replies=[FakeReply(content="ok")])
    assert session._confirmer("anything") is False


# --- a turn that raises --------------------------------------------------


def test_a_failed_turn_is_rewound_and_the_conversation_survives(offline):
    """The same rewind `_run_session` does, and for the same reason: an
    assistant turn whose tool calls never got their `tool` messages makes every
    later request a 400."""
    session = offline(replies=[])
    session.fake_client.chat_with_tools = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("the model went away")
    )
    before = len(session._messages)

    result = session.send("hello")

    assert result.error and "the model went away" in result.error
    assert result.messages[1].notice and "That turn failed" in result.messages[1].notice
    assert len(session._messages) == before
    assert session.transcript[0].text == "hello", "the customer's message is still shown"


def test_a_turn_that_does_not_search_does_not_inherit_the_previous_cards(offline):
    """The distinguishing test, and the first attempt at it was not one.

    "Do the cards stay put across two searches" cannot tell the two designs
    apart: `ChatMessage` is frozen and its `cards` tuple is built when the
    bubble is made, so a `last_search` read at that same moment gives the same
    answer. The mutation survived, and the mutation was right to — the eager,
    frozen capture makes the lazy read unrepresentable in this module.

    What `last_search` *would* get wrong is here: it survives a turn, and the
    activity log does not. Read it, and "what is your returns policy?" comes
    back with the boots from two turns ago underneath it.
    """
    session = offline(
        replies=a_search_turn("boots") + [FakeReply(content="Thirty days.")],
        answers=[a_catalog_answer(1, "Trail Runner", 111, 9499)],
    )

    searched = session.send("find me boots")
    asked = session.send("what is your returns policy?")

    assert [card.name for card in searched.messages[1].cards] == ["Trail Runner"]
    assert asked.messages[1].cards == ()
    # The memory still holds the search, which is what it is for — resolving
    # "the second one" — and that is exactly why it is the wrong thing to draw.
    assert session._memory.last_search is not None
