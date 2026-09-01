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
from shopagent.config import get_settings
from shopagent.money import format_amount
from shopagent.obs.tracing import Tracer
from shopagent.tools.http import CommerceAPIUnreachable
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


# --- the confirmation, with a cart the gate can actually read -------------
#
# The fake below answers `CommerceAPI.request`, which is the one method
# `tools/commerce.py` calls. That is what lets the *real* gate run offline:
# `GuardedRegistry._describe` dispatches `view_cart` through the registry, and
# with a cart behind it the summary a person would be shown is built here
# exactly as it is in production — same dispatch, same `_summarise`, same
# `money.format_amount`.


class FakeCommerceBackend:
    """A cart and an order, over the one method the commerce tools use.

    Built from the shape the real bodies always have rather than from the shape
    the assertions need — `api/schemas.py`'s field names, every money field
    present. A fixture that dropped `line_total_cents` would still satisfy a
    test about the total while leaving the gate's per-line rendering untested,
    which is the blind spot CLAUDE.md records D8 and D10 both paying for.
    """

    VARIANT_ID = 86272
    CART_ID = "11111111-1111-1111-1111-111111111111"
    ORDER_ID = "22222222-2222-2222-2222-222222222222"
    CHECKOUT_URL = "https://checkout.stripe.com/c/pay/cs_test_" + "z" * 60

    def __init__(self, unit_cents: int = 14999) -> None:
        self.unit_cents = unit_cents
        self.quantity = 0
        self.requests: list[tuple[str, str]] = []
        self.ordered = False

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def _line(self) -> dict:
        return {
            "item_id": "33333333-3333-3333-3333-333333333333",
            "variant_id": self.VARIANT_ID,
            "sku": "NR-SMTPRO-42-CHR",
            "product_name": "Summit Peak Pro",
            "variant_label": "size 42, charcoal",
            "quantity": self.quantity,
            "unit_price_cents": self.unit_cents,
            "line_total_cents": self.unit_cents * self.quantity,
        }

    def _body(self, **extra) -> dict:
        items = [self._line()] if self.quantity else []
        return {
            "currency": "eur",
            "items": items,
            "total_cents": sum(item["line_total_cents"] for item in items),
            **extra,
        }

    def request(self, method: str, path: str, json=None):
        self.requests.append((method, path))
        if method == "POST" and path == "/cart":
            return {"cart_id": self.CART_ID, **self._body()}
        if path.endswith("/items") and method == "POST":
            self.quantity += (json or {}).get("quantity", 1)
            return self._body(cart_id=self.CART_ID)
        if method == "GET" and path.startswith("/cart/"):
            return self._body(cart_id=self.CART_ID)
        if method == "POST" and path == "/orders":
            self.ordered = True
            return self._body(order_id=self.ORDER_ID, status="pending")
        if method == "GET" and path.startswith("/orders/"):
            return self._body(order_id=self.ORDER_ID, status="pending")
        if path.endswith("/checkout"):
            return {"checkout_url": self.CHECKOUT_URL, "order_id": self.ORDER_ID}
        raise AssertionError(f"the fake backend was asked for {method} {path}")


@pytest.fixture
def shopping(monkeypatch):
    """A session whose cart is real enough for the gate to summarise."""

    def build(replies=None, unit_cents=14999, **kwargs):
        backend = FakeCommerceBackend(unit_cents=unit_cents)
        # The catalog is on, and it has to be: D9 refuses an `add_to_cart` for
        # a `variant_id` that has not appeared in a tool result in this
        # conversation. A script that added straight to the cart was refused by
        # that guard, which is the guard working — so every script here starts
        # with the search that puts the variant in front of the model.
        catalog = FakeCatalogClient(
            answers=[
                a_catalog_answer(
                    2, "Summit Peak Pro", FakeCommerceBackend.VARIANT_ID, unit_cents
                )
            ]
            * 4
        )
        resources = ui.SharedResources(
            tracer=Tracer(),
            catalog_factory=lambda: catalog,
            commerce_factory=lambda: backend,
        )
        client = FakeClient(replies=replies)
        monkeypatch.setattr(
            ui, "LLMClient", lambda tracker=None: setattr(client, "_tracker", tracker) or client
        )
        session = ui.BrowserSession(resources, catalog_enabled=True, **kwargs)
        session.fake_client = client
        session.backend = backend
        return session

    return build


def _tool(name: str, arguments: dict, said: str | None = None) -> FakeReply:
    return FakeReply(
        content=said,
        tool_calls=[FakeToolCall(id=f"c-{name}", name=name, arguments=json.dumps(arguments))],
    )


def _fills_a_cart_then_checks_out(claimed: str | None = None) -> list[FakeReply]:
    """Three turns: search, add one Summit Peak Pro, then ask to check out.

    The search is not scene-setting. D9's unknown-variant guardrail refuses an
    `add_to_cart` for an id the model has not been shown here, so a script
    without it is refused before the gate is ever reached.

    `claimed` rides along as the model's *narration beside the tool call*
    rather than as a final answer, so the amount guardrail — which only checks
    a final answer — is not what stops it. What this measures is the gate, on
    its own.
    """
    return [
        _tool("search_products", {"query": "trail shoes"}),
        FakeReply(content="Here is the Summit Peak Pro."),
        _tool("add_to_cart", {"variant_id": FakeCommerceBackend.VARIANT_ID, "quantity": 1}),
        FakeReply(content="Added it."),
        _tool("create_checkout", {}, said=claimed),
        FakeReply(content="Waiting for your confirmation."),
    ]


def test_the_summary_a_person_is_shown_comes_from_the_cart(shopping):
    session = shopping(replies=_fills_a_cart_then_checks_out())
    session.send("find me trail shoes")
    session.send("add the Summit Peak Pro")
    session.send("check out")

    pending = session.pending

    assert pending is not None
    assert pending.tool == "create_checkout"
    assert "Summit Peak Pro" in pending.summary
    assert "€149.99" in pending.summary
    assert "1 x" in pending.summary


def test_the_summary_does_not_move_when_the_model_claims_another_total(shopping):
    """The falsification, and the property the whole gate exists for.

    A person approving a figure the model invented is worse than no gate at
    all: it launders the invention through a human and leaves a record saying
    they agreed to it. So the model is made to say €1.00 while asking for the
    checkout, and the summary must still be the cart's own €149.99 — built by
    `GuardedRegistry._describe` from a real `view_cart` dispatch through
    `money.format_amount`.
    """
    lie = "Your total comes to €1.00, placing the order now."
    session = shopping(replies=_fills_a_cart_then_checks_out(claimed=lie))
    session.send("find me trail shoes")
    session.send("add the Summit Peak Pro")
    session.send("check out")

    summary = session.pending.summary

    assert "€1.00" not in summary
    assert "€149.99" in summary
    # And the model really did say it — otherwise this test passes for the
    # wrong reason, which is the failure mode D10 recorded for a probe that
    # never reproduced its own mechanism.
    assert any(lie in message.text for message in session.transcript)


def test_the_summary_is_the_string_the_dialog_prints(shopping):
    """`ui/app.py` renders `pending.summary` verbatim through `st.code`.

    Its line breaks and leading spaces are load-bearing — the gate laid the
    order out as lines and a renderer that reflowed them would be rewriting
    what somebody approved.
    """
    session = shopping(replies=_fills_a_cart_then_checks_out())
    session.send("find me trail shoes")
    session.send("add the Summit Peak Pro")
    session.send("check out")

    summary = session.pending.summary

    assert summary.count("\n") >= 2, "the summary is laid out as lines"
    assert summary == session._memory.pending_confirmation.summary


def test_confirming_lets_the_checkout_run_and_produces_a_payment_link(shopping):
    session = shopping(
        replies=[
            *_fills_a_cart_then_checks_out(),
            _tool("create_checkout", {}),
            FakeReply(content="Your order is placed."),
        ]
    )
    session.send("find me trail shoes")
    session.send("add the Summit Peak Pro")
    session.send("check out")

    result = session.answer_confirmation(True)

    assert session.backend.ordered is True
    assert result.messages[0].payment_url == FakeCommerceBackend.CHECKOUT_URL
    assert session.pending is None


def test_the_payment_link_never_reaches_the_model(shopping):
    """It is read off the conversation's memory, never scraped from prose.

    The model is not given the URL at all — `tools/commerce.py` puts it on the
    memory and returns `payment_link_shown: true` — so there is nothing for a
    renderer to extract and nothing for the model to mistype.
    """
    session = shopping(
        replies=[
            *_fills_a_cart_then_checks_out(),
            _tool("create_checkout", {}),
            FakeReply(content="Your order is placed."),
        ]
    )
    session.send("find me trail shoes")
    session.send("add the Summit Peak Pro")
    session.send("check out")
    session.answer_confirmation(True)

    everything_the_model_saw = json.dumps(session._messages)

    assert FakeCommerceBackend.CHECKOUT_URL not in everything_the_model_saw
    assert "checkout.stripe.com" not in everything_the_model_saw


def test_declining_orders_nothing_and_the_conversation_carries_on(shopping):
    session = shopping(
        replies=[
            *_fills_a_cart_then_checks_out(),
            FakeReply(content="Nothing was ordered."),
            FakeReply(content="Here are some jackets."),
        ]
    )
    session.send("find me trail shoes")
    session.send("add the Summit Peak Pro")
    session.send("check out")

    result = session.answer_confirmation(False)

    assert session.backend.ordered is False
    assert result.messages[0].payment_url is None
    assert session.pending is None
    # And the conversation is usable again.
    assert session.send("show me jackets").messages[1].text == "Here are some jackets."


def test_two_answers_to_one_question_do_not_both_go_through(shopping):
    """Two reruns must not answer the same question twice.

    D10 made `take_confirmation` clear as it reads and `resolve_pending` refuse
    an already-answered question; this asserts the UI does not go around either.
    A double-click on the dialog's button is the case, and it is real: Streamlit
    reruns the fragment on every click.
    """
    session = shopping(
        replies=[
            *_fills_a_cart_then_checks_out(),
            _tool("create_checkout", {}),
            FakeReply(content="Your order is placed."),
        ]
    )
    session.send("find me trail shoes")
    session.send("add the Summit Peak Pro")
    session.send("check out")
    calls_after_asking = len(session.fake_client.seen)

    first = session.answer_confirmation(True)
    turns_driven = len(session.fake_client.seen) - calls_after_asking
    second = session.answer_confirmation(True)

    assert first.messages != ()
    assert second.messages == (), "the second answer drove a turn of its own"
    assert len(session.fake_client.seen) - calls_after_asking == turns_driven


def test_an_answer_to_the_opposite_question_cannot_arrive_late(shopping):
    """Declining after confirming changes nothing, rather than un-ordering."""
    session = shopping(
        replies=[
            *_fills_a_cart_then_checks_out(),
            _tool("create_checkout", {}),
            FakeReply(content="Your order is placed."),
        ]
    )
    session.send("find me trail shoes")
    session.send("add the Summit Peak Pro")
    session.send("check out")
    session.answer_confirmation(True)

    assert session.answer_confirmation(False).messages == ()
    assert session.backend.ordered is True


def test_an_answered_approval_does_not_survive_the_next_customer_message(shopping):
    """D10's rule, from the other side: step 1 covered an *unanswered* one.

    An approval is spendable on exactly one turn. `begin_turn(from_customer=
    True)` is what kills it, and it has to, because an approval that outlives
    its turn is an answer sitting apart from the question it answered.
    """
    session = shopping(
        replies=[
            *_fills_a_cart_then_checks_out(),
            FakeReply(content="Waiting."),
            _tool("create_checkout", {}),
            FakeReply(content="I need you to confirm again."),
        ]
    )
    session.send("find me trail shoes")
    session.send("add the Summit Peak Pro")
    session.send("check out")
    session.answer_confirmation(True)
    # The follow-up turn above did not call `create_checkout`, so the approval
    # was answered and never spent. It is still recorded — but nobody is being
    # asked anything, which is what `pending` reports.
    assert session.pending is None
    assert session._memory.pending_confirmation.answer is True

    # A fresh message, and then a fresh checkout attempt: it must be asked
    # again rather than spending the yes given for the previous turn.
    session.send("actually, go ahead")

    assert session.backend.ordered is False
    assert session.pending is not None


# --- what the page is allowed to do with an amount ------------------------


def test_the_page_formats_no_money_of_its_own():
    """Every figure `app.py` shows was formatted before it got there.

    `VariantCard.price` is `money.format_amount`'s output from `ui/session.py`,
    and `PendingApproval.summary` is the gate's, built from a real `view_cart`.
    A renderer that reached for `format_amount` — or worse, for `/ 100` — would
    be a fourth opinion about what an amount looks like, and the first symptom
    would be a dialog and a payment page disagreeing about one order.
    """
    app = ast.parse((SESSION_PATH.parent / "app.py").read_text())

    assert "money" not in imported_names(app)
    assert not calls_in(app, "format_amount")
    assert not calls_in(app, "_summarise")
    for node in ast.walk(app):
        # `cents / 100` is the exact shape D1 forbade and D9 found in
        # `money.format_amount` itself. It has no business reappearing here.
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv)):
            raise AssertionError(f"app.py divides: {ast.unparse(node)}")


def test_a_second_answer_is_refused_even_when_the_first_was_never_spent(shopping):
    """The narrower half of "two reruns must not answer twice".

    When the follow-up turn *does* call `create_checkout`, `take_confirmation`
    clears the question and a second answer finds nothing — which is why the
    obvious version of this test passes with `resolve_pending`'s
    already-answered check mutated away. The check earns its place in the other
    branch: the model answered without spending the approval, so the question
    is still parked, and a double-click on the dialog would otherwise drive a
    second turn carrying a second CONFIRMED_NOTE.
    """
    session = shopping(
        replies=[
            *_fills_a_cart_then_checks_out(),
            FakeReply(content="Understood."),
            FakeReply(content="A second turn nobody asked for."),
        ]
    )
    session.send("find me trail shoes")
    session.send("add the Summit Peak Pro")
    session.send("check out")
    before = len(session.fake_client.seen)

    first = session.answer_confirmation(True)
    after_first = len(session.fake_client.seen)
    second = session.answer_confirmation(True)

    assert first.messages != ()
    assert after_first > before, "the first answer drove its follow-up turn"
    assert second.messages == ()
    assert len(session.fake_client.seen) == after_first, "a second turn was driven"
    # The approval is still on record and still unspent, which is the state the
    # `answered` check exists for.
    assert session._memory.pending_confirmation.answered is True


# --- what the activity panel is given ------------------------------------


def test_a_refused_tool_call_reaches_the_turn_that_asked_for_it(shopping):
    """The most valuable row the panel can show, at the level it is produced.

    The gate parking a confirmation is a failed `ToolResult`, not an exception,
    so only a wrapper *outside* `GuardedRegistry` sees it — which is why
    `RecordingRegistry` sits outermost. A panel built over anything further in
    would show a checkout that simply did not happen, with no reason.
    """
    session = shopping(replies=_fills_a_cart_then_checks_out())
    session.send("find me trail shoes")
    session.send("add the Summit Peak Pro")
    checkout = session.send("check out")

    (call,) = checkout.messages[1].activity

    assert call.name == "create_checkout"
    assert call.ok is False
    assert "confirm" in call.error
    assert call.duration_ms >= 0.0


def test_a_guardrail_refusal_reaches_the_turn_too(shopping):
    """D9's unknown-variant rule, seen from the panel.

    A `variant_id` the model was never shown is refused before the tool runs,
    and that refusal is a row somebody debugging this shop needs to see — it is
    the difference between "the cart call failed" and "the model made an id up".
    """
    session = shopping(
        replies=[
            _tool("add_to_cart", {"variant_id": 999999, "quantity": 1}),
            FakeReply(content="I could not add that."),
        ]
    )

    result = session.send("add the one I saw yesterday")

    (call,) = result.messages[1].activity
    assert call.name == "add_to_cart"
    assert call.ok is False
    # `error`, not `content`: the panel shows the short machine-facing half,
    # and the long one is written for the model to correct itself from.
    assert "never shown in this conversation" in call.error


def test_a_turn_reports_its_own_cost_and_model_calls(shopping):
    """Read from `UsageTracker`, which has computed both since D1. The panel
    measures nothing of its own."""
    session = shopping(replies=[_tool("search_products", {"query": "x"}),
                                FakeReply(content="Found it.")])

    message = session.send("find something").messages[1]

    assert message.model_calls == 2
    assert message.cost_usd > 0
    assert message.cost_usd == pytest.approx(session.session_cost_usd)


def test_a_refused_call_keeps_no_payload_for_the_panel(shopping):
    session = shopping(
        replies=[
            _tool("add_to_cart", {"variant_id": 999999}),
            FakeReply(content="No."),
        ]
    )
    (call,) = session.send("add it").messages[1].activity
    assert call.content == ""
    assert call.result_chars > 0, "the refusal text still has a length"


# --- one conversation, several traces ------------------------------------


def test_every_turn_of_one_tab_carries_the_same_session_id(offline):
    """The D10 Definition of done, reached the only way Streamlit allows.

    A browser cannot hold one root span open for a whole conversation: a rerun
    runs on a fresh thread and an OTEL span lives in a `contextvar`, so a root
    entered on one rerun's thread cannot be closed on another's. One root per
    turn is the shape that closes where it opened — and `session_id` is what
    Langfuse groups those roots back together with.
    """
    session = offline(replies=[FakeReply(content="a"), FakeReply(content="b")])
    seen = []

    class Recording(Tracer):
        def conversation(self, **fields):
            seen.append(fields.get("session_id"))
            return super().conversation(**fields)

    session._tracer = Recording()
    session.send("one")
    session.send("two")

    assert seen == [session.session_id, session.session_id]
    assert session.session_id


def test_two_tabs_are_two_langfuse_sessions(fresh_process):
    resources = ui.SharedResources(
        tracer=Tracer(), catalog_factory=FakeCatalogClient, commerce_factory=FakeCommerceAPI
    )
    one = ui.BrowserSession(resources, catalog_enabled=False)
    two = ui.BrowserSession(resources, catalog_enabled=False)
    try:
        assert one.session_id != two.session_id
    finally:
        one.close()
        two.close()


def test_the_session_id_is_not_the_shopper(fresh_process):
    """A `uuid4` this process invents, carrying nothing a person wrote.

    `shopper_id` identifies a person and leaves as a digest through
    `redact_identifier`; this identifies a browser tab and nobody. They are two
    parameters for that reason rather than one.
    """
    resources = ui.SharedResources(
        tracer=Tracer(), catalog_factory=FakeCatalogClient, commerce_factory=FakeCommerceAPI
    )
    session = ui.BrowserSession(resources, catalog_enabled=False, shopper_id="ana@example.com")
    try:
        assert "ana" not in session.session_id
        assert len(session.session_id) == 32
    finally:
        session.close()


def test_an_untraced_turn_offers_no_trace_link(offline):
    """Unconfigured tracing is an ordinary state, so the panel has to cope."""
    session = offline(replies=[FakeReply(content="hello")])
    assert session._tracer.enabled is False

    assert session.send("hi").messages[1].trace_url is None


def test_a_traced_turn_carries_the_link_the_tracer_gave_it(offline):
    # Only `trace_url` is overridden. An earlier version of this fake also
    # forced `enabled` true, which made the inert tracer's own `flush()` reach
    # for a client that was never there — a fake built from the shape the
    # assertion wanted rather than the shape the real object has.
    class Linking(Tracer):
        def trace_url(self):
            return "https://cloud.langfuse.com/trace/abc123"

    session = offline(replies=[FakeReply(content="hello")])
    session._tracer = Linking()

    assert session.send("hi").messages[1].trace_url == (
        "https://cloud.langfuse.com/trace/abc123"
    )


# --- the basket beside the conversation (D11 follow-up) ------------------


def _a_filled_basket() -> list[FakeReply]:
    """Two turns: search, then add one Summit Peak Pro. No checkout."""
    return [
        _tool("search_products", {"query": "trail shoes"}),
        FakeReply(content="Here is the Summit Peak Pro."),
        _tool("add_to_cart", {"variant_id": FakeCommerceBackend.VARIANT_ID, "quantity": 2}),
        FakeReply(content="Added two."),
    ]


def _fill(session) -> None:
    session.send("find me trail shoes")
    session.send("add two Summit Peak Pro")


def test_the_basket_panel_reads_the_shop_rather_than_the_transcript(shopping):
    """The defect a panel rendered from the last message would have.

    A `ChatMessage` holds what was true when it was written. The basket changes
    after that — here by a route the conversation never saw, which is the point:
    another client holding the API key can change a cart, and the gate binds the
    model rather than the shop. A panel reading the transcript would go on
    showing two units for the rest of the afternoon.
    """
    session = shopping(replies=_a_filled_basket())
    _fill(session)

    assert session.cart().unit_count == 2

    # Changed behind the conversation's back, exactly as another client would.
    session.backend.quantity = 5

    panel = session.cart()
    assert panel.unit_count == 5
    assert panel.lines[0].quantity == 5


def test_reading_the_basket_asks_the_commerce_api(shopping):
    session = shopping(replies=_a_filled_basket())
    _fill(session)
    before = len(session.backend.requests)

    session.cart()

    made = session.backend.requests[before:]
    assert made == [("GET", f"/cart/{FakeCommerceBackend.CART_ID}")]


def test_reading_the_basket_costs_no_model_call(shopping):
    """Streamlit re-runs this script on every click.

    A panel that asked the model what was in the basket would bill a shopper
    for scrolling, and the cap would stop a conversation nobody had had.
    """
    session = shopping(replies=_a_filled_basket())
    _fill(session)
    calls_before = len(session.fake_client.seen)
    cost_before = session.session_cost_usd

    for _ in range(20):
        session.cart()

    assert len(session.fake_client.seen) == calls_before
    assert session.session_cost_usd == cost_before


def test_the_basket_panel_uses_the_carts_own_id_and_never_makes_one(shopping):
    """The model has never seen a cart id and the panel does not invent one."""
    session = shopping(replies=_a_filled_basket())
    _fill(session)

    session.cart()

    # A `POST /cart` from the panel would be a second basket beside the one the
    # conversation filled, drawn empty next to a conversation that is not.
    assert session.backend.requests.count(("POST", "/cart")) == 1


def test_an_untouched_conversation_has_an_empty_basket(shopping):
    session = shopping(replies=[FakeReply(content="hello")])
    session.send("hi")

    panel = session.cart()

    assert panel.empty
    assert panel.lines == ()
    assert panel.unit_count == 0
    assert panel.error is None
    # No cart exists yet, so nothing was asked of the shop either.
    assert session.backend.requests == []


def test_a_basket_that_cannot_be_read_says_so_instead_of_raising(shopping):
    """The page has to survive the shop being down; the conversation still works."""
    session = shopping(replies=_a_filled_basket())
    _fill(session)

    def refuse(method, path, json=None):
        raise CommerceAPIUnreachable("the shop is not answering")

    session.backend.request = refuse

    panel = session.cart()

    assert panel.error == ui.CART_UNREADABLE
    assert panel.empty


def test_every_amount_in_the_panel_arrives_already_formatted(shopping):
    """`ui/app.py` formats no money, so this is where the figures are made."""
    session = shopping(replies=_a_filled_basket(), unit_cents=14999)
    _fill(session)

    panel = session.cart()

    money = format_amount(14999, get_settings().currency)
    assert panel.lines[0].unit_price == money
    assert panel.total == format_amount(29998, get_settings().currency)
    # The integer itself must not reach a renderer: `29998` reads as thirty
    # thousand to whoever is about to pay three hundred.
    assert "29998" not in panel.total


def test_the_panel_takes_the_total_the_server_computed(shopping):
    """Never a sum over the lines, which would be a second opinion.

    D6 recomputes a cart total from the database on every read precisely so
    that nothing downstream has to. The fake is made to disagree with its own
    lines, and the panel has to report the server.
    """
    session = shopping(replies=_a_filled_basket())
    _fill(session)
    real = session.backend.request

    def disagreeing(method, path, json=None):
        body = real(method, path, json)
        if method == "GET" and path.startswith("/cart/"):
            body = {**body, "total_cents": 111}
        return body

    session.backend.request = disagreeing

    assert session.cart().total == format_amount(111, get_settings().currency)


# --- the button, and what it does not go around --------------------------


def test_the_button_asks_for_a_tool_the_gate_actually_gates():
    """The whole argument, in one assertion.

    The button is defensible only because `create_checkout` is a tool the gate
    stops. The day that name falls out of `CONFIRM_BEFORE` — renamed, split,
    or the gate narrowed — the button silently stops being a request for
    confirmation and becomes a second way to buy something, with no test
    failing anywhere near it.
    """
    from shopagent.agent.guardrails import CONFIRM_BEFORE

    assert ui.CHECKOUT_TOOL in CONFIRM_BEFORE


def test_the_button_dispatches_through_the_registry_the_model_uses():
    """Structural, because the behavioural version cannot see the difference.

    `self._setup.registry` is the `GuardedRegistry` itself and would pass every
    test below: the gate still parks, the summary still comes from the cart.
    What it skips is the tracing and the recording wrapped around it, so a
    checkout started from the button would be missing from the trace and from
    the activity panel — invisible in exactly the surface built to make tool
    calls visible. `self._registry` is the one the model's loop is handed.
    """
    function = next(
        node
        for node in ast.walk(SESSION_TREE)
        if isinstance(node, ast.FunctionDef) and node.name == "request_checkout"
    )
    dispatches = calls_in(function, "dispatch")

    assert len(dispatches) == 1
    (call,) = dispatches
    assert ast.unparse(call.func) == "self._registry.dispatch"


def test_the_button_asks_the_gate_and_buys_nothing(shopping):
    """The falsification target: a button that placed the order itself.

    An implementation that called `POST /orders` — or the tool function behind
    the registry — would leave `backend.ordered` true with no question in front
    of anybody, which is the whole of what this must never do.
    """
    session = shopping(replies=_a_filled_basket())
    _fill(session)

    result = session.request_checkout()

    assert result.pending is not None
    assert result.pending.tool == "create_checkout"
    assert session.backend.ordered is False
    assert ("POST", "/orders") not in session.backend.requests


def test_the_button_costs_no_model_call_to_ask(shopping):
    """The question is the gate's, built from the cart. Nothing is generated."""
    session = shopping(replies=_a_filled_basket())
    _fill(session)
    before = len(session.fake_client.seen)

    session.request_checkout()

    assert len(session.fake_client.seen) == before


def test_the_total_the_button_puts_up_for_approval_comes_from_the_cart(shopping):
    """The property D9 paid for and D10 kept, now reached by a second caller.

    A person approving a figure the model invented is worse than no gate at
    all. The button changes who asks for the checkout and nothing about where
    the number comes from — so the cart is read again here, at the moment of
    asking, and the summary is the gate's own rendering of what came back.
    """
    session = shopping(replies=_a_filled_basket(), unit_cents=14999)
    _fill(session)
    before = len(session.backend.requests)

    result = session.request_checkout()

    assert ("GET", f"/cart/{FakeCommerceBackend.CART_ID}") in session.backend.requests[before:]
    assert format_amount(29998, get_settings().currency) in result.pending.summary
    assert "Summit Peak Pro" in result.pending.summary


def test_confirming_a_button_checkout_places_the_order_and_shows_the_link(shopping):
    """The whole path, ending where the model's own path ends."""
    session = shopping(
        replies=_a_filled_basket()
        + [
            # The follow-up turn the answer drives. The model calls the tool
            # again and the gate spends the approval — which is the protocol,
            # not a shortcut the button took.
            _tool("create_checkout", {}),
            FakeReply(content="Your order is placed — the payment link is below."),
        ]
    )
    _fill(session)
    session.request_checkout()

    result = session.answer_confirmation(True)

    assert session.backend.ordered is True
    assert session.pending is None
    assert result.messages[-1].payment_url == FakeCommerceBackend.CHECKOUT_URL


def test_declining_a_button_checkout_orders_nothing(shopping):
    session = shopping(
        replies=_a_filled_basket() + [FakeReply(content="Nothing was ordered.")]
    )
    _fill(session)
    session.request_checkout()

    result = session.answer_confirmation(False)

    assert session.backend.ordered is False
    assert session.pending is None
    assert "Nothing was ordered." in result.messages[-1].text


def test_the_payment_link_a_button_checkout_produces_never_reaches_the_model(shopping):
    """The URL is the shop's to print, on this path as on the other one."""
    session = shopping(
        replies=_a_filled_basket()
        + [_tool("create_checkout", {}), FakeReply(content="Placed.")]
    )
    _fill(session)
    session.request_checkout()
    session.answer_confirmation(True)

    everything = json.dumps(session._messages)
    assert FakeCommerceBackend.CHECKOUT_URL not in everything
    assert "checkout.stripe.com" not in everything


def test_a_click_while_a_question_is_open_changes_nothing(shopping):
    """The page disables the button; this does not depend on the page doing it.

    A second click must not park a second question over the first, and must not
    reach `begin_turn(from_customer=True)` — which would drop the approval the
    customer is looking at and leave the modal describing nothing.
    """
    session = shopping(replies=_a_filled_basket())
    _fill(session)
    first = session.request_checkout().pending
    before = list(session.backend.requests)

    again = session.request_checkout()

    assert again.pending == first
    assert session.backend.requests == before


def test_a_click_past_the_spend_cap_changes_nothing(shopping):
    """The follow-up turn an answer drives is a model call.

    The door that refuses a typed message has to refuse a click, or the cap
    would be reachable around it in one press.
    """
    # Filled first, then the cap is lowered onto a session that already has a
    # basket. Building it with a cap of nothing instead meant `send` refused
    # the turn that fills the cart, so the click met an empty basket and was
    # refused for that reason — the guard could be mutated away and the test
    # still passed, which is how this was found rather than reasoned about.
    session = shopping(replies=_a_filled_basket())
    _fill(session)
    session._cap_usd = 0.0
    assert session.cap_reached
    assert session.cart().unit_count == 2
    before = list(session.backend.requests)

    result = session.request_checkout()

    assert result.pending is None
    assert session.backend.requests == before
    assert session.backend.ordered is False


def test_a_click_on_an_empty_basket_orders_nothing_and_says_so(shopping):
    """The page disables the button here too; the turn logic still has to hold.

    Nothing to confirm means the gate lets `create_checkout` through, and the
    tool refuses an empty cart in its own words. What a person sees is a notice
    rather than the tool's sentence, which was written for the model.
    """
    session = shopping(replies=[FakeReply(content="hello")])
    session.send("hi")

    result = session.request_checkout()

    assert result.pending is None
    assert session.backend.ordered is False
    assert result.messages[-1].notice == ui.CHECKOUT_NOT_STARTED


def test_a_button_checkout_appears_in_the_activity_of_the_turn_that_settles_it(shopping):
    """The click is not a silent path through the shop.

    `RecordingRegistry` sits outermost, so the `create_checkout` the follow-up
    turn makes is recorded like any other — which is what makes the panel a
    record of what happened rather than of what the model decided.
    """
    session = shopping(
        replies=_a_filled_basket()
        + [_tool("create_checkout", {}), FakeReply(content="Placed.")]
    )
    _fill(session)
    session.request_checkout()

    result = session.answer_confirmation(True)

    names = [call.name for call in result.messages[-1].activity]
    assert "create_checkout" in names


# --- what the page does with the panel -----------------------------------


def _draw_cart_source() -> ast.FunctionDef:
    app = ast.parse((SESSION_PATH.parent / "app.py").read_text())
    return next(
        node
        for node in ast.walk(app)
        if isinstance(node, ast.FunctionDef) and node.name == "_draw_cart"
    )


def test_the_page_offers_no_button_over_a_basket_it_could_not_read():
    """An empty basket and an unreadable one both leave before the button.

    A checkout button over a basket nobody could read is a button whose total
    is unknown, and one over an empty basket is a press that can only produce a
    refusal. Read structurally because the alternative is importing `app.py`,
    which runs the whole page at module scope.
    """
    function = _draw_cart_source()
    button = next(
        call for call in calls_in(function, "button")
        if ast.unparse(call.func).endswith("st.button")
    )

    guarded = {
        ast.unparse(node.test)
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and node.lineno < button.lineno
        and any(isinstance(inner, ast.Return) for inner in node.body)
    }
    assert "panel.error" in guarded
    assert "panel.empty" in guarded


def test_the_page_disables_the_button_while_a_question_or_the_cap_stands():
    function = _draw_cart_source()
    button = next(
        call for call in calls_in(function, "button")
        if ast.unparse(call.func).endswith("st.button")
    )

    disabled = next(
        keyword for keyword in button.keywords if keyword.arg == "disabled"
    )
    guard = next(
        ast.unparse(node.value)
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == ast.unparse(disabled.value)
            for target in node.targets
        )
    )
    assert "pending is not None" in guard
    assert "cap_reached" in guard


def test_the_page_offers_no_way_to_take_a_line_out_of_the_basket():
    """Changing a basket is something you ask for.

    A remove control in the panel would be the second shopping interface the
    whole layout argues against — the same reason a product card has no Add
    button. The one control is the checkout, and it is named.
    """
    function = _draw_cart_source()
    labels = {
        ast.unparse(call.args[0]) if call.args else ""
        for call in calls_in(function, "button")
    }
    assert labels == {"CHECKOUT_LABEL"}
