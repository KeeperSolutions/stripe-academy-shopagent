"""Where a browser session's turn is decided (D11, step 1).

`app.py` renders; this decides. The cut is the one `api/routers/` and
`api/services/` already make, and it is drawn for the same reason plus one
more: **nothing in this file may import `streamlit`**, so a turn can be driven
in a test with no browser, no server and no rerun.
`tests/test_ui_session.py` walks the AST and fails if the import appears, which
is the mechanism `tests/test_evals.py` uses to keep the runner honest.

**Streamlit reruns the entire script on every interaction**, and that is the
fact this module is shaped around. Three things in this project must not be
built twice, and one of them has already cost this repository a day:

- **The tracer.** D10 measured it: Langfuse keeps one resource manager per
  public key, *process-wide*. A second `shutdown()` enqueues a stop sentinel per
  consumer onto a queue whose consumers are already dead, and the next
  `flush()` — which is `queue.join()` — waits for a `task_done()` that cannot
  come. Two eval passes hung there. A tracer per click would reach that state
  in seconds. **Do not call `Tracer.shutdown()` per turn or per session**; the
  one shutdown this process performs is `shutdown_shared_resources()`, at exit.
- **The MCP catalog client.** It is a subprocess with a Postgres pool inside
  it. One per click is a fork bomb with a spinner.
- **The commerce HTTP client.** An `httpx.Client` is a connection pool; one per
  click leaks sockets until the process dies.

So there are two tiers, and which tier a thing belongs to is decided by whether
it holds any of *this conversation's* state:

    per process (`shared_resources`)   per browser session (`BrowserSession`)
    ────────────────────────────────   ──────────────────────────────────────
    Tracer                             ConversationMemory
    MCPToolClient (subprocess)         the registry chain built over it
    CommerceAPI (httpx.Client)         messages, transcript, cost, pending

**The second column is not an optimisation and must not be moved.**
`@st.cache_resource` — which step 2 wraps `shared_resources()` in — is shared
across *every browser session in the process*, not per tab. A
`ConversationMemory` up there would mean one tab's `add_to_cart` landing in
another tab's basket, and a confirmation parked in one tab being spendable in
the other. `agent/memory.py` says "never shared, never global" and means it.

**The registry is still the one `build_tool_setup` returns.** This module calls
the CLI's own function and hands `run_tool_loop` what comes back, so the gate,
the memory, the guardrails, the traced wrappers and the MCP catalog are the ones
a customer gets — the same claim `evals/runner.py` makes and the same reason.
The shared clients reach it through `client_factory` and `api_factory`, behind
a context manager whose exit is a no-op, so the per-session `ExitStack` cannot
close a resource the process still needs.

**A confirmation is answered through `agent/confirmation.py`, not around it.**
D10 rebuilt the gate as two phases precisely for this caller: the question is
parked on the memory and the answer arrives in a *later HTTP request*, which no
blocking callable could ever serve. `send()` returns with `pending` set and
nothing bought; `answer_confirmation()` records the answer through
`resolve_pending` and drives the one follow-up turn. That is exactly what
`_settle_confirmation` in the CLI does and what `_settle` in the eval runner
does.

**The profile is read and never written.** `/remember` and `/forget` stay CLI
commands. A profile is injected into the system prompt, so anything storable in
it is read with the authority of the assistant's own instructions — D9's
argument for a closed domain of five categories and four characters of size.
The one irreducibly free-text field is the display name, and the honest state
of that surface is "narrowed and defended in depth", not "closed". Widening the
number of doors onto it is not something a UI step gets to do quietly, so this
module offers no path to one; `tests/test_ui_session.py` asserts the absence
rather than trusting this paragraph.
"""

from __future__ import annotations

import atexit
import threading
import uuid
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any

from shopagent.agent import confirmation as confirmation_protocol
from shopagent.agent import profile as profiles
from shopagent.agent.activity import ActivityLog, RecordingRegistry, ToolCallRecord
from shopagent.agent.guardrails import GuardedClient
from shopagent.agent.memory import SEARCH_TOOL, results_in
from shopagent.agent.prompt import initial_messages
from shopagent.config import get_settings
from shopagent.llm.client import LLMClient
from shopagent.llm.loop import build_tool_setup, run_tool_loop
from shopagent.llm.usage import UsageTracker
from shopagent.mcp_client.client import MCPToolClient
from shopagent.money import format_amount
from shopagent.obs.instrumentation import TracedClient, TracedRegistry
from shopagent.obs.tracing import Tracer, build_tracer
from shopagent.tools.http import CommerceAPI, CommerceAPIError

CUSTOMER = "customer"
SHOP = "shop"

# The one tool the cart panel's button asks for. Named here rather than
# imported from `agent/guardrails.py`, which holds it as a member of
# `CONFIRM_BEFORE` — a set of tools the gate stops, not a name a caller is
# meant to reach for. What matters is that the two agree, and that is asserted
# rather than arranged: `test_the_button_asks_for_a_tool_the_gate_actually_gates`
# fails if this name ever falls outside `CONFIRM_BEFORE`, which is the moment
# the button would stop being a request to the gate and become a second way to
# buy something.
CHECKOUT_TOOL = "create_checkout"

# What a person is told when the session's cap is reached. Written for a
# shopper rather than for an operator: it says what stopped, that nothing is
# lost, and what to do — and it deliberately does not name a dollar figure the
# shop's own costs are denominated in, which is not this customer's business
# and is in a different currency from every price they have been quoted.
CAP_NOTICE = (
    "This demo session has reached its spending limit, so the assistant has "
    "stopped answering. Everything above is still here to read, and any order "
    "already placed is unaffected. Start a new session to carry on."
)


# What a person is told when the cart cannot be read for the panel. The panel
# is a convenience beside a conversation that still works, so this says what is
# missing and points at the thing that does work, rather than reading as an
# outage.
CART_UNREADABLE = (
    "The basket cannot be shown right now. Ask the assistant what is in it."
)

# When the button's request reached the shop and no question came back. The
# gate parks a question for every basket there is something to confirm about,
# so this is the other case: nothing to check out. Written for a shopper, and
# deliberately vague about which of the several reasons it was — the assistant
# below can say, and it has the conversation to say it in.
CHECKOUT_NOT_STARTED = (
    "The shop could not start a checkout for this basket. Ask the assistant "
    "below and it will say why."
)


# --- tier one: what this process builds once -----------------------------


class _Borrowed:
    """A process-lived resource, lent to a session's `ExitStack`.

    `build_tool_setup` owns whatever it is given: it calls the factory inside
    `stack.enter_context`, and that stack is unwound when the browser session
    ends. That is right for the CLI, where the stack's life *is* the process's,
    and wrong here — closing the MCP subprocess because one tab went away would
    take the catalog out from under every other tab.

    So `__enter__` starts the real resource at most once and `__exit__` does
    nothing. The one place it is really closed is
    `shutdown_shared_resources()`, through the process stack this holds.

    Starting lazily rather than eagerly is deliberate: `MCPToolClient.__enter__`
    spawns a server and handshakes with it, and `build_tool_setup` already
    catches that failing and reports a session with no catalog instead of
    refusing to run. Starting it here would move that failure to import time,
    where nothing is set up to explain it.
    """

    def __init__(self, build: Any, stack: ExitStack, lock: threading.Lock) -> None:
        self._build = build
        self._stack = stack
        self._lock = lock
        self._value: Any = None

    def __call__(self) -> "_Borrowed":
        """Both the factory and the thing it makes.

        `build_tool_setup` calls the factory and enters the result — the shape
        `MCPToolClient` has, where calling the class makes a fresh client per
        session. Here there is one resource and every session gets the same
        handle on it, so the factory returns itself rather than a new object
        that would have to find its way back to the same lazy value.
        """
        return self

    def __enter__(self) -> Any:
        # Streamlit runs each browser session's script on its own thread, so
        # two tabs opened together really do arrive here at once. Without the
        # lock that is two subprocesses, one of which nothing would ever close.
        with self._lock:
            if self._value is None:
                self._value = self._stack.enter_context(self._build())
            return self._value

    def __exit__(self, *exc_info: object) -> None:
        return None


@dataclass(frozen=True)
class SharedResources:
    """Everything one process holds for the life of the process.

    Handed to `BrowserSession` rather than reached for, so a test can supply
    its own and never spawn a subprocess or touch a network.
    """

    tracer: Tracer
    # Called by `build_tool_setup` inside a per-session `ExitStack`. Both hand
    # back a `_Borrowed`, which is why a session ending does not close them.
    catalog_factory: Any
    commerce_factory: Any


_lock = threading.Lock()
_process_stack = ExitStack()
_shared: SharedResources | None = None


def shared_resources() -> SharedResources:
    """The process's shared resources, built at most once.

    Memoised by hand rather than with `functools.lru_cache`, for a reason worth
    stating: this has to be paired with a `cache_clear` that also unwinds the
    `ExitStack`, and an `lru_cache` offers a `cache_clear` that silently would
    not. Two ways to forget one fact, one of which leaks a subprocess.

    Step 2 wraps this in `@st.cache_resource`. That decorator would be enough
    on its own *inside* Streamlit; the memoisation here is what makes the same
    guarantee hold for a test, a script, or anything else that imports this
    module — which is the whole reason the Streamlit call lives one file up.
    """
    global _shared
    with _lock:
        if _shared is None:
            _shared = SharedResources(
                tracer=build_tracer(),
                catalog_factory=_Borrowed(MCPToolClient, _process_stack, _lock),
                commerce_factory=_Borrowed(CommerceAPI, _process_stack, _lock),
            )
        return _shared


def shutdown_shared_resources() -> None:
    """Close the subprocess and the sockets, and shut the tracer down once.

    Registered with `atexit` because Streamlit has no shutdown hook a module can
    reach. Exactly one `Tracer.shutdown()` happens in this process, and it
    happens here — see the module docstring for what a second one costs.
    """
    global _shared
    with _lock:
        shared, _shared = _shared, None
        _process_stack.close()
    if shared is not None:
        shared.tracer.shutdown()


atexit.register(shutdown_shared_resources)


# --- the contract step 2 renders -----------------------------------------


@dataclass(frozen=True)
class VariantCard:
    """One buyable configuration, as a card shows it."""

    variant_id: int
    sku: str
    size: str | None
    color: str | None
    price_cents: int
    # The same integer through `money.format_amount`, resolved here rather than
    # in the renderer. Every amount a person reads in this system goes through
    # that one function, and a template doing `cents / 100` in a browser would
    # be the float this project has refused since D1.
    price: str
    available: int

    @property
    def in_stock(self) -> bool:
        return self.available > 0


@dataclass(frozen=True)
class ProductCard:
    """One product from a search, with its variants, ready to render.

    Colour is present because `catalog/search.py` already returns it per
    variant, alongside `size`, `sku`, `price_cents` and `available`. There is
    no image: the catalog has no image column and no tool returns one, so a
    card is text — adding a picture would mean a schema change and a second
    tool call, which is not something a rendering step decides on its own.
    """

    product_id: int
    name: str
    brand: str | None
    category: str | None
    description: str | None
    variants: tuple[VariantCard, ...]


@dataclass(frozen=True)
class CartLine:
    """One line of the basket, as the panel beside the conversation shows it.

    Both amounts arrive already through `money.format_amount`, for the reason
    `VariantCard.price` does: every figure a person reads in this system comes
    out of that one function, and a renderer doing its own arithmetic is the
    float this project has refused since D1.
    """

    variant_id: int
    product_name: str
    variant_label: str
    quantity: int
    unit_price: str
    line_total: str


@dataclass(frozen=True)
class CartPanel:
    """What is in the basket right now, read from the shop rather than recalled.

    **Read from the commerce API on every draw, never from the transcript.** A
    `ChatMessage` carries what was true when it was written, and a basket
    changes after that — a panel rendered from the last `view_cart` result
    would show a line the customer removed two turns ago and would keep showing
    it. The panel is a live view or it is a lie with a timestamp nobody can see.

    Reading it costs one HTTP request and no model call at all, which is the
    other half of why it can be redrawn on every rerun: Streamlit re-executes
    this script on every click, and a panel that asked the model what was in
    the basket would bill a shopper for scrolling.
    """

    lines: tuple[CartLine, ...]
    total: str
    unit_count: int
    # Set when the cart could not be read. The panel then says so and offers
    # nothing to press — a checkout button over a basket nobody could read is a
    # button whose total is unknown.
    error: str | None = None

    @property
    def empty(self) -> bool:
        return not self.lines


# What each gated question is called, and what the page says while it stands.
#
# Here rather than in `ui/app.py` because it is a decision and not a rendering:
# which question is being asked is a fact about the tool, and the file that
# decides a turn is the one that knows it. It is also the only way to test it,
# since `app.py` runs the whole page at import.
#
# The fallback is neutral rather than an exception, which is the opposite of
# what `confirmation.follow_up_note` does with an unknown tool — and the
# difference is who reads the result. There, the wrong string is an instruction
# to the model to place an order nobody asked about. Here it is a heading over
# a summary the person is reading anyway, so vague is survivable and a blank
# page is not.
_QUESTIONS = {
    "create_checkout": (
        "Confirm this purchase",
        "Waiting for your confirmation. Nothing has been ordered and nothing "
        "has been charged.",
    ),
    "request_refund": (
        "Confirm this refund",
        "Waiting for your confirmation. No refund has been requested and your "
        "order is unchanged.",
    ),
}

_UNNAMED_QUESTION = (
    "Confirm this",
    "Waiting for your confirmation. Nothing has happened yet.",
)


@dataclass(frozen=True)
class PendingApproval:
    """The question in front of the customer right now.

    `summary` is the gate's own text, built from a real tool result and
    rendered through `money.format_amount` — never from anything the model
    wrote. That is the property D9 paid for and D10 kept through the rewrite: a
    person approving a figure the model invented is worse than no gate at all,
    because it launders the invention through a human and leaves a record
    saying they agreed. The page renders this string and must not compose its
    own.
    """

    tool: str
    summary: str

    @property
    def title(self) -> str:
        """What the dialog is headed. "Confirm this purchase" over a refund is
        a heading that contradicts the summary underneath it."""
        return _QUESTIONS.get(self.tool, _UNNAMED_QUESTION)[0]

    @property
    def waiting(self) -> str:
        """What the page says behind the dialog while the question stands.

        Per tool because the reassurance differs and reverses: a parked
        purchase has charged nothing, while a parked refund leaves an order
        that is still paid — and "nothing has been charged" would be telling
        somebody their money is not where it is.
        """
        return _QUESTIONS.get(self.tool, _UNNAMED_QUESTION)[1]


@dataclass(frozen=True)
class ChatMessage:
    """One bubble in the transcript.

    **The cards belong to the message, not to the session.** The tempting
    alternative is for a renderer to read `ConversationMemory.last_search` when
    it draws a bubble, and it is wrong for a reason D9 wrote down: every new
    search *replaces* the previous one on purpose, because "the second one" can
    only mean the second row of the list the customer is looking at now. A
    transcript rendered from it would show the newest results under every older
    answer. So the rows are captured when they are produced and stay on the
    message that produced them, and `last_search` goes on meaning what it means.
    """

    role: str
    text: str
    cards: tuple[ProductCard, ...] = ()
    activity: tuple[ToolCallRecord, ...] = ()
    # What this turn cost and how many model calls it took. Read from
    # `UsageTracker`, which has computed both since D1 — nothing is measured a
    # second time here.
    cost_usd: float = 0.0
    model_calls: int = 0
    # The Stripe payment page in the bytes the shop issued, never relayed
    # through the model: asked twice for one session, the model reproduced the
    # 475-character URL correctly once and changed a character the second time.
    # Measured on PR #9; `tools/commerce.py` puts it on the memory instead.
    payment_url: str | None = None
    # Something the shop is saying about itself rather than something the
    # assistant said — the spend cap, a failed turn, a missing catalog.
    notice: str | None = None
    # Where this turn's trace is, when tracing is on. `None` is the ordinary
    # answer — an unconfigured Langfuse is a normal state, so the panel that
    # renders this has to cope with its absence anyway.
    trace_url: str | None = None


@dataclass(frozen=True)
class TurnResult:
    """What one call produced. The contract `app.py` consumes.

    `messages` is only what this turn added; the whole conversation is
    `BrowserSession.transcript`. Both are given because a renderer that appends
    and a renderer that redraws are both reasonable, and neither should have to
    diff.
    """

    messages: tuple[ChatMessage, ...]
    # Set when the gate parked a question. The turn is over and nothing was
    # bought; answer it with `BrowserSession.answer_confirmation`.
    pending: PendingApproval | None
    session_cost_usd: float
    cap_usd: float
    # Two facts, not one, and a test conflated them once. `cap_reached` is the
    # *state after* this call — what a renderer reads to decide whether to
    # disable its input box. `refused` is what happened *on* this call: the cap
    # was already reached at the door, so no model call was made at all. The
    # turn that first crosses the cap has `cap_reached` true and `refused`
    # false, and it is a real answer.
    cap_reached: bool
    refused: bool = False
    # An exception this turn raised, already formatted. The conversation
    # survives it: the partial turn is rewound and everything before it is
    # still readable.
    error: str | None = None


# --- tier two: one browser session ---------------------------------------


@dataclass
class _BrowserConfirmer:
    """Somebody is reachable — just not from inside a dispatch.

    `GuardedRegistry` needs to know a person exists (`can_confirm`), because a
    gate that cannot reach anybody must refuse rather than allow. In a browser
    that person exists but answers in a *later request*, so this is passed to
    `build_tool_setup` to make `can_confirm` true and is invoked only later,
    from `answer_confirmation`, once the answer is actually known.

    `answer is None` returns False, and that is the same rule
    `_ask_to_confirm` applies to end-of-input: the safe answer to "could not
    ask" is the answer to "they said no". Reaching this branch would mean a
    caller resolved a confirmation without setting one, which must not buy
    anything.
    """

    answer: bool | None = None
    asked: list[str] = field(default_factory=list)

    def __call__(self, summary: str) -> bool:
        self.asked.append(summary)
        return bool(self.answer)


class BrowserSession:
    """One browser tab's conversation, and the only thing that drives a turn.

    Owns a `ConversationMemory`, a message list, a transcript and a
    `UsageTracker` — everything with this conversation's lifetime — over the
    process-wide resources it is handed. Close it with `close()` when the tab
    goes away; the shared clients survive that by design.
    """

    def __init__(
        self,
        resources: SharedResources | None = None,
        *,
        catalog_enabled: bool | None = None,
        client_factory: Any = None,
        shopper_id: str | None = None,
        spend_cap_usd: float | None = None,
    ) -> None:
        settings = get_settings()
        self._resources = resources if resources is not None else shared_resources()
        self._tracer = self._resources.tracer
        self._cap_usd = (
            spend_cap_usd if spend_cap_usd is not None else settings.ui_spend_cap_usd
        )
        self._shopper_id = shopper_id if shopper_id is not None else settings.shopper_id

        # One conversation, several traces. A turn opens and closes its own
        # root — see `_drive` for why it cannot be one root for the tab — and
        # this is what puts them back together in Langfuse. A `uuid4` rather
        # than the shopper's id: that one identifies a person and is digested
        # on its way out; this one identifies a browser tab and nothing else.
        self._session_id = uuid.uuid4().hex
        self._stack = ExitStack()
        self._confirmer = _BrowserConfirmer()
        self._tracker = UsageTracker()
        self._activity = ActivityLog()
        self._transcript: list[ChatMessage] = []

        # The CLI's own call. `client_factory` is overridable for a test that
        # wants a catalog which fails to start; everything else is what
        # `main()` builds, which is what makes the loop below the loop a
        # customer runs.
        self._setup = build_tool_setup(
            self._stack,
            catalog_enabled=catalog_enabled,
            client_factory=client_factory or self._resources.catalog_factory,
            api_factory=self._resources.commerce_factory,
            confirm=self._confirmer,
        )
        self._memory = self._setup.memory

        # Wrapped in the order `_run_session` wraps them, for the reason it
        # does: `TracedClient` *inside* `GuardedClient`, so the corrected retry
        # the amount guardrail can send shows as the second billed call it is.
        # `RecordingRegistry` goes outermost so the panel sees the gate's own
        # refusals, which are `ToolResult`s from `GuardedRegistry.dispatch`
        # rather than exceptions.
        self._registry = RecordingRegistry(
            TracedRegistry(self._setup.registry, self._tracer), self._activity
        )
        self._client = GuardedClient(
            TracedClient(LLMClient(tracker=self._tracker), self._tracer),
            self._memory,
            self._tracer,
        )

        # Read, never written. See the module docstring.
        self._profile, self._profile_note = profiles.load_for_session(self._shopper_id)
        self._messages = initial_messages(
            self._setup.catalog_available, profile=self._profile
        )
        self._tools = self._registry.openai_schemas()

    # --- what a renderer reads -------------------------------------------

    @property
    def transcript(self) -> tuple[ChatMessage, ...]:
        return tuple(self._transcript)

    @property
    def pending(self) -> PendingApproval | None:
        """The question in front of the customer, if there is one."""
        parked = self._memory.pending_confirmation if self._memory else None
        if parked is None or parked.answered:
            return None
        return PendingApproval(tool=parked.tool, summary=parked.summary)

    @property
    def session_id(self) -> str:
        """This tab's conversation, as Langfuse groups it."""
        return self._session_id

    @property
    def session_cost_usd(self) -> float:
        return self._tracker.total_cost_usd

    @property
    def cap_usd(self) -> float:
        return self._cap_usd

    @property
    def cap_reached(self) -> bool:
        return self.session_cost_usd >= self._cap_usd

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._registry.names())

    @property
    def catalog_available(self) -> bool:
        return self._setup.catalog_available

    @property
    def notes(self) -> tuple[str, ...]:
        """What the shop has to say about how it started, for a status line."""
        found = [self._setup.note, self._profile_note]
        if not self._tracer.enabled:
            from shopagent.obs.tracing import UNCONFIGURED_NOTE

            found.append(UNCONFIGURED_NOTE)
        return tuple(note for note in found if note)

    @property
    def profile(self) -> profiles.Profile | None:
        """What the shop remembers, for display. There is no setter."""
        return self._profile

    def cart(self) -> CartPanel:
        """The basket, read from the shop over the tools' own client.

        Three things about where this reads from, and each of them is the
        reason it is not read from somewhere easier.

        **The commerce API, not the transcript.** A message holds what was true
        when it was written; a basket does not stay that way. `self._setup.api`
        is the same client the five commerce tools were built over, so the
        panel and the agent are looking at one shop through one connection
        pool.

        **`memory.cart_id`, not an id of its own.** The model has never seen a
        cart id and never will — that is D9's rule and the reason the tools
        hold it. A panel that made its own cart would draw an empty basket
        beside a conversation that had filled one.

        **No model call, ever.** One `GET`, and the numbers come back already
        computed by the API, which recomputes a cart total from the database on
        every read for exactly this reason. Streamlit re-runs this script on
        every click, so anything here that reached the model would bill a
        shopper for scrolling.
        """
        currency = get_settings().currency
        cart_id = self._memory.cart_id if self._memory else None
        if cart_id is None:
            # Not an error and not a failure to read: a shopper who has added
            # nothing has an empty basket, which is the same answer `view_cart`
            # gives the model in the same situation.
            return CartPanel(lines=(), total=format_amount(0, currency), unit_count=0)

        api = self._setup.api
        if api is None:
            return CartPanel(
                lines=(), total=format_amount(0, currency), unit_count=0,
                error=CART_UNREADABLE,
            )

        try:
            body = api.request("GET", f"/cart/{cart_id}")
        except CommerceAPIError:
            # Narrow on purpose: this catches the shop being unreachable, slow,
            # or refusing, which are the states a panel has to survive. Anything
            # else is a fault in this process and belongs in the traceback the
            # page would otherwise never show.
            return CartPanel(
                lines=(), total=format_amount(0, currency), unit_count=0,
                error=CART_UNREADABLE,
            )

        return _panel(body, currency)

    # --- driving a turn ---------------------------------------------------

    def send(self, text: str) -> TurnResult:
        """One customer message, driven to an answer.

        Returns without asking anybody anything: if the gate parked a
        confirmation, `pending` is set and the caller puts it to the person in
        its own time. That is the half of the D10 protocol built for exactly
        this caller — the answer arrives in a later request, and no callable
        could have blocked for it.
        """
        text = text.strip()
        if not text:
            return self._nothing_happened()

        if self.cap_reached:
            # Checked at the door, before any model call. The customer's own
            # message is still shown, because a refusal under a bubble that
            # vanished would read as the shop losing it — but it is deliberately
            # not appended to `self._messages`: nothing is going to answer it,
            # and a user message with no assistant turn after it is a message
            # the next request would answer out of order.
            return self._refuse(text)

        # A customer message. Anything a person was asked to approve and never
        # answered lapses here — `ConversationMemory.begin_turn`, and it is
        # load-bearing rather than tidy: an approval that outlives its turn is
        # an answer sitting apart from the question it answered.
        self._memory.begin_turn(from_customer=True)
        customer = ChatMessage(role=CUSTOMER, text=text)
        self._transcript.append(customer)
        shop, error = self._drive(
            lambda: self._messages.append({"role": "user", "content": text})
        )
        return self._result((customer, shop), error)

    def request_checkout(self) -> TurnResult:
        """Ask the gate to check this basket out, on the customer's own click.

        **This is a second way to *start* a checkout and deliberately not a
        second way to *make* one.** D11 decided the cart panel would be
        read-only, on the argument that a second route to payment contradicts a
        demo built to show an agent. That decision is reversed here for exactly
        one implementation, and which implementation is the whole of the
        reasoning:

        - **Not** `POST /orders` and `POST /orders/{id}/checkout` from the
          page. That really is a second route: it reaches the commerce API
          without the gate, without the memory and without a summary anybody
          approved, and it stays refused.
        - **Not** typing "proceed to checkout" into the conversation as though
          the customer had. It adds no route, but it inherits the variance D10
          measured and D11 hit live — the model sometimes answers a request to
          check out with prose instead of a tool call — so a button that
          sometimes does nothing is a button nobody trusts.
        - **This**: the same `create_checkout`, dispatched through the same
          `GuardedRegistry.dispatch` the model reaches, which parks the same
          question built from the same `view_cart` read and rendered through
          the same `money.format_amount`.

        What is bypassed is one thing and it is nameable: the model's decision
        to call the tool. The customer expresses that decision by clicking,
        which is a better signal than a sentence somebody has to hope is
        parsed. **Everything the gate protects is still in front of them** —
        the summary comes from the cart and not from prose, a person still
        answers, `_spend` still re-reads the basket and refuses an approval
        given for a different one, and the order is still placed by
        `create_checkout` under `place_order`'s locks.

        `begin_turn(from_customer=True)` because a click is the customer
        acting, and it carries that method's other effect on purpose: an
        approval nobody answered lapses here, so a click can never spend one
        parked earlier. The page disables the button while a question is open,
        and this does not rely on that.

        Returns with `pending` set and nothing bought — the same half-finished
        state `send` returns when the model asks. The caller puts the question
        to the customer and answers it through `answer_confirmation`, which is
        D10's protocol unchanged and not a third implementation of it.
        """
        if self.pending is not None or self.cap_reached:
            # A question is already in front of them, or this session has
            # stopped spending. Either way the click changes nothing: the
            # follow-up turn an answer drives is a model call, and letting one
            # start past the cap would spend money the door already refused.
            return self._nothing_happened()

        self._memory.begin_turn(from_customer=True)
        self._activity.begin_turn()
        self._registry.dispatch(CHECKOUT_TOOL, {})

        if self.pending is not None:
            # The ordinary outcome: a question is parked and nothing ran.
            return self._result(())

        # The gate let the call through, which it does when there is nothing to
        # confirm — an empty basket, or an order already placed in this
        # conversation whose payment page `create_checkout` resumes. The link
        # is taken off the memory here for the same reason a turn takes it:
        # left there, it would be printed again under every later answer.
        link = self._memory.take_checkout_url() if self._memory else None
        shop = ChatMessage(
            role=SHOP,
            text="",
            activity=tuple(self._activity.calls),
            payment_url=link,
            notice=None if link else CHECKOUT_NOT_STARTED,
        )
        self._transcript.append(shop)
        return self._result((shop,))

    def answer_confirmation(self, approved: bool) -> TurnResult:
        """Carry a person's yes or no back to the model, through the protocol.

        `resolve_pending` then one turn carrying `follow_up_note` — the same two
        calls `_settle_confirmation` makes in the CLI and `_settle` makes in the
        eval runner. Nothing here writes to the memory's pending state itself:
        an approval recorded by hand would pass today and be the first thing to
        break when the protocol moves.

        Exactly one follow-up turn, because an approval is good for exactly one
        turn. A second question raised inside it is left parked and lapses at
        the customer's next message.
        """
        self._confirmer.answer = bool(approved)
        answered = confirmation_protocol.resolve_pending(self._memory, self._confirmer)
        if answered is None:
            return self._nothing_happened()

        self._memory.begin_turn(from_customer=False)
        note = confirmation_protocol.follow_up_note(answered)
        # A system message rather than a user one: the customer pressed a
        # button in the shop's own interface, and recording that as speech would
        # put words in the transcript they never typed.
        shop, error = self._drive(
            lambda: self._messages.append({"role": "system", "content": note})
        )
        return self._result((shop,), error)

    def close(self) -> None:
        """End this conversation. The shared resources are untouched."""
        self._stack.close()

    # --- the turn itself --------------------------------------------------

    def _drive(self, append_prompt: Any) -> tuple[ChatMessage, str | None]:
        """Append whatever starts this turn, run the loop, and read the result.

        A turn is traced as one observation. The CLI opens a conversation span
        for its whole REPL and closes it on the way out; this cannot, and the
        reason is Streamlit rather than taste: a rerun runs on a fresh thread,
        an OTEL span is put in the current context by a `contextvar`, and a
        span entered on one rerun's thread cannot be closed on another's. One
        span per turn is entered and closed on the same thread, always.
        """
        self._activity.begin_turn()
        calls_before = len(self._tracker.calls)
        cost_before = self._tracker.total_cost_usd
        history_length = len(self._messages)
        error: str | None = None
        said: list[str] = []
        trace_url: str | None = None

        append_prompt()
        try:
            with self._tracer.conversation(
                shopper_id=self._shopper_id,
                model=self._client.model,
                session_id=self._session_id,
            ):
                trace_url = self._tracer.trace_url()
                run_tool_loop(self._client, self._registry, self._messages, self._tools)
        except Exception as exc:  # noqa: BLE001 - a broken turn must not end a session
            error = f"{type(exc).__name__}: {exc}"
            # Removed whole rather than patched up: an assistant turn whose tool
            # calls never got their `tool` messages makes every later request a
            # 400. The same rewind `_run_session` does.
            del self._messages[history_length:]
        else:
            said = [
                str(message["content"])
                for message in self._messages[history_length:]
                if message.get("role") == "assistant" and message.get("content")
            ]
        finally:
            # After the answer, so a turn is visible while the next one is being
            # typed. Flushed, never shut down — see the module docstring.
            self._tracer.flush()

        shop = ChatMessage(
            role=SHOP,
            text="\n\n".join(said),
            cards=self._cards_of_this_turn(),
            activity=tuple(self._activity.calls),
            cost_usd=self._tracker.total_cost_usd - cost_before,
            model_calls=len(self._tracker.calls) - calls_before,
            # Read outside the failure branch on purpose: a tool that ran before
            # an exception still placed the order, and the customer still needs
            # its payment page.
            payment_url=self._memory.take_checkout_url() if self._memory else None,
            notice=None if error is None else f"That turn failed: {error}",
            trace_url=trace_url,
        )
        self._transcript.append(shop)
        return shop, error

    def _cards_of_this_turn(self) -> tuple[ProductCard, ...]:
        """The search results produced during this turn, in the order returned.

        The *last* search of the turn, when there was more than one, for the
        same reason `last_search` keeps only the newest: the cards under an
        answer are the list that answer is about. Older ones stay on the older
        messages they were captured onto, which is the whole point of capturing
        rather than reading back.
        """
        payloads = self._activity.results_of(SEARCH_TOOL)
        if not payloads:
            return ()
        currency = get_settings().currency
        return tuple(_card(row, currency) for row in results_in(payloads[-1]))

    # --- assembling an answer ---------------------------------------------

    def _refuse(self, text: str) -> TurnResult:
        customer = ChatMessage(role=CUSTOMER, text=text)
        shop = ChatMessage(role=SHOP, text="", notice=CAP_NOTICE)
        self._transcript.extend((customer, shop))
        return TurnResult(
            messages=(customer, shop),
            pending=None,
            session_cost_usd=self.session_cost_usd,
            cap_usd=self._cap_usd,
            cap_reached=True,
            refused=True,
        )

    def _nothing_happened(self) -> TurnResult:
        return TurnResult(
            messages=(),
            pending=self.pending,
            session_cost_usd=self.session_cost_usd,
            cap_usd=self._cap_usd,
            cap_reached=self.cap_reached,
        )

    def _result(
        self, messages: tuple[ChatMessage, ...], error: str | None = None
    ) -> TurnResult:
        return TurnResult(
            messages=messages,
            pending=self.pending,
            session_cost_usd=self.session_cost_usd,
            cap_usd=self._cap_usd,
            cap_reached=self.cap_reached,
            error=error,
        )


def _panel(body: dict, currency: str) -> CartPanel:
    """One cart body from the API, turned into what the panel draws.

    Reads the names `api/schemas.py` publishes — `unit_price_cents` and
    `line_total_cents` are already the flattened, resolved numbers this side of
    the boundary is meant to read, so nothing is renamed a third time here.
    The only work done is `money.format_amount`, and it is done once per figure
    so that no template ever holds an integer number of cents.
    """
    items = [item for item in body.get("items", []) if isinstance(item, dict)]
    lines = tuple(
        CartLine(
            variant_id=int(item["variant_id"]),
            product_name=str(item.get("product_name", "")),
            variant_label=str(item.get("variant_label", "")),
            quantity=int(item.get("quantity", 0)),
            unit_price=format_amount(item.get("unit_price_cents"), currency),
            line_total=format_amount(item.get("line_total_cents"), currency),
        )
        for item in items
        if "variant_id" in item
    )
    return CartPanel(
        lines=lines,
        # The server's total, never a sum computed here. D6 recomputes it from
        # the database on every read precisely so that nothing downstream has
        # to, and a panel that added the lines up itself would be a second
        # opinion about what the basket costs.
        total=format_amount(body.get("total_cents", 0), body.get("currency", currency)),
        unit_count=sum(line.quantity for line in lines),
    )


def _card(row: dict, currency: str) -> ProductCard:
    variants = tuple(
        VariantCard(
            variant_id=int(variant["variant_id"]),
            sku=str(variant.get("sku", "")),
            size=variant.get("size"),
            color=variant.get("color"),
            price_cents=int(variant.get("price_cents", 0)),
            price=format_amount(variant.get("price_cents"), currency),
            available=int(variant.get("available", 0)),
        )
        for variant in row.get("variants", [])
        if isinstance(variant, dict) and "variant_id" in variant
    )
    return ProductCard(
        product_id=int(row.get("product_id", 0)),
        name=str(row.get("name", "")),
        brand=row.get("brand"),
        category=row.get("category"),
        description=row.get("description"),
        variants=variants,
    )
