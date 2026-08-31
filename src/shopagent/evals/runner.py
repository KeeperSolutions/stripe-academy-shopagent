"""Drive every scenario, report pass/fail, and leave nothing behind (D10, step 3).

**The runner enters the loop by the door the CLI uses.** `build_tool_setup` is
called here exactly as `main()` calls it, and `run_tool_loop` is handed the
registry it returns — so the gate, the memory, the guardrails, the traced
wrappers and the MCP catalog are the ones a customer gets. A runner that built
its own registry would be measuring a shop nobody uses, and every result it
produced would be evidence about that shop instead.

That is asserted rather than promised: `tests/test_evals.py` walks this
module's AST and fails if it constructs a registry of its own, which is the
same mechanism `tests/test_lifecycle.py` uses to keep `transition()` behind one
service function.

**A confirmation is answered the way the CLI answers it** — `resolve_pending`,
then one more turn carrying `follow_up_note` — because that is the protocol D10
step 1 built and the browser will use on D11. `ScriptedConfirmer` answers in
place of a person and keeps every summary it was shown, which is what lets
scenario 5 assert something about *what* was put in front of somebody rather
than only that something was.

**Cleanup is by id and through the front door.** Every scenario cancels its own
order and deletes its own rows, matched by the ids this conversation created —
never a truncate, which would take the rows of anything else running. A *paid*
order cannot be cancelled: `paid -> cancelled` is not in the transition table
and never will be. Scenario 10 therefore pays with a signed
`checkout.session.completed` and is undone with a signed `charge.refunded`,
which moves it `paid -> refunded` through `apply_transition` and releases the
reservation by the mechanism D8 built for it. Nothing here decrements
`inventory.reserved` by hand.

Anything that could not be undone is *reported*, never swallowed: the run ends
by naming it, because a cleanup nobody checks is a cleanup that stops working
silently. The D10 step 1 collection guard is the second half of that — it stops
the next `pytest` run cold if a row survives.
"""

from __future__ import annotations

import faulthandler
import hashlib
import hmac
import json
import sys
import time
import uuid
from contextlib import ExitStack
from dataclasses import dataclass, field

import httpx
from sqlalchemy import text

from shopagent.agent import confirmation as confirmation_protocol
from shopagent.agent.confirmation import ScriptedConfirmer
from shopagent.agent.prompt import initial_messages
from shopagent.config import get_settings
from shopagent.db import get_engine
from shopagent.evals import expectations as checks
from shopagent.evals.spec import Scenario, load_scenarios
from shopagent.llm.client import LLMClient
from shopagent.llm.loop import build_tool_setup, run_tool_loop
from shopagent.llm.usage import UsageTracker
from shopagent.obs.tracing import Tracer, build_tracer
from shopagent.payments.stripe_svc import STRIPE_API_VERSION

# How long a scenario's HTTP calls may take. Generous: these run against a
# local API and a slow answer is not what any scenario is about.
TIMEOUT_SECONDS = 20.0

# Extra seconds on top of the longest a configured model call can legitimately
# take, before a stalled run is assumed to be stuck rather than slow.
STUCK_MARGIN_SECONDS = 30.0


def stuck_after_seconds() -> float:
    """When silence stops being slowness and becomes a hang.

    Derived from the model timeouts rather than written down, so the two cannot
    drift: `(connect + read) x (1 + retries)` is the worst a single
    `chat_with_tools` can take now that `llm/client.py` bounds it, and anything
    past that plus a margin is something no configured timeout will end.

    This exists because a D10 eval pass hung for ten minutes and had to be
    diagnosed from `lsof` and a Langfuse trace, after the first hypothesis
    turned out to be wrong. A stack trace would have said it in one line.
    """
    settings = get_settings()
    worst_call = (
        settings.openai_connect_timeout_seconds + settings.openai_read_timeout_seconds
    ) * (1 + settings.openai_max_retries)
    return worst_call + STUCK_MARGIN_SECONDS


@dataclass
class Result:
    """One scenario, run."""

    scenario: Scenario
    verdicts: list[checks.Verdict] = field(default_factory=list)
    cost_usd: float = 0.0
    calls: int = 0
    trace_id: str | None = None
    skipped: str | None = None
    error: str | None = None
    leftovers: list[str] = field(default_factory=list)
    # Reported rather than asserted: whether the amount guardrail actually
    # fired. See `_every_amount_traceable` for why this is an observation and
    # not an expectation.
    guardrail_fired: bool = False

    @property
    def passed(self) -> bool:
        return (
            self.skipped is None
            and self.error is None
            and not self.leftovers
            and all(verdict.passed for verdict in self.verdicts)
        )

    @property
    def status(self) -> str:
        if self.skipped:
            return "SKIP"
        if self.error:
            return "ERROR"
        return "PASS" if self.passed else "FAIL"


def run_all(names: list[str] | None = None) -> list[Result]:
    """Every scenario, or the named ones. One command's worth of work.

    Arms `faulthandler` first. A run that stops making progress prints every
    thread's stack to stderr instead of being a process somebody has to guess
    about — `repeat=True`, so a long hang says so more than once and the
    dumps bracket where it stopped.
    """
    faulthandler.dump_traceback_later(
        stuck_after_seconds(), repeat=True, exit=False, file=sys.stderr
    )
    scenarios = load_scenarios()
    if names:
        known = {scenario.name for scenario in scenarios}
        unknown = [name for name in names if name not in known]
        if unknown:
            raise SystemExit(
                f"no such scenario: {', '.join(unknown)}\n"
                f"known: {', '.join(sorted(known))}"
            )
        scenarios = [scenario for scenario in scenarios if scenario.name in names]
    # **One tracer for the whole run, shut down once.** Langfuse keeps one
    # resource manager per public key, *process-wide*, so a tracer is not a
    # per-unit-of-work object however much it looks like one.
    #
    # This file used to build and shut one down per scenario. What that does is
    # measured rather than reasoned about: the first `shutdown()` stops the
    # score-ingestion consumer threads; the second one enqueues a stop sentinel
    # per consumer onto a queue nothing is left to drain, leaving
    # `unfinished_tasks == 1`; and the third scenario's `flush()` — which is
    # `queue.join()` — waits for a `task_done()` that will never come. Two eval
    # passes hung there, in scenario three both times, and the second one's
    # `faulthandler` dump named the line.
    #
    # `flush()` per scenario stays: that is what makes a trace visible while the
    # run is still going, and it is safe as long as the consumers are alive.
    # `tests/test_evals.py::test_the_run_builds_one_tracer_and_hands_it_to_every_scenario`
    # is what fails if anybody reaches for `shutdown()` here again, and
    # `tests/test_tracing.py::test_a_second_shutdown_strands_the_queue_a_later_flush_waits_on`
    # is the Langfuse measurement that guard rests on.
    tracer = build_tracer()
    try:
        return [run_one(scenario, tracer) for scenario in scenarios]
    finally:
        tracer.shutdown()
        # Cancelled on the way out so a report written after a slow run is not
        # followed by a stack dump of a process that is merely finishing.
        faulthandler.cancel_dump_traceback_later()


def run_one(scenario: Scenario, tracer: Tracer) -> Result:
    """Drive one scenario end to end, then undo it.

    The tracer is passed in rather than built here, and required rather than
    defaulted: see `run_all` for what building one per scenario costs. A caller
    with nothing to trace passes an inert `Tracer()`.
    """
    result = Result(scenario=scenario)

    missing = _unmet(scenario)
    if missing:
        result.skipped = missing
        return result

    tracker = UsageTracker()
    observed = checks.Observed()
    setup = None

    try:
        with ExitStack() as stack:
            # The CLI's own call, with the CLI's own confirmer replaced by one
            # that answers from the scenario. Everything else — the catalog
            # switch, the commerce tools, the gate, the memory — is what
            # `main()` builds.
            confirmer = None if scenario.confirms == "never" else ScriptedConfirmer(
                answer=scenario.confirms == "yes"
            )
            setup = build_tool_setup(stack, confirm=confirmer)
            observed.memory = setup.memory
            if not setup.catalog_available:
                result.skipped = f"the catalog did not start: {setup.note}"
                return result

            with tracer.conversation(shopper_id=scenario.name, model="eval") as span:
                result.trace_id = _trace_id(tracer)
                _drive(scenario, setup, confirmer, tracker, tracer, observed, result)
                span.update(metadata={"scenario": scenario.name, "asks": scenario.asks})
    except Exception as exc:  # noqa: BLE001 - one broken scenario must not end the run
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        # Flushed, never shut down. See `run_all`.
        tracer.flush()
        result.cost_usd = tracker.total_cost_usd
        result.calls = len(tracker.calls)
        if setup is not None:
            observed.order_status = _order_status(setup.memory.order_id)
            result.leftovers = clean_up(setup.memory)

    if result.error is None and result.skipped is None:
        result.verdicts = [
            checks.check(expectation, observed) for expectation in scenario.expectations
        ]
    return result


def _drive(scenario, setup, confirmer, tracker, tracer, observed, result) -> None:
    """The conversation itself, one turn at a time."""
    from shopagent.agent.guardrails import FALLBACK_PREFIX, GuardedClient
    from shopagent.obs.instrumentation import TracedClient, TracedRegistry

    # Wrapped in the order `_run_session` wraps them, for the reason it does:
    # `TracedClient` inside `GuardedClient`, so a corrected retry shows as the
    # second billed call it is.
    registry = _Recording(TracedRegistry(setup.registry, tracer), observed)
    client = GuardedClient(TracedClient(LLMClient(tracker=tracker), tracer), setup.memory, tracer)
    messages = initial_messages(setup.catalog_available)

    for turn in scenario.turns:
        if turn.is_action:
            ACTIONS[turn.value](setup)
            continue

        setup.memory.begin_turn(from_customer=True)
        messages.append({"role": "user", "content": turn.value})
        run_tool_loop(client, registry, messages, registry.openai_schemas())
        _settle(scenario, setup, confirmer, client, registry, messages, observed)
        observed.answers.append(_last_said(messages))

    if confirmer is not None:
        observed.confirmations = list(confirmer.asked)
    result.guardrail_fired = any(
        answer.startswith(FALLBACK_PREFIX) for answer in observed.answers
    )


def _settle(scenario, setup, confirmer, client, registry, messages, observed) -> None:
    """Answer a parked confirmation exactly as `_run_session` does."""
    answered = confirmation_protocol.resolve_pending(setup.memory, confirmer)
    if answered is None:
        return
    setup.memory.begin_turn(from_customer=False)
    messages.append(
        {"role": "system", "content": confirmation_protocol.follow_up_note(answered)}
    )
    run_tool_loop(client, registry, messages, registry.openai_schemas())


def _last_said(messages) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message["content"])
    return ""


class _Recording:
    """Everything the loop asks of a registry, plus a transcript.

    A forwarding wrapper for the reason `tests/test_agent_chain.py` uses one:
    the loop takes the registry as a parameter, which is the seam D2 built and
    D5 used, so recording needs no change to anything the customer runs. It
    wraps the *real* setup's registry rather than replacing it — see this
    module's docstring.
    """

    def __init__(self, registry, observed: checks.Observed) -> None:
        self._registry = registry
        self._observed = observed
        self._turn = 0

    def __getattr__(self, name):
        return getattr(self._registry, name)

    def dispatch(self, name, raw_args=None):
        result = self._registry.dispatch(name, raw_args)
        self._observed.dispatches.append(
            checks.Dispatch(
                turn=self._turn,
                name=name,
                arguments=raw_args,
                ok=result.ok,
                content=result.content,
            )
        )
        return result


# --- actions a customer cannot perform ------------------------------------


def simulate_payment(setup) -> None:
    """Mark this conversation's order paid the way Stripe would.

    A real card is a browser and a person, which is a demo rather than an eval.
    What can be automated is the half that matters here: a signed
    `checkout.session.completed` through `POST /webhooks/stripe`, verified by
    the same HMAC the real one is and dispatched by the same handler. The order
    reaches `paid` through the only path that may move it there — D8's rule
    that an order's status changes only through a webhook is not bypassed, it
    is used.

    The session id is read from the order rather than invented, because
    `handle_checkout_completed` refuses an event whose session is not the one
    the order points at — a guard that exists so an order on its second
    checkout is not moved by the first session's event.
    """
    order_id = setup.memory.order_id
    if not order_id:
        raise RuntimeError("simulate_payment: no order was placed in this conversation")

    session_id, _ = _order_stripe_ids(order_id)
    if not session_id:
        raise RuntimeError(f"simulate_payment: order {order_id} has no Checkout Session")

    payment_intent = f"pi_eval_{uuid.uuid4().hex[:24]}"
    _post_event(
        "checkout.session.completed",
        {
            "id": session_id,
            "object": "checkout.session",
            "status": "complete",
            "payment_status": "paid",
            "payment_intent": payment_intent,
            "amount_total": _order_total(order_id),
            "metadata": {"order_id": str(order_id)},
        },
    )


ACTIONS = {"simulate_payment": simulate_payment}


def _post_event(event_type: str, data: dict) -> httpx.Response:
    """Sign an event the way Stripe documents and post it to the local API.

    The signing secret is the configured one, so the endpoint verifies this
    exactly as it verifies a real delivery. `livemode` is false and is checked
    by the handler — a live event is recorded and never dispatched.
    """
    settings = get_settings()
    body = json.dumps(
        {
            "id": f"evt_eval_{uuid.uuid4().hex[:24]}",
            "object": "event",
            # The version this project pins, so a simulated event is shaped
            # like the ones the handler already sees.
            "api_version": STRIPE_API_VERSION,
            "livemode": False,
            "type": event_type,
            "data": {"object": data},
        }
    ).encode()

    timestamp = int(time.time())
    digest = hmac.new(
        settings.stripe_webhook_secret.encode(),
        f"{timestamp}.".encode() + body,
        hashlib.sha256,
    ).hexdigest()

    response = httpx.post(
        f"{settings.commerce_api_base_url}/webhooks/stripe",
        content=body,
        headers={
            "Stripe-Signature": f"t={timestamp},v1={digest}",
            "Content-Type": "application/json",
        },
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response


# --- undoing the run ------------------------------------------------------


def clean_up(memory) -> list[str]:
    """Undo exactly this scenario, by id. Returns what could not be undone.

    Ordered so the reservation is released before the rows that record it are
    deleted. A cancel or a refund that fails stops the deletion of that order:
    a row removed without its release leaves units unsellable with nothing in
    the database left to explain why.
    """
    leftovers: list[str] = []
    order_id, cart_id = memory.order_id, memory.cart_id

    if order_id and cart_id is None:
        # `create_checkout` releases the cart id once the order exists. The
        # order remembers which cart it came from, so ask the database.
        with get_engine().begin() as connection:
            row = connection.execute(
                text("SELECT cart_id FROM orders WHERE id = :id"), {"id": order_id}
            ).first()
        if row is not None:
            cart_id = str(row[0])

    if order_id and not _released(order_id, leftovers):
        return leftovers + [f"cart {cart_id} (kept; its order still holds stock)"]

    try:
        with get_engine().begin() as connection:
            if order_id:
                connection.execute(
                    text("DELETE FROM order_items WHERE order_id = :id"), {"id": order_id}
                )
                connection.execute(text("DELETE FROM orders WHERE id = :id"), {"id": order_id})
                connection.execute(
                    text("DELETE FROM processed_events WHERE event_id LIKE 'evt_eval_%'")
                )
            if cart_id:
                connection.execute(
                    text("DELETE FROM cart_items WHERE cart_id = :id"), {"id": cart_id}
                )
                connection.execute(text("DELETE FROM carts WHERE id = :id"), {"id": cart_id})
    except Exception as exc:  # noqa: BLE001 - the message is the point
        leftovers.append(
            f"order {order_id} / cart {cart_id} could not be deleted "
            f"({type(exc).__name__}: {exc}). Run: python scripts/manual_test_state.py restore"
        )
    return leftovers


def _released(order_id: str, leftovers: list[str]) -> bool:
    """Get this order's reservation back, whatever state it is in.

    `pending` cancels. `paid` cannot — `paid -> cancelled` is not in the
    transition table and never will be, because once a charge settles the only
    way back is a refund. So a paid order is refunded, and by the same door the
    payment came through: a signed `charge.refunded` for the full amount, which
    `apply_transition` turns into `paid -> refunded` and a release of exactly
    the units the order held. Nothing here touches `inventory.reserved`.
    """
    status = _order_status(order_id)
    if status in ("cancelled", "refunded"):
        return True

    if status == "paid":
        return _refunded(order_id, leftovers)

    headers = {"X-API-Key": get_settings().shopagent_api_key}
    try:
        response = httpx.post(
            f"{get_settings().commerce_api_base_url}/orders/{order_id}/cancel",
            headers=headers,
            timeout=TIMEOUT_SECONDS,
        )
        if response.is_success:
            return True
        leftovers.append(
            f"order {order_id} is {status} and could not be cancelled "
            f"({response.status_code}). Its stock is still reserved. "
            "Run: python scripts/manual_test_state.py restore"
        )
    except httpx.HTTPError as exc:
        leftovers.append(
            f"order {order_id} could not be cancelled ({type(exc).__name__}). "
            "Its stock is still reserved. Run: python scripts/manual_test_state.py restore"
        )
    return False


def _refunded(order_id: str, leftovers: list[str]) -> bool:
    """Undo a simulated payment with a simulated full refund.

    Full, and it has to be: `charge.refunded` fires for a partial refund too,
    and a partial one deliberately changes nothing — there is no status between
    `paid` and `refunded`, so acting on one would free the whole reservation
    for a fraction of the money. `amount_refunded == amount` and
    `refunded: true` are what make this the full case.

    `payment_intent` is the order's own, because a refund is attributed by
    PaymentIntent and not by `metadata.order_id`: an order id says which order
    a charge is *about*, not that it is the payment that order recorded.
    """
    _, payment_intent = _order_stripe_ids(order_id)
    if not payment_intent:
        leftovers.append(
            f"order {order_id} is paid with no PaymentIntent recorded, so its "
            "reservation cannot be released here. "
            "Run: python scripts/manual_test_state.py restore"
        )
        return False

    total = _order_total(order_id)
    try:
        _post_event(
            "charge.refunded",
            {
                "id": f"ch_eval_{uuid.uuid4().hex[:24]}",
                "object": "charge",
                "payment_intent": payment_intent,
                "amount": total,
                "amount_refunded": total,
                "refunded": True,
                "metadata": {"order_id": str(order_id)},
            },
        )
    except httpx.HTTPError as exc:
        leftovers.append(
            f"order {order_id} is paid and the simulated refund failed "
            f"({type(exc).__name__}). Its stock is still reserved. "
            "Run: python scripts/manual_test_state.py restore"
        )
        return False

    if _order_status(order_id) == "refunded":
        return True
    leftovers.append(
        f"order {order_id} did not become refunded after a full simulated "
        "refund. Its stock is still reserved. "
        "Run: python scripts/manual_test_state.py restore"
    )
    return False


# --- reading the shop's state ---------------------------------------------


def _order_status(order_id: str | None) -> str | None:
    if not order_id:
        return None
    with get_engine().connect() as connection:
        row = connection.execute(
            text("SELECT status FROM orders WHERE id = :id"), {"id": order_id}
        ).first()
    return str(row[0]) if row else None


def _order_stripe_ids(order_id: str) -> tuple[str | None, str | None]:
    with get_engine().connect() as connection:
        row = connection.execute(
            text(
                "SELECT stripe_checkout_session_id, stripe_payment_intent_id "
                "FROM orders WHERE id = :id"
            ),
            {"id": order_id},
        ).first()
    return (row[0], row[1]) if row else (None, None)


def _order_total(order_id: str) -> int:
    with get_engine().connect() as connection:
        row = connection.execute(
            text("SELECT total_amount_cents FROM orders WHERE id = :id"), {"id": order_id}
        ).first()
    return int(row[0]) if row else 0


def _unmet(scenario: Scenario) -> str | None:
    """Why this scenario cannot run, in words, or `None`."""
    settings = get_settings()
    if "stripe_webhook" in scenario.needs and not settings.stripe_webhook_secret:
        return "STRIPE_WEBHOOK_SECRET is not configured, so a payment cannot be simulated"
    try:
        httpx.get(f"{settings.commerce_api_base_url}/health", timeout=5.0).raise_for_status()
    except httpx.HTTPError as exc:
        return (
            f"the commerce API is not answering ({type(exc).__name__}). "
            "Run: uvicorn shopagent.api.main:app --reload"
        )
    return None


def _trace_id(tracer) -> str | None:
    if not tracer.enabled:
        return None
    try:
        return tracer._client.get_current_trace_id()
    except Exception:  # noqa: BLE001 - a missing trace id is not a failed scenario
        return None


# --- the report -----------------------------------------------------------


def render(results: list[Result]) -> str:
    """The pass/fail table, and every failed claim underneath it."""
    lines = ["", f"{'scenario':<44} {'result':<7} {'calls':>5} {'cost':>10}  trace", "-" * 100]
    for result in results:
        lines.append(
            f"{result.scenario.name:<44} {result.status:<7} {result.calls:>5} "
            f"${result.cost_usd:>9.6f}  {result.trace_id or '-'}"
        )
    lines.append("-" * 100)
    total = sum(result.cost_usd for result in results)
    passed = sum(1 for result in results if result.status == "PASS")
    lines.append(
        f"{passed}/{len(results)} passed · "
        f"{sum(r.calls for r in results)} calls · ${total:.6f}"
    )

    for result in results:
        if result.status == "PASS":
            continue
        lines += ["", f"### {result.scenario.name} — {result.status}", f"  asks: {result.scenario.asks}"]
        if result.skipped:
            lines.append(f"  skipped: {result.skipped}")
        if result.error:
            lines.append(f"  error: {result.error}")
        for verdict in result.verdicts:
            lines.append(f"  {verdict}")
        for leftover in result.leftovers:
            lines.append(f"  LEFT BEHIND: {leftover}")

    fired = [r.scenario.name for r in results if r.guardrail_fired]
    lines += ["", f"amount guardrail fired in: {', '.join(fired) if fired else 'no scenario'}"]
    return "\n".join(lines)
