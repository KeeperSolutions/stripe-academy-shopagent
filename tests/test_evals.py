"""The eval runner is code, so it has tests (D10, step 3).

Nothing here spends a token. What is checked is the machinery: that a bad
scenario file is refused by name, that every expectation evaluates the way it
claims to — falsified, one by one, by feeding it a run that should fail it —
that the runner cleans up by id, and that it enters the agent loop by the same
door the CLI does.

That last one is a structural check rather than a promise, for the reason
`tests/test_lifecycle.py` walks the AST to keep `transition()` behind one
service function: a runner that quietly built its own registry would go on
passing every behavioural test while measuring a shop nobody uses.
"""

from __future__ import annotations

import ast
import json
import re
import textwrap
from pathlib import Path

import pytest

from shopagent.agent.memory import ConversationMemory
from shopagent.evals import expectations as checks
from shopagent.evals import runner
from shopagent.obs.tracing import Tracer
from shopagent.evals.spec import (
    EXPECTATIONS,
    Expectation,
    ScenarioError,
    load_scenarios,
)

RUNNER_SOURCE = Path(runner.__file__).read_text()


def code_only(source: str) -> str:
    """The module's code with its comments and docstrings removed.

    The first version of the structural checks below searched the raw file and
    three of them failed on the module's own prose — the docstring says "never
    a truncate" and the cleanup explains that it "never touches
    `inventory.reserved`". A check that a *comment* can satisfy or break is not
    a check on the code, and it fails in both directions: it would also pass
    over `TRUNCATE` written inside a string the moment somebody rephrased the
    sentence around it.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree))


RUNNER_CODE = code_only(RUNNER_SOURCE)


def write(tmp_path, body: str) -> Path:
    path = tmp_path / "scenarios.yaml"
    path.write_text(textwrap.dedent(body))
    return path


GOOD = """
- name: a_scenario
  asks: Whether the loader works.
  turns:
    - say: hello
  expect:
    - tools_called: [view_cart]
"""


# --- the file a person reads ----------------------------------------------


def test_the_shipped_scenarios_load():
    """The file in the repository is the one this suite is about."""
    scenarios = load_scenarios()

    assert len(scenarios) == 10, "the plan asks for ten"
    assert all(scenario.asks for scenario in scenarios)
    assert all(scenario.turns for scenario in scenarios)
    assert all(scenario.expectations for scenario in scenarios)


def test_every_expectation_the_file_uses_is_implemented():
    """The two halves of the vocabulary cannot drift.

    A key declared in `spec.EXPECTATIONS` with no function behind it would
    raise at run time, in a paid run, after the tokens were spent.
    """
    assert set(EXPECTATIONS) == set(checks.CHECKS)


@pytest.mark.parametrize(
    "body, message",
    [
        ("- name: x\n  asks: y\n  turns: [{say: hi}]\n  expect: [{nonsense: 1}]", "nonsense"),
        ("- asks: y\n  turns: [{say: hi}]\n  expect: [{tools_called: []}]", "has no name"),
        ("- name: x\n  turns: [{say: hi}]\n  expect: [{tools_called: []}]", "has no `asks:`"),
        ("- name: x\n  asks: y\n  expect: [{tools_called: []}]", "has no `turns:`"),
        ("- name: x\n  asks: y\n  turns: [{say: hi}]", "has no `expect:`"),
        ("- name: x\n  asks: y\n  turns: [{shout: hi}]\n  expect: [{tools_called: []}]", "verb"),
        ("- name: x\n  asks: y\n  turns: [{do: explode}]\n  expect: [{tools_called: []}]", "explode"),
        (
            "- name: x\n  asks: y\n  confirms: maybe\n  turns: [{say: hi}]\n"
            "  expect: [{tools_called: []}]",
            "confirms",
        ),
        (
            "- name: x\n  asks: y\n  needs: [a_gpu]\n  turns: [{say: hi}]\n"
            "  expect: [{tools_called: []}]",
            "needs",
        ),
        (
            "- name: x\n  asks: y\n  turns: [{say: hi}]\n  expect: [{order_status: 5}]",
            "expected str",
        ),
        ("[]", "non-empty list"),
    ],
)
def test_a_scenario_that_cannot_be_trusted_is_refused_by_name(tmp_path, body, message):
    """Every refusal says which scenario and what about it.

    A loader that ignored an expectation it did not recognise would report a
    pass for a claim nothing checked, which is worse than having no suite.
    """
    with pytest.raises(ScenarioError, match=message):
        load_scenarios(write(tmp_path, body))


def test_a_true_argument_is_not_accepted_where_a_number_is_wanted(tmp_path):
    """`bool` is a subclass of `int`, and `cart_total_cents: true` would mean 1.

    The same trap `agent/memory.py` documents for `in_stock: true` being
    recorded as the amount 1.
    """
    with pytest.raises(ScenarioError, match="expected a number"):
        load_scenarios(
            write(
                tmp_path,
                "- name: x\n  asks: y\n  turns: [{say: hi}]\n"
                "  expect: [{cart_total_cents: true}]",
            )
        )


def test_two_scenarios_with_one_name_are_refused(tmp_path):
    """`--only` names a scenario, so a duplicate name makes it ambiguous."""
    with pytest.raises(ScenarioError, match="duplicate"):
        load_scenarios(write(tmp_path, GOOD + GOOD))


def test_a_missing_file_says_where_it_looked(tmp_path):
    with pytest.raises(ScenarioError, match="no scenario file"):
        load_scenarios(tmp_path / "nothing.yaml")


# --- the expectations, each falsified -------------------------------------


def observed(**kwargs) -> checks.Observed:
    dispatches = [
        checks.Dispatch(turn=1, name=name, arguments=args, ok=ok, content=content)
        for name, args, ok, content in kwargs.pop("calls", [])
    ]
    return checks.Observed(dispatches=dispatches, **kwargs)


def verdict(key, argument, run) -> checks.Verdict:
    return checks.check(Expectation(key=key, argument=argument), run)


SEARCH = json.dumps(
    {
        "count": 2,
        "results": [
            {"name": "Trail Runner GTX", "variants": [{"variant_id": 11, "price_cents": 9499}]},
            {"name": "Road Lite", "variants": [{"variant_id": 22, "price_cents": 7999}]},
        ],
    }
)
CART = json.dumps({"currency": "eur", "items": [{"variant_id": 11}], "total_cents": 9499})


def test_tools_called_passes_and_fails():
    run = observed(calls=[("search_products", {}, True, SEARCH)])

    assert verdict("tools_called", ["search_products"], run).passed
    assert not verdict("tools_called", ["create_checkout"], run).passed


def test_tools_not_called_passes_and_fails():
    run = observed(calls=[("add_to_cart", {}, True, CART)])

    assert verdict("tools_not_called", ["create_checkout"], run).passed
    assert not verdict("tools_not_called", ["add_to_cart"], run).passed


def test_tools_in_order_is_a_subsequence_not_an_exact_list():
    """The variance rule, asserted.

    D9 measured a model checking stock three times between a search and an add.
    Requiring an exact list would fail that run for a reason the scenario is
    not about; requiring a subsequence keeps the only claim that is invariant —
    the search happened before the add, because the add needs an id only the
    search could supply.
    """
    run = observed(
        calls=[
            ("search_products", {}, True, SEARCH),
            ("check_stock", {}, True, "{}"),
            ("check_stock", {}, True, "{}"),
            ("add_to_cart", {}, True, CART),
        ]
    )

    assert verdict("tools_in_order", ["search_products", "add_to_cart"], run).passed
    assert not verdict("tools_in_order", ["add_to_cart", "search_products"], run).passed


def test_search_results_cost_at_most_cents_reads_every_variant():
    run = observed(calls=[("search_products", {}, True, SEARCH)])

    assert verdict("search_results_cost_at_most_cents", 10000, run).passed
    assert not verdict("search_results_cost_at_most_cents", 8000, run).passed


def test_a_search_that_returned_nothing_cannot_satisfy_a_price_bound():
    """Vacuous truth is the failure mode of every "all of them" assertion.

    An empty result set satisfies "every price is under the bound" and proves
    nothing, so it is a failure rather than a pass.
    """
    empty = observed(calls=[("search_products", {}, True, json.dumps({"count": 0, "results": []}))])

    assert not verdict("search_results_cost_at_most_cents", 10000, empty).passed
    assert not verdict("search_returned_results", True, empty).passed


def test_added_variant_is_search_row_checks_the_list_the_model_was_shown():
    memory = ConversationMemory()
    memory.observe("search_products", {}, SEARCH)
    run = observed(
        calls=[("search_products", {}, True, SEARCH), ("add_to_cart", {"variant_id": 22}, True, CART)],
        memory=memory,
    )

    assert verdict("added_variant_is_search_row", 2, run).passed
    assert not verdict("added_variant_is_search_row", 1, run).passed


def test_added_variant_is_search_row_fails_when_nothing_was_added():
    memory = ConversationMemory()
    memory.observe("search_products", {}, SEARCH)

    assert not verdict(
        "added_variant_is_search_row", 2, observed(memory=memory)
    ).passed


def test_cart_total_cents_reads_the_last_cart_result():
    run = observed(
        calls=[
            ("add_to_cart", {}, True, json.dumps({"total_cents": 18998})),
            ("remove_from_cart", {}, True, CART),
        ]
    )

    assert verdict("cart_total_cents", 9499, run).passed
    assert not verdict("cart_total_cents", 18998, run).passed


def test_order_status_reads_the_database_not_the_tool():
    assert verdict("order_status", "paid", observed(order_status="paid")).passed
    assert not verdict("order_status", "paid", observed(order_status="pending")).passed
    assert verdict("order_status", "none", observed(order_status=None)).passed


def test_confirmation_expectations_read_what_a_person_was_shown():
    asked = observed(confirmations=["  About to place this order:\n  Total: €94.99"])

    assert verdict("confirmation_requested", True, asked).passed
    assert verdict("confirmation_summary_matches", r"Total: €94\.99", asked).passed
    assert not verdict("confirmation_summary_matches", r"Total: €5\.00", asked).passed
    assert not verdict("confirmation_requested", True, observed()).passed
    assert not verdict("confirmation_summary_matches", ".", observed()).passed


def test_every_amount_traceable_reads_the_same_rule_the_guardrail_does():
    memory = ConversationMemory()
    memory.observe("view_cart", {}, json.dumps({"total_cents": 9499}))

    good = observed(answers=["That comes to €94.99."], memory=memory)
    bad = observed(answers=["Three pairs come to €284.97."], memory=memory)

    assert verdict("every_amount_traceable", True, good).passed
    assert not verdict("every_amount_traceable", True, bad).passed


def test_answer_matches_reads_the_last_answer():
    run = observed(answers=["Here are three.", "Which trip did you have in mind?"])

    assert verdict("answer_matches", r"\?", run).passed
    assert not verdict("answer_matches", r"^Here", run).passed


def test_a_check_that_raises_is_a_failure_and_not_a_crash():
    """One broken expectation must not end a paid run."""
    # A pattern the loader would refuse, constructed here directly: this is
    # about the last line of defence, not about the file. The first version of
    # this test fed unparseable JSON, which `_added_variant_is_search_row`
    # handles gracefully — so nothing raised, and narrowing the `except` to
    # `ZeroDivisionError` survived the mutation untouched.
    run = observed(answers=["anything"])

    result = verdict("answer_matches", "(unclosed", run)

    assert not result.passed
    assert "raised" in result.detail


def test_a_pattern_that_is_not_a_regex_is_refused_before_the_run(tmp_path):
    """Cheaper than the alternative: a typo caught after a paid scenario looks
    exactly like a scenario that failed."""
    with pytest.raises(ScenarioError, match="not a valid regular expression"):
        load_scenarios(
            write(
                tmp_path,
                "- name: x\n  asks: y\n  turns: [{say: hi}]\n"
                "  expect: [{answer_matches: \"(unclosed\"}]",
            )
        )


def test_a_gracefully_handled_bad_payload_is_a_plain_failure():
    """The other half: not everything unusual has to raise."""
    run = observed(calls=[("add_to_cart", "{not json", True, CART)])

    result = verdict("added_variant_is_search_row", 1, run)

    assert not result.passed
    assert "not JSON" in result.detail


# --- the runner drives the shop the customer uses -------------------------


def calls_in(source: str, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def test_the_runner_builds_no_registry_of_its_own():
    """The claim, checked structurally rather than promised in a docstring.

    A runner that constructed its own `ToolRegistry` would answer every
    behavioural test correctly while measuring a shop with no gate, no memory
    and no catalog — and every number it produced would be evidence about that
    shop instead of this one.
    """
    for forbidden in ("ToolRegistry", "GuardedRegistry", "RememberingRegistry"):
        assert not calls_in(RUNNER_SOURCE, forbidden), (
            f"the eval runner constructs a {forbidden} of its own; it must take "
            f"the one `build_tool_setup` returns, the way the CLI does"
        )


def test_the_runner_calls_the_same_two_functions_the_cli_calls():
    assert calls_in(RUNNER_SOURCE, "build_tool_setup"), "the runner must use the CLI's setup"
    assert calls_in(RUNNER_SOURCE, "run_tool_loop"), "the runner must drive the CLI's loop"


def test_the_runner_answers_a_confirmation_through_the_protocol():
    """Not through `_ask_to_confirm`, and not by writing to the memory itself.

    `resolve_pending` plus a turn carrying `follow_up_note` is the whole
    protocol, and it is what D11's browser will call. A runner that set
    `memory._pending.answer` directly would pass today and be the first thing
    to break when the protocol moves.
    """
    assert "resolve_pending" in RUNNER_CODE
    assert "follow_up_note" in RUNNER_CODE

    reached = {
        node.attr
        for node in ast.walk(ast.parse(RUNNER_SOURCE))
        if isinstance(node, ast.Attribute) and node.attr.startswith("_pending")
    }
    assert not reached, f"the runner reaches into the memory's private state: {reached}"


def test_the_runner_never_truncates_a_table():
    """Cleanup is by id. A truncate would take the rows of whatever else ran."""
    lowered = RUNNER_CODE.lower()

    assert "truncate" not in lowered
    for statement in ("delete from order_items", "delete from orders", "delete from carts"):
        assert f"{statement} where" in lowered, f"{statement!r} must be constrained by id"


def test_no_delete_matches_a_prefix_rather_than_an_id():
    """A `LIKE` in a cleanup is a truncate wearing a WHERE clause.

    `DELETE FROM processed_events WHERE event_id LIKE 'evt_eval_%'` looks
    constrained and takes the idempotency claims of every eval process sharing
    this database. A concurrent run would find its own handled events
    forgotten, and Stripe redelivering one of them would be processed twice —
    the exact thing that table exists to prevent. The runner records the ids it
    signed and deletes those. Raised by review on PR #10.
    """
    lowered = RUNNER_CODE.lower()

    assert " like " not in lowered, "cleanup matches a prefix instead of an id"
    assert "evt_eval_%" not in lowered
    assert "delete from processed_events where event_id = any(" in lowered


def test_the_runner_releases_stock_through_the_lifecycle_and_not_by_hand():
    """`inventory.reserved` is moved by `apply_transition` or not at all.

    A paid order cannot be cancelled, so the temptation is to delete the row
    and decrement the column — which is the D8 mistake `manual_test_state.py`
    exists to document, one step worse because it would be automated. The
    runner refunds instead, through a signed event, and the release happens
    where every other release happens.
    """
    assert "inventory" not in RUNNER_CODE.lower(), (
        "the runner moves inventory.reserved itself; releasing is "
        "`apply_transition`'s job and nowhere else's"
    )
    assert "charge.refunded" in RUNNER_CODE


def test_a_scenario_that_cannot_run_is_skipped_with_a_reason(monkeypatch):
    """A skip is not a pass, and it says what to do about it."""
    scenario = load_scenarios()[-1]
    assert "stripe_webhook" in scenario.needs

    monkeypatch.setattr(
        "shopagent.evals.runner.get_settings",
        lambda: type("S", (), {"stripe_webhook_secret": None})(),
    )
    result = runner.run_one(scenario, Tracer())

    assert result.status == "SKIP"
    assert "STRIPE_WEBHOOK_SECRET" in result.skipped
    assert not result.passed


def test_a_run_that_left_a_row_behind_cannot_pass():
    """A cleanup nobody checks is a cleanup that stops working silently."""
    scenario = load_scenarios()[0]
    clean = runner.Result(scenario=scenario)
    dirty = runner.Result(scenario=scenario, leftovers=["order abc could not be cancelled"])

    assert clean.passed
    assert not dirty.passed
    assert dirty.status == "FAIL"


def test_the_report_names_every_failed_claim():
    """A table that only says FAIL sends a reader back to the tokens."""
    scenario = load_scenarios()[0]
    failed = runner.Result(
        scenario=scenario,
        verdicts=[
            checks.Verdict(
                expectation=Expectation("tools_called", ["search_products"]),
                passed=False,
                detail="never called ['search_products']",
            )
        ],
    )

    report = runner.render([failed])

    assert "FAIL" in report
    assert scenario.name in report
    assert "never called" in report
    assert "0/1 passed" in report


# --- the one scenario whose claim depends on the catalog ------------------


@pytest.mark.db
def test_the_semantic_scenario_has_no_lexical_overlap(engine):
    """A semantic claim that a keyword search could satisfy is not one.

    `a_semantic_query_finds_what_shares_no_words_with_it` asserts that a query
    whose words appear nowhere comes back with results, which is only evidence
    about embeddings while the premise holds. The premise is about the catalog,
    and the catalog is regenerated by a seed — so it can stop holding without
    anybody touching this file, and the scenario would quietly become a test
    that keyword search works.

    Checked two ways, because they fail differently. A substring sweep catches
    the word appearing at all, including inside another one. Postgres's own
    `to_tsvector`/`plainto_tsquery` is the stronger half: it is *the* keyword
    search this claim is contrasted with, applied with the same stemming a
    real one would use, so a zero here means a keyword search genuinely
    returns nothing for this sentence.
    """
    from sqlalchemy import text as sql

    scenario = next(
        s for s in load_scenarios() if s.name == "a_semantic_query_finds_what_shares_no_words_with_it"
    )
    # Read from the scenario rather than repeated here: a copy of the query in
    # this file is a second record of it, and the two would drift the first
    # time somebody reworded the YAML.
    query = next(turn.value for turn in scenario.turns if turn.verb == "say")

    catalog = "name || ' ' || description || ' ' || category || ' ' || brand"
    with engine.connect() as connection:
        matches = connection.execute(
            sql(
                f"SELECT count(*) FROM products WHERE to_tsvector('english', {catalog}) "
                f"@@ plainto_tsquery('english', :query)"
            ),
            {"query": query},
        ).scalar_one()

        words = [
            word
            for word in re.findall(r"[a-z']+", query.lower())
            if len(word) >= 5 and word not in {"something", "there"}
        ]
        assert words, "the query has no content words to check"
        present = [
            word
            for word in words
            if connection.execute(
                sql(f"SELECT count(*) FROM products WHERE lower({catalog}) LIKE :like"),
                {"like": f"%{word}%"},
            ).scalar_one()
        ]

    assert matches == 0, (
        f"a keyword search finds {matches} product(s) for {query!r}, so a "
        f"non-empty result proves nothing about the embedding"
    )
    assert not present, f"{present} appear(s) in the catalog text"


def test_the_runner_dumps_a_stack_before_anybody_has_to_guess():
    """Armed from the model timeouts, not from a number typed here.

    The first full eval pass hung for ten minutes and was diagnosed from
    `lsof` and a Langfuse trace — after the first hypothesis turned out to be
    wrong. The threshold has to sit above the longest a *configured* call can
    take, or a slow scenario would print stacks for no reason; deriving it from
    the same settings `llm/client.py` reads is what keeps that true when
    somebody changes the timeout.
    """
    from shopagent.config import get_settings

    settings = get_settings()
    worst_call = (
        settings.openai_connect_timeout_seconds + settings.openai_read_timeout_seconds
    ) * (1 + settings.openai_max_retries)

    assert runner.stuck_after_seconds() > worst_call, (
        "a dump before the model's own timeout would fire on a slow scenario"
    )
    assert "dump_traceback_later" in RUNNER_CODE
    assert "cancel_dump_traceback_later" in RUNNER_CODE, (
        "an armed dump left running would fire after the report"
    )


def test_the_run_builds_one_tracer_and_hands_it_to_every_scenario(monkeypatch):
    """The guard on the fix, at the seam that failed.

    `run_one` used to build its own tracer and shut it down, ten times in one
    process. Langfuse keeps one resource manager per public key process-wide,
    so the third scenario's flush waited on a queue two shutdowns had
    stranded — see
    `tests/test_tracing.py::test_a_second_shutdown_strands_the_queue_a_later_flush_waits_on`
    for the measurement.

    What is asserted here is the shape that avoids it: one tracer, built once,
    passed to every scenario, shut down once at the end. Before the fix this
    fails outright, because `run_all` called `run_one` with no tracer to give.
    """
    from shopagent.obs.tracing import Tracer

    given = []
    shut_down = []

    class Counting(Tracer):
        def shutdown(self):
            shut_down.append(1)

    monkeypatch.setattr(runner, "build_tracer", Counting)
    monkeypatch.setattr(
        runner,
        "run_one",
        lambda scenario, tracer: given.append(tracer) or runner.Result(scenario=scenario),
    )

    results = runner.run_all()

    assert len(results) == 10
    assert len(given) == 10
    assert len({id(tracer) for tracer in given}) == 1, "a tracer per scenario"
    assert shut_down == [1], f"shutdown() was called {len(shut_down)} times, not once"
