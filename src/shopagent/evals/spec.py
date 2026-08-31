"""What a scenario is, and what it is allowed to claim (D10, step 3).

`evals/scenarios.yaml` is the file a person reads to find out what this suite
asserts, so the loader's job is to make sure the file can be trusted at face
value: a key nobody implements, an argument of the wrong type, a turn with no
verb — each is refused here by name, at load time, before a single token is
spent. An eval suite that silently ignores an expectation it does not recognise
is worse than no suite, because it reports a pass for a claim nothing checked.

**The vocabulary is closed and small on purpose.** Every key below is used by a
scenario that exists; nothing is here "for later". A general expression
language would let a scenario say anything and would move the interesting part
of the suite out of the YAML and into whoever writes the expressions — which is
exactly the readability this file is meant to protect.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from shopagent.config import REPO_ROOT

SCENARIOS_PATH = REPO_ROOT / "evals" / "scenarios.yaml"

# The two things a turn can be. `say` is a customer message; `do` is an action
# the shop performs that no customer message could — today only the simulated
# payment scenario 10 needs, which is a signed webhook rather than a card.
TURN_VERBS = frozenset({"say", "do"})

# The actions `do:` accepts. A closed set, because an action runs code with
# side effects and "whatever string is in the YAML" is not a thing to dispatch
# on.
ACTIONS = frozenset({"simulate_payment"})

# What a scenario may answer when the shop asks it to confirm a purchase.
# `never` means no confirmer at all, which the gate treats as a refusal — the
# D9 rule that a gate which cannot reach a person is not a gate.
CONFIRMATIONS = frozenset({"yes", "no", "never"})

# Extra requirements beyond the API, Postgres and OpenAI that every scenario
# needs. Named so a scenario can be skipped with a reason rather than failing.
REQUIREMENTS = frozenset({"stripe_webhook"})

# Every expectation key, with the type of its argument. The runner in
# `expectations.py` has one function per key and a test asserts the two sets
# are equal, so a key added here without an implementation fails offline rather
# than passing vacuously in a paid run.
EXPECTATIONS: dict[str, type | tuple[type, ...]] = {
    # What the model did.
    "tools_called": list,
    "tools_not_called": list,
    "tools_in_order": list,
    # What the tools answered.
    "search_returned_results": bool,
    "search_results_cost_at_most_cents": int,
    "added_variant_is_search_row": int,
    "cart_total_cents": int,
    # What the shop's state became.
    "order_status": str,
    # What the guardrails did.
    "confirmation_requested": bool,
    "confirmation_summary_matches": str,
    "every_amount_traceable": bool,
    # What the customer read. Only where the text *is* the thing measured —
    # see the module docstring of `runner.py`.
    "answer_matches": str,
}


# The expectations whose argument is a regular expression, compiled at load
# time so a bad pattern is refused before any tokens are spent.
REGEX_EXPECTATIONS = frozenset({"confirmation_summary_matches", "answer_matches"})


class ScenarioError(ValueError):
    """A scenario file that cannot be trusted. Raised with the name in it."""


@dataclass(frozen=True)
class Turn:
    """One step of a conversation: something said, or something done."""

    verb: str
    value: str

    @property
    def is_action(self) -> bool:
        return self.verb == "do"


@dataclass(frozen=True)
class Expectation:
    """One claim, as written in the YAML."""

    key: str
    argument: Any

    def __str__(self) -> str:
        return f"{self.key}: {self.argument}"


@dataclass(frozen=True)
class Scenario:
    """One scenario, validated.

    `asks` is prose and is not optional. A scenario whose name is the only
    explanation of it is one nobody can tell is still measuring what it was
    written for — the same reason every rule in CLAUDE.md carries its argument.
    """

    name: str
    asks: str
    turns: tuple[Turn, ...]
    expectations: tuple[Expectation, ...]
    confirms: str = "no"
    needs: tuple[str, ...] = field(default_factory=tuple)


def load_scenarios(path: Path | None = None) -> list[Scenario]:
    """Every scenario in the file, or a `ScenarioError` naming the bad one."""
    path = path or SCENARIOS_PATH
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise ScenarioError(f"no scenario file at {path}") from exc
    except yaml.YAMLError as exc:
        raise ScenarioError(f"{path} is not valid YAML: {exc}") from exc

    if not isinstance(raw, list) or not raw:
        raise ScenarioError(f"{path} must be a non-empty list of scenarios")

    scenarios = [_scenario(entry, index) for index, entry in enumerate(raw, start=1)]
    names = [scenario.name for scenario in scenarios]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ScenarioError(f"duplicate scenario name(s): {', '.join(sorted(duplicates))}")
    return scenarios


def _scenario(entry: Any, index: int) -> Scenario:
    where = f"scenario #{index}"
    if not isinstance(entry, dict):
        raise ScenarioError(f"{where} is not a mapping")

    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ScenarioError(f"{where} has no name")
    where = f"scenario {name!r}"

    asks = entry.get("asks")
    if not isinstance(asks, str) or not asks.strip():
        raise ScenarioError(f"{where} has no `asks:` — say what it is for")

    unknown = set(entry) - {"name", "asks", "turns", "expect", "confirms", "needs"}
    if unknown:
        raise ScenarioError(f"{where} has unknown key(s): {', '.join(sorted(unknown))}")

    confirms = entry.get("confirms", "no")
    if confirms not in CONFIRMATIONS:
        raise ScenarioError(
            f"{where} has confirms: {confirms!r}; expected one of "
            f"{', '.join(sorted(CONFIRMATIONS))}"
        )

    needs = entry.get("needs", []) or []
    if not isinstance(needs, list) or any(item not in REQUIREMENTS for item in needs):
        raise ScenarioError(
            f"{where} has needs: {needs!r}; expected a list from "
            f"{', '.join(sorted(REQUIREMENTS))}"
        )

    return Scenario(
        name=name,
        asks=asks.strip(),
        turns=tuple(_turn(item, where, position) for position, item in enumerate(_list(entry.get("turns"), where, "turns"), start=1)),
        expectations=tuple(
            _expectation(item, where) for item in _list(entry.get("expect"), where, "expect")
        ),
        confirms=confirms,
        needs=tuple(needs),
    )


def _list(value: Any, where: str, key: str) -> list:
    if not isinstance(value, list) or not value:
        raise ScenarioError(f"{where} has no `{key}:` — it must be a non-empty list")
    return value


def _turn(item: Any, where: str, position: int) -> Turn:
    if not isinstance(item, dict) or len(item) != 1:
        raise ScenarioError(
            f"{where} turn {position} must be one of {', '.join(sorted(TURN_VERBS))}"
        )
    (verb, value), = item.items()
    if verb not in TURN_VERBS:
        raise ScenarioError(
            f"{where} turn {position} has verb {verb!r}; expected "
            f"{' or '.join(sorted(TURN_VERBS))}"
        )
    if not isinstance(value, str) or not value.strip():
        raise ScenarioError(f"{where} turn {position} ({verb}) is empty")
    if verb == "do" and value not in ACTIONS:
        raise ScenarioError(
            f"{where} turn {position} does {value!r}; expected one of "
            f"{', '.join(sorted(ACTIONS))}"
        )
    return Turn(verb=verb, value=value.strip())


def _expectation(item: Any, where: str) -> Expectation:
    if not isinstance(item, dict) or len(item) != 1:
        raise ScenarioError(f"{where} has an expectation that is not a single key: {item!r}")
    (key, argument), = item.items()
    if key not in EXPECTATIONS:
        raise ScenarioError(
            f"{where} expects {key!r}, which nothing implements. Known: "
            f"{', '.join(sorted(EXPECTATIONS))}"
        )
    wanted = EXPECTATIONS[key]
    # `bool` is a subclass of `int` in Python, so an expectation wanting an int
    # would silently accept `true` — the same trap `agent/memory.py` documents
    # for `in_stock: true` being recorded as the amount 1.
    if wanted is int and isinstance(argument, bool):
        raise ScenarioError(f"{where} expects {key}: {argument!r}; expected a number")
    if not isinstance(argument, wanted):
        raise ScenarioError(
            f"{where} expects {key}: {argument!r}; expected {wanted.__name__}"
        )
    if key in REGEX_EXPECTATIONS:
        # Compiled here so a typo in a pattern is a refusal before the run,
        # rather than a FAIL after a scenario has been paid for. It is the same
        # argument the whole loader makes: an eval file has to be trustworthy
        # at face value, and "this expectation never actually ran" is the one
        # outcome that looks like a result.
        try:
            re.compile(argument)
        except re.error as exc:
            raise ScenarioError(
                f"{where} expects {key}: {argument!r}, which is not a valid "
                f"regular expression ({exc})"
            ) from exc
    return Expectation(key=key, argument=argument)
