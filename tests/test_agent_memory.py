"""What one conversation remembers outside the message list (D9, step 3).

Two things are held here and they have deliberately different lifetimes, which
is the whole design and the reason this file exists:

*The last search is replaced.* "The second one" can only ever mean the second
row of the most recent list. Keeping older lists would let a reference made
three messages ago resolve against a list the customer is no longer looking
at, and put the wrong shoe in the basket — silently, which is the worst
available outcome for a resolution nobody can see.

*Seen variant ids accumulate.* They answer a different question — "did this id
ever appear in front of the model in this conversation?" — and that question
is about the whole conversation. A variant found by a search, checked for
stock and added to a cart six messages later has not become unseen because
another search happened in between. Step 5 is what uses this; nothing refuses
anything yet.

Nothing here validates model output. The surface is built in this step and the
rule that reads it is written in step 5, so that the two are separately
reviewable.
"""

from __future__ import annotations

import json

import pytest

from shopagent.agent.memory import SEARCH_TOOL, ConversationMemory, RememberingRegistry
from shopagent.tools.registry import ToolResult, ToolSpec
from pydantic import BaseModel


def results(*names, base=1000):
    """Search results shaped as `catalog/search.py` returns them.

    `base` keeps two searches from minting the same variant ids. It matters:
    the first version of this helper derived ids from the row's position, so
    two different searches produced the same numbers and the test for
    accumulation passed against a set that had never grown.
    """
    return [
        {
            "product_id": 100 + index,
            "name": name,
            "variants": [
                {"variant_id": base + index * 10 + n, "sku": f"{name[:3].upper()}-{n}"}
                for n in range(2)
            ],
        }
        for index, name in enumerate(names)
    ]


def search_payload(*names, base=1000):
    return json.dumps({"count": len(names), "results": results(*names, base=base)})


# --- the last search is the only search ----------------------------------


def test_a_new_search_replaces_the_previous_one():
    memory = ConversationMemory()
    memory.observe(SEARCH_TOOL, {"query": "shoes"}, search_payload("Trail Runner", "Summit Peak"))
    memory.observe(SEARCH_TOOL, {"query": "jackets"}, search_payload("Storm Guard"))

    assert [row["name"] for row in memory.last_search.results] == ["Storm Guard"]
    assert memory.last_search.arguments == {"query": "jackets"}


def test_an_ordinal_beyond_the_end_of_the_list_says_how_long_it_was():
    memory = ConversationMemory()
    memory.observe(SEARCH_TOOL, {"query": "shoes"}, search_payload("Trail Runner", "Summit Peak"))

    reference = memory.nth_from_last_search(5)

    assert not reference.resolved
    assert "2" in reference.message


def test_an_ordinal_before_any_search_says_to_search_first():
    reference = ConversationMemory().nth_from_last_search(1)

    assert not reference.resolved
    assert SEARCH_TOOL in reference.message


def test_an_ordinal_counts_from_one_the_way_a_person_does():
    memory = ConversationMemory()
    memory.observe(SEARCH_TOOL, {"query": "shoes"}, search_payload("Trail Runner", "Summit Peak"))

    assert memory.nth_from_last_search(1).result["name"] == "Trail Runner"
    assert memory.nth_from_last_search(2).result["name"] == "Summit Peak"
    assert not memory.nth_from_last_search(0).resolved


# --- seen ids accumulate -------------------------------------------------


def test_seen_variant_ids_survive_a_new_search():
    """The other lifetime, and the reason the two are not one field."""
    memory = ConversationMemory()
    memory.observe(SEARCH_TOOL, {"query": "shoes"}, search_payload("Trail Runner"))
    first = set(memory.seen_variant_ids)
    memory.observe(SEARCH_TOOL, {"query": "jackets"}, search_payload("Storm Guard", base=2000))

    assert first
    assert first <= memory.seen_variant_ids
    assert len(memory.seen_variant_ids) > len(first)


def test_a_variant_id_anywhere_in_a_result_is_seen():
    """Not only searches: check_stock and a cart put ids in front of the model too."""
    memory = ConversationMemory()

    memory.observe("check_stock", {"variant_id": 77}, json.dumps({"variant_id": 77, "available": 3}))
    memory.observe("view_cart", {}, json.dumps({"items": [{"variant_id": 88, "quantity": 1}]}))

    assert {77, 88} <= memory.seen_variant_ids


def test_a_result_that_is_not_json_is_remembered_as_nothing():
    """A tool may answer prose. That is not a reason to lose the conversation."""
    memory = ConversationMemory()

    memory.observe("get_time", {}, "It is 14:02 in Tokyo.")

    assert memory.seen_variant_ids == frozenset()
    assert memory.last_search is None


# --- one memory per conversation -----------------------------------------


def test_two_memories_in_one_process_share_nothing():
    first, second = ConversationMemory(), ConversationMemory()

    first.cart_id = "cart-a"
    first.observe(SEARCH_TOOL, {}, search_payload("Trail Runner"))

    assert second.cart_id is None
    assert second.last_search is None
    assert second.seen_variant_ids == frozenset()


# --- filled by dispatching, not by the caller remembering ----------------


class NoArgs(BaseModel):
    pass


def registry_with(payload, memory, name=SEARCH_TOOL):
    registry = RememberingRegistry(memory)
    registry.register(
        ToolSpec(name=name, description="d", args_model=NoArgs, fn=lambda: json.loads(payload))
    )
    return registry


def test_dispatching_a_search_records_it():
    """Nothing in the agent loop has to remember to call the memory."""
    memory = ConversationMemory()
    registry = registry_with(search_payload("Trail Runner", "Summit Peak"), memory)

    registry.dispatch(SEARCH_TOOL, {})

    assert [row["name"] for row in memory.last_search.results] == ["Trail Runner", "Summit Peak"]


def test_a_failed_call_records_nothing():
    """A refusal is not a list the customer is looking at."""
    memory = ConversationMemory()
    registry = RememberingRegistry(memory)
    registry.register(
        ToolSpec(
            name=SEARCH_TOOL,
            description="d",
            args_model=NoArgs,
            fn=lambda: ToolResult(ok=False, content="Error: nothing", error="nothing"),
        )
    )

    registry.dispatch(SEARCH_TOOL, {})

    assert memory.last_search is None
    assert memory.seen_variant_ids == frozenset()


def test_the_registry_still_never_raises():
    memory = ConversationMemory()
    registry = RememberingRegistry(memory)
    registry.register(
        ToolSpec(name="boom", description="d", args_model=NoArgs,
                 fn=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    )

    result = registry.dispatch("boom", {})

    assert not result.ok
    assert "boom" in result.content
