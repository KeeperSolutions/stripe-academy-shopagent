"""The two-phase confirmation protocol itself (D10, step 1).

`tests/test_guardrails.py` asserts what the gate *decides*; this file asserts
the protocol underneath it — what a caller has to do, what it may leave out,
and what happens when it does the wrong thing. The two are separate because
the protocol has three callers and only one of them is the CLI: the eval runner
answers from a rule, and D11's browser will answer in a later HTTP request.
Anything asserted here has to be true for all three, so nothing here imports
`llm/loop.py`.
"""

from __future__ import annotations

from shopagent.agent.confirmation import (
    CONFIRMATION_LIFETIME_TURNS,
    CONFIRMED_NOTE,
    DECLINED_NOTE,
    ScriptedConfirmer,
    follow_up_note,
    resolve_pending,
)
from shopagent.agent.memory import ConversationMemory


def parked(summary="  About to place this order:\n  Total: €189.98"):
    """A memory with one question in front of a person, mid-turn."""
    memory = ConversationMemory()
    memory.begin_turn(from_customer=True)
    memory.park_confirmation("create_checkout", summary)
    return memory


# --- what a caller has to do ---------------------------------------------


def test_resolving_shows_the_summary_and_records_the_answer():
    memory = parked()
    confirmer = ScriptedConfirmer(answer=True)

    pending = resolve_pending(memory, confirmer)

    assert confirmer.asked == ["  About to place this order:\n  Total: €189.98"]
    assert pending is not None
    assert pending.answer is True


def test_a_turn_with_nothing_parked_asks_nobody_anything():
    """Callers resolve after every turn, so the quiet case has to be free."""
    memory = ConversationMemory()
    memory.begin_turn(from_customer=True)
    confirmer = ScriptedConfirmer(answer=True)

    assert resolve_pending(memory, confirmer) is None
    assert confirmer.asked == []


def test_resolving_twice_asks_once():
    """A caller that resolves the same question twice changes nothing.

    Not an exception, deliberately: the second call is a bug in the caller and
    the honest report of it is that nothing moved, not a crash in the middle of
    somebody's conversation.
    """
    memory = parked()
    confirmer = ScriptedConfirmer(answer=False)

    resolve_pending(memory, confirmer)
    second = resolve_pending(memory, confirmer)

    assert second is None
    assert len(confirmer.asked) == 1
    assert memory.pending_confirmation.answer is False


def test_answering_twice_keeps_the_first_answer():
    """Below `resolve_pending`, which has its own check for this.

    The memory is reachable directly — the eval runner and D11's request
    handler both hold one — so "the answer is written once" has to be true of
    the store and not only of the one helper that happens to guard it. A second
    answer overwriting the first would mean an approval could be turned into a
    refusal, or a refusal into a purchase, by a caller that simply asked twice.
    """
    memory = parked()

    assert memory.answer_confirmation(True) is not None
    assert memory.answer_confirmation(False) is None
    assert memory.pending_confirmation.answer is True


def test_a_caller_with_no_confirmer_answers_nothing_rather_than_yes():
    """Unreachable through the gate, which refuses before parking anything.

    Handled anyway, because the failure mode of the other choice is a purchase:
    an unanswered question read as an approval is the one outcome this whole
    file exists to make impossible.
    """
    memory = parked()

    assert resolve_pending(memory, None) is None
    assert memory.pending_confirmation.answer is None


def test_the_note_the_model_reads_says_which_way_it_went():
    memory = parked()
    resolve_pending(memory, ScriptedConfirmer(answer=True))
    assert follow_up_note(memory.pending_confirmation) == CONFIRMED_NOTE

    memory = parked()
    resolve_pending(memory, ScriptedConfirmer(answer=False))
    assert follow_up_note(memory.pending_confirmation) == DECLINED_NOTE


def test_neither_note_carries_an_amount_or_a_summary():
    """The person saw the figures; the model does not need them and must not
    be handed a new source for one.

    The summary is built from `view_cart` and read by a human. Copying it into
    a system message would create a second place an amount can enter the
    conversation — one the amount guardrail would then accept, because it
    would have been in the context all along.
    """
    for note in (CONFIRMED_NOTE, DECLINED_NOTE):
        assert "€" not in note
        assert not any(character.isdigit() for character in note)


# --- what a caller may not get away with ---------------------------------


def test_a_scripted_confirmer_keeps_every_summary_it_was_shown():
    """The runner asserts on what a person would have seen, not only on flow."""
    confirmer = ScriptedConfirmer(answer=False)

    confirmer("first")
    confirmer("second")

    assert confirmer.asked == ["first", "second"]


def test_an_answer_is_spendable_on_exactly_one_turn():
    memory = parked()
    resolve_pending(memory, ScriptedConfirmer(answer=True))

    pending = memory.pending_confirmation
    assert pending.spendable_on_turn == memory.turn + CONFIRMATION_LIFETIME_TURNS
    assert CONFIRMATION_LIFETIME_TURNS == 1


def test_taking_an_answer_for_a_different_tool_gets_nothing():
    """The question named a tool, and an approval is an approval of that call."""
    memory = parked()
    resolve_pending(memory, ScriptedConfirmer(answer=True))
    memory.begin_turn(from_customer=False)

    assert memory.take_confirmation("remove_from_cart") is None
    assert memory.take_confirmation("create_checkout") is not None


def test_taking_an_unanswered_question_gets_nothing():
    memory = parked()

    assert memory.take_confirmation("create_checkout") is None
    assert memory.pending_confirmation is not None, "and it stays parked to be answered"


def test_a_second_question_replaces_the_first_rather_than_queueing():
    """One question in front of a person at a time.

    A queue would mean an approval given for the summary on screen being spent
    on a different one behind it.
    """
    memory = parked("first")
    memory.park_confirmation("create_checkout", "second")

    assert memory.pending_confirmation.summary == "second"


def test_a_fresh_memory_has_nothing_parked_and_no_turns():
    memory = ConversationMemory()

    assert memory.pending_confirmation is None
    assert memory.turn == 0


def test_two_conversations_do_not_share_an_approval():
    """The same reason a memory is a plain object rather than a global."""
    first, second = parked(), ConversationMemory()
    resolve_pending(first, ScriptedConfirmer(answer=True))

    assert second.pending_confirmation is None
