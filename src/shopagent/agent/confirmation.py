"""The two-phase confirmation protocol (D10, step 1).

D9 built the gate as a blocking call: `GuardedRegistry.dispatch` reached a
`confirm` callable in the middle of a tool dispatch, and in the CLI that
callable read a line from stdin. It was the right answer to the right
question — a person has to approve a purchase, and an argument the model sets
is a suggestion with a type annotation — and it is the wrong *shape* for
everything that comes after it.

Two things forced this open, and neither is a matter of taste.

**An eval runner cannot answer a prompt in the middle of a dispatch.**
Scenario 5 of the D10 suite is "checkout without confirmation → the agent asks
for confirmation", and a runner driving that scenario has nobody at a keyboard.
It could pass a callable that answers instantly, which is what makes the
blocking design *look* sufficient — but then the eval is written against a
mechanism that is about to change, and a green eval over yesterday's mechanism
proves nothing about tomorrow's.

**A browser cannot answer one either.** D11 puts this conversation behind
HTTP, where the answer arrives in a *later request* than the one that asked.
There is no callable that can block for it, so a protocol built on blocking has
to be replaced rather than adapted. It is being replaced once, now, before
anything is written on top of it.

**What the two phases are.**

1. `create_checkout` meets the gate. The gate reads the cart, builds the
   summary a person will read, *parks* it on the conversation's memory and
   returns a `ToolResult` saying a confirmation has been requested. The tool
   does not run. Nothing blocks.
2. Whatever is presenting the conversation — the CLI, the runner, a browser
   later — sees `memory.pending_confirmation`, puts it to whoever it can reach,
   and records the answer. The next `create_checkout` then passes or is
   refused.

Parking state the model never sees on the conversation's memory is not a new
idea here: `checkout_url` is already there for the same reason, and `cart_id`
before it. This is the fourth piece.

**What an expiry means, and why there is one.** An answered confirmation is
spendable on exactly one turn, and *any* customer message drops it, answered or
not. The failure that argues for it is one seen in another codebase: a
classifier that decides "did they say yes" by the shape of the word rather than
by what the word is answering will happily read a "yes" aimed at some other
question as authorisation to spend money. This gate is immune to that by
construction — it asks the question itself, explicitly, and reads the answer to
that question and no other. But a pending state that outlives its turn gives
the immunity back: an approval sitting in memory while the conversation moves on
is once again an answer separated from its question. So it does not outlive it.

**Nothing here knows about a terminal.** `resolve_pending` takes a callable
that turns a summary into a yes or a no; the CLI's happens to print and read a
line, and the runner's answers from a rule. A browser's will do neither, and it
will not need this file to change.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

# A confirmer is anything that can turn "here is what they are about to buy"
# into a yes or a no. `None` means nobody can be asked at all, which the gate
# treats as a refusal rather than as permission — see `GuardedRegistry`.
Confirmer = Callable[[str], bool]

# How many turns an answered confirmation is good for. One: the turn driven
# immediately after the person answered. See the module docstring for why this
# is a number rather than "until it is used".
CONFIRMATION_LIFETIME_TURNS = 1


@dataclass
class PendingConfirmation:
    """One question put to a person, and their answer once it exists.

    `summary` is built by the gate from a real tool result and is the only text
    a person is shown. It is deliberately not something the model wrote: a
    person approving a figure the model invented is worse than no gate at all,
    because it launders the invention through a human and leaves a record
    saying they agreed to it.
    """

    tool: str
    summary: str
    raised_on_turn: int
    answer: bool | None = None
    # Set when the answer is recorded, to the one turn on which it may be
    # spent. `None` while unanswered.
    spendable_on_turn: int | None = None

    @property
    def answered(self) -> bool:
        return self.answer is not None


@dataclass
class ScriptedConfirmer:
    """A confirmer with no person behind it, for a runner (D10, step 1).

    The eval suite needs both answers — scenario 5 is a checkout that is *not*
    confirmed — and it needs to read what was shown, because "the agent asked
    for confirmation" is a claim about the summary and not only about the flow.
    So every summary it was given is kept.

    A class rather than `lambda summary: True` because of that record. The
    lambda is still perfectly good wherever the summary does not matter.
    """

    answer: bool
    asked: list[str] = field(default_factory=list)

    def __call__(self, summary: str) -> bool:
        self.asked.append(summary)
        return self.answer


# What the model is told once a person has answered. A system note rather than
# a user message, because the customer did not type it — what they did was
# press a key in an interface the shop owns, and attributing that to them as
# speech would put words in the transcript they never said.
#
# These are close in wording to the `ToolResult` the gate returns for a
# declined purchase, and they are deliberately separate strings: one is the
# answer to a tool call the model made, the other is the shop telling the model
# something it did not ask about. Merging them would mean one sentence having
# to make sense in both positions, and it would stop being possible to change
# what a refused *call* says without changing what a refused *purchase* says.
CONFIRMED_NOTE = (
    "The shop showed the customer the order and its total and asked them to "
    "confirm it. They confirmed. Call create_checkout now to place the order — "
    "the confirmation is recorded and this call will go through."
)

DECLINED_NOTE = (
    "The shop showed the customer the order and its total and asked them to "
    "confirm it. They declined. Nothing was ordered and nothing was charged. "
    "Tell them that plainly, and do not call create_checkout again unless they "
    "ask for it in a later message."
)


def resolve_pending(memory: object, confirm: Confirmer | None) -> PendingConfirmation | None:
    """Put whatever is parked to a person, and record what they said.

    Returns the answered confirmation, or `None` when there was nothing to ask
    about — which is the ordinary case for every turn that did not reach a
    checkout, so a caller can call this after every turn without checking
    first.

    `confirm` being `None` here cannot happen through the gate, which refuses a
    purchase outright when nobody can be asked and so never parks anything. It
    is handled anyway, because a caller that wires the two halves up
    inconsistently should get nothing rather than an exception in the middle of
    a conversation.
    """
    pending = getattr(memory, "pending_confirmation", None)
    if pending is None or pending.answered or confirm is None:
        return None
    memory.answer_confirmation(bool(confirm(pending.summary)))
    return pending


def follow_up_note(pending: PendingConfirmation) -> str:
    """The system message that carries a person's answer back to the model."""
    return CONFIRMED_NOTE if pending.answer else DECLINED_NOTE
