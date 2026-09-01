"""The browser UI (D11, step 2).

    streamlit run src/shopagent/ui/app.py

**This file renders and decides nothing.** Every question a turn raises — what
the tools did, what a turn cost, whether the cap was reached, whether a
confirmation is parked — is answered by `ui/session.py` and arrives here as a
`TurnResult`. If something here starts looking like a rule, it belongs one file
down; that is the same cut `api/routers/` and `api/services/` make, and it is
what lets step 1's logic be tested with no Streamlit at all.

It lives beside the module it renders rather than at the repository root. The
root would buy a shorter command and cost the thing this project keeps: every
path in CLAUDE.md's layout table is package-relative, `ui/session.py` is
already there, and a bare `app.py` at the top of a repo says nothing about
which application it is. The command is written down once, in the README.

**Chat is the only interface.** There is no grid beside the conversation and no
way to browse the catalog around the agent, which is why there is no
`GET /products` to render from. Search results appear *inside* the message that
produced them, and **a card is not clickable**: nothing enters a basket except
by asking. A card with an "Add" button would be a second way to shop, and the
whole layout is the claim that there is only one.

**The one control that is not a message is the basket's Checkout button.** It
adds no route to payment: it dispatches `create_checkout` through the same
`GuardedRegistry` the model reaches, so the gate parks the same question built
from the same `view_cart` read, and a person still approves the same summary.
What it skips is the model's decision to call the tool, which the customer has
just made by clicking. `BrowserSession.request_checkout` holds the argument in
full, including the two implementations that were refused. Nothing else in the
panel is a control — no line can be removed there, because changing a basket is
something you ask for.

**The page is centred, and that is `st.chat_input`'s doing.** Streamlit pins a
`chat_input` called from an app's main body to the bottom of the page, and
renders it inline — scrolling away with the content — as soon as it is nested
in a container or a column. So the readable column is not built out of
`st.columns`; it is Streamlit's own `layout="centered"`, which the input pins
itself to the bottom of. Fighting that with columns would have traded a pinned
input for a centred one.

**The confirmation gate is answered here and implemented nowhere here.** When
`create_checkout` meets the gate, the question is parked on the conversation's
memory; this page reads it, puts it in a modal, and hands the answer back
through `BrowserSession.answer_confirmation` — which is `resolve_pending` plus
one follow-up turn, the same two calls the CLI and the eval runner make. The
summary in that modal is printed verbatim and is never rebuilt here. Until
somebody answers, nothing is ordered and nothing is charged, which is the safe
half of the D10 protocol: an unanswered approval is never spendable.

(The paragraph this replaces said the gate was "not wired up here, that is step
3". It was written on step 2 and step 3 wired it up without coming back for
it — a docstring describing the file as it was two commits ago, which is the
kind of drift this project spends its comments arguing against.)
"""

from __future__ import annotations

import html

import streamlit as st

from shopagent.ui import cards as layout
from shopagent.ui import colors
from shopagent.ui import session as turns

TITLE = "ShopAgent"
# Discreet, and text rather than a mark: there is no logo file in this
# repository, and inventing one would be worse than a wordmark. It sits in the
# caption beside the cost, not in the heading — the shop is what the page is,
# and whose shop it is is a footnote.
OWNER = "Keeper Solutions"

# Three openings, and every one of them is a line the D10 eval suite drove
# against a live model and a real catalog — scenario 1's price filter,
# scenario 2's semantic search, and the first turn of scenario 3's ordinal
# sequence. Invented examples would be a promise this shop has not tested; an
# empty screen offering one that does not work is worse than an empty screen.
OPENERS = (
    "find me some trail running shoes",
    "show me running shoes under 100 euros",
    "I need something for when it's pouring outside",
)

# The dialog's heading and the sentence behind it both come from
# `PendingApproval` — see `ui/session.py`, where they are chosen. Which
# question is being asked is a fact about the tool, and this file decides
# nothing. Two tools reach the gate now, and "Confirm this purchase" over a
# refund would be a heading contradicting the summary underneath it.

# The panel beside the conversation. A basket, and one button.
CART_TITLE = "Your basket"
CART_EMPTY = "Nothing in it yet. Ask for something and it will appear here."
# Said under the button rather than beside every line, because it is one rule
# about the whole panel: this is a view, and the only way to change what is in
# it is to ask. There is no remove control for the same reason there is no Add
# button on a card — see `_draw_cart`.
CART_READ_ONLY = "Ask the assistant to change or remove anything."
CHECKOUT_LABEL = "Checkout"


# --- what this process holds, and what this tab holds --------------------


@st.cache_resource
def _shared():
    """The tracer, the catalog subprocess and the HTTP client, once per process.

    `st.cache_resource` is shared by every browser session in the process,
    which is exactly right for these three and exactly wrong for a
    conversation — see `ui/session.py`, which is where the two tiers are
    decided. This wraps that function and adds no caching of its own; a second
    scheme here would be a second answer to "how many tracers are there", and
    D10 measured what the wrong answer costs.
    """
    return turns.shared_resources()


def _session() -> turns.BrowserSession:
    """This tab's conversation, built once and kept across reruns.

    `st.session_state` is per browser session, so two tabs get two carts —
    which is the isolation `agent/memory.py` requires and the reason this is
    not in the cache above.
    """
    if "conversation" not in st.session_state:
        st.session_state.conversation = turns.BrowserSession(_shared())
    return st.session_state.conversation


# --- drawing -------------------------------------------------------------


def _swatch(color: str | None) -> str:
    """A small coloured disc, or nothing when the variant has no colour.

    The only value interpolated is a hex from `ui/colors.py`. No product name,
    no colour name and nothing else the catalog holds ever reaches this string
    — which is what makes `unsafe_allow_html` safe at the call site, and why
    `colors.swatch` returns a hex rather than a fragment of markup.
    """
    fill = colors.swatch(color)
    if fill is None:
        return ""
    # A pale swatch would vanish into a light page, so it is outlined rather
    # than darkened: changing the fill would answer "what colour is this shoe"
    # with a lie about the shoe.
    alpha = "0.55" if colors.needs_outline(color) else "0.25"
    return (
        f'<span style="display:inline-block;width:0.85em;height:0.85em;'
        f"border-radius:50%;background:{fill};"
        f"border:1px solid rgba(128,128,128,{alpha});"
        f'vertical-align:-0.09em;margin-right:0.45em;"></span>'
    )


def _amount(text: str) -> str:
    """An already-formatted amount, made safe to put in markdown.

    Streamlit renders markdown, and a pair of unescaped dollars around a number
    is inline LaTeX to it — the header caption lost its spend cap into a maths
    block on D11 for exactly this, and it was found by looking at the page
    rather than by a test. Every amount on this page arrives formatted by
    `money.format_amount`, and in a shop priced in dollars that string starts
    with the character that opens the block.

    This escapes and computes nothing. The figure is whatever came in.
    """
    return html.escape(text).replace("$", "\\$")


def _group_line(group: layout.VariantGroup) -> str:
    """One colour of one product at one price, with all of its sizes.

    Three sizes of one black shoe is one line rather than three — see
    `ui/cards.py` for why density is the thing being fixed and filtering
    against the model's prose is not. Everything from the catalog goes through
    `html.escape` before it meets the swatch's markup, and `group.price` is
    already `money.format_amount`'s output: no arithmetic happens in this file.
    """
    parts = [html.escape(part) for part in (group.color, group.label) if part]
    head = " · ".join(parts)
    price = _amount(group.price)

    if group.all_sold_out:
        # Nothing left in this colour. Said in words, because a row that was
        # only dimmed would read as a rendering fault.
        gone = html.escape(", ".join(group.sold_out))
        return (
            f'<span style="opacity:0.45;">{_swatch(group.color)}{head}'
            f'{" · " if head else ""}{gone} — {price}</span>'
            f'<span style="opacity:0.55;"> · sold out</span>'
        )

    line = f"{_swatch(group.color)}{head} — {price}"
    if group.sold_out:
        # Struck through and still there. Dropping it would answer "do you have
        # 42?" by omission, and "there is no 42" is a different sentence from
        # "the 42 is gone".
        missing = html.escape(", ".join(group.sold_out))
        line += (
            f'<span style="opacity:0.5;"> · <s>{missing}</s> sold out</span>'
        )
    return line


def _draw_card(card: turns.ProductCard) -> None:
    """One product, with its variants. Nothing here is clickable, on purpose."""
    with st.container(border=True):
        st.markdown(f"**{html.escape(card.name)}**", unsafe_allow_html=True)
        heading = " · ".join(part for part in (card.brand, card.category) if part)
        if heading:
            st.caption(heading)
        for group in layout.group_variants(card.variants):
            st.markdown(_group_line(group), unsafe_allow_html=True)
        if card.description:
            st.caption(card.description)


def _draw_cards(cards: tuple[turns.ProductCard, ...]) -> None:
    """The results, with everything past the first couple behind a fold.

    Nothing is hidden and the fold says how much it holds — a fold that did not
    name its own size would read as the end of the list.
    """
    shown, folded = layout.split_for_display(cards)
    for card in shown:
        _draw_card(card)
    if folded:
        with st.expander(f"{len(folded)} more result(s) the shop found"):
            for card in folded:
                _draw_card(card)


def _draw_activity(message: turns.ChatMessage) -> None:
    """What the agent actually did on this turn, under a fold.

    **Collapsed by default and attached to its own turn.** Collapsed, because
    a conversation is what this page is and a tool log under every message
    would bury it. Per turn rather than one panel at the bottom, because a
    single panel can only ever describe the last turn — and the question this
    project is built to answer, which tools ran in what order and what they
    cost, is asked about a turn that has already scrolled past.

    **A refused call is the most valuable row here.** The gate parking a
    confirmation and the unknown-variant guardrail both come back as failed
    `ToolResult`s rather than exceptions, so `RecordingRegistry` — which sits
    outermost for exactly this reason — sees them, and they are drawn in the
    error colour with the reason the tool gave.

    Monospace, and this is the one place in the page where that is right:
    everything here is an identifier, a number or a JSON fragment.
    """
    if not message.activity and not message.model_calls:
        return

    failed = sum(1 for call in message.activity if not call.ok)
    summary = (
        f"{len(message.activity)} tool call(s) · {message.model_calls} model call(s) · "
        f"\\${message.cost_usd:.6f}"
    )
    if failed:
        summary += f" · {failed} refused"

    with st.expander(summary, expanded=False):
        for index, call in enumerate(message.activity, start=1):
            mark = "ok " if call.ok else "REF"
            head = f"{index}. [{mark}] {call.name}  ({call.duration_ms:.0f} ms)"
            if call.ok:
                st.markdown(f"`{head}`")
            else:
                # Red, because a refusal is the row somebody scrolled here for.
                st.markdown(f":red[`{head}`]")
            if call.arguments:
                st.code(call.arguments, language="json")
            if call.error:
                st.markdown(f":red[{html.escape(call.error)}]")
            elif call.result_chars:
                # The size, not the payload: one catalogue search is 4,728
                # characters and a panel that inlined them would be the thing
                # it was meant to make readable.
                st.caption(f"{call.result_chars:,} characters returned")
        if message.trace_url:
            st.link_button("Open this turn in Langfuse", message.trace_url)


def _draw_message(message: turns.ChatMessage) -> None:
    role = "user" if message.role == turns.CUSTOMER else "assistant"
    with st.chat_message(role):
        if message.text:
            st.markdown(message.text)
        if message.notice:
            st.warning(message.notice)
        if message.cards:
            _draw_cards(message.cards)
        if message.payment_url:
            # Read from `ChatMessage.payment_url`, which `tools/commerce.py`
            # wrote onto the conversation's memory — never scraped out of the
            # model's answer. The model is not given the URL at all: asked
            # twice for one session it reproduced 475 characters correctly once
            # and changed one of them the second time, which Stripe answers
            # with a 401. Measured on PR #9.
            #
            # A button rather than the raw string, because a payment page is
            # something to open, and 475 characters of opaque URL in the middle
            # of a conversation is something to scroll past.
            st.link_button(
                "Pay with Stripe", message.payment_url, type="primary",
                use_container_width=True,
            )
        if message.role == turns.SHOP:
            _draw_activity(message)


@st.dialog("Confirm", dismissible=False, width="medium")
def _ask_to_confirm(session: turns.BrowserSession, pending: turns.PendingApproval) -> None:
    """Put the parked question to the customer, and carry their answer back.

    **The summary is printed verbatim and is never rebuilt here.**
    `agent/guardrails.py` made it from a real `view_cart` dispatch, rendered
    through `money.format_amount`, and that is the whole point of the gate: a
    person approving a figure the model invented is worse than no gate at all,
    because it launders the invention through a human and leaves a record
    saying they agreed. `st.code` rather than `st.markdown`, so the leading
    spaces and the line breaks survive — markdown would collapse the lines into
    one paragraph and quietly reflow somebody's order.

    The answer goes through `session.answer_confirmation`, which is
    `confirmation.resolve_pending` plus one follow-up turn — the protocol D10
    built for exactly this caller, and the same two calls the CLI and the eval
    runner make. Nothing here writes to the conversation's memory.

    `dismissible=False` and `st.rerun(scope="app")`: a dialog that can be
    clicked away would leave the question parked with the chat input disabled
    behind it, which is a customer with no way forward and no way out.
    Declining is the way out, and it orders nothing. The `scope="app"` matters
    because `st.dialog` is a fragment — the default rerun would redraw the
    dialog and not the transcript the answer just produced.
    """
    # `st.dialog`'s own title is fixed at decoration time and cannot vary per
    # call, so the question's real heading is the first thing inside it. The
    # generic one above it cannot be removed either: Streamlit raises
    # `StreamlitAPIException: A non-empty title argument has to be provided for
    # dialogs`, measured with a throwaway app rather than assumed.
    #
    # So the modal carries "Confirm", then "Confirm this refund", then the
    # gate's own "About to refund this whole order:" — three headings for one
    # fact, which this project objects to everywhere else. It is kept anyway,
    # and the reason is the action rather than the layout: dropping the middle
    # one would leave the only per-tool signal inside a grey code block, and a
    # person clicking quickly could approve a refund thinking it was a
    # purchase. Neither is reversible. Repetition is the cheaper mistake.
    st.subheader(pending.title, anchor=False)
    st.code(pending.summary, language=None)
    decline, confirm = st.columns(2)
    if decline.button("Cancel", use_container_width=True, key="confirm-no"):
        with st.spinner("Cancelling…"):
            session.answer_confirmation(False)
        st.rerun(scope="app")
    if confirm.button(
        "Confirm", type="primary", use_container_width=True, key="confirm-yes"
    ):
        with st.spinner("Placing the order…"):
            session.answer_confirmation(True)
        st.rerun(scope="app")


def _draw_cart(session: turns.BrowserSession, pending: turns.PendingApproval | None) -> bool:
    """The basket, in the sidebar. Returns True if the customer asked to check out.

    **Why the sidebar.** The criterion was that it must not break the
    conversation and must be visible without scrolling, and the main column can
    only satisfy one of those at a time: drawn above the transcript it pushes
    the conversation down the page, and drawn below it, it sits at the end of a
    thread that grows all afternoon. The sidebar is neither — it keeps its own
    scroll, stays put while the conversation moves, and Streamlit collapses it
    on a narrow screen. It also leaves `st.chat_input` alone, which matters
    more here than it looks: that input pins itself to the bottom of the page
    only while it is called from the main body, and a panel that had to be
    nested in a container to sit above it would have unpinned it.

    **Nothing here removes anything.** There is no per-line control, for the
    same reason a product card has no Add button: changing what is in a basket
    is something the customer asks for, and a panel that could take a line out
    would be the second shopping interface this page exists to argue against.
    The one button is a checkout, and what that button does — and does not
    bypass — is `BrowserSession.request_checkout`.
    """
    panel = session.cart()
    asked = False

    with st.sidebar:
        st.subheader(CART_TITLE, anchor=False)

        if panel.error:
            st.warning(panel.error)
            return False

        if panel.empty:
            st.caption(CART_EMPTY)
            return False

        for line in panel.lines:
            name = html.escape(line.product_name)
            label = html.escape(line.variant_label)
            st.markdown(f"**{name}**")
            st.caption(
                f"{label} · {line.quantity} × {_amount(line.unit_price)} "
                f"= {_amount(line.line_total)}"
            )

        st.divider()
        st.markdown(f"**Total** · {_amount(panel.total)}")
        st.caption(f"{panel.unit_count} item(s)")

        # Disabled for two states that are one sentence apart and not the same
        # thing. A question already open: answering it is the way forward, and
        # a second click would drop the approval the customer is looking at.
        # The cap reached: the follow-up turn an answer drives is a model call,
        # and the door that refuses a message has to refuse a click too.
        blocked = pending is not None or session.cap_reached
        if st.button(
            CHECKOUT_LABEL,
            type="primary",
            use_container_width=True,
            disabled=blocked,
            key="cart-checkout",
        ):
            asked = True
        st.caption(CART_READ_ONLY)

    return asked


def _draw_openers(session: turns.BrowserSession) -> str | None:
    """The empty screen. Returns a line the customer picked, if they picked one."""
    st.caption("Ask for anything in the shop — searching, sizes, a basket, an order.")
    chosen = None
    for index, opener in enumerate(OPENERS):
        if st.button(opener, key=f"opener-{index}", use_container_width=True):
            chosen = opener
    return chosen


# --- the page ------------------------------------------------------------

st.set_page_config(page_title=TITLE, page_icon="🛍️")

session = _session()

# `anchor=False`: a heading otherwise grows a link icon beside it, which is a
# deep link into a page that has one screen and no sections.
st.title(TITLE, anchor=False)
# `\$` rather than `$`: Streamlit renders markdown, and a pair of unescaped
# dollars around a number is inline LaTeX to it — the first draft of this line
# put the session cost in a maths block and swallowed the cap entirely. Found
# by looking at the page rather than by reasoning about it.
st.caption(
    f"{OWNER} · session \\${session.session_cost_usd:.6f} of "
    f"\\${session.cap_usd:.2f} · {len(session.tool_names)} tools"
)
for note in session.notes:
    st.caption(f"[{note}]")

pending = session.pending

# Drawn before the transcript so the click is handled on the run that produced
# it. Streamlit places it in the sidebar wherever it is called from, so this
# says nothing about where it appears — only about when it is read.
wants_checkout = _draw_cart(session, pending)
if wants_checkout:
    with st.spinner("Reading your basket…"):
        session.request_checkout()
    # Straight back to the top of the script, where `session.pending` is read
    # again and the dialog is opened over a page already in its new state.
    st.rerun()

for message in session.transcript:
    _draw_message(message)

if pending is not None:
    # Behind the modal, so the page still says what state it is in — and it is
    # deliberately the sentence without the summary. The summary belongs to the
    # dialog; printing it twice on one screen would be two things to approve.
    with st.chat_message("assistant"):
        st.info(pending.waiting)

picked = _draw_openers(session) if not session.transcript else None

# Disabled while a question is open, and that is a rule rather than a
# courtesy. `ConversationMemory.begin_turn(from_customer=True)` drops a pending
# confirmation — deliberately, since D10 — so a message sent now would silently
# void the question the customer is looking at, and the modal would vanish with
# nothing said about why.
blocked = session.cap_reached or pending is not None
if session.cap_reached:
    placeholder = "Session limit reached"
elif pending is not None:
    placeholder = "Answer the confirmation to carry on"
else:
    placeholder = "Message ShopAgent"

typed = st.chat_input(placeholder, disabled=blocked)
if session.cap_reached:
    st.caption("This session has stopped spending. Nothing further is charged.")

# Called last, after the transcript and the input, so the modal is drawn over a
# page that is already in its answered state. Only one dialog may be open per
# script run, which is exactly the number of questions the gate ever parks.
if pending is not None:
    _ask_to_confirm(session, pending)

asked = typed or picked
if asked:
    with st.spinner("Thinking…"):
        result = session.send(asked)
    if result.refused:
        st.toast("That message was not sent to the assistant, so nothing was charged.")
    st.rerun()
