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

**The page is centred, and that is `st.chat_input`'s doing.** Streamlit pins a
`chat_input` called from an app's main body to the bottom of the page, and
renders it inline — scrolling away with the content — as soon as it is nested
in a container or a column. So the readable column is not built out of
`st.columns`; it is Streamlit's own `layout="centered"`, which the input pins
itself to the bottom of. Fighting that with columns would have traded a pinned
input for a centred one.

**The confirmation gate is not wired up here.** `create_checkout` parks a
question on the conversation's memory and this page shows that it is waiting,
deliberately without answering it — that is step 3. Nothing is ordered and
nothing is charged in the meantime, which is the safe half of the D10 protocol:
an unanswered approval is never spendable.
"""

from __future__ import annotations

import html

import streamlit as st

from shopagent.ui import colors
from shopagent.ui import session as turns

TITLE = "ShopAgent"

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

WAITING_FOR_YOU = (
    "The shop is asking you to approve this purchase. Nothing has been ordered "
    "and nothing has been charged."
)


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


def _variant_line(variant: turns.VariantCard) -> str:
    """One row of a card: colour, size, price, and whether it can be bought.

    Everything from the catalog goes through `html.escape` before it meets the
    swatch's markup. `variant.price` is already `money.format_amount`'s output
    — no arithmetic happens in this file, which is the D1 rule about money
    reaching its last mile.
    """
    parts = [part for part in (variant.color, variant.size) if part]
    label = " · ".join(parts) or variant.sku
    body = f"{html.escape(label)} — {html.escape(variant.price)}"
    if variant.in_stock:
        return f"{_swatch(variant.color)}{body}"
    # Dimmed and named. A greyed row with nothing said would read as a
    # rendering fault rather than as stock information.
    return (
        f'<span style="opacity:0.45;">{_swatch(variant.color)}{body}</span>'
        f'<span style="opacity:0.55;"> · out of stock</span>'
    )


def _draw_card(card: turns.ProductCard) -> None:
    """One product, with its variants. Nothing here is clickable, on purpose."""
    with st.container(border=True):
        st.markdown(f"**{html.escape(card.name)}**", unsafe_allow_html=True)
        heading = " · ".join(part for part in (card.brand, card.category) if part)
        if heading:
            st.caption(heading)
        for variant in card.variants:
            st.markdown(_variant_line(variant), unsafe_allow_html=True)
        if card.description:
            st.caption(card.description)


def _draw_message(message: turns.ChatMessage) -> None:
    role = "user" if message.role == turns.CUSTOMER else "assistant"
    with st.chat_message(role):
        if message.text:
            st.markdown(message.text)
        if message.notice:
            st.warning(message.notice)
        for card in message.cards:
            _draw_card(card)
        if message.payment_url:
            # Printed in the bytes the shop issued, never relayed by the model:
            # asked twice for one session, the model changed a character of the
            # 475-character URL. Measured on PR #9.
            st.markdown("**Pay here**")
            st.code(message.payment_url, language=None)


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
    f"session \\${session.session_cost_usd:.6f} of \\${session.cap_usd:.2f} · "
    f"{len(session.tool_names)} tools"
)
for note in session.notes:
    st.caption(f"[{note}]")

for message in session.transcript:
    _draw_message(message)

if session.pending is not None:
    # Shown, not answered. The control that answers it is step 3; until then an
    # unanswered approval is simply never spendable, which is the safe half of
    # the protocol rather than a gap in it.
    with st.chat_message("assistant"):
        st.info(WAITING_FOR_YOU)
        st.code(session.pending.summary, language=None)

picked = _draw_openers(session) if not session.transcript else None

typed = st.chat_input(
    "Message ShopAgent" if not session.cap_reached else "Session limit reached",
    disabled=session.cap_reached,
)
if session.cap_reached:
    st.caption("This session has stopped spending. Nothing further is charged.")

asked = typed or picked
if asked:
    with st.spinner("Thinking…"):
        result = session.send(asked)
    if result.refused:
        st.toast("That message was not sent to the assistant, so nothing was charged.")
    st.rerun()
