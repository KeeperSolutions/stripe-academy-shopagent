"""The two pages Stripe redirects a shopper back to (D7, step 5; D11 follow-up).

Unauthenticated, because Stripe sends a browser here and a browser carries no
`X-API-Key`. That is safe only because these pages read and never write: the
worst a stranger can do with a session id is read the status of a payment they
would have had to make.

**Neither page changes an order, and that rule has not moved.** This is the
point D7 was built around and it survives the page now knowing the order's real
status. A success redirect is a URL anybody can open — Stripe substitutes
`{CHECKOUT_SESSION_ID}` into it and then the browser follows it, with nothing
about that proving money moved. Only a signed delivery from Stripe, arriving on
this server's own webhook route, may move an order to `paid`. A page that
flipped the status because a browser arrived would be trusting the customer's
browser with the shop's money.

**What changed on D11 is that it reports the status instead of asserting one.**
The page used to print a fixed sentence saying the order was still awaiting
confirmation — which went stale within about a second, because the signed
delivery routinely lands before a person has finished reading, and the page
would insist on `pending` over an order already `paid`. It read the *session*
and guessed the *order*. Now it reads both and states each. Reporting is not
deciding: every branch below is a `SELECT`.

**Nothing here names this repository.** These are the only pages in the project
a customer sees who never opened the code, and the previous text sent them the
project's own vocabulary — a development day by number, the name of a webhook,
an HTTP route to call. None of that is a fact about their order.
"""

from __future__ import annotations

import html
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from shopagent.api.db import get_session
from shopagent.api.lifecycle import OrderStatus
from shopagent.api.services.events import SETTLED_PAYMENT_STATUSES
from shopagent.api.models import Order
from shopagent.config import get_settings
from shopagent.money import format_amount
from shopagent.payments import stripe_svc
from shopagent.payments.stripe_svc import MissingStripeKey

router = APIRouter(prefix="/checkout", tags=["checkout"])

_PAGE = """<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font: 16px/1.6 system-ui, sans-serif; max-width: 34rem; margin: 4rem auto; padding: 0 1rem; }}
  .note {{ background: #f4f4f5; border-left: 3px solid #71717a; padding: .75rem 1rem; margin: 1.5rem 0; }}
  dt {{ font-weight: 600; margin-top: .5rem; }}
  dd {{ margin: 0 0 0 1rem; font-family: ui-monospace, monospace; }}
  a.back {{ color: #3f3f46; }}
</style>
<h1>{heading}</h1>
{body}
{back}
"""

# Said on every page, and it is the first thing a returning shopper needs. The
# payment button opens Stripe in a new tab, so the conversation is usually still
# open in the tab behind this one — and going back to it is strictly better than
# following the link below, which starts a *new* session with an empty
# transcript. So the sentence leads with the tab and offers the link as the
# fallback it is.
#
# "should still be open" rather than "is": whether that tab exists is a fact
# about the shopper's browser, not about this server, and somebody who navigated
# in the same tab is being told something false by the confident version.
_BACK = (
    '<p class="note">Your conversation should still be open in the tab you came '
    "from — switch back to it to carry on. "
    '<a class="back" href="{url}">Open the shop</a> if it is not there any more; '
    "that starts a fresh conversation.</p>"
)


def _render(title: str, heading: str, body: str, *, back: bool = True) -> HTMLResponse:
    link = _BACK.format(url=html.escape(get_settings().ui_base_url, quote=True)) if back else ""
    return HTMLResponse(_PAGE.format(title=title, heading=heading, body=body, back=link))


# What each status means, said to the person who just came back from paying.
#
# Every status is here, including the two a shopper is unlikely to arrive on.
# They are not unreachable: a refund issued while somebody sat on the payment
# page, or an order cancelled from elsewhere, both land here, and a page that
# assumed otherwise would fall through to a sentence describing none of them.
# `fulfilled` is in the same category — nothing in this project transitions
# into it automatically, which is a fact about this project and not a promise
# about the row this page is reading.
#
# **Nothing here promises anything this system does not control**, and the rule
# had to be applied line by line rather than once. Four sentences were removed
# for failing it, and one of them was measured against Stripe rather than
# reasoned about:
#
#   - "Stripe has emailed you a receipt." No receipt was emailed. Checked
#     against the live payment of 2026-09-01: `charge.receipt_number` is null,
#     and Stripe sets it only once a receipt has actually been sent.
#     `receipt_email` is null on the PaymentIntent and on the Charge, because
#     nothing in `payments/checkout.py` sets it — the shopper's address reaches
#     `session.customer_details.email` and stops there. In test mode Stripe does
#     not email receipts at all unless the dashboard is configured to, which is
#     a setting this repository neither reads nor owns.
#   - "This usually takes a few seconds." True of a card and false of a
#     delayed-notification method, which settles in days — and which methods are
#     offered is a dashboard setting this code deliberately does not restrict.
#   - "You will be told the moment it lands." There is no push of any kind. The
#     assistant answers when it is asked, through `check_order_status`.
#   - "If you were charged, the payment will be returned to your card."
#     Nobody issues that refund. `paid -> cancelled` is not in the transition
#     table, so a cancelled order was never paid — and the page still does not
#     claim the opposite, because telling a charged shopper they were not
#     charged is the one answer this file already refuses to give.
#
# "Refunded in full" is kept, and it is the one money claim here that *is* this
# system's own: only a full refund moves an order to `refunded`, and a partial
# one changes nothing and is logged.
_STATUS_TEXT = {
    OrderStatus.PAID: (
        "Payment received",
        "<p>Your payment went through and your order is confirmed.</p>",
    ),
    # `pending` is two different situations and only one of them is a payment
    # in flight, so this entry is never used on its own — `_pending_text`
    # chooses between them from what Stripe says about the *session*. Kept in
    # the table so every status has an entry and the lookup below cannot fall
    # through, and set to the neutral half, which is the one that is safe to
    # show when nothing else is known.
    OrderStatus.PENDING: (
        "This order has not been paid",
        "<p>This order is waiting for payment. Nothing has been charged for it "
        "yet. Ask in the conversation and the assistant will give you the "
        "payment link again.</p>",
    ),
    OrderStatus.FULFILLED: (
        "Order complete",
        "<p>This order has been paid and is complete.</p>",
    ),
    OrderStatus.CANCELLED: (
        "This order was cancelled",
        "<p>This order is cancelled and its items have been put back on sale, "
        "so nothing is owed on it. Ask in the conversation if you would like to "
        "order the same items again.</p>",
    ),
    OrderStatus.REFUNDED: (
        "This order was refunded",
        "<p>This order has been refunded in full. How long the money takes to "
        "appear is up to your bank.</p>",
    ),
}

_UNKNOWN_ORDER = (
    "We could not find that order",
    "<p>The payment page came back with an order this shop cannot find. "
    "<strong>That does not say whether a payment went through.</strong> If you "
    "were charged, your card statement is the record — the reference below is "
    "what identifies the payment.</p>",
)


# What a `pending` order means once the session is taken into account. An order
# is `pending` from the moment it is placed, and this URL is one anybody can
# open — a shopper who reached the payment page and backed out lands here with
# an `unpaid` session and a `pending` order, and telling them "you do not need
# to pay again" would be telling somebody who has not paid that they have.
# Raised by review on PR #11.
#
# The allow-list is `SETTLED_PAYMENT_STATUSES`, imported rather than respelled:
# it is the same question the webhook asks before moving an order to `paid`,
# and two spellings of "did the money arrive" is the drift this file exists to
# argue against. `no_payment_required` is in it, which is why this is not
# `== "paid"`.
_PAYMENT_IN_FLIGHT = (
    "Payment is being confirmed",
    "<p>Your payment is being confirmed, and you do not need to pay again. "
    "Refresh this page to check, or ask in the conversation and the assistant "
    "will look it up for you.</p>",
)


def _pending_text(payment_status: str | None) -> tuple[str, str]:
    """Which of the two `pending` stories this shopper is in."""
    if payment_status in SETTLED_PAYMENT_STATUSES:
        return _PAYMENT_IN_FLIGHT
    return _STATUS_TEXT[OrderStatus.PENDING]


def _order_status(session: Session, order_id: str) -> OrderStatus | None:
    """Read one order's status. A `SELECT` and nothing else.

    Deliberately not `render_order`: this page needs one column, and asking for
    the whole snapshot would put an order's lines on a page reachable by
    anybody holding a session id.
    """
    try:
        key = uuid.UUID(order_id)
    except (ValueError, AttributeError, TypeError):
        return None
    order = session.get(Order, key)
    return OrderStatus(order.status) if order is not None else None


@router.get("/success", response_class=HTMLResponse)
def checkout_success(
    session_id: str | None = None,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """Report what Stripe says about the session and what this shop says about
    the order. Change neither.

    `session_id` arrives because `success_url` carries
    `{CHECKOUT_SESSION_ID}`, which Stripe substitutes on redirect. Reading the
    session back is what makes this page show the real outcome rather than
    assuming one from the fact that the browser arrived here; reading the order
    back is what stops the page contradicting it a second later.
    """
    if not session_id:
        return _render(
            "Checkout",
            "No payment to look up",
            "<p>This page is where Stripe sends you after a payment, and it "
            "needs the reference Stripe adds to the address.</p>",
        )

    try:
        checkout = stripe_svc.retrieve_checkout_session(session_id)
    except MissingStripeKey:
        # Deliberately does not say "nothing was charged", which would be a
        # guess dressed as a reassurance. A key can be removed or rotated after
        # a session was created and paid; this server then cannot tell what
        # happened, and telling a charged shopper that they were not charged is
        # the worst of the three possible answers.
        return _render(
            "Checkout",
            "This payment cannot be checked right now",
            "<p>This shop cannot reach the payment provider, so it cannot look "
            "your payment up. <strong>That does not mean the payment did not go "
            "through</strong> — check your card statement before paying "
            "again.</p>",
        )
    except Exception:
        # A bad or foreign session id. Say so without leaking whether it merely
        # belongs to somebody else.
        return _render(
            "Checkout",
            "This payment cannot be checked right now",
            "<p>That payment reference could not be read. <strong>This does not "
            "say whether a payment went through</strong> — it says only that "
            "this shop could not find out.</p>",
        )

    order_id = (checkout.metadata._data or {}).get("order_id")
    status = _order_status(session, str(order_id)) if order_id else None

    if status is None:
        heading, explanation = _UNKNOWN_ORDER
    elif status is OrderStatus.PENDING:
        heading, explanation = _pending_text(checkout.payment_status)
    else:
        heading, explanation = _STATUS_TEXT[status]

    body = f"""
{explanation}
<dl>
  <dt>Amount</dt><dd>{html.escape(format_amount(checkout.amount_total, checkout.currency))}</dd>
  <dt>Order</dt><dd>{html.escape(str(order_id or "unknown"))}</dd>
</dl>
"""
    return _render("Checkout", heading, body)


@router.get("/cancel", response_class=HTMLResponse)
def checkout_cancel() -> HTMLResponse:
    """Where Stripe sends a shopper who backed out. Nothing was charged."""
    return _render(
        "Checkout cancelled",
        "Checkout cancelled",
        # "Nothing was charged" is safe here and nowhere else on this page:
        # Stripe sends a browser to `cancel_url` from an *open* session, which
        # is a session nobody has paid.
        #
        # What was removed is the two sentences after it. "Pay for it whenever
        # you like" is contradicted by this system's own behaviour — a Checkout
        # Session expires, and `checkout.session.expired` cancels the order and
        # releases its stock. And "say so in the conversation and the items will
        # be released" promised something the assistant cannot do: there are
        # six commerce tools and none of them cancels an order.
        "<p>Nothing was charged. Your order is still open and its items are "
        "still held, so the same checkout can be started again.</p>"
        "<p>Leaving this page is not the same as cancelling the order. An order "
        "left unpaid does not stay open forever — when it lapses, the items go "
        "back on sale on their own.</p>",
    )
