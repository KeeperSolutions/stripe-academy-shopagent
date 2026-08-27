"""The two pages Stripe redirects a shopper back to (D7, step 5).

Unauthenticated, because Stripe sends a browser here and a browser carries no
`X-API-Key`. That is safe only because these pages read and never write: the
worst a stranger can do with a session id is read the status of a payment they
would have had to make.

**Neither page changes an order.** This is the point the whole day is built
around, and the pages say so in as many words. A success redirect is a URL
anybody can open — Stripe substitutes `{CHECKOUT_SESSION_ID}` into it and then
the browser follows it, with nothing about that proving money moved. Only D8's
webhook, which arrives from Stripe over a signed channel, may move an order to
`paid`. A success page that flipped the status would be trusting the customer's
browser with the shop's money.

So the page reports what Stripe says about the *session* and states plainly
that the order is still awaiting confirmation. The two are different claims and
showing them side by side is the honest version.
"""

from __future__ import annotations

import html

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

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
</style>
<h1>{heading}</h1>
{body}
"""


# Currencies whose smallest unit *is* the unit: no decimal part exists, so
# dividing by 100 would invent one. Not exhaustive — these are the ones a
# shop like this plausibly meets. Anything unlisted is treated as two decimals,
# which is right for every currency this project actually sells in.
ZERO_DECIMAL_CURRENCIES = frozenset(
    {"bif", "clp", "djf", "gnf", "jpy", "kmf", "krw", "mga", "pyg", "rwf",
     "ugx", "vnd", "vuv", "xaf", "xof", "xpf"}
)


def format_amount(minor_units: int | None, currency: str | None) -> str:
    """Render a Stripe amount for a person to read.

    Stripe speaks minor units and so does this project, all the way from
    `price_cents` through `amount_total` — which is deliberate and is why there
    is no rounding anywhere in the money path. A page is where that stops being
    right: `4200 USD` reads as four thousand dollars, and the shopper who just
    paid $42 has no way to tell which one happened.

    So the conversion lives here, at the boundary, and nowhere else — the same
    place the `amount_cents` to `price_cents` rename happens for the same
    reason.
    """
    if minor_units is None:
        return "—"

    code = (currency or "").lower()
    if code in ZERO_DECIMAL_CURRENCIES:
        return f"{minor_units:,} {code.upper()}"
    return f"{minor_units / 100:,.2f} {code.upper()}"


def _render(title: str, heading: str, body: str) -> HTMLResponse:
    return HTMLResponse(_PAGE.format(title=title, heading=heading, body=body))


@router.get("/success", response_class=HTMLResponse)
def checkout_success(session_id: str | None = None) -> HTMLResponse:
    """Report what Stripe says about the session. Change nothing.

    `session_id` arrives because `success_url` carries
    `{CHECKOUT_SESSION_ID}`, which Stripe substitutes on redirect. Reading the
    session back is what makes this page show the real outcome rather than
    assuming one from the fact that the browser arrived here.
    """
    if not session_id:
        return _render(
            "Checkout",
            "No session to look up",
            "<p>This page expects a <code>session_id</code>, which Stripe adds "
            "when it redirects here after a payment.</p>",
        )

    try:
        session = stripe_svc.retrieve_checkout_session(session_id)
    except MissingStripeKey:
        # Deliberately does not say "nothing was charged", which would be a
        # guess dressed as a reassurance. A key can be removed or rotated after
        # a session was created and paid; this server then cannot tell what
        # happened, and telling a charged shopper that they were not charged is
        # the worst of the three possible answers.
        return _render(
            "Checkout",
            "This payment cannot be checked right now",
            "<p>The server is not configured to talk to Stripe, so it cannot "
            "look this session up. <strong>That does not mean the payment did "
            "not go through</strong> — check your card statement or your email "
            "receipt from Stripe before paying again.</p>",
        )
    except Exception:
        # A bad or foreign session id. Say so without leaking whether it merely
        # belongs to somebody else.
        return _render(
            "Checkout",
            "This payment cannot be checked right now",
            "<p>That session id could not be read. <strong>This does not say "
            "whether a payment went through</strong> — it says only that this "
            "server could not find out.</p>",
        )

    paid = session.payment_status == "paid"
    order_id = (session.metadata._data or {}).get("order_id", "unknown")

    heading = "Payment received" if paid else "Payment not completed"
    body = f"""
<dl>
  <dt>Stripe session</dt><dd>{html.escape(session.status or "")}</dd>
  <dt>Payment status</dt><dd>{html.escape(session.payment_status or "")}</dd>
  <dt>Amount</dt><dd>{html.escape(format_amount(session.amount_total, session.currency))}</dd>
  <dt>Order</dt><dd>{html.escape(str(order_id))}</dd>
</dl>
<p class="note">
  <strong>The order has not been marked paid yet, and this page did not mark
  it.</strong> A redirect is a URL anybody can open, so it is not proof that
  money moved. The order changes state only when Stripe delivers a signed
  webhook to this server, which is what Day 8 builds. Until then the order
  stays <code>pending</code> even when Stripe says the payment succeeded.
</p>
"""
    return _render("Checkout", heading, body)


@router.get("/cancel", response_class=HTMLResponse)
def checkout_cancel() -> HTMLResponse:
    """Where Stripe sends a shopper who backed out. Nothing was charged."""
    return _render(
        "Checkout cancelled",
        "Checkout cancelled",
        "<p>Nothing was charged. The order is still open and its items are "
        "still reserved, so the same checkout can be started again.</p>"
        '<p class="note">Cancelling the checkout page is not the same as '
        "cancelling the order. <code>POST /orders/{id}/cancel</code> is what "
        "releases the reserved stock.</p>",
    )
