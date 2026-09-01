# ShopAgent

A conversational shopping assistant — training project (Agentic Commerce Training).
Users browse the catalog, manage a cart and complete checkout entirely through
natural language: the catalog is exposed over **MCP**, cart and orders over a
**REST API**, and payment runs through **Stripe** with a webhook flipping the order
status to `paid`.

## Architecture

```
                       a person at a terminal
                                 │
   "add the second one" ─────────┤   the payment page, printed
                                 │   verbatim from the tool result —
                                 ▼   the model never sees the URL
   ┌────────────────────────────────────────────────────────────────┐
   │ llm/loop.py :: run_tool_loop                                   │
   │ a while, a message list, a dispatch. Byte-stable since D2      │
   │ (161bdc1c…9d00): nothing below sits inside it, all of it wraps │
   └───────┬─────────────────────────────────────┬──────────────────┘
           │ the conversation                    │ one tool call
           ▼                                     ▼
   ┌─────────────────────────┐   ┌──────────────────────────────────┐
   │ GuardedClient   D9·D10  │   │ GuardedRegistry      D9 → D10    │
   │ no amount in an answer  │   │ create_checkout is parked, not   │
   │ that no tool produced:  │   │ run. The total a person approves │
   │ one corrective retry,   │   │ is read from view_cart and       │
   │ then a fallback naming  │   │ formatted here — never taken     │
   │ the figure it could     │   │ from the model's prose. One      │
   │ not trace.              │   │ approval, good for one turn.     │
   ├─────────────────────────┤   ├──────────────────────────────────┤
   │ TracedClient      D10   │   │ RememberingRegistry        D9    │
   │ inside the guard, so    │   │ last_search, seen variant ids,   │
   │ a corrected turn is     │   │ seen amounts, cart_id, order_id, │
   │ billed twice and        │   │ checkout_url — state the model   │
   │ traced twice.           │   │ is never handed.                 │
   └───────┬─────────────────┘   ├──────────────────────────────────┤
           │                     │ TracedRegistry            D10    │
           ▼                     │ watches; it does not decide.     │
   OpenAI, Chat Completions      └───────────────┬──────────────────┘
   (the loop stays here, not                     │
    on somebody's server)                        │
                    ┌────────────────────────────┴─────┐
                    │ MCP, stdio                       │ HTTP + X-API-Key
                    ▼ a subprocess                     ▼ another service
   ┌───────────────────────────────────┐  ┌─────────────────────────────────┐
   │ mcp_client/ ──▶ mcp_server/       │  │ tools/commerce.py ──▶ tools/    │
   │ registers whatever the server     │  │ http.py ──▶ api/ (FastAPI)      │
   │ lists; naming a tool here is      │  │ carts, orders, checkout, cancel,│
   │ forbidden, and it is a thin       │  │ refund. The rules live in       │
   │ wrapper only — the search         │  │ api/services/, which imports no │
   │ lives one layer down, in          │  │ FastAPI — so a webhook and an   │
   │ catalog/.                         │  │ agent tool reach them too.      │
   └────────────────┬──────────────────┘  └────────────┬────────────────────┘
                    │                                  │
                    └────────────────┬─────────────────┘
                                     ▼
                      Postgres 16 + pgvector
                      catalog — seed-built, disposable
                      carts · orders · processed_events — real data
                                     ▲
                                     │ only a signed webhook may write `paid`
                    Stripe ──────────┘ api/routers/webhooks.py → lifecycle.py
```

**Two protocols, on purpose.** The catalog reaches the model through **MCP**
because it is somebody else's data in the shape this project wanted to learn:
a server publishes tools, the client registers whatever it is given, and
nothing on this side may match on a tool name. That constraint is the point —
swap the server and the agent gains its tools without a line changing here.
The catalog is also read-only and idempotent, so the worst a bad call costs is
a wasted round trip.

Cart and checkout reach it over **HTTP**, because they *write*, and because
the writes have to be reachable by callers that are not an agent at all.
`place_order` locks inventory under `SELECT ... FOR UPDATE`; only a signed
webhook may move an order to `paid`; a person with `curl` and the API key can
do everything the model can. Putting those behind MCP would have made this
project's own agent the only client of its own shop, and would have hidden the
protections that bind everyone rather than just the model.

**The gate and the guardrails bind the model, not the shop**, and the diagram
places them accordingly: they sit between `run_tool_loop` and the tools, above
both protocols. They exist because a model can be talked into spending money —
not because HTTP is dangerous. Nothing in `agent/` is a substitute for the
inventory lock, the lifecycle table or the webhook signature, which sit below
and apply to every caller.

**`run_tool_loop` is in none of it.** Its source hashes to `161bdc1c…9d00` on
`main` and on every branch since D2. D5 changed where tools come from, D9
changed what the model is told and what it may say, D10 added tracing — each
of them by wrapping, and the hash is how that claim is checked rather than
asserted.

## Prerequisites

- Python 3.11+
- Docker (Docker Desktop / Docker Engine with `docker compose`)

## Setup

```bash
# 1. virtualenv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 2. dependencies
pip install -r requirements.txt

# 3. env
cp .env.example .env     # .env is never committed; fill in your keys locally
#
# OPENAI_API_KEY and SHOPAGENT_API_KEY are required — nothing starts without
# them. Three that D9 added, all optional:
#   COMMERCE_API_BASE_URL   where the agent's cart tools reach the API.
#                           Defaults to http://localhost:8000; separate from
#                           APP_BASE_URL, which is the public URL Stripe
#                           redirects a browser to.
#   SHOPPER_ID              who the CLI shops as, and the key of the profile it
#                           remembers between conversations. Blank means no
#                           long-term memory, which is not an error. It is a
#                           label, not a credential: it authenticates nobody,
#                           and it is redacted before a trace leaves the
#                           process.
#   CURRENCY                the shop's currency, `eur` by default. Changing it
#                           after seeding means a reseed: `prices` rows carry
#                           the currency they were written with.
#
# Four that D10 added, all optional:
#   LANGFUSE_PUBLIC_KEY     a project on https://cloud.langfuse.com, or your
#   LANGFUSE_SECRET_KEY     own instance via LANGFUSE_HOST. Leave them blank
#   LANGFUSE_HOST           and the agent runs untraced and says so once in
#                           its banner — observability is one part of this
#                           system, not a precondition for the rest.
#   TRACE_REDACT_TEXT       `true` by default. Every field a person wrote —
#                           the customer's messages, the model's answers, the
#                           system prompt with the profile name in it, the
#                           `query` argument, SHOPPER_ID — is replaced with a
#                           salted digest before the trace leaves this
#                           process. What still travels is this shop's own
#                           data and this process's own measurements: tool
#                           names and their other arguments, tool results,
#                           amounts, order ids, product names, tokens, cost,
#                           latency and which guardrail refused what.
#                           Set it false on your own machine when you need to
#                           read a trace as a conversation; the default must
#                           not be the setting somebody has to remember.
#
# Three more, with defaults chosen rather than inherited:
#   OPENAI_CONNECT_TIMEOUT_SECONDS  10
#   OPENAI_READ_TIMEOUT_SECONDS     90
#   OPENAI_MAX_RETRIES              2   a dead connection is given up on
#                                       within (10+90) x 3 = 300s. That bounds
#                                       a stall, not a request: these are
#                                       per-phase inactivity timeouts, so a
#                                       reply arriving slowly can outlast them.
#                                       The SDK's own default is read=600s with
#                                       two retries, which turns a dropped
#                                       connection into a thirty-minute
#                                       silence. Measured: a D10 eval pass sat
#                                       there for ten minutes.

# 4. database (Postgres 16 + pgvector)
docker compose up -d
docker compose exec db psql -U shopagent -d shopagent \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 5. catalog (schema, 30 products, vectors + HNSW index)
python scripts/create_schema.py
python scripts/seed_catalog.py
python scripts/embed_catalog.py

# 6. migrations (commerce tables only; create_all cannot alter an existing one)
#    Two of them today: 0001 adds the D7 customer columns to `orders`, 0002
#    creates `processed_events`. Both are idempotent, so the loop is safe to
#    re-run and safe on a database that already has them.
for f in migrations/*.sql; do
  docker compose exec -T db psql -U shopagent -d shopagent -f /dev/stdin < "$f"
done
python scripts/create_schema.py   # exits 2 if a column or foreign key is missing

# 7. verify
docker compose exec db psql -U shopagent -d shopagent \
  -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
pytest tests/ -v          # 1224 pass; add -m network for the ones that cost money
```

Step 6 matters on any database that predates a schema change: `create_all`
creates tables that do not exist and never alters one that does, so a database
built before D7 would silently lack `orders.customer_email` and one built
before D8 would have every table except `processed_events`. On a genuinely
fresh database step 5 has already built both, and the migrations are then
no-ops that print `already exists, skipping` — which is what makes running them
unconditionally the simpler instruction. The migrations are
idempotent, so running all of them every time is safe, and re-running
`create_schema.py` afterwards is what confirms it — it compares the models
against the live database and exits 2 on any gap. See CLAUDE.md for why there
is no migrations table.

Postgres listens on `localhost:5432` (user / password / db: `shopagent`), with data
stored in the named volume `pgdata`. Stop it with `docker compose down` — the volume
is preserved.

## Commands

**The agent needs the commerce API running.** Since D9 the CLI carries cart
and checkout tools that reach `uvicorn` over HTTP, so start the API in another
terminal before the agent. Without it the catalog still works and the cart
tools answer the model with an explanation rather than a traceback, which is
deliberate — but nothing can be bought.

```bash
# the agent (start `uvicorn` first, see below)
python -m shopagent.llm.loop            # CLI agent: 11 tools
MCP_CATALOG_ENABLED=false \
  python -m shopagent.llm.loop          # same CLI without the catalog: 7 tools

# the same agent in a browser (D11). `uvicorn` must be running first, for the
# same reason the CLI needs it: the cart and checkout tools reach it over HTTP.
streamlit run src/shopagent/ui/app.py   # on :8501

# inside the agent
#   /tools            the tools the model can call
#   /profile          what the shop remembers about you between conversations
#   /remember k=v     record one field: display_name, shoe_size, clothing_size,
#                     favourite_categories (shoes, jackets, bags, accessories,
#                     equipment). Needs SHOPPER_ID set.
#   /forget k         clear one field
#   /reset            clear the history and re-read the profile
#   /cost             tokens and dollars for this session

# the catalog MCP server on its own
python scripts/run_mcp_server.py        # serves on stdio; the agent starts this itself

# the MCP Inspector, against that server
npx @modelcontextprotocol/inspector \
  .venv/bin/python scripts/run_mcp_server.py               # web UI on :6274
npx @modelcontextprotocol/inspector --cli \
  .venv/bin/python scripts/run_mcp_server.py --method tools/list   # non-interactive

# the commerce API (carts, orders, checkout, cancel, refund)
uvicorn shopagent.api.main:app --reload --port 8000      # docs on :8000/docs

# Stripe, test mode only; needs STRIPE_SECRET_KEY in .env
python scripts/sync_stripe_catalog.py --dry-run          # plan, write nothing
python scripts/sync_stripe_catalog.py                    # Products + Prices

# webhooks: run this in a second terminal alongside uvicorn, then put the
# whsec_... it prints into .env as STRIPE_WEBHOOK_SECRET
stripe listen --forward-to localhost:8000/webhooks/stripe
stripe trigger checkout.session.completed                # a fixture delivery
stripe events resend evt_...                             # the same event again

# before and after driving the API by hand
python scripts/manual_test_state.py snapshot
python scripts/manual_test_state.py restore

# catalog data
python scripts/create_schema.py         # pgvector + create_all, idempotent
python scripts/seed_catalog.py          # 30 products; --reset to rebuild
python scripts/embed_catalog.py         # vectors + HNSW index; --force to redo

# evals — what this shop is claimed to do, settled by a run
python scripts/run_evals.py --list      # free: every scenario and its claims
python scripts/run_evals.py             # costs money: ~$0.012 for all ten
python scripts/run_evals.py --only the_happy_path_reaches_a_payment_page
                                        # the report is also written to
                                        # notes/eval-report.txt, because D9
                                        # paid twice for a run it read with
                                        # `tail`

# tests
pytest tests/ -v                        # 1267 collected: 1224 pass, 20 skip,
                                        # 23 deselected because they cost money
pytest tests/ -m network                # the 4 embedding tests and the 3 chain runs;
                                        # these cost money and need uvicorn running
pytest tests/ -m stripe                 # the 16 that call Stripe in test mode (free)
pytest tests/ -m db                     # the 460 that need Postgres; they skip
                                        # with a reason when it is unreachable
```

The eval run needs `uvicorn` and Postgres up, the same as the agent, and
`STRIPE_WEBHOOK_SECRET` for the one scenario that pays. It drives the CLI's own
entry point — `build_tool_setup` then `run_tool_loop` — so the gate, the memory,
the guardrails, the traced wrappers and the MCP catalog are the ones a customer
gets rather than a copy assembled for the test. Each scenario undoes itself **by
id**, never by truncating a table, and a paid order is *refunded* rather than
deleted, because `paid -> cancelled` is not in the transition table.

Paying goes through a Stripe Checkout Session: `POST /orders/{id}/checkout`
returns a URL, and the test card is `4242 4242 4242 4242` with any future date
and any CVC. The order stays `pending` when the browser lands back on the
success page — that redirect is a URL anybody can open — and moves to `paid`
only when `POST /webhooks/stripe` receives a signed `checkout.session.completed`
from Stripe. `line_items` are built from the `order_items` snapshot rather than
from the Stripe Prices that `sync_stripe_catalog.py` writes; CLAUDE.md explains
why those are two separate things.

`POST /orders/{id}/cancel` is the other way an order ends, and the only one a
person drives directly. It applies to a `pending` order — never a `paid` one,
because once a charge settles the way back is a refund — releases the reserved
stock, and expires the Checkout Session so the payment URL stops working. A
second cancel is 409: `cancelled` is terminal, which is what stops the same
reservation being handed back twice.

Refunds work the same way round. `POST /orders/{id}/refund` asks Stripe for a
full refund and answers **202**: the order is still `paid` when it returns, and
`charge.refunded` is what moves it to `refunded` and releases the reserved
stock. A partial refund — which only the Stripe dashboard can issue here —
arrives as the same event and deliberately changes nothing, because there is no
status between `paid` and `refunded` and releasing the whole reservation for
part of the money would be worse than leaving it alone. It is logged at ERROR.

**`stripe listen` cannot be pinned to an API version.** It renders events at
the account's default version, or at Stripe's newest with `--latest`, and there
is no `--stripe-version` flag — so neither necessarily matches the
`STRIPE_API_VERSION` this repo pins. The mismatch is harmless (a signature is
computed over bytes) and the webhook logs a warning naming both versions, which
means that warning appears on every local delivery. Do not go looking for a
flag to silence it.

The API and the catalog reach the model over different protocols on purpose:
products come through MCP, carts and orders over HTTP. `/docs` is the fastest way
to drive the second — paste `SHOPAGENT_API_KEY` into **Authorize** once and every
cart and order call carries it, while `/health` needs no key. Screenshots of that
walkthrough are in `docs/screenshots/`, alongside the `d11-*` set showing the
browser UI: the conversation with product cards, the confirmation dialog, the
agent activity panel, and one order paid end to end.

The agent spawns the catalog server itself, so `run_mcp_server.py` is only needed
to drive the server by hand — from the Inspector, or to tell a catalog fault
apart from an agent fault. `MCP_CATALOG_ENABLED=false` is what makes "the product
answers come from MCP" demonstrable: the same binary then has two tools instead
of six and says the catalogue is unavailable.

Pass the server path, not `-m shopagent.mcp_server.server`, to the Inspector: it
parses `-m` as one of its own flags and the module never starts.

## The browser UI (D11)

```bash
uvicorn shopagent.api.main:app --reload --port 8000   # first, always
streamlit run src/shopagent/ui/app.py                 # then, on :8501
```

**`uvicorn` is not optional.** The cart, checkout and order tools reach it over
HTTP, so without it the catalog still answers and every commerce tool tells the
model — in words — that the shop is unreachable. The page will load and nothing
can be bought.

**Chat is the whole interface.** There is no grid beside the conversation and no
catalog to browse around the agent, which is why the API has no `GET /products`.
Search results appear as cards *inside* the message that produced them, and a
card is not clickable: nothing enters a basket except by asking. Variants are
grouped by colour — `black · 41, 42, 43 — €94.99` — with sold-out sizes struck
through rather than dropped, because "there is no 42" and "the 42 is gone" are
different sentences.

**Every turn carries a collapsed activity panel**: which tools ran in what
order, their arguments, how long each took, whether it was refused and why, the
turn's cost and model-call count, and a link to its Langfuse trace. Closed, the
page is a conversation; open, it is what this project is for. A refused call is
the most useful row in it — the confirmation gate parking a question and the
unknown-variant guardrail both appear there with their reasons.

**A purchase is confirmed in a modal, and the total in it comes from
`view_cart`** through `money.format_amount` — never from anything the model
wrote. The chat input is disabled while that question is open, because a new
message would silently void it. The payment link is rendered from state the
shop wrote, never from the model's prose; the model is not given the URL at all.

**A customer can ask for a refund, and it goes through the same gate.**
"I want a refund" refunds the order placed in *this* conversation, in full —
there is no way to refund one line, because there is no status between `paid`
and `refunded` for a partly refunded order to sit in. The shop shows the order
and asks before anything is requested, the same two-phase protocol a checkout
uses. What comes back is a refund that has been **requested**: the order stays
`paid` until Stripe confirms the money went back, exactly as when a refund is
issued from the Stripe dashboard. An order from an earlier conversation cannot
be refunded here — nothing links an order to a shopper, which is written up as
an open gap.

**The basket sits in the sidebar and is read from the shop, not from the
conversation.** Lines, quantities and a total, re-read over HTTP on every draw
and costing no model call — a panel rendered from the last message would keep
showing a line that was removed two turns ago. Nothing can be taken out of it:
changing a basket is something you ask for, the same reason a card has no Add
button.

Its one button is **Checkout**, and it is not a second route to payment. It
dispatches `create_checkout` through the same guarded registry the model
reaches, so the same gate parks the same question, built from the same
`view_cart` read. What it skips is the model's decision to call the tool — which
the customer has just made by clicking. It is disabled on an empty basket, while
a confirmation is open, and once the spend cap is reached.

**The page Stripe redirects back to reads your order and reports it.** Paid,
still confirming, cancelled or refunded — a sentence for each, and none of them
names anything from this repository. It writes nothing: only a signed delivery
from Stripe moves an order to `paid`. It also says the conversation is waiting
in the tab you came from, because the payment button opens a new one and
following the page's own link starts a fresh session with an empty transcript.

| Setting | Default | What it does |
|---|---|---|
| `UI_SPEND_CAP_USD` | `0.50` | what one browser session may spend on model calls. Checked at the door of a turn, never inside one, so the ceiling is the cap plus one turn — `run_tool_loop` is not opened for it. When it is reached the input is disabled and the conversation stays readable. |
| `UI_BASE_URL` | `http://localhost:8501` | where this page runs, so the checkout pages can offer a way back to it. A third URL, separate from `APP_BASE_URL` and `COMMERCE_API_BASE_URL`, because the browser interface is a separate server on a separate port. |

`.streamlit/config.toml` sets `toolbarMode = "minimal"`, which drops the Deploy
button and the developer menu while keeping the rerun control — the one thing in
that bar a person watching a slow turn wants.

Measured end to end on D11: search, add to cart, confirm in the dialog, pay with
`4242 4242 4242 4242`, and the agent answers "your payment went through, the
order is paid" from `check_order_status` — a status written by a signed
`checkout.session.completed` and by nothing else. Five webhook deliveries, all
200. `docs/screenshots/d11-06-paid-end-to-end.png`.

Driven again through the basket button in the follow-up: two items, **Checkout**
from the sidebar, confirm, pay, and back — €244.98, `pending → paid`, five
deliveries, $0.002371 for the whole conversation. And once more for refunds:
buy, pay, then "I want a refund" — the gate asks *Confirm this refund* over the
order's own total, and the agent answers "your full refund of €94.99 has been
requested and is on its way, it is not completed yet". `charge.refunded` moved
the order to `refunded` and put the reserved unit back. $0.001844.
`docs/screenshots/d11-07-basket-panel-with-checkout.png`,
`d11-07b-button-reaches-the-gate.png`,
`d11-08-success-page-reads-the-order.png` and
`d11-09-refund-dialog.png`.

## Eval results

One full pass on 2026-08-31, `gpt-5.6-luna`: **8 of 10 scenarios passed**, 68
model calls, **$0.011674**. The two that failed did not fail for the same kind
of reason, and the number on its own would suggest they did.

| Scenario | Result | Calls | Cost |
|---|---|---|---|
| `a_price_limit_becomes_a_filter` | PASS | 2 | $0.000298 |
| `a_semantic_query_finds_what_shares_no_words_with_it` | PASS | 2 | $0.000337 |
| `the_second_one_means_the_second_row` | **FAIL** | 6 | $0.001075 |
| `a_size_the_shop_does_not_stock_is_refused` | PASS | 4 | $0.000577 |
| `a_checkout_is_put_to_a_person_before_it_happens` | PASS | 11 | $0.001827 |
| `no_amount_reaches_the_customer_that_no_tool_produced` | PASS | 6 | $0.001719 |
| `the_happy_path_reaches_a_payment_page` | PASS | 11 | $0.001587 |
| `an_ambiguous_request_is_answered_with_a_question` | **FAIL** | 2 | $0.000484 |
| `removing_a_line_changes_the_total` | PASS | 11 | $0.001854 |
| `an_order_reads_paid_only_after_the_webhook` | PASS | 13 | $0.001914 |

**`the_second_one_means_the_second_row` — model variance.** The scenario asks
for "the second one" after a search and a size check, and asserts that
`add_to_cart` runs with the id in row two. The model called `search_products`
and `check_stock`, resolved the ordinal **correctly** — it named Summit Peak
Pro, which was the second row — and then declined to act, asking the customer
to choose between two products instead. The reason it gave for declining
("its variant wasn't the second result in the stock check") describes no
guardrail in this codebase; nothing refused it, and there were zero
unknown-variant refusals in the whole pass. D9 measured the same phrase working
twice. So the claim is unchanged and the run is what moved.

**`an_ambiguous_request_is_answered_with_a_question` — the claim is wrong, not
the shop.** Asked for "something for my trip", the model listed options and
ended "Tell me your trip type, preferred item, and budget." It bought nothing
and guessed nothing: `tools_not_called: [add_to_cart, create_checkout]` passed.
What failed is `answer_matches: "\?"` — the scenario operationalised "asks for
clarification" as "contains a question mark", and a request for clarification
in the imperative carries none. That is a test asserting the presence of a
character rather than the correctness of a behaviour, the same class of defect
D9 recorded when `"dollar" in description` went on passing in a euro shop.

**Neither was tuned away, and that is the point of writing them down.** The
prompt, the tool descriptions, the guardrails and the code were not touched to
make a scenario green, and neither expectation was widened to accept what
happened. A suite that is edited until it passes measures the editing. The
second failure names a real repair — the expectation needs to describe asking
rather than punctuation — and it is filed as an open gap rather than patched in
the same breath as the run that found it.

**The amount guardrail has still never fired against a real model**, five days
running. Scenario 6 asks for arithmetic and passed, but not by the model
declining: it worked out that three pairs at €94.99 come to €284.97, *called
`add_to_cart` with quantity 3 first*, and quoted the €284.97 that came back in
the tool result. The figure was traceable because the shop had produced it, so
there was nothing to catch. The rule holds exactly as written and the branch
behind it remains unexercised outside offline tests.

## What is not finished

The full list, with the reasoning behind each, is in
[JOURNAL.md](JOURNAL.md#known-gaps). The ones worth knowing before using any of
this:

- **Two eval scenarios fail**, for the two different reasons above. The suite
  is red on purpose rather than green by adjustment.
- **The amount guardrail's fallback has never run against a real model.** It is
  proven to do the right thing when handed a bad answer; it is not proven that
  a real model produces one.
- **The rule about amounts cannot tell "read from a tool" from "computed, then
  confirmed by a tool".** Scenario 6 shows the second case passing, correctly by
  the letter and by accident in spirit — the order of operations decides.
- **The suite's database guard cannot tell an eval leftover from a real Stripe
  delivery.** `stripe listen` writes `processed_events` rows on its own while it
  runs. The message says so; the guard cannot distinguish them.
- **No demo video.** Deliberately deferred rather than dropped: the web
  interface arrives on D11, and a recording of the CLI would be a week stale on
  the day after it was made. It gets made against the interface people will
  actually see.

Findings, decisions and open questions from each day are in
[JOURNAL.md](JOURNAL.md).
