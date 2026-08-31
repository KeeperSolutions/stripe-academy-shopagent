# ShopAgent

A conversational shopping assistant — training project (Agentic Commerce Training).
Users browse the catalog, manage a cart and complete checkout entirely through
natural language: the catalog is exposed over **MCP**, cart and orders over a
**REST API**, and payment runs through **Stripe** with a webhook flipping the order
status to `paid`.

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
#                           long-term memory, which is not an error.
#   CURRENCY                the shop's currency, `eur` by default. Changing it
#                           after seeding means a reseed: `prices` rows carry
#                           the currency they were written with.

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
pytest tests/ -v          # 925 tests; add -m network for the ones that cost money
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
python -m shopagent.llm.loop            # CLI agent: 10 tools
MCP_CATALOG_ENABLED=false \
  python -m shopagent.llm.loop          # same CLI without the catalog: 7 tools

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

# tests
pytest tests/ -v                        # 925, offline and database
pytest tests/ -m network                # the 4 embedding tests and the 3 chain runs;
                                        # these cost money and need uvicorn running
pytest tests/ -m stripe                 # the 16 that call Stripe in test mode (free)
```

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
walkthrough are in `docs/screenshots/`.

The agent spawns the catalog server itself, so `run_mcp_server.py` is only needed
to drive the server by hand — from the Inspector, or to tell a catalog fault
apart from an agent fault. `MCP_CATALOG_ENABLED=false` is what makes "the product
answers come from MCP" demonstrable: the same binary then has two tools instead
of six and says the catalogue is unavailable.

Pass the server path, not `-m shopagent.mcp_server.server`, to the Inspector: it
parses `-m` as one of its own flags and the module never starts.

Findings, decisions and open questions from each day are in
[JOURNAL.md](JOURNAL.md).
