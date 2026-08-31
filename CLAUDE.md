# CLAUDE.md

ShopAgent is a conversational commerce agent — find products, manage a cart and
check out through Stripe, all in natural language. It is a ten-day training
project, built without an agent framework so the agent loop stays visible.

## Language

Everything committed to this repo is in English: code, comments, docstrings,
test names, CLI strings, system prompts, README, JOURNAL and commit messages.
The planning notes under `notes/` are the one exception, and they are not
tracked.

## Layout

| Path under `src/shopagent/` | Day | Purpose |
|---|---|---|
| `config.py` | D1 | pydantic-settings; the only reader of the environment |
| `db.py` | D3 | engine, session factory, pgvector extension; shared with D6 |
| `llm/client.py` | D1 | OpenAI wrapper; the only importer of `openai` |
| `llm/usage.py` | D1 | token and cost accounting per session |
| `llm/loop.py` | D1→D2 | the agent loop; grows through the week |
| `tools/registry.py` | D2 | `ToolSpec`, `ToolRegistry`, `ToolResult` |
| `tools/basic.py` | D2 | `get_time`, `calculator` |
| `tools/commerce.py` | D9 | cart and checkout, as tools, over HTTP |
| `tools/http.py` | D9 | the only importer of `httpx` under `tools/` |
| `money.py` | D7→D9 | minor units to something a person reads |
| `catalog/` | D3 | models, seed data, embeddings, search |
| `mcp_server/` | D4 | exposes `catalog/search.py` as MCP tools |
| `mcp_client/` | D5 | client, schema adapter, registration into the registry |
| `api/main.py` | D6 | the FastAPI app, CORS, `/health`, router mounts |
| `api/db.py` | D6 | the request-scoped session dependency |
| `api/deps.py` | D6 | `X-API-Key` authentication |
| `api/models.py` | D6 | carts and orders, on the catalog's `Base` |
| `api/lifecycle.py` | D6 | `OrderStatus` and the transitions between them |
| `api/schemas.py` | D6→D8 | request and response bodies; where names change |
| `api/routers/cart.py` | D6 | the cart endpoints |
| `api/routers/orders.py` | D6→D8 | orders, checkout, cancel, refund |
| `api/services/cart.py` | D6 | cart rules; the advisory stock check |
| `api/services/orders.py` | D6→D8 | `place_order`, `apply_transition`, cancel, refund |
| `payments/stripe_svc.py` | D7 | the Stripe SDK layer; the only importer of `stripe` |
| `payments/catalog_sync.py` | D7 | mirrors the catalog into Stripe; nothing bills from it |
| `payments/checkout.py` | D7 | Checkout Sessions built from the order snapshot |
| `payments/customers.py` | D7 | attaching a buyer to an order |
| `api/routers/checkout_pages.py` | D7 | the two pages Stripe redirects a browser back to |
| `api/routers/webhooks.py` | D8 | `POST /webhooks/stripe` — verify, claim, dispatch |
| `api/services/events.py` | D8 | idempotency and what each event type means |
| `agent/prompt.py` | D9 | the system prompt; the only file that says it |
| `agent/memory.py` | D9 | one conversation's state outside the message list |
| `agent/profile.py` | D9 | what the shop remembers about a customer |
| `agent/guardrails.py` | D9 | the confirmation gate and output validation |
| `obs/` | D10 | Langfuse tracing |

Search logic belongs in `catalog/`, never inside the MCP server. The server is
a thin wrapper, which is what lets D5 swap transports without touching logic.

## Conventions

**Configuration goes through `shopagent.config.get_settings()`.** `os.getenv`
and `os.environ` appear nowhere else. A new variable is added there as a typed
field, then to `.env.example`, and only then used. `MCP_CATALOG_ENABLED`
(D5, default true) is the catalog's off switch: false runs the same CLI with the
two local tools and no server, which is what makes "the product answers come
from MCP" a claim that can be demonstrated rather than asserted.

**`openai` is imported only in `llm/client.py`.** Changing provider then means
editing one file rather than hunting through the tree.

**Money is an `int` of minor units (`price_cents`), never a float.** Stripe
works in the smallest unit, so this avoids conversion entirely; a float here
turns into a rounding bug at checkout.

**`amount_cents` and `price_cents` are two names on purpose — do not unify
them.** `prices.amount_cents` is a database column: one row per currency per
variant, with an `active` flag, so a price that was once charged stays
readable — at most one of them active at a time, which a partial unique index
on `(variant_id, currency) WHERE active` enforces, because a second active row
in one currency reaches the model as the same sku twice at two prices.
`price_cents` is a field in the flattened result the search functions return
and the model reads — one number, already resolved to the active price
in the session currency. The rename happens exactly where the layer changes.
Making them the same word would hide that boundary, and the first symptom would
be a tool handing the model a row that has three prices attached.

**The catalog is seed-generated and disposable, so there is no Alembic.**
`catalog/seed.py` is the source of truth for every product row, not the
database. A schema change is therefore drop the tables, `create_all`, reseed —
never a migration, because nothing of value lives in those four tables that the
seed cannot rebuild. This holds for `products`, `variants`, `prices` and
`inventory` only. The `carts` and `orders` tables arriving on D6 hold what real
users did, which no script can regenerate; the moment they exist, altering them
needs a migration path and this rule stops applying to them. Concretely:
`scripts/seed_catalog.py --reset` issues `DELETE FROM products` and lets the
cascade clear the rest, which is safe only while nothing outside the catalog
points at a variant. D6 paid that debt — see the next rule.

**`carts`, `cart_items`, `orders` and `order_items` are not seed data.** They
hold what a shopper actually did, and no script regenerates them, so the rule
above stops at the catalog's four tables. From the first real order, changing
this half of the schema needs a migration path — `create_all` is still the
mechanism only because these tables are young, not because dropping them is
ever acceptable.

**Schema changes to those tables go in `migrations/`, as numbered idempotent
SQL.** "No Alembic" was a rule about the catalog, and for a long time it was
the only rule there was — which left changing a commerce table with a
prohibition and no replacement. The gap looked like a decision, so D7's two
`orders` columns were added with an `ALTER` typed into a terminal and reported
in conversation, and nothing in the repository recorded that a database built
before them needed anything at all.

The convention, which D8's `processed_events` is the next to need:

- **One numbered file per change**, `migrations/NNNN_short_description.sql`.
  The number is the order they were written, not a version anything reads.
- **Every statement is idempotent**: `ADD COLUMN IF NOT EXISTS`,
  `CREATE TABLE IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`. Re-running a
  migration must be a no-op, not an error.
- **Applies to `carts`, `cart_items`, `orders`, `order_items` and every table
  added later that holds real data.** The catalog's four keep their own rule —
  drop, `create_all`, reseed — because `catalog/seed.py` can rebuild them and
  nothing of a customer's is in there.

Applied with, from the repository root:

```bash
docker compose exec -T db psql -U shopagent -d shopagent \
  -f /dev/stdin < migrations/0001_d7_order_customer_columns.sql
```

**The repository does not track which migrations have run, on purpose.** A
migrations table is a second record of the schema, and the failure it produces
is the one this whole area exists to prevent: the ledger says a migration was
applied while the column is not there, and every check that trusts the ledger
agrees. Idempotent SQL plus drift detection covers the same ground with no such
state — `scripts/create_schema.py` asks the *database* what it has, compares it
against the models, exits **2** on any missing column or mismatched foreign
key, and can be run at any time by anyone. Re-running every migration in order
is always safe, which is what makes the ledger unnecessary rather than merely
absent.

That trade has an edge, and it is worth knowing where: it holds only while
migrations are additive. The day one has to backfill or transform existing rows,
`IF NOT EXISTS` stops being able to express "already done", and a record of what
ran becomes the only way to know. That is the moment to introduce Alembic — not
before, and not by hand.

`create_all` remains the mechanism for creating tables that do not exist yet.
It never alters one that does, which is why it reports a table as "already
present" while that table is missing the column added last week, and why the
first symptom without the check above is `UndefinedColumn` from an ordinary
read.

That is enforced mechanically rather than remembered. `order_items.variant_id`
is `ON DELETE RESTRICT`, so Postgres refuses `DELETE FROM products` while any
order line points into the catalog — against any client, psql included.
`assert_no_orders()` in `api/models.py` is the courtesy in front of it:
`scripts/seed_catalog.py --reset` calls it and stops with a sentence naming how
many orders are in the way, instead of an `IntegrityError` naming a constraint.
`cart_items.variant_id` is `ON DELETE CASCADE` on purpose — the opposite answer
to the same question, because a product that no longer exists cannot be bought
and has no business sitting in a basket.

**`routers/` speaks HTTP, `services/` speaks the domain, and `services/`
imports no FastAPI.** A router parses, calls one service function and maps a
domain exception to a status code; it holds no rule about what a cart may
contain. The reason is not tidiness: D8's Stripe webhook and D9's agent tools
call `place_order` and `add_item` outside any request, where an
`HTTPException` would have nobody to catch it and a 500 would appear where the
log should have said what was refused. `api/lifecycle.py` follows the same rule
and goes further, taking anything with a `status` attribute rather than an ORM
row — which is what lets the whole transition table be swept offline.

**`lifecycle.transition()` is the only way an order's status changes.**
Assigning `order.status` directly bypasses the table of allowed transitions,
and the two absences in that table are the ones that matter: `paid → pending`
and `paid → cancelled`. Once a charge settles the only way back is `refunded`,
which is a movement of money and a status of its own, and refusing `paid →
paid` is what makes D8 idempotent when Stripe delivers the same event twice.

Status columns are `Enum(..., native_enum=False, create_constraint=True)` —
VARCHAR plus a CHECK, never a Postgres enum type. A native enum changes with
`ALTER TYPE`, which `create_all` never issues, and D8 adding a status is a
realistic week rather than a hypothetical one. `create_constraint=True` is
spelled out because SQLAlchemy 2.0 defaults it to `False`, and a bare VARCHAR
that accepts any string at all is the failure that looks like it works.

**Authentication is attached to the router mount, never to a route.** Routers
are included with `dependencies=[Depends(require_api_key)]`, so a route added
later is protected by where it lives rather than by whoever wrote it
remembering a decorator. A missing key is **401, not 403**: 403 is the answer
to a known caller who may not do this, and it tells a client that retrying with
a credential is pointless. That is why `deps.py` declares
`APIKeyHeader(auto_error=False)` and raises by hand — `auto_error=True` answers
403 to a missing header — and declaring the scheme is also what puts the
Authorize button in `/docs`. The comparison is `secrets.compare_digest`, never
`==`, because string equality returns at the first differing byte and how long
that takes measures how much of the key was right.

An unusable key kills the process at import: `min_length=1` on the setting
rejects a blank `.env` value, `configured_api_key()` rejects a whitespace-only
one that passes a length check and authenticates nobody, and `api/main.py`
calls it at module scope so uvicorn dies while loading rather than starting and
refusing every request afterwards. `/health` is unauthenticated and touches no
database — a liveness probe that queries reports the database's latency as the
process's liveness, so a slow Postgres reads as a dead API and the wrong thing
gets restarted.

**The rename from `amount_cents` to `price_cents` happens in
`api/schemas.py`.** `prices.amount_cents` and `order_items.unit_amount_cents`
are database columns; `unit_price_cents` and `total_cents` are what a reader
gets — one number, already resolved to the active price in the session
currency. Never copy a column name through to a response because it was
convenient; that is the boundary the two names exist to mark.

A cart total is computed from the database on every read, never from a request
and never cached on the cart, because a stored total is a number that is right
until a price changes. An order's total is the opposite: written once from the
snapshot and never recomputed.

**`order_items` snapshots enough to render an order with no join to the
catalog.** `sku`, `product_name`, `variant_label`, `unit_amount_cents`,
`currency` and `quantity` are copied at order time, because prices change and
products get renamed while an order is a record of an event rather than a view
over current data. This is asserted by recording the SQL `render_order`
actually issues and failing if any of the four catalog tables appears — not by
checking that the response fields came back populated, which an implementation
that joins the catalog would satisfy perfectly right up to the day a row
changes. The row locks in `place_order` are tested the same way and for the
same reason: `FOR UPDATE` and its `ORDER BY variant_id` are invisible in
single-threaded behaviour, so an implementation that dropped them would pass
every behavioural test and fail only in production.

**`inventory.reserved` rises when an order is placed; `quantity` is never
decremented.** Units leave `quantity` when they physically ship, and this
project has no fulfilment flow — `fulfilled` is a status nothing transitions
into automatically, so there is no moment at which decrementing would be
correct. Available stock is `quantity - reserved` everywhere, so a reservation
already makes the units unsellable. The *stock* check in `services/cart.py` is
advisory and says so in its docstring: no lock, no write, and two requests can
both be told there is room. The authoritative check is `place_order`'s, under
`SELECT ... FOR UPDATE` in the transaction that writes `reserved`.

The cart *row*, by contrast, is locked on every write. `add_item` and
`remove_item` take `FOR UPDATE` on `carts` before reading the status, because
`place_order` locks the same row, snapshots the items it finds and flips the
status to `ordered` — and an unlocked add can read `open`, be descheduled, and
commit its line after that snapshot, leaving an ordered cart holding an item on
no order. `render_cart` deliberately does not lock: a read that took a write
lock would queue every `GET /cart` behind an in-flight checkout for nothing.
Raised by review on PR #6.

**A table joins `Base.metadata` when its class is imported, so both model
modules have to be.** `api/models.py` registers on the same `Base` as
`catalog/models.py` — one base, one `create_all`, one schema. The consequence
is that `scripts/create_schema.py` and `tests/conftest.py` both import
`shopagent.api.models` for the side effect alone: importing `Base` by itself
builds the catalog's four tables and none of the commerce ones, which is a
schema that looks complete until the first missing relation.

**Ids that leave the process are UUIDs.** `carts`, `cart_items`, `orders` and
`order_items` have `UUID` primary keys, while the catalog keeps its integers.
The difference is exposure: an order id reaches the model in conversation and
travels to Stripe as `metadata.order_id` on D7, and a serial integer in that
position invites a shopper to try the neighbouring number. A UUID also exists
before the row is flushed, which is what makes it safe to put in a response.

**`payments/` imports no FastAPI, and splits the SDK from the decisions.**
`stripe_svc.py` is the only importer of `stripe` and holds nothing but calls —
which is what lets every layer above it be tested by replacing one name.
`checkout.py`, `catalog_sync.py` and `customers.py` hold the rules. Same
reason `api/services/` exists: D8's webhook and D9's agent tools reach payments
outside any HTTP request, where an `HTTPException` would have nobody to catch
it.

**`line_items` are built from the `order_items` snapshot, never from
`stripe_price_id`.** This is the most important rule of D7. `catalog_sync.py`
writes a Stripe Price for every variant and the checkout deliberately does not
read one: `order_items` froze the price at order time — D6 proves that by
recording the SQL and failing if the catalog is touched — while a Stripe Price
is a separate object that a re-sync, a dashboard edit or a local price change
can move. Charging from it would mean the shopper pays Stripe's number while
`orders.total_amount_cents` claims another, and the two would diverge silently.
`price_data` carries the snapshot into the session instead, so there is exactly
one number. It is checked rather than trusted: `build_line_items` refuses to
return lines whose sum is not the order's total, and nothing is charged.

**The catalog sync is an isolated deliverable and no checkout depends on it.**
`scripts/sync_stripe_catalog.py` exists so the catalog is visible in the
dashboard and so the Products/Prices API is exercised. A stale or missing
Stripe object therefore cannot produce a wrong charge, which is what makes it
acceptable for the script to report price drift rather than repair it — a
Stripe Price is immutable, so repairing means creating a replacement and
archiving the original, and a script that silently retires objects in somebody
else's account is one nobody can reason about.

**Idempotency uses two different mechanisms for two different windows.** A
Stripe idempotency key covers 24 hours and protects a script that died between
creating an object and storing its id; it is derived from the local row, never
random, and the amount is part of a Price's key because a Price is immutable
and a different amount is a different object. A stored id — `stripe_product_id`,
`stripe_price_id`, `orders.stripe_checkout_session_id` — is durable and is what
makes a second run free and lets a shopper return the next day to the session
they left. Neither substitutes for the other.

**`metadata.order_id` is mandatory on every Checkout Session, and it is sent
twice because Stripe does not propagate it.** A session's `metadata` stays on
the session: verified against a real payment, the PaymentIntent and Charge it
produced both came back with `metadata: {}`. So the checkout also passes
`payment_intent_data={"metadata": {"order_id": ...}}`, and a second payment
confirmed the identifier then reaches the PaymentIntent *and* its Charge.

The distinction matters when reading this code: **nothing propagates
automatically, and the explicit copy is why all three objects carry it.** D8
took that up — `services/events.py::order_id_from` reads `order_id` the same
way off a Session, a PaymentIntent and a Charge, and `charge.refunded` carries
nothing else to attribute a refund by. It works only for as long as that
parameter keeps being sent, which is what the offline test on the outgoing
payload exists to guard. No PaymentIntent exists
until a shopper starts paying, so nothing on an unpaid session can prove the
copy arrived.

This is also why the project uses a Checkout Session rather than a Payment
Link — a Payment Link is a durable URL bound to a Price, reusable by anyone
holding it and carrying no per-order metadata at all.

**`lifecycle.transition()` returns `TransitionEffects` and is called only
through `api/services/orders.py::apply_transition`.** The function stays pure
and touches no database, which is what keeps the whole transition table
sweepable offline; the price of that purity is that acting on the result is an
obligation on the caller, and an obligation spread across callers is one D8's
webhook will eventually forget — leaving stock reserved against an order that
will never ship. Concentrating it in one service function turns forgetting into
"did not call the service", which is visible.
`tests/test_lifecycle.py` walks the AST of `src/shopagent` and fails if
anything else calls `transition()`.

Releasing is the mirror of reserving and runs under the same `SELECT ... FOR
UPDATE` ordered by `variant_id`. Releasing twice would hand back units the
order never held; what prevents it is the transition table, not a check inside
the release — `cancelled` and `refunded` are terminal, so a second attempt is
refused before any stock moves.

**Test mode is the only mode.** `config.py` rejects a `STRIPE_SECRET_KEY`
beginning `sk_live_` or `rk_live_` at configuration time, which is the last
moment the mistake is free. `in_test_mode()` is the second layer and reads
`livemode` from Stripe itself, because a prefix is a string this repo compares
and `livemode` is not. It reads the balance rather than the account: `GET
/v1/account` returns no `livemode` field.

**`STRIPE_API_VERSION` is pinned and must be re-pinned deliberately.** Stripe
advances the default per account and per signup date, so an unpinned client
returns a differently shaped object one morning with nothing in this repo
having changed. A test asserts the pin equals `stripe.api_version`, so
upgrading the SDK fails until somebody has read the changelog — note that
comparing the client's effective version against the constant proves nothing on
its own, because an unpinned client currently resolves to the same string.

**A missing Stripe key is not a startup failure.** Unlike `SHOPAGENT_API_KEY`,
which gates every request, payments are one part of the system: a cart that
cannot be browsed because Stripe is unconfigured would be the wrong failure.
`get_client()` raises `MissingStripeKey` at the moment something needs it, and
the checkout route maps that to **503** — the capability is absent, the server
is not broken.

**A catalog drop removes the commerce tables' foreign keys, and `create_all`
does not restore them.** `DROP TABLE ... CASCADE` on `products` also drops
`order_items_variant_id_fkey` — the `ON DELETE RESTRICT` that stops a reset
taking order history with it — because the constraint lives on `order_items`,
which `create_all` will not touch since it already exists. The guard in
`seed_catalog.py` stays in place, so the protection looks intact from the one
direction anybody checks. **The documented "drop, `create_all`, reseed" path is
therefore incomplete: run `scripts/create_schema.py` afterwards and check its
exit code**, which is 2 when a foreign key is missing or has the wrong delete
action. `tests/test_schema_constraints.py` asserts the same thing from
`pg_constraint` with a hand-written table of expectations, because a check
derived from the models passes whenever the models and the database are wrong
together.

**A stored vector does not record which model made it.** `products.embedding`
is 1536 floats and nothing else — no model name, no timestamp. Changing
`EMBEDDING_MODEL` therefore means drop, reseed and re-embed, not a migration,
which is the same rule the catalog already lives under. Note that
`scripts/embed_catalog.py --force` does **not** detect a changed model: it
re-embeds what it is pointed at, so running it after a model change over a
catalog that was only partly re-embedded leaves two models' vectors in one
column, silently comparable and quietly wrong. A dimension change fails loudly
instead, because the column type is `VECTOR(1536)`.

**The webhook handler takes no body parameter, and never may.** FastAPI reads
a request body once and hands a declared parameter the *parsed* result; a
Stripe signature is an HMAC over the bytes that arrived. Verifying against a
re-serialised body is dangerous precisely because it mostly works —
`json.dumps` reproduces many payloads exactly — so it would pass every test,
every `stripe listen` session and every demo, and then fail intermittently on
whichever real event carried a float, a non-ASCII character or a key order the
encoder did not preserve. The handler therefore takes `Request` and reads the
raw bytes itself — through `read_capped_body`, never `await request.body()`,
for the size reason the next rule gives.
`tests/test_webhooks.py` fails if a parameter is added:
once by name, and once on FastAPI's own `route.body_field`, which is the
assertion with teeth. Dependencies are allowed — `Depends(get_session)` is not
a body — and the guard checks for that rather than for an exact parameter list.

**It reads the stream under a cap, not `await request.body()`.** The signature
*is* the credential on this route, and it cannot be checked until the body has
arrived — so everything read before that point was sent by somebody holding
nothing. `request.body()` buffers the whole delivery before the check can
happen, which turns "send a very large body" into memory and HMAC work an
anonymous caller can ask for. `read_capped_body` reads `request.stream()` and
gives up the moment the total passes `MAX_WEBHOOK_BODY_BYTES`, answering
**413**. `Content-Length` is consulted first only as a fast path: it is a
header the sender writes and a chunked request need not carry one, so it can
shorten the work but never enforces the limit — there is a test that lies in
that header and is still refused. 256 KiB against a measurement rather than a
guess: the largest of the last hundred events on this account is 4,145 bytes.
A constant rather than a setting, by the configuration rule above — a limit
somebody can raise from a `.env` is a limit that stops holding on the day it
matters. Raised in review on PR #8.

**Insert first, then work — never check-then-act.** The obvious shape is to
look the event id up and do the work if it is absent, and two concurrent
redeliveries both read "absent" before either writes. Retries are the condition
this table exists for, so that race is the normal operating condition rather
than a corner case. `services/events.py::record_event` inserts into
`processed_events` and lets the primary key arbitrate; a duplicate comes back
as `False`, not as an exception the caller has to interpret.

The insert is wrapped in `session.begin_nested()`, and that is load-bearing. A
unique violation aborts the *entire* Postgres transaction — the next statement
on that connection returns `InFailedSqlTransaction` and SQLAlchemy raises
`PendingRollbackError` for everything after it. A duplicate is the one outcome
this code expects to continue past, so without a SAVEPOINT the failure would
not appear on the duplicate at all: it would appear on whatever the request
touched next.

**The claim and the work are one transaction.** `process_event` runs inside the
transaction holding that event's `processed_events` row, and the router rolls
both back together on any exception. A row that outlived a failed handler would
tell Stripe's retry — the mechanism that exists to recover from exactly this —
that the event was already handled, and the payment would silently never land.
The rollback is in the handler rather than left to `get_session`, because the
invariant belongs to the code that owns both halves.

**Three independent layers make this idempotent, and they catch different
things.** Proven separately against a live account rather than argued:

- **`processed_events` primary key** catches the *same event id* arriving
  twice — Stripe's own redelivery. The handler is never entered.
- **The transition table** catches the *same work* arriving under a different
  event id: a manual replay, or two event types describing one payment.
  `paid -> paid` is not in the table, so it is refused after the claim
  succeeded. This is why `lifecycle.py` refuses a status transitioning to
  itself instead of absorbing it as a no-op.
- **The `SELECT ... FOR UPDATE` inside `apply_transition`** catches two
  transitions racing on one order, which neither of the above can see: both
  hold distinct event ids and both read a legal starting status.

None of the three subsumes another, and removing any one leaves a real hole.

**500 means retry, 200 means stop, and the choice is about Stripe not about
us.** Stripe redelivers anything that is not 2xx, with backoff, for three days.
So a *transient* failure — the database is unreachable, Stripe timed out,
anything unexpected — propagates and becomes a 500, which is what makes the
retry useful. A *permanent* one — an event type nothing handles, an order id
absent from the metadata, an order that does not exist here, a transition the
lifecycle refuses — is logged and answered **200**, because no redelivery
improves it and three days of retries would only bury the log line that
explains it. `handle_event` catches almost nothing, which is how that split
stays honest.

**An order's status changes only through a webhook.** No endpoint that calls
Stripe also writes the status it expects to result. `POST /orders/{id}/refund`
issues the refund and answers **202**: the refund has been accepted, not
completed. Measured on a real card, the order stayed `paid` for between 1.1
and 2.1 seconds after the 202 before `charge.refunded` arrived — and on other
payment methods a refund can be pending for days. The response names the
unchanged `order_status` for that reason: it is the field a caller would
otherwise assume had moved. This is the same rule the success redirect
follows, and it means a refund issued from the Stripe dashboard lands in
exactly the same place as one issued here.

**`checkout.session.expired` is the only path where stock is released without
a person, so it is the most defensive handler.** `cancelled` is terminal, and
getting this wrong is unrecoverable in both directions at once: the money is
real and the reservation is gone. Two guards, neither optional. The event is
not trusted about payment — the session is fetched from Stripe and
`payment_status` read from the answer, because Stripe can expire a session
whose payment is in flight and delivery order is not guaranteed. And the
event's session must be the one the order currently points at, or an order
that started a second checkout would be cancelled by the first session's
expiry while the shopper is on the new payment page. The comparison is an
allow-list: only `unpaid` may cancel, because `payment_status` has a third
value (`no_payment_required`) and `!= "unpaid"` and `== "paid"` differ exactly
there.

**The session comparison happens twice, and only the second one decides.** The
first runs before the Stripe call and exists to skip a round trip. The
authoritative one runs after it, on a row re-read through
`orders.lock_order` — `SELECT ... FOR UPDATE` with `populate_existing` — and
is held until the commit. The reason is the gap the network call opens: a
retrieve takes long enough for a shopper to start a second checkout, and
`_reusable_session` will happily build one because the first session is
expired. Cancelling on the earlier read then releases the stock of an order
whose new payment page is open and chargeable. It is the same defect shape as
the transition preflight — an unlocked read deciding a write — and it was found
the same way, in review on PR #8.

`populate_existing` on that re-read is load-bearing rather than defensive:
`_load_order` has already put this order in the Session's identity map, so
without it the locked select returns the instance with the attributes it was
loaded with, and the whole point of waiting for the lock is lost.
`test_a_second_checkout_started_during_the_stripe_call_stops_the_expiry` fails
without it, which is how that was established rather than assumed.

**The session that paid is not always the one the order points at, and the
other one has to be closed.** That guard has a consequence the guard itself
creates. An order whose session expired with a payment still in flight is
correctly left `pending` — Stripe reports it as not `unpaid`, so nothing is
cancelled — and the shopper who returns gets a *second* Checkout Session,
because `_reusable_session` sees an expired one and makes a new one. When the
first session finally completes, the order is genuinely paid while the second
session is open and chargeable, and a shopper still looking at that payment
page pays for the same order twice. `_reconcile_paying_session` therefore
expires the session the order points at, records the one that actually paid,
and only then moves the order. Raised in the second review round on PR #8.

Three things about it are deliberate. It runs behind `_may_become`, so an
order that cannot become paid never causes somebody else's session to be
expired on the way to being told so. The catch around the expiry is narrow —
`stripe_svc.InvalidRequestError` only, which is Stripe refusing to expire a
session that is not open; a connection failure is not that and must keep
propagating into a 500 so the delivery is redelivered. And when the expiry is
refused the payment is still accepted, because the money that arrived is real:
the alternative to a possible double charge is a definite unpaid order. That
case is the one thing here no code can fix, so it is logged at ERROR naming
both sessions, and a person refunds it.

**A refund is attributed by PaymentIntent, not by `metadata.order_id`.**
`order_id` says which order a charge is *about*; it does not say the charge is
the payment that order recorded, and the two come apart in exactly the case
this file already documents — a superseded Checkout Session that was paid
anyway leaves two Charges carrying the same `order_id`. Refunding the
duplicate is what a person reconciling does first, and its own `amount` and
`amount_refunded` balance perfectly, so without a check it reads as a full
refund of the order: terminal status, whole reservation released, and the
recorded payment still charged. `_charge_belongs_to` compares
`charge.payment_intent` against `orders.stripe_payment_intent_id` and refuses
anything else, including an order with no PaymentIntent recorded — `refunded`
is terminal and releases stock, so an attribution that cannot be verified is
not one to act on. Raised in review on PR #8.

The fixture is the other half of that story. `refunded_event` in the tests did
not carry `payment_intent` at all, so no test could have noticed the check was
missing; a fixture that omits a field the real object always has is a blind
spot with the shape of coverage.

**A zero-total order is paid with nothing to refund, and that is not a
defect.** `total_amount_cents >= 0` and `prices.amount_cents >= 0` both permit
zero, and Stripe settles such a checkout as `no_payment_required`, which
`SETTLED_PAYMENT_STATUSES` accepts deliberately — so the order becomes `paid`
with no PaymentIntent, legitimately. `refund_order` used to call that state
impossible and tell the caller to refund a payment in the dashboard that never
existed. It is a plain 409 now, and the ERROR is kept for a positive total,
where the same state really does mean a bug here. Raised in review on PR #8:
this codebase allowed the order to exist and then declared it unreachable.

**Only a full refund moves an order to `refunded`.** `charge.refunded` fires
for a partial refund too — measured against one real charge refunded twice:

    partial   amount=18998  amount_refunded=100    refunded=False
    full      amount=18998  amount_refunded=18998  refunded=True

The event type says nothing about completeness. A partially refunded order is
not finished — the shopper still has the goods — and there is no status
between `paid` and `refunded`, so acting on one would drive a terminal status
and hand back the whole reservation for a fraction of the money: $1 returned
on a $190 order would free every unit. So a partial refund changes nothing and
is logged at **ERROR**, because this system genuinely cannot represent what
happened and that line is the only record. The decision is arithmetic
(`amount_refunded >= amount`) with the `refunded` flag as a cross-check:
numbers cannot be quietly redefined, whereas a flag can be deprecated into
absence and `getattr` on an absent flag would silently mean "never full". When
the two disagree, nothing moves.

**`checkout.session.completed` does not mean the money arrived.** For
delayed-notification payment methods Stripe sends it with
`payment_status="unpaid"` and settles later through
`checkout.session.async_payment_succeeded`, or fails through
`checkout.session.async_payment_failed`. Nothing in `payments/checkout.py`
restricts `payment_method_types`, so which methods are offered is a dashboard
setting this code does not control — reading `payment_status` removes the
assumption instead of documenting it. The allow-list is `paid` and
`no_payment_required`, the same shape the expiry guard uses and for the same
reason. Both async events are handled, because a guard without them would
strand such an order at `pending` for ever: `completed` has already been and
gone by the time the funds confirm.

**`checkout.session.async_payment_failed` cancels the order, and a declined
card does not.** The two look like the same event and leave the session in
opposite states. `payment_intent.payment_failed` happens while the session is
still `open`: the shopper can try another card, and if they never do, the
session expires and `checkout.session.expired` cancels the order — so logging
and doing nothing is complete. The async failure only ever arrives *after*
`checkout.session.completed`, so the session is `complete`, and a complete
session never expires while `_reusable_session` in `payments/checkout.py`
refuses to start a new checkout for an order holding one. Logging and doing
nothing there is not a lighter answer but a permanent leak: no retry path, no
release path, the order and its reserved units stuck for ever. Raised in
review on PR #8.

The policy is therefore explicit — the payment is over, so the order is
cancelled and the units go back on sale. It is the second path where stock is
released without a person, and it asks Stripe nothing, unlike the expiry
handler. Two things make that safe. The event type is itself Stripe's verdict
on the payment, where `expired` says nothing about money and is exactly why
that handler cannot trust it. And if the money did arrive the order is already
`paid`, so `paid -> cancelled` is refused by the transition table before any
stock moves — the guard is the table rather than a check somebody has to
remember. The session-identity guard is the same one the expiry handler uses,
for the same reason.

**A live-mode event is recorded and never dispatched.** `config.py` refuses an
`sk_live_` key, which reads like "live events cannot reach this code" and does
not cover this path at all: `STRIPE_WEBHOOK_SECRET` is a separate credential
and a live endpoint's signing secret begins `whsec_` exactly like a test one,
so it verifies just as well. `handle_event` therefore stops on
`event.livemode`, logs at ERROR and returns — 200, because the delivery is
genuine and the configuration is what has to change. A guard's coverage is a
property of the path it sits on, not of the sentence describing it.

**A column that belongs to a transition is written inside `apply_transition`,
after the locked check.** `handle_checkout_completed` records
`stripe_payment_intent_id`, and the damaging case is concrete: a second
`checkout.session.completed` for an order already `paid` overwriting the
PaymentIntent the refund endpoint spends, with one from a session that may
never have been charged. Two rounds of review on PR #8 took two attempts at
the same mistake, and the second is the instructive one.

The obvious shape — assign the column, then call `_move`, which checks the
transition first — is wrong because the assignment has already happened: a
refusal returns without undoing it, and the router's `commit()` writes it to
an order whose status never changed. Moving the assignment into `_move`, in
front of its own check, fixed that and left a subtler version standing.
`_move`'s check is unlocked and advisory; the authoritative one is inside
`apply_transition`, behind `SELECT ... FOR UPDATE`. An attribute assigned
before that statement runs is dirty on the Session, **SQLAlchemy autoflushes
it as part of issuing the select**, and the `UPDATE` is therefore already in
the transaction by the time the locked check refuses. `session.expire()`
afterwards drops the attribute and not the statement, so the router commits
it. Only an assignment made *after* the locked check cannot outlive a refusal,
which is why `updates` travel through `_move` to `apply_transition` and are
applied there.

The general form is worth stating, because it is not specific to this column:
**any write made before an autoflushing read is already in the transaction,
whatever the code does with the attribute afterwards.** A check that runs
between them decides nothing about what commits.
`test_a_transition_refused_under_the_lock_writes_no_column_either` simulates
the race by neutering the preflight, which is exactly the state a caller is in
when a concurrent delivery moved the order between the two checks.

The lost-race branch inside `_move` still expires the instance rather than
rolling the session back, now for one reason instead of two: it drops the
status this handler read before the wait. A rollback there discarded this
event's `processed_events` claim while the handler still returned normally and
the router still answered 200 — the delivery recorded nowhere and Stripe told
to stop, which are the two things that must never both be true.

**`apply_transition` asks for `populate_existing` under its lock.** The order
is usually already in the Session's identity map, and an ORM select that
returns an instance it already holds is not obliged to overwrite attributes
loaded earlier. If it did not, a caller that waited behind a concurrent
transition would evaluate the status from *before* the wait and two refunds
could each release the same reservation. SQLAlchemy 2.0.52 does refresh under
`with_for_update()`; asking explicitly turns that from an observation into part
of the statement, and
`test_two_concurrent_transitions_release_the_reservation_once` is what would
catch it changing — two connections and one order, rather than a test that
reads the emitted SQL for `FOR UPDATE`.

**`payment_intent.succeeded` deliberately changes nothing.** One payment
produces it *and* `checkout.session.completed`, in an order nobody controls.
If both drove the transition, whichever lost would be refused by the
transition table on every successful payment — a permanent stream of warnings
describing the system working. `checkout.session.completed` is the primary
because it says the *checkout* finished rather than that a charge settled, it
carries the session this project created, and its `payment_status` is what the
expiry guard reads. It is kept as a handler rather than left to the
unknown-type branch so the log distinguishes "nothing happened because nothing
handles this" from "nothing happened because nothing should".

**Editing an applied migration reaches no database that already ran it.**
`0002_d8_processed_events.sql` is the convention's first *table* rather than
its first columns, and that difference makes the file easy to think
unnecessary: `create_all` checks tables one at a time and builds any it does
not find, so a fresh clone gets `processed_events` without running anything —
and so would a database created before D8. This document previously claimed
`create_all` would leave such a database alone, which is wrong and was
corrected in review on PR #8. The migration's justification is not that
nothing else can create the table; it is that the repository has to *record*
what changed. `create_all` is a script that will build whatever it finds
missing without saying so, and reading it tells nobody which change a
deployment needs.
The consequence for later: adding a fifth column to `processed_events` means
`0003` with `ADD COLUMN IF NOT EXISTS`, **not** an edit to `0002`. Idempotency
makes re-running safe; it does not make a rewrite retroactive.

**The system prompt lives in `agent/prompt.py`, never in `llm/loop.py`.** The
loop is a mechanism — a `while`, a message list, a dispatch — and it has been
byte-stable since D2 on purpose, which is a claim D5 and D9 both leaned on when
they changed where tools come from. What the assistant is told is policy, it is
the part of the system most likely to be edited, and editing a sentence about
quoting a price should not touch the file whose stability is an argument.
`initial_messages()` assembles four blocks: role, catalog-or-not, cart, money.

Two things are kept out of it by test rather than by intention. Nothing about
confirming a checkout — that is a gate in code, which can see *who* said yes
where an instruction cannot, and a prompt that got there first would leave
nobody able to say which of the two was stopping a purchase. Nothing that
restates a tool's failure message either: `tools/commerce.py` writes those
against measured failures, and a paraphrase here would be the copy that goes
stale.

The one exception carved into the D1 rule is money. `never do arithmetic in
your head` is from D2, where the model answered `5 factorial` from memory; but
every amount in this system arrives as an integer number of minor units, and
turning 9499 into $94.99 is arithmetic somebody has to do. A base rule the
model must break to answer at all is a rule it stops reading, so the conversion
is named as the only one permitted and producing a *new* amount — a total, a
difference, a comparison — is refused outright.

**A diagnostic tool is not advertised to the model.** `ping` was in the
catalog server's `tools/list` from D5 to D9 with no commercial meaning, and the
cost was never that the model called it — it never did. The tool list is what
the model reads to work out what it can do, so a name in it that means nothing
is one it must rule out on every turn. `MCP_EXPOSE_PING` (default false)
decides one `add_tool` call in `mcp_server/server.py`; the tool itself is
unchanged and still separates "the server is unreachable" from "the catalog is
broken" for whoever is debugging.

The fix belongs on the server and nowhere else. Filtering by name in
`mcp_client/` would make this project's client know about this project's
server, and registering whatever a server lists is the property D5 exists to
demonstrate. What actually let this sit for four days is that no test said what
the tool list *was*: every test named the tools it cared about. There are now
two that name the whole set, offline against a fake catalog client and against
the real server under `db`, so the next unintended publication fails rather
than quietly costing the model a decision.

**`agent/` is policy and `llm/loop.py` is mechanism, and the split is a rule
rather than an accident.** `run_tool_loop` has been byte-stable since D2 — a
`while`, a message list, a dispatch — and that stability is an argument two
days have leaned on: D5 changed where tools come from and D9 changed what the
model is told, what it remembers and what it is allowed to say, and neither
touched it. Everything D9 added went in around it instead: the prompt into
`agent/prompt.py`, the memory into a registry subclass, the output validation
into a client wrapper. `tests/test_loop_mcp.py` pins the signature at
`(client, registry, messages, tools)`, and the day something has to be added
there, the honest move is to say so rather than to widen the function quietly.

**Matching on a tool's name is forbidden in `mcp_client/` and correct in
`agent/`.** D5 refused to filter `ping` out of the tool list in the client,
because registering whatever a server lists is the property that module exists
to demonstrate — a client that knows this project's server is not a client. D9
fixed the same problem on the server, which is where the entry said it
belonged.

`agent/memory.py` then names `search_products` and `agent/guardrails.py` names
`create_checkout`, `view_cart` and `add_to_cart`, and that is not the same
thing. `agent/` is the layer whose job is to know what the tools *mean*: it
decides what to tell the model they are for, which of them spends money, and
what a bad answer looks like. A layer forbidden to name a tool could do none of
that. The rule is about `mcp_client/`, not about the word.

**A guardrail is code. An instruction is not a guardrail, and D9 measured why.**
`create_checkout`'s description used to say "get an explicit yes first". The
customer said "yes, order it"; the model showed the cart and asked for the yes
again, because the sentence never said the previous message could be that yes.
An instruction that states a precondition without stating how it is satisfied
cannot be satisfied — it is not weak, it is unsatisfiable. The sentence is
gone and `agent/guardrails.py` intercepts the call instead, shows a person what
they are buying, and asks. There is no `confirmed` argument on purpose: an
argument the model sets is a suggestion with a type annotation.

The same reasoning closed D2's older debt. The prompt said never do arithmetic
in your head; the model did it anyway and was right, silently. So an amount in
an answer is now checked against the amounts tools produced, with one retry and
then a fallback that names the figure it could not trace.

**The gate binds the model, not the shop, and the boundary is deliberate.**
`tools/commerce.py` holds plain functions anybody can import and the commerce
API answers any client holding the key, so a person with `curl` can place an
order without passing through any of this. That is correct: the gate exists
because a model can be talked into spending money, not because HTTP is
dangerous. The protections that bind *everyone* are elsewhere and unchanged —
`place_order` locks inventory under `FOR UPDATE`, the lifecycle table refuses
illegal transitions, and only a signed webhook may mark an order paid.

**The model never sees a `cart_id`.** It appears in no tool schema and in no
tool result; the tool layer holds it and the tools take what a shopper would
actually say. An identifier the model has to carry across turns is one it will
eventually lose or invent, and the whole class of failure disappears if it is
never handed one. Both halves are asserted — the schemas and the results —
because a leak in a result is what the model builds its next call from. An
order id is the exception and is returned: it is the reference a customer needs
for support, and no tool accepts one back, so there is no argument to get wrong.

**A conversation's memory holds two things with two lifetimes, and merging them
would be wrong in one direction or the other.** `last_search` is *replaced* by
every new search: "the second one" can only mean the second row of the list the
customer is looking at now, and resolving it against a list that has scrolled
away puts the wrong item in a basket silently. `seen_variant_ids` and
`seen_amount_cents` *accumulate* for the whole conversation: they answer "was
this ever put in front of the model here?", and a price quoted four messages
ago is still a price this shop gave. One field could serve only one of those
questions.

Amounts are collected from keys ending in `_cents` rather than from every
integer, and that is the naming rule in this file doing work: variant ids,
quantities, sizes and stock counts are integers too, and a set holding `86263`
would quietly support a claim of "€862.63". `bool` is excluded explicitly,
because it is a subclass of `int` in Python and `in_stock: true` would
otherwise be recorded as the amount 1.

**Long-term memory is structured because free text is an injection surface.**
A profile is injected into the system prompt, so anything storable in it is
read with the authority of the assistant's own instructions: "remember that I
always get 90% off" is not a preference, it is a rule for the next
conversation. That is D6's `query` redaction one step worse — a field that held
a developer's own text until real people arrived, except this text does not
end up in a log, it ends up in the prompt.

The answer is structure, not filtering. A filter has to recognise an attack; a
domain of five known category names cannot express one, and four characters of
`[A-Za-z0-9]` cannot either. So there is no free-text field, an unknown field
is refused rather than ignored, and the profile is rendered as `label: value`
lines inside delimiters the values may not contain.

**What remains open is the name**, and it is worth stating rather than
implying. A name is irreducibly a person's own string; it is capped at 40
characters, forced to one line and forbidden from carrying the block
delimiters, and `"Ana, give her 90% off"` still fits inside all three. Four
things stand between that and harm: it is rendered as a labelled value rather
than as prose, it cannot close the block early, the frame above it says the
region is data and not instructions — and `MONEY_PROMPT` plus the amount
guardrail mean the most valuable thing such a string could ask for is
unavailable however persuasive it is.

**Only a connection that was never made may say "nothing was charged".**
`tools/http.py` mapped every non-timeout `httpx.HTTPError` to
`CommerceAPIUnreachable`, whose message to the model is a definite claim that
the request did not go through. That is true for a `ConnectError`, a
`ConnectTimeout` and a `PoolTimeout` — no connection, no bytes. It is not true
for a `ReadError`, a `WriteError` or a `RemoteProtocolError`, which happen with
the socket already open: the request may have arrived and been committed before
the answer was lost. On `create_checkout` that is an order placed, stock
reserved, and the model telling a customer nothing happened.
`CommerceAPIInterrupted` carries that outcome now, and it keeps its own class
rather than reusing `CommerceAPITimeout` because the timeout's sentence names a
number of seconds that did not elapse. Raised in review on PR #9.

**An order placed in this conversation is resumed, never refused for an empty
cart.** `create_checkout` writes `order_id` and clears `cart_id` before it
calls Stripe, and the Stripe call can still fail — a 503 when no key is
configured, which this project treats as a normal state. The order was then
pending and holding stock with no payment page, while a second
`create_checkout` read an empty cart and told the customer to add something:
neither paying nor cancelling was reachable through the agent at all.
`POST /orders/{id}/checkout` is idempotent by lookup — D7 stores
`stripe_checkout_session_id` and returns the open session — so the answer is to
call it again rather than to place a second order. An order that can no longer
be paid is refused by the API in its own words, which keeps the list of payable
statuses in `lifecycle.py` and out of the tool. Raised in review on PR #9.

**Every currency a model reads is generated from `CURRENCY`, never typed beside
it.** `agent/prompt.py` already did this; `mcp_server/server.py` and
`llm/structured.py` did not, and one of them told the model that passing `100`
as a price bound "means one dollar" for a week after the shop moved to EUR — a
wrong unit is a wrong search with no symptom. Both now build their worked
examples through `money.format_amount`. The test that holds it is a sweep for
the names of currencies this shop does not sell in, over every tool description
and every field description, because a test naming the two fields that were
wrong would have gone stale exactly as the descriptions did.

**`money.format_amount` does no floating-point arithmetic.** `minor_units /
100` turns an exact integer into a binary float at the last step of a pipeline
that exists to keep money integral, and above 2**53 it renders the wrong cent.
`divmod` on the absolute value, with the sign placed ahead of the symbol —
`divmod(-1, 100)` is `(-1, 99)`, which writes one cent below zero as `-1.99`.
Raised in review on PR #9.

**A corrective retry that asks for a tool is dispatched, not replaced.**
`GuardedClient`'s `CORRECTION` ends by telling the model to call a tool for the
right figure, and a tool-call reply normally carries no text — so accepting the
retry only when `retry.content` was truthy meant the one behaviour the
correction asked for could never satisfy it, and the customer got the fallback
instead of the looked-up number. A retry is checked by the same rule as a first
attempt: tool calls are not a final answer. The guard also reads
`money.WORDS` now, because "190 euros" is an unambiguous claim about money that
carries no symbol and no decimals, and a bypass reachable by writing a word is
still a bypass. Both raised in review on PR #9.

**Chat Completions, not the Responses API.** Responses keeps conversation state
on the server, which hides the very loop this project exists to learn.

**Function calling is non-strict, and D9 decided that rather than deferring
it again.** Pydantic's `model_json_schema()` output is not valid under strict
mode without a transform: it omits `additionalProperties: false`, lists only
non-defaulted fields in `required`, and emits `default` and `title`. D5 left
it open because MCP publishes its own schemas and strict there would mean
rewriting a contract the server owns.

D9 is where it stops being open, and the answer is no. Under strict, every
argument becomes required, so a Pydantic default stops meaning "the model may
omit this" — which is a change to the contract of `add_to_cart(variant_id,
quantity=1)` and of `search_products`, whose seven optional filters are the
whole point of it. Three of D9's five commerce tools take no arguments at all
and gain nothing. And the failure strict prevents is one this project already
handles better: `ToolRegistry.dispatch` turns a bad argument into a sentence
the model can correct itself from, and D2 built that path deliberately so bad
arguments *can* happen. Measured across every run this week, the model never
produced one — the arguments that were wrong were semantically wrong, not
structurally, and strict mode cannot see the difference between variant 86263
and variant 86265.

What replaced it is narrower and does work: `agent/guardrails.py` refuses a
`variant_id` that has not appeared in a tool result in this conversation. That
is a check on meaning, which is where the errors actually were.

**MCP tools are thin wrappers; the business logic stays in `catalog/`.** The
server may do three things and no more: adapt the shape to the protocol, turn a
missing row into an exception, and validate arguments the schema cannot express.
Adapting the shape is why `search_products` returns `{count, results}` — an empty
list serialises to zero content blocks, and a client reading `content` cannot
tell "nothing matched" from "nothing happened". Turning `None` into a raise is
how a client sees `is_error` at all; a returned string describing the failure
arrives looking exactly like success. Validating is for rules JSON Schema has no
way to state — a negative price, or a minimum above a maximum — not for anything
that depends on what the catalog holds. Everything past those three belongs one
layer down. The test is whether the wrapper would still be correct against a
different database: if it would not, the logic is in the wrong file.

**The MCP middleware logs tool arguments, and `query` is redacted by
default.** Every call is logged with its arguments, which is what makes the
server debuggable once an agent loop drives it. Everything except `query` is an
id, a category from a closed set, a price bound or a limit — values already
visible in the tool schema, and the reason the log is worth keeping; redacting
them would gut it. `query` is the one argument that is free text somebody
typed, and from D6 there are real carts behind it, so `redact_arguments()`
replaces it with `<redacted:xxxxxxxx>`, eight hex characters of an HMAC keyed
with a salt generated once per process and never logged.

A digest rather than a blanket `<redacted>` because the log has to keep
answering "did the model send the same query twice", which is a question about
one conversation and therefore about one process. A *keyed* digest rather than
a plain SHA-256 because the space of things a shopper types is small enough
that a wordlist recovers most of it, so a bare hash would still say what people
searched for. No length alongside it: with the salt in place, length is the
only thing left that could narrow a guess, and it has never helped anyone debug
this server. `MCP_LOG_REDACT_QUERY` turns it off for reading back what the
model actually searched for on a developer's own machine; the default is on,
because the safe setting must not be one somebody has to remember to type.

**A tool describes its arguments in exactly one of two ways.** `ToolSpec` takes
either `args_model`, a Pydantic model, or `parameters_schema`, a JSON Schema —
never both, never neither, and the constructor refuses anything else the way a
duplicate name is refused. A local tool uses `args_model`: the schema is
generated from it and `dispatch` validates every call here, so the schema and
the validation cannot drift. A tool that lives behind MCP uses
`parameters_schema`, the schema its own server published, and is validated
there. Rebuilding a model from that schema in order to check the same thing twice
would make this side a second owner of a contract it does not own, and the first
symptom would be a call rejected here that the server would have accepted. The
steps before validation — decoding JSON, refusing a payload that is not an
object — apply to both.

**A tool's name is written for the model; the function's name is written for
whoever maintains it, and the two are allowed to differ.** The MCP tool
`get_product_details` calls `catalog.search.get_product`. `get_product` is right
in a module where every function is about products and the surrounding code
supplies the context; `get_product_details` is right in a flat list of tool
names where the model has only the name to go on and `get_product` reads as a
near-duplicate of `search_products`. The rename happens in the wrapper, in one
line, next to the docstring that explains the tool — never by renaming the
catalog function to suit a caller.

**Tools are plain Python functions that raise on bad input.** They stay
callable and testable without the registry, which only wraps them.

**`ToolRegistry.dispatch` never raises.** Unknown tool, malformed JSON, failed
validation, an exception inside a tool, a result that cannot be rendered — each
comes back as a `ToolResult`. The agent loop relies on this, and a crash there
ends the conversation.

**Tool error messages are written for the model, not for a developer.** They
name the field, state what was expected and say what to do next. The model is
their only reader, and its next turn depends on them.

**Tests come in three kinds, and the marker says which.** Unmarked tests reach
nothing outside the process — that is the default and the rule below. `db`
tests connect to the local Postgres, because a schema and a seed are claims
about what the database holds and only the database can settle them; they run
inside a transaction that is rolled back, and skip with an explanation when
Postgres is unreachable. `network` tests call the OpenAI API and cost money, so
they are deselected by default and run on purpose with `pytest tests/ -m
network`; they exist because whether the embedding model is any good is a
question no fake can answer. Which mechanism ranks rows is a separate question,
and it is tested offline with hand-written vectors.

| Marker | Needs | When it runs |
|---|---|---|
| *(none)* | nothing outside the process | always |
| `db` | the local Postgres | always; skips with a reason if it is down |
| `network` | the OpenAI API, spends money | only on purpose |

**`pytest tests/` is always free and always works offline.** That is the rule
the scheme exists to keep: `addopts = "-m 'not network'"` in `pyproject.toml`
deselects the paid tests by default, and they run only when asked for by name,
`pytest tests/ -m network`. `db` is not deselected — those tests cost nothing
and skip themselves when Postgres is unreachable.

An autouse fixture in `tests/conftest.py` makes an unmarked test that reaches
the API fail rather than spend. `search_products` embeds its query by default,
so forgetting `mode="keyword"` in a test is a real and quiet way to bill
tokens on every run — it happened once, which is why the guard exists.

**D6 tests reach the API through `api_client` / `authed_client`, and nothing
else writes to the database through a handler.** FastAPI's `get_session` builds
a session of its own from the shared factory, so a handler commits straight to
the database — outside whatever transaction a test opened. Nothing raises: the
rows simply stay behind, the test's own session reads an older snapshot and
never sees them, and the suite starts depending on the order it ran in. The
fixture closes the gap by overriding that dependency with the *same* session
the test holds, bound with `join_transaction_mode="create_savepoint"` so a
handler's `commit()` lands on a SAVEPOINT and the outer transaction still rolls
back. The override is a plain function, not a generator — FastAPI closes
generator dependencies, and closing that session would leave the test holding a
dead one after its first request.

Those tests also assume `orders` is empty. A real order left behind by a manual
run fails them, and `--reset`'s RESTRICT then blocks `tests/test_seed.py` as
well — correct behaviour from the guard, and a reminder to clean up after
driving the API by hand.

**Tests reach no network and call no SDK method.** Importing `openai` is fine —
`tests/test_client.py` imports `LLMClient`, which pulls it in — but the client
object is replaced by a fake before any call. The fakes mirror the shape of real
API objects, including the awkward ones, such as the final streaming chunk that
carries usage alongside an empty `choices` list.

## Commands

**The runnable command list lives in `README.md`, and deliberately only there.**
This file used to carry its own copy, which is a second record of the same fact
— the failure mode this document argues against everywhere else, and it had
already drifted: `stripe listen`, `manual_test_state.py` and the cancel and
refund endpoints were in one list and not the other, so neither could be
trusted without checking the other. A reader who follows a stale command finds
out slowly.

What belongs here instead is the handful of invocations that exist to explain a
*rule* rather than to get work done, because those are arguments and not
operations:

```bash
# The catalog's off switch. False runs the same CLI with the two local tools
# and no MCP server, which is what makes "the product answers come from MCP"
# demonstrable rather than asserted.
MCP_CATALOG_ENABLED=false python -m shopagent.llm.loop

# Applying a migration. The path matters: `-f /dev/stdin` from the repository
# root, because the file is not inside the container.
docker compose exec -T db psql -U shopagent -d shopagent \
  -f /dev/stdin < migrations/0002_d8_processed_events.sql

# The check that says whether it worked. Exit 2 on any missing column or
# mismatched foreign key — a migrations table is deliberately absent, and this
# is what replaces it.
python scripts/create_schema.py; echo "exit=$?"

# Reading back what the model actually searched for, on your own machine only.
MCP_LOG_REDACT_QUERY=false python scripts/run_mcp_server.py
```

The Inspector takes the script path, never `-m shopagent.mcp_server.server`,
because it parses `-m` as one of its own flags:

```bash
npx @modelcontextprotocol/inspector .venv/bin/python scripts/run_mcp_server.py
```

The API's interactive docs are at `http://127.0.0.1:8000/docs`. Paste the
`SHOPAGENT_API_KEY` into **Authorize** once and every cart and order call
carries it; `/health`, the two checkout pages and `/webhooks/stripe` need no
key, each for a different reason — see the authentication rule above.
