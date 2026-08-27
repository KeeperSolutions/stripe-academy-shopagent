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
| `tools/commerce.py` | D9 | cart and checkout, over HTTP |
| `catalog/` | D3 | models, seed data, embeddings, search |
| `mcp_server/` | D4 | exposes `catalog/search.py` as MCP tools |
| `mcp_client/` | D5 | client, schema adapter, registration into the registry |
| `api/main.py` | D6 | the FastAPI app, CORS, `/health`, router mounts |
| `api/db.py` | D6 | the request-scoped session dependency |
| `api/deps.py` | D6 | `X-API-Key` authentication |
| `api/models.py` | D6 | carts and orders, on the catalog's `Base` |
| `api/lifecycle.py` | D6 | `OrderStatus` and the transitions between them |
| `api/schemas.py` | D6, D8 | request and response bodies; where names change |
| `api/routers/` | D6, D8 | HTTP only — parse, call a service, map an error |
| `api/services/` | D6, D8 | cart and order logic; imports no FastAPI |
| `payments/stripe_svc.py` | D7 | the Stripe SDK layer; the only importer of `stripe` |
| `payments/catalog_sync.py` | D7 | mirrors the catalog into Stripe; nothing bills from it |
| `payments/checkout.py` | D7 | Checkout Sessions built from the order snapshot |
| `payments/customers.py` | D7 | attaching a buyer to an order |
| `agent/` | D9 | memory, guardrails |
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
automatically, and the explicit copy is why all three objects carry it.** D8 may
therefore subscribe to `checkout.session.completed`,
`payment_intent.succeeded` or `charge.succeeded` and attribute any of them —
but only for as long as that parameter keeps being sent, which is what the
offline test on the outgoing payload exists to guard. No PaymentIntent exists
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

**Chat Completions, not the Responses API.** Responses keeps conversation state
on the server, which hides the very loop this project exists to learn.

**Function calling is non-strict for now.** Pydantic's `model_json_schema()`
output is not valid under strict mode without a transform: it omits
`additionalProperties: false`, lists only non-defaulted fields in `required`,
and emits `default` and `title`. D5 came and went without it — MCP
publishes its own schemas, so strict there would mean rewriting a contract the
server owns. Revisit on D9.

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

```bash
docker compose up -d              # Postgres 16 + pgvector
pip install -r requirements.txt
python scripts/create_schema.py   # pgvector + create_all, idempotent
python scripts/seed_catalog.py    # 30 products; --reset to rebuild
python scripts/embed_catalog.py   # vectors + HNSW index; --force to redo
pytest tests/ -v                  # offline and database tests
pytest tests/ -m network          # the four that call the API and cost money
python -m shopagent.llm.loop      # the CLI agent (local tools + MCP catalog)
python scripts/run_mcp_server.py  # the catalog MCP server alone, on stdio
uvicorn shopagent.api.main:app --reload --port 8000   # the commerce API
python scripts/sync_stripe_catalog.py --dry-run       # plan the Stripe catalog sync
python scripts/sync_stripe_catalog.py                 # create what is missing
pytest tests/ -m stripe           # the Stripe test-mode tests; needs a key
```

The API's interactive docs are at `http://127.0.0.1:8000/docs`. Paste the
`SHOPAGENT_API_KEY` into **Authorize** once and every cart and order call
carries it; `/health` needs no key. `MCP_LOG_REDACT_QUERY=false` puts the raw
search query back in the catalog server's log, which is worth doing only on
your own machine.

`MCP_CATALOG_ENABLED=false` runs the same CLI without the catalog server. The
Inspector takes the script path, never `-m shopagent.mcp_server.server`, because
it parses `-m` as one of its own flags:

```bash
npx @modelcontextprotocol/inspector .venv/bin/python scripts/run_mcp_server.py
```
