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
| `mcp_client/` | D5 | loads tools from MCP dynamically |
| `api/` | D6, D8 | FastAPI cart and orders; Stripe webhooks |
| `payments/` | D7 | Stripe SDK |
| `agent/` | D9 | memory, guardrails |
| `obs/` | D10 | Langfuse tracing |

Search logic belongs in `catalog/`, never inside the MCP server. The server is
a thin wrapper, which is what lets D5 swap transports without touching logic.

## Conventions

**Configuration goes through `shopagent.config.get_settings()`.** `os.getenv`
and `os.environ` appear nowhere else. A new variable is added there as a typed
field, then to `.env.example`, and only then used.

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
points at a variant. D6 owes that script a guard — it must refuse to run while
any order exists, rather than taking a customer's order lines down with the
catalog.

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
and emits `default` and `title`. Revisit on D5/D9.

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
python -m shopagent.llm.loop      # the CLI agent
```
