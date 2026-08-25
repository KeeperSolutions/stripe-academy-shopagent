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

# 4. database (Postgres 16 + pgvector)
docker compose up -d
docker compose exec db psql -U shopagent -d shopagent \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 5. catalog (schema, 30 products, vectors + HNSW index)
python scripts/create_schema.py
python scripts/seed_catalog.py
python scripts/embed_catalog.py

# 6. verify
docker compose exec db psql -U shopagent -d shopagent \
  -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
pytest tests/ -v          # 227 tests; add -m network for the 4 that call the API
```

Postgres listens on `localhost:5432` (user / password / db: `shopagent`), with data
stored in the named volume `pgdata`. Stop it with `docker compose down` — the volume
is preserved.

## Journal

### Day 1 — findings

**Chat Completions, not the Responses API.** Responses keeps conversation state
on the server. The point of D1–D2 is to see the agent loop from the inside — we
hold the message list ourselves, send it on every call ourselves, and append
`tool` replies ourselves. Responses would hide exactly that. Worth revisiting on
D9, when a framework enters the picture anyway.

**`gpt-5.6-luna` rejects `temperature`.** Any explicit value returns a 400:

> `Unsupported value: 'temperature' does not support 0.7 with this model. Only the default (1) value is supported.`

So `temperature` in `LLMClient` is optional (`None`) and is sent only when
given. The client never guesses what a model supports — that is the caller's
decision.

**Temperature experiment** (run on `gpt-4o-mini`, since `gpt-5.6-luna` refuses
it; same prompt three times per temperature). At `temperature=0` all three
answers were byte-identical — the model is deterministic, which is exactly what
repeatable work needs (extraction, classification, tool arguments). At `1.2` all
three differed, and that variety came at a price: one answer drifted into a
different language than the prompt, another returned a name unrelated to the
product category. Takeaway: the agent loop runs at a low temperature, high
values are reserved for places that explicitly call for variation. The whole
experiment (6 calls, 258 tokens) cost **$0.0000684**.

**The cached input discount is per-model, not global.** The first version of the
tracker assumed a flat 10% — wrong. The `gpt-5.6-*` family pays 10% of the full
input price, `gpt-4o-mini` pays 50%. That is why `PRICING` holds three numbers
per model (`input`, `output`, `cached_input`) instead of a percentage derived
from one. Identical usage on two models with the same input/output prices can
still cost different amounts.

**Model prices** in `llm/usage.py` were last checked on **2026-08-14**
(`platform.openai.com/pricing` + `pricepertoken.com`). A model missing from
`PRICING` does not break the tracker — its cost is 0 and it lands in
`unknown_models`, with a warning in `summary()`.

### Day 2 — findings

**`gpt-5.6-luna` refuses function tools on Chat Completions by default.** The
call comes back `400`: *"Function tools with reasoning_effort are not supported
for gpt-5.6-luna in /v1/chat/completions. To use function tools, use
/v1/responses or set reasoning_effort to 'none'."* Since staying on Chat
Completions is a deliberate choice here, the second option applies:
`OPENAI_REASONING_EFFORT` is sent alongside the tool schemas, and only then. It
is configuration rather than a constant because a model that does not know the
parameter returns `400` if it is sent at all.

**The tool loop is a `while`, and one input can cost several calls.** Asking for
a time in Tokyo *and* a duration in minutes took 2 model calls and 1,270 tokens
(**$0.000333**) — the model requested both tools in a single reply, so the two
ran in parallel rather than in sequence. A six-turn conversation over the same
two tools came to 12 calls and 9,140 tokens (**$0.00206**). Every call in a turn
is priced, not just the last one.

**Parallel tool calls are not chaining.** Two tools in one reply is one round
trip; chaining is the model reading a result before deciding what to call next.
Forcing it needs a second tool that cannot start without the first one's output:
*"What time is it in Tokyo right now? Then calculate how many minutes are left
until midnight there."* gave 3 model calls, with the calculator receiving
`24*60-(2*60+22+54/60)` — arithmetic built from the timestamp `get_time` had
just returned.

**The two features have different constraints, and the difference was measured
rather than assumed.** After the tools 400, the obvious guess was that
`reasoning_effort='none'` would be needed for `response_format` too. It is not:
structured output works with no `reasoning_effort` sent at all. The restriction
is specific to function tools. Guessing would have added a parameter nothing
required and left the wrong reason for it in the code.

**Strict `json_schema` enforces less than the docs imply.** Raw Pydantic output
is rejected — *"Invalid schema for response_format: In context=(),
'additionalProperties' is required to be supplied and to be false"* — but
probing one change at a time showed only two are actually enforced:
`additionalProperties: false` on every object, and every property listed in
`required`. `title` and `default` were accepted when tried. `llm/structured.py`
strips them anyway: neither is in the documented strict subset, so today's
tolerance is not something to build on, and they cost tokens on every call.

**Tool schemas stay non-strict while `response_format` is strict.** Not an
inconsistency. Strict mode would stop the model producing a bad tool argument —
and catching bad arguments is the thing D2 is for. `ToolRegistry.dispatch`
exists to turn a rejected argument into a message the model can correct itself
from, and that path only gets exercised if bad arguments can happen. Structured
output has no such loop: there is one shot at the answer, so the schema does the
work up front.

**Dollars become cents in exactly one place.** `parse_product_query` is the only
code that ever sees a dollar figure; everything downstream receives
`max_price_cents` as an integer. Pinned at three levels, because one is not
enough: the system prompt states the rule with worked examples ("under $100" is
`10000`), `Field(strict=True)` stops Pydantic quietly accepting `100.0` as `100`
the way lax mode does, and `ge=0` rejects a negative bound. `"something warm for
winter, not too expensive"` returned `None` for both price fields rather than
inventing a number.

### Day 3 — findings

**Semantic search answers a question keyword search cannot see.** The same
string through both modes, against the same thirty products:

```
query: "something to run in when it's raining"

  keyword  (ILIKE)     0 results
  semantic (<=>)       5 results
     1. Trail Runner GTX       shoes         9499c   GORE-TEX, spray, slick stone
     2. Storm Pace 4           shoes         8999c   wet mornings, standing water
     3. Runner's Cap           accessories   2999c   water-repellent brim
     4. Cloud Sprint 2         shoes         7499c
     5. Packable Wind Jacket   jackets       4999c

query: "keeping my gear dry on a boat"

  keyword  (ILIKE)     0 results
  semantic (<=>)       5 results
     1. Dry Duffel 40L         bags          9999c   welded, roll-top, open deck
     2. Hydration Vest 5L      equipment    11999c
     3. Packable Wind Jacket   jackets       4999c
     4. Compact Dry Bag 10L    equipment     1999c
     5. Storm Pace 4           shoes         8999c
```

The zero on the keyword side is the whole point and it is not an accident: the
letters `rain` appear nowhere in the seed, and `tests/test_seed.py` fails if
anyone adds them. Nothing lexical connects the query to a GORE-TEX membrane, so
ILIKE has nothing to match, while the two shoes whose copy is about wet ground
come back first and second from the vector search. The ranking beyond the top
two is looser — a cap outranks a shoe on the first query, and a hydration vest
places second on a query about keeping kit dry — which is the honest shape of
embedding search on thirty items: the top of the list is right, the tail is
approximate.

**The whole catalog costs $0.000027 to embed.** Thirty products,
`name + brand + category + description` joined, 1,336 tokens in one batched
`text-embedding-3-small` call. Re-running the script spends nothing at all: it
selects the products whose `embedding` is NULL, finds none, and never opens a
connection to the API. `--force` is the way to pick up an edited description,
because editing prose does not make a stored vector NULL. `text-embedding-3-small`
had to be added to `PRICING` in `llm/usage.py`, with the output price at 0.0 —
an embedding call has no completion to bill.

**The HNSW index is decorative at this size.** It exists because the syntax is
worth knowing:

```sql
CREATE INDEX ix_products_embedding_hnsw ON products USING hnsw (embedding vector_cosine_ops);
```

With thirty rows Postgres will sequential-scan the table faster than it can
descend a graph, and the planner is free to ignore the index entirely — which
it should. The detail that does matter at any size is `vector_cosine_ops`: an
index built for one operator class is silently unused by a query ranking with a
different operator, and `search.py` ranks with `<=>`. It is also built after
the vectors exist, since an index over an all-NULL column has nothing to build.

**Filtering stays in SQL, ranking only changes the ORDER BY.** The trap here is
easy to fall into and quiet when you do: rank the five nearest products, then
drop the ones over the price limit in Python, and a search for running shoes
under $100 returns two results while the shop holds four. Every filter is a
WHERE predicate, the distance is an ORDER BY, and LIMIT runs last. There is a
test that pins it — the nearest product by construction is the one too
expensive to qualify, and it must not come back.

**Prompt caching still has nothing to bite on.** The D2 gap is unchanged, and
D3 was never going to change it: the catalog is not exposed as tools until D4.
Measured again on the current build — 535 prompt tokens, `cached_tokens` 0 on
both a first and an immediately repeated call, against OpenAI's 1,024-token
minimum. The accounting works and is tested; the prompt is simply too small.
D5 is where it should finally engage, when MCP supplies the catalog schemas.

### Known gaps

**The CLI does not stream while tools are in play.** `chat_with_tools` is a
blocking call; `stream_chat` still exists and is still tested, but nothing
drives it now. Streaming a tool call means accumulating deltas per `index`: the
function name arrives in one chunk, the arguments in fragments spread over the
next several, and the `id` only once. Reassembling that is bookkeeping that
would have buried the chaining D2 exists to demonstrate. Worth revisiting once
the loop itself is settled.

**Function calling currently runs without reasoning, and that is unresolved.**
`reasoning_effort='none'` is the price of using function tools on Chat
Completions with `gpt-5.6-luna`. For D2 it costs nothing visible: two
independent tools, and the model still chained them correctly. D9 is the
worry — five commerce tools with real interdependencies (search → check stock →
add to cart → view cart → checkout), where picking the next call *is* the
reasoning. Whether a non-reasoning model holds that chain together is untested.
Options if it does not: move to the Responses API, which supports tools with
reasoning but keeps conversation state server-side and would hide the loop this
project exists to show, or switch to a model without the restriction. Neither
is free, and the decision is deferred rather than made.

**A prompt instruction is not a guardrail.** The system prompt tells the model
never to do arithmetic in its head. Asked for *"the sine of 30 degrees multiplied
by 4"* and *"5 factorial"*, it made **zero tool calls** and answered `2` and
`120` from memory. Both were correct, which is the uncomfortable part — nothing
in the output marked them as unverified. The calculator genuinely cannot express
either operation, so the model was choosing between a useless refusal and a
right answer and chose well; the failure is that it chose *silently*. Two
tempting fixes are both wrong. Sharpening the wording would make these two
examples comply and teach nothing, since the instruction being ignored is
already explicit. Teaching the calculator `sin(...)` means allowing `ast.Call`,
which is the single rule keeping every injection vector out. The answer belongs
in `agent/guardrails.py` on D9 — validate the output in code instead of asking
the model to behave. It is the same shape as the price rule waiting there: an
amount that appears in an answer without appearing in the context has to be
blocked, not discouraged. Cheap to learn on a sine; expensive to learn on a
checkout total.

**Prompt caching never engaged.** The system prompt and both tool schemas repeat
on every call, which is exactly the shape caching rewards, yet `cached_tokens`
was `0` in every call measured. The prompt peaked at 975 tokens and OpenAI's
cache has a 1,024-token minimum, so nothing qualified — the mechanism works and
is tested, it simply has nothing to bite on yet. That should change on D3–D5,
when the catalogue tools push the schemas past the threshold; the accounting is
already in `llm/usage.py`.

**Tool schemas stay non-strict.** `llm/structured.py` has the transform, and
`response_format` uses it, but `tools/registry.py` still sends raw Pydantic
output with `strict` unset. Under strict every tool argument would become
required, so a Pydantic default would stop meaning "the model may omit this" —
a change to the tool contract rather than a formatting fix. Revisit on D5/D9,
when MCP supplies the schemas anyway.
