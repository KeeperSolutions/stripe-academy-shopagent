# Journal

What each day of building ShopAgent turned up: decisions and the reasoning
behind them, measurements, and the things that turned out not to work. Written
as it happened, so an entry is what was true that day — the running list of
what is still open is at the end.

Setup and commands live in [README.md](README.md).

## Day 1 — findings

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

## Day 2 — findings

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

## Day 3 — findings

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

**The HNSW index is decorative at this size — but it had to be made usable
anyway.** It exists because the syntax is worth knowing:

```sql
CREATE INDEX ix_products_embedding_hnsw ON products USING hnsw (embedding vector_cosine_ops);
```

With thirty rows Postgres will sequential-scan the table faster than it can
descend a graph, and the planner is free to ignore the index entirely — which
it should. The detail that does matter at any size is `vector_cosine_ops`: an
index built for one operator class is silently unused by a query ranking with a
different operator, and `search.py` ranks with `<=>`. It is also built after
the vectors exist, since an index over an all-NULL column has nothing to build.

**Decorative is not the same as unusable, and the first version of this query
was unusable.** Code review caught it and `EXPLAIN` settled it. An HNSW index
serves exactly one shape — `ORDER BY embedding <=> $1` with a `LIMIT`, over a
plain scan of the table — and the query had neither half of it: the distance
was wrapped in `min()` behind a `GROUP BY`, and a `products.id` tie-break
followed it. Each alone is enough to lose the index:

```
ORDER BY embedding <=> $1                 -> Index Scan using ix_products_embedding_hnsw
ORDER BY embedding <=> $1, products.id    -> Sort  (whole table)
min(embedding <=> $1) after GROUP BY      -> Sort  (whole table)
```

So the variant filters moved into an `EXISTS` subquery, which leaves the outer
statement a scan of `products`, and the tie-break came out. Losing it costs
nothing real — exact ties between float distances do not happen — and it was
the difference between an index scan and sorting every qualifying row.

Thirty products cannot show this: the planner sequential-scans either shape,
correctly. Loading 2,000 synthetic products into a transaction and rolling it
back afterwards makes the choice real, and the two plans separate:

```
new:  Limit -> Nested Loop Semi Join -> Index Scan using ix_products_embedding_hnsw
                                          Order By: (embedding <=> $1)
old:  Limit -> Sort (2030 rows) -> HashAggregate
```

The same measurement turned up a subtler thing. An index scan never returns
rows whose vector is NULL, because they are not in the index, while a
sequential scan sorts them last. The same query would therefore return 29 or 30
rows depending on the plan Postgres picked. Products without a vector are now
excluded from a semantic search explicitly, so both plans agree.

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

## Day 4 — findings

**`FastMCP` does not exist in `mcp` 2.0.** The plan, and every tutorial written
before the 2.0 release, opens with `from mcp.server.fastmcp import FastMCP`.
That module was renamed: it is `mcp.server.mcpserver` now, the class is
`MCPServer`, and the import that works is `from mcp.server import MCPServer`.
Grepping the installed package for `FastMCP` returns nothing at all, so the
failure is an immediate `ModuleNotFoundError` rather than a deprecation warning.
Decorators and handler signatures came through the rename unchanged, which is
why this cost an import line and not a redesign — but a v1 tutorial does not run
against a v2 pin, and `mcp>=1.2` in `requirements.txt` would happily have
installed either. It is `mcp>=2,<3` now.

**The `Args:` section of a docstring never reaches the parameter schema.** The
plan says "docstrings and types are the contract — MCP generates the schemas the
model sees from them". Half true, and the half that is false is the expensive
one. A tool's *docstring* becomes its `description`; the *argument* schema is
built from the type hints alone. An `Args:` block describing each parameter is
read by nobody: the properties come out carrying a type and a title and no
`description` at all. Per-parameter text requires
`Annotated[..., Field(description=...)]`, which is a Pydantic annotation and not
a docstring convention. Checked by generating both forms side by side before
writing anything, which is the only reason `search_products` does not ship with
seven undocumented properties. Both channels are used now — the docstring says
when to call a tool, the `Field` says what one argument means.

**A returned string describing a failure is indistinguishable from success.**
The obvious way to report "no such product" is to return a sentence saying so.
It arrives at the client as `isError: false` with that sentence as the content —
byte-identical in shape to a successful answer, and the model has only prose to
tell them apart. The only way a client sees an error is an exception: the SDK
catches whatever a tool raises, wraps it as `ToolError`, and the protocol
handler turns that into `CallToolResult(is_error=True)`. So `get_product_details`
and `check_stock` raise on an unknown id rather than returning `None` or a
message. Worth pinning precisely, because the two layers disagree on purpose:
`server.call_tool(...)` — the bare Python API — *raises*, while the same call
over the transport *returns* `is_error`. A test asserts both, since step 3's
error handling is built on the distinction.

**An empty list serialises to zero content blocks.** `search_products` returning
`[]` produced `content: []` — not an empty string, not an empty array rendered
as text, but no blocks at all. `structuredContent` still carried `{"result": []}`,
so a client reading structured output was fine; a client reading `content`, which
is the shape the D5 adapter will hand the model, received literally nothing. "No
matches" and "the call did nothing" then look the same, and the difference is
operational: the first means widen a filter, the second means retry. The fix is
an envelope — `search_products` returns `{count, results}` — so one object always
renders as one block and `count: 0` states the outcome in words. No filtering and
no reshaping of an individual product: the envelope exists purely to make
emptiness legible.

**The SDK's logging handler quietly truncated the logs.** `MCPServer.__init__`
calls the SDK's own `configure_logging`, which installs a `RichHandler` whenever
`rich` is importable — and it is, as a transitive dependency. That handler
formats for a terminal: it wraps to the console width, and with stderr a pipe
rather than a tty it falls back to a default width and cuts. What sits at the end
of a log line here is the arguments and the duration, so exactly the diagnostic
payload was being dropped while the line still looked like a log entry. A
`logging.basicConfig` in `main()` did nothing about it, because the handler was
already installed at import time and `basicConfig` is a no-op once handlers
exist. `force=True` with an explicit stderr handler fixes it. A log that drops
its payload is worse than no log, because nothing about it looks broken.

**A bare `dict` return annotation produces no output schema.** `-> dict` yields
`outputSchema: None` and no `structuredContent`; `-> dict[str, Any]` yields a
free-form object schema, and a `TypedDict` yields a real one with named
properties. Two of the four tools were shipping without an output contract on the
strength of an annotation that looked complete. The return types are
`dict[str, Any]` now, and `search_products` returns a `SearchResults` TypedDict so
that `count` and `results` appear in the schema rather than only in prose.

**Edge cases: which are errors, and which are simply empty.** The line drawn is
that an error is what the model *must* change to get any answer at all, and an
empty result is what valid arguments return when the shop holds nothing like
that. The distinction matters because the two demand opposite next moves —
correct the argument, or relax a filter.

| Case | Outcome | Why |
|---|---|---|
| `max_price_cents` < 0 | error | No catalogue holds a product priced below zero. Returning nothing would read as "nothing that cheap" and invite the model to search lower — the one direction that cannot help |
| `min_price_cents` < 0 | error | Same |
| `min` > `max` | error | The range is empty by construction and stays empty however much the shop grows, so "widen the filter" leads nowhere |
| `limit` outside 1-50 | clamped, succeeds | The model still gets a usable answer, just a different size. Refusing would cost a turn to teach it something the schema already states |
| `query` empty or whitespace | succeeds, browses | An empty query means "no query". `catalog.search_products` skips embedding for it, so it costs nothing — worth a test, because that guard is invisible from outside and easy to delete |
| `size`/`color` with no match | empty result | Arguments valid, shop has none. Exactly what the envelope was built for |
| unknown `product_id`/`variant_id` | error | The id is wrong; repeating it cannot start working |

**Pydantic's validation message is written for a developer, and is left mostly
alone.** A wrong argument type never reaches a tool body — the SDK validates
against the generated model first — so the text the model reads is Pydantic's
own, ending in a link to `errors.pydantic.dev` and tagged with an internal type
code. Rewriting it properly would mean loosening the schema types so the error
never fires, which makes the mistake *more* likely in order to make the message
prettier. Instead a small middleware trims the trailing noise and the generated
model name while keeping the field name and the expected type. It fires only on
text carrying Pydantic's own header, so the hand-written messages pass through
untouched, and a test asserts that.

**The catalog tools open their own database sessions, and they close.** A
long-lived server leaking a connection per call is a real failure mode, so it was
measured rather than assumed: 40 consecutive `check_stock` calls through a real
stdio subprocess held the backend count in `pg_stat_activity` at exactly 1
throughout and 0 after shutdown, and `engine.pool.status()` reported 0 checked
out after 50 mixed operations. `_session_for` from D3 does the work — the tool
passes no session, so `session_scope` opens one and closes it in a `finally`.

## Known gaps

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

**Price validation lives in the MCP wrapper, not in `catalog/search.py`.** A
negative bound or a minimum above a maximum is rejected in
`mcp_server/server.py`, because D4 is where the plan puts edge cases and because
changing `search_products` would change the contract D3 tests. The cost is that
the rule is not where the function is: a caller reaching
`catalog.search_products` directly — which is what D9 does behind its own
tools — still gets a silent empty list for `max_price_cents=-500`. Either the
validation moves down into `catalog/`, or D9 repeats it in its own wrapper. The
first is tidier and is a change to D3's tested surface, so it is a decision
rather than a chore.

**The `limit` clamp is silent.** `search.py` clamps to 1-50, so a model asking
for 100 gets at most 50 and is told nothing about it. The parameter description
now says the clamp exists and points at `count` as the authority on how many came
back, which is a docstring rather than a signal in the response — a model that
ignores the description learns nothing from the result either. Making it explicit
needs a third field in the envelope, which was deliberately not added while the
shape is this new.

**The upper half of that clamp has never actually fired.** The catalog holds 30
products, so `limit=100` returns everything that matched and never reaches the
cap of 50 — the clamp is verified by reading `max(1, min(int(limit), MAX_LIMIT))`
and by the lower bound, where `limit=0` and `limit=-5` both return one result.
The 50 ceiling is asserted in a test as a range rather than observed, and will
stay that way until the catalog outgrows it.

**The MCP middleware logs `query`, which stops being safe on D6.** Every tool
call is logged with its arguments, and that is deliberate: `query` is the one
argument that shows what the model understood the shopper to want, so redacting
it would gut the log precisely where D5 needs it. It is safe today only because
the text is a developer's own — typed into the Inspector or a test. D6 brings
real carts and real customers, and the same field becomes something a stranger
wrote. Before this server sees production traffic, `query` needs redaction, a
hash, or a config flag; the ids, prices and timings can stay. Raised by review
on PR #4 and deferred on purpose rather than missed.
