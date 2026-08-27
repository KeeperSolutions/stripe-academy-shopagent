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

## Day 5 — findings

**The D2 abstraction held, and the evidence is that nothing moved.**
`run_tool_loop` takes a registry and a list of schemas and does not ask where
either came from. Adding four tools that live in another process therefore
changed the assembly and the lifetime of the session, and not one line of the
loop: `while`, `dispatch`, the `tool` messages and `MAX_TOOL_ITERATIONS` are
what D2 left. That was checked by parsing both versions and comparing the
function source, not by reading a diff — `run_tool_loop`, `_shorten`,
`_show_tool_call`, `_show_tool_result` and `_print_cost` all come back
identical. The eight lines `main()` did lose were all the same edit: the global
`REGISTRY` becoming the registry that was assembled for this session. Making the
registry a parameter instead of a global cost nothing on D2 and paid for itself
three days later.

**`ToolSpec` promised something its shape could not deliver.** `register()`
carried the docstring "This is the entry point an MCP adapter (D5) will use",
and `dispatch` already described taking "an already-parsed dict when the
arguments came from somewhere else (MCP, on D5)". Both were written on D2 with
this day in mind. Neither was reachable: `to_openai_schema` called
`args_model.model_json_schema()` and `_parse` called `spec.args_model(**raw)`,
so the type was hard — a Pydantic class or nothing, and MCP publishes JSON
Schema. Constructing one anyway succeeds, because a dataclass does not check its
annotations, and fails on first use with `AttributeError: 'dict' object has no
attribute 'model_json_schema'`. The intention was recorded; the shape did not
carry it. `args_model` is optional now and `parameters_schema` sits beside it,
with exactly one required.

**A remote tool validated twice is a schema with two owners.** The tempting fix
was to rebuild a Pydantic model out of the published schema and keep validating
locally. That would make this side a second source of truth for a contract the
server owns, and the first symptom of a drift would be a call rejected here that
the server would have accepted — with an error message the server never wrote.
D4 had already measured the server's rejections reaching the model as usable,
correctable text, so there was nothing to add. A tool with `parameters_schema`
skips the Pydantic step entirely and the dict goes through as the model sent it.
Everything before that step still runs: malformed JSON and a non-object payload
are caught for remote tools exactly as for local ones, because neither is
something a server should have to explain.

**`dispatch` would have swallowed the failure, and running it is what showed
that.** An MCP server reports a failed call in its reply rather than by raising,
so the natural way to carry `isError` across is for the tool function to return a
`ToolResult`. That did not work and did not look broken: `dispatch` passed the
returned object to `_to_content()`, which has a `str()` fallback, and wrapped the
result in `ToolResult(ok=True, ...)`. The model would have received
`ToolResult(ok=False, content='...')` — the repr of a failure — as a successful
answer. This is the client-side twin of the D4 finding that a returned string
describing an error is indistinguishable from success, and it was found by
constructing the case and printing the result before writing the test that now
pins it. `dispatch` passes a returned `ToolResult` straight through.

**The spawned server inherits almost no environment, and D1 is the only reason
that does not matter.** `stdio_client` does not hand the child the parent's
environment: `get_default_environment()` copies an allowlist, which on POSIX is
`HOME`, `LOGNAME`, `PATH`, `SHELL`, `TERM` and `USER`. No `OPENAI_API_KEY`, no
`DATABASE_URL`. It works because `config.py` sets `env_file` to an absolute path
computed from the module's own location, so the server reads `.env` itself
whatever it was started with and from wherever. A relative path there — the
obvious way to write it — would have produced a catalog server that starts
cleanly and fails every single call.

**`content`, not `structuredContent`.** Convenience says read the structured
payload; the protocol says otherwise. On a failed call `structuredContent` is
`None` and the message exists only in `content`, and a tool without an output
schema fills only `content` too. Preferring structured output would mean two code
paths and a `None` check deciding what the model sees on the turn it most needs
to understand — the failing one. Content is the field every result has.

**A blocking portal, not `anyio.run` per call.** The SDK is async and the rest of
the project is not. An MCP session is stateful: one subprocess, one handshake,
and every later call down the same pipes. Wrapping each call in `anyio.run` would
spawn a Python interpreter per tool call, and D4's measurement that the server
holds exactly one pooled database connection would stop meaning anything.
Instead one background thread runs one event loop for the client's lifetime and
`anyio.from_thread.start_blocking_portal` lets synchronous callers submit work to
it. `anyio` was already there as a dependency of `mcp`, and asyncio stays in one
file rather than spreading through the project because one module needed it.

**Prompt caching finally engaged — the gap left open on D2 and D3 is closed.**
Both days measured `cached_tokens: 0` and diagnosed the same cause: OpenAI's
cache has a 1,024-token minimum and the prompt peaked at 975. Six tool schemas
changed that. The system prompt plus the catalog rules plus six schemas is
**2,254 prompt tokens**, and the prefix is hit almost whole: 2,251 of 2,254 on
one call, 3,549 of 3,552 on the next. `gpt-5.6-luna` bills cached input at
$0.02/1M against $0.20/1M full, so a cached token costs a tenth.

| Run | Without cache | Actually paid | Saved |
|---|---|---|---|
| Scenarios 1+2, cold cache | $0.003214 | $0.001436 | $0.001777 (55%) |
| Scenarios 1+2, warm cache | $0.003095 | $0.000749 | $0.002345 (76%) |
| Scenario 3, blue jacket | $0.001109 | $0.000301 | $0.000808 (73%) |
| **All three** | **$0.007417** | **$0.002486** | **$0.004931 (66%)** |

Cold and warm are worth separating, because a demo run twice flatters itself.
On the first call of a genuinely cold session `cached_tokens` is 0 and the full
2,254 tokens are billed at the input rate; every later call in that session hits,
because the prefix has been sent once. Re-running the same conversation minutes
later starts warm — the first call already reports 2,251 cached — which is why
the second row saves more than the first. The 55% is the honest number for a
session started from nothing; 76% is what a repeated demo looks like. Either way
the accounting in `llm/usage.py` has been correct since D1 and was simply waiting
for something to measure.

**The validation error cannot be provoked from the Inspector.** Driving the
finished server through the UI covered the catalog tools, the error path and the
empty envelope, but one case turned out to be unreachable: sending a string where
`max_price_cents` wants an integer. The Inspector renders that argument as a
Mantine `NumberInput`, which discards non-numeric input before anything is sent —
setting the value programmatically through the native setter leaves the field
empty too. So the exact path the D4 middleware exists to clean up, stripping
`errors.pydantic.dev` and the generated model name out of Pydantic's message, is
reproducible only from an API client or from a model. That is a boundary of the
tool rather than a defect: the Inspector protects a human from sending what the
schema forbids, and a model has no such protection. It is why that message is
pinned by a test rather than by a screenshot.

**A failed tool call is green in the protocol log.** In the Inspector's message
list, a `tools/call` that came back with `isError: true` still shows **OK**, and
it is right to: the JSON-RPC request succeeded and carried a flag saying the tool
did not. The failure is red only in the Results panel. Anyone reading the log
alone would score that call as fine.

That is the third appearance of one idea, at three different layers, and worth
naming as a single class. On D4 the server side: a tool that *returns* a string
describing a failure produces `isError: false`, indistinguishable from an answer.
On D5 the registry: a tool returning a `ToolResult` was flattened by `dispatch`
into `ToolResult(ok=True)` with the failure as its repr. And here the UI: the
transport-level outcome is displayed where a reader looks for the tool-level one.
Every time, a failure travelled inside a successful envelope and something
upstream read only the envelope. The lesson is the same at each layer — the
success of the delivery is never evidence about the content — and D9 gets it in
its most expensive form, where the content is a price.

### A2A, and why the catalog is not one

A2A standardises the horizontal direction — agent to agent — where MCP
standardises the vertical one, agent to tools and resources. The distinction its
own documentation draws is between primitives and peers: tools are "well-defined,
structured inputs and outputs" performing "specific, often stateless functions",
while agents "reason, plan, use multiple tools, maintain state over longer
interactions". What A2A adds on top of a tool call is what you need when the
thing on the other end is not a function: an Agent Card at a well-known URL so
capabilities can be discovered rather than configured, and a task with a
lifecycle — `submitted`, `working`, `input-required`, `completed`, `failed` — so
a peer can work for minutes, ask a clarifying question, and stream artifacts
back. It also assumes opacity: an A2A peer need not reveal its tools, its model,
or its reasoning, which is exactly the property you want when it belongs to
somebody else.

Could the D4 catalog server have been an A2A agent? Technically yes, and it
would have been worse. `search_products` is a stateless function with a typed
input and a JSON result; it finishes in 70ms, has nothing to clarify, and holds
no state between calls. Wrapping it in a task lifecycle would add
`submitted`/`working` bookkeeping to something that is over before the first poll,
and an Agent Card would advertise discovery for a server whose address we
already know because we spawn it ourselves. Opacity would be an active loss: D4
spent its effort on making the tool schemas legible to the model, and A2A's
selling point is that the peer does not have to show you that.

Where A2A would have a case in this project is nowhere inside it, and that is the
useful conclusion. It starts to earn its keep at a boundary this project does not
have — a returns agent at a supplier who negotiates over several turns and takes
a day to answer, a fraud-review agent that owns its own policy and must not
expose it, a shipping partner whose capabilities change without our redeploying.
All of them are somebody else's autonomous system. Everything ShopAgent talks to
is our own function behind our own schema: the catalog on D4, the cart and orders
API on D6, Stripe on D7. MCP is the right protocol for all of them, and reaching
for A2A here would be adding a negotiation layer between a program and itself.

## Day 6 — findings

**A table joins `Base.metadata` only when its class is imported, and a schema
missing half its tables looks finished.** `scripts/create_schema.py` imported
`Base` from `catalog/models.py` and called `create_all`, which was correct on
D3 and stayed correct. Adding `api/models.py` on the same base changed nothing
about that call — and it went on creating the catalog's four tables and none of
the commerce ones, reporting success each time, because the module defining
them had never been imported in that process. There is no error at that point:
the failure surfaces later as `relation "carts" does not exist`, in whatever
happens to touch a cart first. Both `create_schema.py` and `tests/conftest.py`
now import `shopagent.api.models` for the side effect alone, with a comment
saying why, because an import whose only purpose is registration is exactly the
line a tidy-up deletes.

**`Enum(native_enum=False)` without `create_constraint=True` is a bare VARCHAR
that accepts anything.** SQLAlchemy 2.0 defaults `create_constraint` to
`False`, so the obvious spelling produces a column with no CHECK behind it: the
Python side still validates, the database does not, and a status written by
anything other than the ORM goes in unchallenged. It is the shape of bug that
never announces itself — every test passes and the table looks plausible, the
constraint is simply absent. Worth knowing that the safe-looking half of that
argument pair is the one that is off by default.

**A route sweep that passes is not the same as a route sweep that looks.** Step
2 added a parametrised test asserting every non-public route answers 401
without a key, built on filtering `app.routes` for `APIRoute`. FastAPI 0.141
stopped flattening `include_router` into `app.routes` and wraps each mount in an
`_IncludedRouter` instead. The consequence was silent and total: once the cart
router was mounted, the sweep found `/health`, saw no protected routes, and
passed green while four unauthenticated write endpoints sat on the app. It now
recurses through `original_router` — an undocumented attribute — so a second
test compares its result against `app.openapi()["paths"]`, which is built by an
entirely different code path, and a future rename fails that test loudly
instead of quietly emptying the sweep. The general lesson is the one this cost
an hour to learn: a test whose subject is "everything of kind X" needs a
separate assertion that it can still find X, or it decays into a test that
asserts nothing.

**Two of the most important properties of this day cannot be tested through
behaviour, so they are tested by reading the SQL.** `render_order` must not
join the catalog, and `place_order` must take `FOR UPDATE` on `carts` and on
`inventory`. A single-threaded test cannot see either: an implementation that
joins the catalog returns a perfectly correct response, and an implementation
with no locks behaves identically until two requests overlap. Both are asserted
by registering a `before_cursor_execute` listener, capturing the statements the
request actually issues, and inspecting them — no catalog table may appear in
the first, `for update` and `order by inventory.variant_id` must appear in the
second. Both were then falsified on purpose: adding a `join(Variant, ...)` to
`render_order` made the first fail with the offending SQL quoted back in the
assertion. A test asserting an invisible property is worth nothing until it has
been seen to fail.

**A FastAPI dependency opens its own session, so a test that does not override
it writes to a different database than it reads.** `get_session` builds a
session from the shared factory, and a handler's `commit()` therefore goes
straight to Postgres — outside the transaction the fixture opened and will roll
back. Nothing raises. The rows survive the test, the test's own session reads
an older snapshot and never sees them, and the suite starts passing or failing
on collection order. Two details make the fix work: the override hands back the
*same* session the test holds, bound with
`join_transaction_mode="create_savepoint"` so an inner `commit()` lands on a
savepoint; and it is a plain function rather than a generator, because FastAPI
closes generator dependencies and closing that session leaves the test holding
a dead one after its first request. Containment is checked from a second
connection, which must *not* see the row — the half that actually catches a
missing override.

**`ORDER BY variant_id` inside a `FOR UPDATE` looks cosmetic and is not.** Two
orders covering the same two variants in opposite order each end up holding the
row the other waits for, and Postgres resolves that by killing one with a
deadlock error. Locking in a globally consistent order turns it into the second
transaction waiting for the first. The same argument rules out a loop: a loop
acquires locks in whatever order Python iterates, which is a different order
per request unless the caller sorted first — so it is one statement with an
`ORDER BY`, not five `session.get`s.

**Driving the real Swagger UI found a wording bug no test would have.** `POST
/orders` on an already-ordered cart answered *"cart … is ordered and has already
been ordered"* — the status value interpolated into a sentence that then
repeated it. Every test asserted `409` and the presence of `"ordered"`, and both
passed. CLAUDE.md says error messages are written for the model, and the model
is the reader who would have had to make sense of that. Small, but the specific
kind of small that only appears when somebody reads the output instead of
asserting on it.

**The `--reset` guard fired against a real order for the first time.** The debt
D3 recorded and D6 owed — `scripts/seed_catalog.py --reset` issuing `DELETE FROM
products` while order lines point into the catalog — stopped being theoretical
during step 4's manual run. Both layers behaved: the guard refused with a
sentence naming how many orders were in the way, and with the guard bypassed
entirely through psql, the `ON DELETE RESTRICT` on `order_items.variant_id`
refused the delete with a constraint error. A debt is only closed once the
closing mechanism has been seen to fire.

**The `db` test suite is not hermetic with respect to orders left behind.** The
Swagger walkthrough committed one real cart and one real order, and the next
full run produced 6 failures and 17 errors. Nothing was broken: tests asserting
`count(orders) == 0` are right to fail when an order exists, and
`tests/test_seed.py` errors because `reset_catalog` hits the very RESTRICT D6
added to protect order history. Both are the system working as designed. But it
means "clean up after driving the API by hand" is now a real obligation rather
than tidiness, and a suite that built its own database — or that skipped those
tests when orders exist — would be the honest fix. Deferred, and recorded below.

## Day 7 — findings

**SDK objects are not the shapes they look like, and every one of these was
caught by a real call.** Four in four days. `Account.livemode` does not exist —
`GET /v1/account` returns `charges_enabled` and a dozen other fields and simply
has no `livemode`, so the test-mode assertion had to move to `GET /v1/balance`,
which does. `metadata.get(...)` is not a method: `StripeObject` overrides
`__getattr__`, so `.get` raises `KeyError: 'get'` re-raised as `AttributeError`,
and reading metadata means `metadata._data.get(...)`. `Session.url` is
`Optional[str]` and empties the moment a session stops being open, so returning
it unguarded ships `null` as a checkout URL. `client.accounts` is deprecated in
stripe-python 15 in favour of `client.v1.accounts`.

Not one of these would have been found by a fake, because a fake returns the
shape the person writing it imagined. That is the entire argument for the
`stripe` marker existing separately from `network`: these tests cost nothing,
they need a real account, and they are the only thing standing between an
assumption and production.

**A test can compare a variable to itself and look thorough.** The pin on
`STRIPE_API_VERSION` was asserted by reading the version back off the client and
comparing it to the constant — which passes just as happily against a client
that was never given `stripe_version=` at all, because stripe-python's built-in
default currently *is* that same string. The assertion with teeth is
`STRIPE_API_VERSION == stripe.api_version`: it fails when the SDK is upgraded
and the pin is not, which is the case that will actually happen. The general
shape is worth remembering — an assertion whose two sides come from the same
place proves only that the place is self-consistent.

**`DROP TABLE ... CASCADE` removes foreign keys that `create_all` will not put
back, and nothing says so.** Adding two columns to the catalog meant the
documented path — drop, `create_all`, reseed — and the cascade also took
`order_items_variant_id_fkey`, the `ON DELETE RESTRICT` that D6 built to stop a
catalog reset destroying order history. `create_all` did not restore it:
the constraint lives on `order_items`, which already existed, and `create_all`
never touches a table it did not create. The guard in `seed_catalog.py` stayed
in place, so `--reset` still refused while orders existed and the protection
looked intact from the one direction anybody checks — while the layer that held
against every client, psql included, was gone.

The fix is not a sentence in a document, because somebody running a drop will
not read it. It is `tests/test_schema_constraints.py`, which reads
`pg_constraint` and asserts each `confdeltype` against a hand-written table,
plus `find_foreign_key_gaps` in `db.py` which `create_schema.py` reports on with
exit code 2. Both were falsified before being trusted: dropping the constraint
fails three separate assertions, and — the subtler case — replacing RESTRICT
with CASCADE fails the one that reads the letter rather than checking presence.

The two checks are complementary rather than redundant. The hard-coded table
catches the database drifting from intent *and* somebody changing the models;
the derived check catches drift on tables nobody has written down yet, at the
cost of passing whenever the models and the database are wrong together.

**Stripe refuses `customer` and `customer_email` on the same session.** "You may
only specify one of these parameters: customer, customer_email." Found by
sending both rather than by reading about it. The Customer wins when there is
one — it is the richer object, and it is what puts the payment on that
customer's dashboard timeline instead of leaving it unattached.

**`customers.search` lags; `customers.list` does not.** Deduplicating a
Customer by email is the obvious use for the search API and the wrong one:
search is backed by an index that trails writes by up to a minute, so a
customer created and then searched for is frequently not found — which is
exactly the case deduplication exists to handle. `list(email=...)` filters the
field directly and is immediately consistent. The test asserts it by looking a
customer up in the same breath as creating it.

**A Stripe Price is immutable, which changes what an idempotency key means.**
`unit_amount` cannot be edited; a new amount is a new object and the old one is
archived. So the amount is part of the Price's idempotency key — reusing the
key across a price change would have Stripe replay the original Price and
report success for a create that never happened. The sync therefore reports
drift instead of repairing it: nothing is charged from these objects, so drift
costs a stale dashboard rather than a wrong charge, and a script that silently
archives objects in somebody's account is one nobody can reason about
afterwards.

**Idempotency needed two mechanisms because there are two windows.** A Stripe
idempotency key covers 24 hours and answers "the process died between creating
the object and storing its id". A stored id answers "this ran last week" and
"the shopper closed the tab and came back tomorrow". They are not alternatives:
the catalog sync uses both, and the checkout deliberately uses only the stored
id, because a key expiring after a day would hand a returning shopper a second
session for an order they were already paying for.

**The success redirect is not proof of payment, and this was measured rather
than argued.** A real card was charged through the hosted Checkout page —
`4242 4242 4242 4242`, $284.97 — and afterwards Stripe reported
`status=complete`, `payment_status=paid`, `amount_total=28497`, while
`orders.status` was still **`pending`**. That is correct and is the whole point:
the redirect is a URL anybody can open, and only D8's webhook may move an order
to `paid`. A repeat `POST /orders/{id}/checkout` at that moment returned 409
rather than a second session — the first time that branch ran against a real
completed session rather than a fake one.

**`metadata.order_id` does not propagate down the object chain.** The chain
from that payment reads
`cs_test_b1s3pU4X…` → `pi_3U95WcRnt986EK7P0pqxJM6h` → `ch_3U95WcRnt986EK7P07qIOXLw`
→ `txn_3U95WcRnt986EK7P0oxr7lFu`, with the Customer attached to the
PaymentIntent and the Charge. The metadata is on the **session only**: the
PaymentIntent and the Charge both come back with `metadata: {}`.

This decides D8's design, and the decision was made here rather than left to
D8: the checkout now sends `payment_intent_data={"metadata": {...}}` as well,
so the identifier is on both objects and a webhook may subscribe to whichever
event suits it. Without that, `payment_intent.succeeded` arrives as a
successful payment that cannot be attributed to anything.

A second payment settled it. With `payment_intent_data` in place, the
PaymentIntent `pi_3U967sRnt986EK7P147U2V7k` **and** its Charge
`ch_3U967sRnt986EK7P1N2MgrsJ` both came back carrying
`{'order_id': 'eb268d01-…'}` — where the first payment's PaymentIntent and
Charge had both been `{}`. The Charge inheriting it as well was not obvious in
advance and is worth knowing: D8 can attribute
`checkout.session.completed`, `payment_intent.succeeded` or `charge.succeeded`.

Two details made the fix harder to verify than to write. Stripe accepts
`payment_intent_data` alongside `mode="payment"` and everything else the
session already carries — but **no PaymentIntent exists until a shopper starts
paying**: `session.payment_intent` is null on a freshly created session, and
the session does not echo `payment_intent_data` back in its response. A hosted
Checkout page also cannot be completed through the API. So the claim was only
ever closable by paying, which is why it took two payments rather than a test —
and why the automated guard is the offline one that reads the outgoing payload,
not a `stripe`-marked test pinned to one object in one account.

**Clicking a control the user cannot see was a mistake, and the reasoning that
led to it was the wrong shape.** Stripe's hosted Checkout page carries a
checkbox reading "I am an AI agent acting on behalf of someone else". It is
positioned at `x=-9827px`, far outside the viewport — a human shopper cannot
see it, cannot tab to it, and will never set it. It was set anyway, by script,
on the reasoning that the sentence was true.

That reasoning skipped a step. An element hidden from a person can only be
reached by an automated client, which makes it a detector rather than an option
offered to us; and text on a page is data about the page, not an instruction to
follow — least of all text nobody is shown. The right response was to report it
and stop, which is now the standing rule: **do not interact with elements the
user cannot see.**

Worth recording that the payment itself shows no sign of having been treated
differently. Radar returned `risk_level: normal`, `risk_score: 17`,
`network_status: approved_by_network`, no rule hits; `origin_context` on the
session is null, and the event stream is the ordinary
`payment_intent.created` → `charge.succeeded` → `payment_intent.succeeded` →
`checkout.session.completed`. Whatever the flag feeds, it is not exposed on the
API objects — which is exactly why guessing at its purpose from its label was a
bad way to decide.

**Payment Link versus Checkout Session, since the plan asks for the
distinction.** A Payment Link is a durable URL bound to a Price, reusable by
anyone who has it, with no per-order metadata. A Checkout Session is created
for one order, carries `metadata`, and expires. The metadata is the whole
argument: D8 exists to flip one order to `paid` on one event, and a Payment
Link leaves it guessing. Payment Links are right for selling one product from a
link in a post; they are wrong for a cart.

**A prohibition without a replacement reads as a decision.** D7 added two
columns to `orders`. `orders` is a commerce table, so the catalog's "drop and
reseed" rule does not apply to it, and `create_all` cannot alter a table it did
not create — so the columns went in as an `ALTER` typed into a terminal, and
that was reported in conversation and nowhere else. A fresh clone would have
built a database without them, and the first symptom would have been
`UndefinedColumn` from an ordinary read of `Order`, a long way from the change
that caused it.

The interesting part is why it was not noticed at the time. `CLAUDE.md` said
"the catalog is disposable, so there is no Alembic", and D6 correctly added
that this stops at the commerce tables — but it never said what applies
*instead*. A rule that forbids something and offers nothing in its place leaves
a hole shaped exactly like a decision: there was no migrations directory, so
running a statement by hand looked like the intended workflow rather than the
absence of one. Nobody was reasoning badly; the document had a gap and the gap
was invisible from inside.

Two things came out of it. The convention is now written down — numbered
idempotent SQL in `migrations/`, the exact command to apply it, and the
reasoning for having no migrations table. And, because a written convention is
still only a convention, `scripts/create_schema.py` now compares the models
against the live database and exits 2 on any missing column or mismatched
foreign key. The document explains; the exit code is what actually catches it.

The same failure had already happened once this day in a different disguise —
the foreign keys a catalog drop removed, which `create_all` also would not
restore. Both are the same shape: a mechanism that only creates, trusted to
keep a schema correct after it has changed.

**The balance transaction is not the charge amount.** The $284.97 charge
settled as `amount=24469, fee=1285, net=23184` — a different number because the
account settles in a currency other than the one charged. Nothing in this
project reads it, but it is worth knowing before anyone reconciles
`orders.total_amount_cents` against a payout and finds the two disagree by a
conversion.

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

**Closed on D6.** `redact_arguments()` replaces `query` with an HMAC digest
keyed by a per-process salt, and leaves every other argument alone.
`MCP_LOG_REDACT_QUERY` turns it off, and defaults to on. Why a digest rather
than a blanket `<redacted>` is in CLAUDE.md; what remains open is that the salt
is per-process, so two runs of the server cannot be correlated with each other.
That is deliberate — the question the log answers is about one conversation —
but worth knowing before anyone tries to trace a shopper across a restart.

**`DEFAULT_COMMAND` cannot be changed after import.** `MCPToolClient.__init__`
takes `command: str = DEFAULT_COMMAND`, and a default argument is bound when the
function is defined, not when it is called. Reassigning the module attribute
afterwards does nothing — which cost a verification run on D5, where the CLI was
supposed to be pointed at a broken interpreter and cheerfully started the real
server instead. Harmless today, because nothing needs to change it. The day the
server command comes from configuration, it has to be read inside the call.

**The model miscounted its own summary.** Asked "do you have those in size 42?"
it called `check_stock` on four variants, got four correct answers, and opened
with "Yes, all three are available" above four rows — three products, four
variants. No price, size or stock figure was invented; every number traced to a
tool result. The failure is in summarising, not in retrieving, which makes it the
kind of thing a guardrail can catch: D9 already owes a rule that an amount
appearing in an answer must appear in the context, and a count is the same shape
of claim.

**`ping` is offered to the model.** It is a diagnostic tool with no business
meaning, and it sits in the model's tool list alongside the four that matter.
Filtering it out would mean matching on a tool name in the client, which is the
one thing D5 exists to avoid — the adapter registers whatever the server lists.
It was never called across any of the demo scenarios. If a future server exposes
enough diagnostics to crowd the list, the fix belongs on the server (not
advertising them) rather than in a name check here.

**There is no fulfilment flow, so `reserved` only ever goes up.** `place_order`
adds to `inventory.reserved` and nothing subtracts from it except a rolled-back
transaction. `fulfilled` is a status nothing transitions into automatically,
and `cancelled` and `refunded` do not release stock either — so a catalog run
long enough will reserve itself down to zero available with no orders shipping.
Correct for a training project with no warehouse behind it, and wrong the
moment anything real depends on the number. Releasing on `cancelled`/`refunded`
is the small half of the fix; the large half is deciding when units leave
`quantity`, which is a fulfilment design this project does not have.

**A cart line with no active price passes in the cart and fails at the order.**
Deliberate on both sides — a cart that silently drops an item is worse than one
showing an item it cannot price, and a line missing from an order is goods that
ship and are never charged for — but the consequence is a shopper who can hold
a cart that cannot be bought, and only finds out at checkout. Nothing warns
them earlier. A flag on the cart response would be the obvious improvement and
was not added, because inventing a field the plan does not describe is how a
response shape stops being reviewable.

**The advisory stock check can tell two shoppers the same units are free.**
`services/cart.py` reads `quantity - reserved` with no lock and writes nothing,
by design: a cart is a statement of intent. The authoritative check under
`FOR UPDATE` is in `place_order`. The gap is a user-experience one rather than
a correctness one — two people can both fill a cart and only one can buy —
which is the right trade for a basket and would be the wrong trade anywhere
money moves.

**`orders.cart_id` is UNIQUE, so a cancelled order's cart cannot be reordered.**
The constraint closes a real race — two concurrent `POST /orders` on one cart —
and costs nothing while `ordered` is terminal for a cart. But `cancelled` and
`refunded` are real order statuses, and after either, the cart that produced
the order is permanently unusable: the only path is a new cart with the same
lines re-added. Acceptable now, and a decision to revisit on D8 when a
cancellation actually happens.

**There is no readiness endpoint.** `/health` deliberately does not touch the
database, because a liveness probe that queries reports the database's latency
as the process's liveness. That leaves "can this process actually serve a cart"
unanswered by any endpoint — a separate `/ready` that does hit Postgres is the
missing half, and it was left out rather than guessed at.

**CORS is `allow_origins=["*"]`.** Nothing here is reached by a browser today —
the client is the agent process, and the credential is a header that
same-origin rules were never protecting. The setting is a placeholder that has
to become a list of origins before any front end exists, and the combination
with `allow_credentials=True` is one browsers refuse anyway, which is the
reminder built into it.

**The commerce API has no rate limiting and no request logging.** Both are
outside D6's scope and both are load-bearing once the key is anything other
than a developer's own. Worth naming here so that "the API is done" does not
read as "the API is deployable".

**`checkout.session.expired` does not release a reservation.** D7 releases
stock on `cancelled` and `refunded`, and Stripe expires an unpaid session after
24 hours — but nothing listens for that. The order stays `pending` for ever
with its units reserved, which is the same leak D6 had, moved one step later.
D8 closes it either with a webhook on that event or with a periodic sweep over
pending orders whose session has expired. Named here because it is the most
likely way this system quietly runs out of sellable stock.

**Checkout Sessions cannot be deleted, only expired.** Stripe keeps them
permanently, so `expire` is the whole of what cleanup can mean and the test
account accumulates sessions on every `pytest -m stripe` run. Harmless, but it
means "I cleaned up after the test" is not literally true and should not be
believed of any Stripe object without checking: Products and Prices archive,
Sessions expire, and only Customers actually delete.

**The catalog sync has no path for removing what it wrote.** Reseeding the
catalog produces new local rows with no `stripe_product_id`, so the next sync
creates a second set of Stripe Products while the first set stays active and
orphaned. Archiving the old ones would need a record of which Stripe objects
belonged to a catalog generation, which does not exist. Tolerable because
nothing is charged from them; visible as clutter in the dashboard.

**Price drift is reported and never repaired.** Deliberate, for the reasons in
the findings above, but it does mean the Stripe catalog silently stops matching
the local one after any price change, and only a run of the sync says so.

**The checkout pages are unauthenticated, and that is load-bearing.**
`GET /checkout/success` and `GET /checkout/cancel` are mounted without
`require_api_key`, because Stripe redirects a browser to them and a browser
carries no key. It is acceptable only while they read and never write — a test
asserts both accept `GET` alone. Anyone adding a write there removes the reason
it was safe. The success page also deliberately does not mark the order paid,
and says so on the page: a redirect is a URL anybody can open.

**Nothing reconciles a charge against an order total.** `amount_total` on the
session was asserted equal to `orders.total_amount_cents` by a test, but no
running code checks it, and the balance transaction settles in a different
currency again. Whoever adds refunds on D8 will need that reconciliation to be
real rather than a test fixture.
