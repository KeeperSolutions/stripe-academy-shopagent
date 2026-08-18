# CLAUDE.md

ShopAgent is a conversational commerce agent — find products, manage a cart and
check out through Stripe, all in natural language. It is a ten-day training
project, built without an agent framework so the agent loop stays visible.

## Language

Everything committed to this repo is in English: code, comments, docstrings,
test names, CLI strings, system prompts, README and commit messages. The
planning notes under `notes/` are the one exception, and they are not tracked.

## Layout

| Path under `src/shopagent/` | Day | Purpose |
|---|---|---|
| `config.py` | D1 | pydantic-settings; the only reader of the environment |
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

**Tests reach no network and call no SDK method.** Importing `openai` is fine —
`tests/test_client.py` imports `LLMClient`, which pulls it in — but the client
object is replaced by a fake before any call. The fakes mirror the shape of real
API objects, including the awkward ones, such as the final streaming chunk that
carries usage alongside an empty `choices` list.

## Commands

```bash
docker compose up -d              # Postgres 16 + pgvector
pip install -r requirements.txt
pytest tests/ -v                  # no network needed
python -m shopagent.llm.loop      # the CLI agent
```
