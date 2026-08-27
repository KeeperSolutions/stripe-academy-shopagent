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
pytest tests/ -v          # 362 tests; add -m network for the 4 that call the API
```

Postgres listens on `localhost:5432` (user / password / db: `shopagent`), with data
stored in the named volume `pgdata`. Stop it with `docker compose down` — the volume
is preserved.

## Commands

```bash
# the agent
python -m shopagent.llm.loop            # CLI agent: local tools + the MCP catalog
MCP_CATALOG_ENABLED=false \
  python -m shopagent.llm.loop          # same CLI, local tools only, no server

# the catalog MCP server on its own
python scripts/run_mcp_server.py        # serves on stdio; the agent starts this itself

# the MCP Inspector, against that server
npx @modelcontextprotocol/inspector \
  .venv/bin/python scripts/run_mcp_server.py               # web UI on :6274
npx @modelcontextprotocol/inspector --cli \
  .venv/bin/python scripts/run_mcp_server.py --method tools/list   # non-interactive

# catalog data
python scripts/create_schema.py         # pgvector + create_all, idempotent
python scripts/seed_catalog.py          # 30 products; --reset to rebuild
python scripts/embed_catalog.py         # vectors + HNSW index; --force to redo

# tests
pytest tests/ -v                        # 362, offline and database
pytest tests/ -m network                # the 4 that call the API and cost money
```

The agent spawns the catalog server itself, so `run_mcp_server.py` is only needed
to drive the server by hand — from the Inspector, or to tell a catalog fault
apart from an agent fault. `MCP_CATALOG_ENABLED=false` is what makes "the product
answers come from MCP" demonstrable: the same binary then has two tools instead
of six and says the catalogue is unavailable.

Pass the server path, not `-m shopagent.mcp_server.server`, to the Inspector: it
parses `-m` as one of its own flags and the module never starts.

Findings, decisions and open questions from each day are in
[JOURNAL.md](JOURNAL.md).
