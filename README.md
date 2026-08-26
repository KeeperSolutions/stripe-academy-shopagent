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
pytest tests/ -v          # 254 tests; add -m network for the 4 that call the API
```

Postgres listens on `localhost:5432` (user / password / db: `shopagent`), with data
stored in the named volume `pgdata`. Stop it with `docker compose down` — the volume
is preserved.

Findings, decisions and open questions from each day are in
[JOURNAL.md](JOURNAL.md).
