-- D7: record who is buying on the order itself.
--
-- `orders` is a commerce table and therefore NOT under the catalog's
-- "seed-generated and disposable" rule: it holds what a shopper actually did,
-- and no script regenerates it. A schema change here is an ALTER applied to
-- the existing table, never a drop and `create_all`.
--
-- Both columns are nullable, so this is safe to apply to a table with rows in
-- it: existing orders simply have no buyer recorded, which is exactly what was
-- true of them before D7.
--
-- Apply with:
--   docker compose exec -T db psql -U shopagent -d shopagent \
--     -f /dev/stdin < migrations/0001_d7_order_customer_columns.sql
--
-- Then run `python scripts/create_schema.py` and check its exit code: it
-- reports any column the models declare and the database lacks.
--
-- Idempotent: IF NOT EXISTS means re-running it is a no-op rather than an
-- error, which matters because nothing in this project records which
-- migrations have already run.

ALTER TABLE orders ADD COLUMN IF NOT EXISTS customer_email     VARCHAR(320);
ALTER TABLE orders ADD COLUMN IF NOT EXISTS stripe_customer_id VARCHAR(255);
