-- D8: remember which Stripe webhook deliveries have already been handled.
--
-- `processed_events` holds what actually happened to this server, so it falls
-- on the commerce side of the line: no script regenerates it, and a change to
-- it is an ALTER applied from this directory rather than a drop and
-- `create_all`. The catalog's four tables keep their own rule.
--
-- This is the first *table* the convention creates rather than the first
-- columns it adds, and the two cases differ in one way worth stating.
-- `create_all` does build a table that does not exist, so a fresh clone gets
-- this one without running anything here — which is exactly what makes the
-- file easy to skip. It is not skippable: a database created before today
-- already has every other table, `create_all` leaves it alone, and nothing
-- would say the new one is missing until the first insert failed.
--
-- Apply with, from the repository root:
--   docker compose exec -T db psql -U shopagent -d shopagent \
--     -f /dev/stdin < migrations/0002_d8_processed_events.sql
--
-- Then run `python scripts/create_schema.py` and check its exit code.
--
-- Idempotent: IF NOT EXISTS throughout, so re-running is a no-op rather than
-- an error. Nothing in this project records which migrations have run, and
-- that is only safe while every one of them can be applied twice.

CREATE TABLE IF NOT EXISTS processed_events (
    -- Stripe's `evt_...`. The primary key is the mechanism, not a formality:
    -- the handler inserts before it does the work, so a second delivery of the
    -- same event is refused here rather than by a check in the application
    -- that two concurrent retries would both pass.
    event_id     VARCHAR(255) NOT NULL,
    event_type   VARCHAR(255) NOT NULL,
    livemode     BOOLEAN      NOT NULL,
    processed_at TIMESTAMPTZ  NOT NULL DEFAULT now(),

    CONSTRAINT processed_events_pkey PRIMARY KEY (event_id)
);
