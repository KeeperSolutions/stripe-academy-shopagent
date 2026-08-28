-- D8: remember which Stripe webhook deliveries have already been handled.
--
-- `processed_events` holds what actually happened to this server, so it falls
-- on the commerce side of the line: no script regenerates it, and a change to
-- it is an ALTER applied from this directory rather than a drop and
-- `create_all`. The catalog's four tables keep their own rule.
--
-- This is the first *table* the convention creates rather than the first
-- columns it adds, and it is worth being exact about what that changes.
-- `create_all` checks tables one at a time, so it would build this one on any
-- database that lacks it — a fresh clone and a pre-D8 database alike. An
-- earlier version of this comment claimed otherwise, and review corrected it.
--
-- So this file is not the only way the table can come to exist. What it is, is
-- the recorded change: the repository's own statement of what a database built
-- before D8 needs, applied the way every other commerce change is applied, and
-- the thing `0003` will extend when a column is added here. Reaching for
-- `create_all` instead means running a script that will silently build
-- anything else it finds missing, which is the opposite of a change somebody
-- can read before it runs.
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
