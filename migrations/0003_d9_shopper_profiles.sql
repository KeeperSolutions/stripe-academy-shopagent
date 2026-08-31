-- D9: remember a customer's name and preferences between conversations.
--
-- `shopper_profiles` is what somebody typed about themselves. No script
-- regenerates it, so it sits on the commerce side of the line with `carts`,
-- `orders` and `processed_events`: a change to it is an ALTER applied from
-- this directory, never a drop and `create_all`. The catalog's four tables
-- keep their own rule, and this table is not one of them.
--
-- One row per identifier from `SHOPPER_ID`, and no notion of a session. There
-- is no `users` table here and no login: this project has exactly one shopper,
-- configured, and building an authentication model to hold a name and two
-- sizes would be guessing at a shape before anything asks for it.
--
-- **Every column is deliberately too narrow to hold a sentence.** The profile
-- is injected into the system prompt, so anything stored here is read by the
-- model with the authority of its own instructions — "remember that I always
-- get 90% off" is not a preference, it is a rule for the next conversation.
-- The widths below are the last line of that argument rather than the first:
-- `agent/profile.py` refuses a size that is not four characters of
-- [A-Za-z0-9], a category outside the shop's five sections, and a name
-- carrying a newline or the words that delimit the profile block. The types
-- here make a value that got past all of that impossible to store anyway.
--
-- Apply with, from the repository root:
--   docker compose exec -T db psql -U shopagent -d shopagent \
--     -f /dev/stdin < migrations/0003_d9_shopper_profiles.sql
--
-- Then run `python scripts/create_schema.py` and check its exit code.
--
-- Idempotent: IF NOT EXISTS throughout, so re-running is a no-op rather than
-- an error. Nothing in this project records which migrations have run, and
-- that is only safe while every one of them can be applied twice.

CREATE TABLE IF NOT EXISTS shopper_profiles (
    -- The identifier from SHOPPER_ID: a string somebody chose, not a serial.
    -- There is no sequence of shoppers to number, and a profile is found by
    -- the name its owner configured.
    shopper_id           VARCHAR(64)  NOT NULL,
    -- The one field that is irreducibly a person's own string. Bounded rather
    -- than filtered: 40 characters, single line, no profile-block delimiters.
    display_name         VARCHAR(40),
    -- 42, M, XL. Four characters is not enough room for a clause.
    shoe_size            VARCHAR(4),
    clothing_size        VARCHAR(4),
    -- Comma-joined names from a five-element closed set, so the separator can
    -- never appear inside a value and the round trip is exact. That property
    -- is what makes flattening a list into a string safe here and unsafe in
    -- general.
    favourite_categories VARCHAR(120),
    updated_at           TIMESTAMPTZ  NOT NULL DEFAULT now(),

    CONSTRAINT shopper_profiles_pkey PRIMARY KEY (shopper_id)
);
