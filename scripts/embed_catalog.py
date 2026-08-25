"""Embed the catalog and build the vector index (D3, step 4).

    python scripts/embed_catalog.py            # only products without a vector
    python scripts/embed_catalog.py --force    # re-embed everything

The default run is free once the catalog is embedded: it selects the products
whose `embedding` is NULL, finds none, and makes no API call at all. Use
`--force` after editing a description or changing `EMBEDDING_MODEL`, since
neither of those makes an existing vector NULL.

The HNSW index is created at the end, after the vectors exist — an index over
an empty column has nothing to build. On thirty products it is decorative; see
the note in the README.
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy.exc import OperationalError, ProgrammingError

from shopagent.catalog.embeddings import (
    HNSW_INDEX_NAME,
    embed_products,
    embedded_count,
    ensure_hnsw_index,
    missing_embedding_count,
)
from shopagent.db import get_engine, session_scope
from shopagent.llm.client import LLMClient
from shopagent.llm.usage import UsageTracker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-embed every product, including those that already have a vector",
    )
    args = parser.parse_args(argv)

    # A tracker of this run's own, so the cost printed is this run's cost and
    # not a number carried over from somewhere else in the process.
    tracker = UsageTracker()
    client = LLMClient(tracker=tracker)

    try:
        with session_scope() as session:
            missing = missing_embedding_count(session)
            if not args.force and missing == 0:
                print(
                    f"Every product already has a vector "
                    f"({embedded_count(session)} of them). Nothing to do."
                )
                print("Use --force to embed them again.")
            else:
                summary = embed_products(session, force=args.force, client=client)
                print("Embedding run")
                for line in summary.as_lines():
                    print(line)

            embedded = embedded_count(session)

        if embedded:
            ensure_hnsw_index()
            print(f"\nHNSW index {HNSW_INDEX_NAME}: ready")

        print(f"\nUsage this run: {tracker.summary()}")
    except OperationalError as exc:
        print(f"Cannot reach the database at {get_engine().url}.", file=sys.stderr)
        print("Is Postgres up? Try: docker compose up -d", file=sys.stderr)
        print(f"\n{exc.orig}", file=sys.stderr)
        return 1
    except ProgrammingError as exc:
        print("The catalog tables are missing.", file=sys.stderr)
        print("Run: python scripts/create_schema.py", file=sys.stderr)
        print(f"\n{exc.orig}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
