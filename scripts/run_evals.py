"""Run the eval scenarios and print a pass/fail table (D10, step 3).

    python scripts/run_evals.py                    # every scenario
    python scripts/run_evals.py --only happy_path  # one, by name
    python scripts/run_evals.py --list             # what exists, without running

Costs money: every scenario is a real conversation with a real model against
the local API and the local catalog. `--list` is free and is what to reach for
when the question is "what does this suite claim".

The report is written to a file as well as printed, because D9 paid for a chain
run twice after reading the first one with `tail` and losing the part that
mattered.
"""

from __future__ import annotations

import argparse
import sys

from shopagent.config import REPO_ROOT
from shopagent.evals.runner import render, run_all
from shopagent.evals.spec import ScenarioError, load_scenarios

REPORT_PATH = REPO_ROOT / "notes" / "eval-report.txt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", metavar="NAME",
                        help="run just this scenario; repeatable")
    parser.add_argument("--list", action="store_true", help="show the scenarios and exit")
    arguments = parser.parse_args()

    try:
        scenarios = load_scenarios()
    except ScenarioError as refused:
        print(f"[the scenario file is not usable] {refused}", file=sys.stderr)
        return 2

    if arguments.list:
        for scenario in scenarios:
            print(f"{scenario.name}\n    {scenario.asks}")
            for expectation in scenario.expectations:
                print(f"      - {expectation}")
        return 0

    results = run_all(arguments.only)
    report = render(results)
    print(report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report + "\n")
    print(f"\n[report written to {REPORT_PATH}]")

    # Non-zero when anything did not pass, so this is usable from a shell.
    # A skip is not a pass and is not a failure either — it is counted here as
    # a failure to *run*, which is what it is.
    return 0 if all(result.status == "PASS" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
