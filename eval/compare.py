"""Compare two saved scenario runs on the scenarios they share.

    uv run python eval/compare.py eval/results/scenarios_A.json eval/results/scenarios_B.json

It answers "is A really better than B?" with McNemar's exact test and a bootstrap interval
on the difference. A scenario counts as passed only when every run of it passed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from enquiry_desk.evals.runner import per_scenario
from enquiry_desk.evals.stats import compare


def main() -> None:
    a_path, b_path = Path(sys.argv[1]), Path(sys.argv[2])
    a, b = json.loads(a_path.read_text(encoding="utf-8")), json.loads(b_path.read_text(encoding="utf-8"))
    result = compare(per_scenario(a["results"]), per_scenario(b["results"]))
    print(f"A: {a['desk_model']} ({a_path.name})\nB: {b['desk_model']} ({b_path.name})")
    print(f"Shared scenarios: {result['shared_scenarios']}. A passed {result['a_passed']}, B passed {result['b_passed']}.")
    print(f"Only A passed: {result['only_a_passed']}\nOnly B passed: {result['only_b_passed']}")
    low, high = result["difference_interval_95"]
    print(f"Difference in pass rate (A minus B): {result['difference']:+.0%}, 95% interval {low:+.0%} to {high:+.0%}.")
    print(f"McNemar exact p = {result['mcnemar_p']}.")
    if result["mcnemar_p"] >= 0.05:
        print("The data do not show a difference between A and B. That is not the same as showing they are equal.")
    else:
        print("The difference is unlikely to be chance alone, on these scenarios.")


if __name__ == "__main__":
    main()
