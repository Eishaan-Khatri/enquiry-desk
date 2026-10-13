"""The statistics the evaluation reports. Plain Python, so each can be checked by hand.

Three questions, three tools:
  How sure is this pass rate?              a bootstrap interval over scenarios
  Is the desk consistent, not just lucky?  the share of scenarios that pass every run
  Is model A really better than model B?   McNemar's exact test on the same scenarios
"""

from __future__ import annotations

import random
from math import comb


def bootstrap_interval(values: list[float], *, draws: int = 5000, level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """A percentile bootstrap interval for the mean of `values`.

    Resample the scenarios with replacement many times and take the mean each time. The
    middle 95% of those means is the interval. It makes no assumption about the shape of the
    data, which suits pass rates on a few dozen scenarios.

    One value per scenario, never one per run: three runs of one scenario are not three
    independent observations, and counting them as such would make the interval too narrow.
    """
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(draws))
    tail = (1 - level) / 2
    return means[int(tail * draws)], means[min(draws - 1, int((1 - tail) * draws))]


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact p-value for paired pass/fail results.

    Scenarios that both models pass, or both fail, say nothing about which is better. Only
    the disagreements count: `only_a` scenarios that A passed and B failed, and `only_b` the
    other way. If the models were equally good, each disagreement would be a fair coin toss,
    so the p-value is how often a fair coin gives a split at least this lopsided.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    one_tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * one_tail)


def summarise(per_scenario: dict[str, list[bool]]) -> dict:
    """`per_scenario` maps a scenario id to the pass/fail of each of its runs."""
    if not per_scenario:
        return {"scenarios": 0, "runs": 0}
    rates = [sum(runs) / len(runs) for runs in per_scenario.values()]
    low, high = bootstrap_interval(rates)
    return {
        "scenarios": len(per_scenario),
        "runs": sum(len(runs) for runs in per_scenario.values()),
        "pass_rate": round(sum(rates) / len(rates), 3),
        "interval_95": [round(low, 3), round(high, 3)],
        # A scenario that passes two runs in three is a desk a business cannot rely on.
        "passed_every_run": sum(all(runs) for runs in per_scenario.values()),
        "passed_no_run": sum(not any(runs) for runs in per_scenario.values()),
    }


def compare(a: dict[str, list[bool]], b: dict[str, list[bool]]) -> dict:
    """Paired comparison of two suites on the scenarios they share.

    A scenario counts as passed for a model only if every run passed.
    """
    shared = sorted(set(a) & set(b))
    only_a = [s for s in shared if all(a[s]) and not all(b[s])]
    only_b = [s for s in shared if all(b[s]) and not all(a[s])]
    diffs = [float(all(a[s])) - float(all(b[s])) for s in shared]
    low, high = bootstrap_interval(diffs)
    return {
        "shared_scenarios": len(shared),
        "a_passed": sum(all(a[s]) for s in shared),
        "b_passed": sum(all(b[s]) for s in shared),
        "only_a_passed": only_a,
        "only_b_passed": only_b,
        "difference": round(sum(diffs) / len(diffs), 3) if diffs else 0.0,
        "difference_interval_95": [round(low, 3), round(high, 3)],
        "mcnemar_p": round(mcnemar_exact(len(only_a), len(only_b)), 4),
    }
