"""Run the scenario suite: a second model plays each caller, and the end state is checked.

    uv run python eval/run_scenarios.py                      # every scenario, one run each
    uv run python eval/run_scenarios.py --runs 3             # three runs each, for consistency
    uv run python eval/run_scenarios.py --only fact --limit 4
    uv run python eval/run_scenarios.py --judge              # also ask a model about unsupported claims

The desk uses the model in .env. The caller and the judge use the same server unless
DESK_EVAL_BASE_URL, DESK_EVAL_API_KEY and DESK_EVAL_MODEL say otherwise. A caller that is
weaker than the desk makes a poor test, so use the strongest model you have for it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from enquiry_desk.config import get_settings
from enquiry_desk.evals.runner import per_scenario, run_scenario, unique_phone
from enquiry_desk.evals.scenarios import load_scenarios
from enquiry_desk.evals.stats import summarise
from enquiry_desk.llm.wire import ChatClient, LLMError
from enquiry_desk.runtime import open_desk

HERE = Path(__file__).parent


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, default=HERE / "scenarios.yaml")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--only", help="run one category")
    parser.add_argument("--limit", type=int, help="run the first N scenarios after filtering")
    parser.add_argument("--judge", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    scenarios = [s for s in load_scenarios(args.scenarios) if not args.only or s.category == args.only]
    scenarios = scenarios[: args.limit] if args.limit else scenarios
    helper = ChatClient(
        os.environ.get("DESK_EVAL_BASE_URL", settings.llm_base_url),
        os.environ.get("DESK_EVAL_API_KEY", settings.llm_api_key),
        os.environ.get("DESK_EVAL_MODEL", settings.llm_model),
        timeout=settings.llm_timeout_s,
        min_gap_s=settings.llm_min_gap_s,
        max_retries=settings.llm_max_retries,
    )
    stamp = datetime.now().strftime("%m%d%H%M")
    results = []
    skipped: list[str] = []
    counter = 0
    async with open_desk(settings) as runtime:
        for scenario in scenarios:
            for run in range(1, args.runs + 1):
                counter += 1
                try:
                    result = await run_scenario(
                        runtime, helper, scenario, run, unique_phone(counter), judge_chat=helper if args.judge else None
                    )
                except LLMError as error:
                    # The caller model failed, usually on a rate limit. That is not a result
                    # about the desk, so the scenario is left out and reported as not run.
                    print(f"SKIP  {scenario.id:34} run {run}  not run: {str(error)[:70]}", flush=True)
                    skipped.append(scenario.id)
                    continue
                if any(turn["usage"].get("errors") and "having trouble on my side" in turn["desk"] for turn in result.turns):
                    print(f"SKIP  {scenario.id:34} run {run}  not run: the desk's model call failed", flush=True)
                    skipped.append(scenario.id)
                    continue
                results.append(result)
                failed = [name for name, ok in result.checks.items() if not ok]
                print(
                    f"{'PASS' if result.passed else 'FAIL'}  {scenario.id:34} run {run}  {result.state['turns']} turns  "
                    f"{result.seconds:6.1f} s" + (f"   failed: {', '.join(failed)}" if failed else ""),
                    flush=True,
                )
    await helper.aclose()

    overall = summarise(per_scenario(results))
    by_category = defaultdict(list)
    for result in results:
        by_category[result.category].append(result)
    print(f"\nDesk model {settings.llm_model}, caller model {helper.model}")
    print(f"Pass rate {overall['pass_rate']:.0%}, 95% interval {overall['interval_95'][0]:.0%} to {overall['interval_95'][1]:.0%}, "
          f"over {overall['scenarios']} scenarios and {overall['runs']} runs.")
    if skipped:
        print(f"Not run: {len(skipped)} ({', '.join(skipped)})")
    if args.runs > 1:
        print(f"Passed every run: {overall['passed_every_run']} of {overall['scenarios']}. Passed no run: {overall['passed_no_run']}.")
    for category, items in sorted(by_category.items()):
        print(f"  {category:16} {sum(r.passed for r in items)} of {len(items)} runs")
    failures = defaultdict(int)
    for result in results:
        for name, ok in result.checks.items():
            failures[name.split(" ")[0] if name.startswith(("states", "never", "lead.")) else name] += not ok
    print("Failed checks by kind:", {name: count for name, count in sorted(failures.items(), key=lambda x: -x[1]) if count})

    out = HERE / "results" / f"scenarios_{settings.llm_model.replace(':', '_').replace('/', '_')}_{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(
            {"desk_model": settings.llm_model, "caller_model": helper.model, "search_first": settings.always_search_first,
             "runs_per_scenario": args.runs, "not_run": skipped, "summary": overall, "results": [r.as_dict() for r in results]},
            indent=1, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    turns = [turn for r in results for turn in r.turns]
    costs = [turn["usage"]["cost_usd"] for turn in turns if turn["usage"].get("cost_usd") is not None]
    if costs and results:
        per_call = sorted(sum(t["usage"]["cost_usd"] or 0 for t in r.turns) for r in results)
        print(f"List-price cost: ${sum(costs):.4f} for {len(results)} calls, median ${per_call[len(per_call) // 2]:.5f} a call.")
    times = sorted(t["usage"]["model_ms"] + t["usage"]["tool_ms"] for t in turns)
    firsts = sorted(t["usage"]["first_token_ms"] for t in turns if t["usage"].get("first_token_ms"))
    if times and firsts:
        print(f"Turn time (model and tools): median {times[len(times) // 2] / 1000:.1f} s, 95th percentile {times[int(0.95 * len(times))] / 1000:.1f} s. "
              f"First token: median {firsts[len(firsts) // 2] / 1000:.2f} s.")
    checked = sum(bool(t["problems"]) for t in turns)
    print(f"Reply check objected in {checked} of {len(turns)} turns.")
    print(f"Saved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
