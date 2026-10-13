"""Scoring a model's tool calls against the right ones.

Four numbers, from loose to strict:
  valid        the output could be parsed at all
  right_tools  it named the right tools, arguments aside
  exact        every call matches: the name and every argument
  restraint    on requests that need no tool, it called none

`exact` is the one that matters in the desk: a booking with the right function and the wrong
slot is a wrong booking.
"""

from __future__ import annotations

import json
import re

from enquiry_desk.evals.stats import bootstrap_interval

TAGGED = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_calls(text: str) -> list[dict] | None:
    """The tool calls in a model's raw text. None means it tried to call and wrote something unreadable."""
    blocks = TAGGED.findall(text)
    if not blocks:
        stripped = text.strip()
        if not stripped.startswith(("{", "[")):
            return []  # plain words: no call was attempted
        blocks = [stripped]
    calls: list[dict] = []
    for block in blocks:
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            return None
        for item in parsed if isinstance(parsed, list) else [parsed]:
            if not isinstance(item, dict) or "name" not in item:
                return None
            calls.append({"name": item["name"], "arguments": item.get("arguments", item.get("parameters", {}))})
    return calls


def canonical(value):
    """Arguments in one form, so that 5 and "5", or keys in another order, do not count as errors."""
    if isinstance(value, dict):
        return {key: canonical(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonical(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        return text.lower()


def same_calls(predicted: list[dict], gold: list[dict]) -> bool:
    """True when the two lists hold the same calls. The order of calls does not matter."""

    def key(call: dict) -> str:
        return json.dumps({"name": call["name"], "arguments": canonical(call.get("arguments") or {})}, sort_keys=True)

    return sorted(map(key, predicted)) == sorted(map(key, gold))


def score_one(output: str, gold: list[dict]) -> dict[str, bool | None]:
    calls = parse_calls(output)
    valid = calls is not None
    calls = calls or []
    return {
        "valid": valid,
        "right_tools": valid and sorted(c["name"] for c in calls) == sorted(g["name"] for g in gold),
        "exact": valid and same_calls(calls, gold),
        # Only defined for requests that need no tool. None keeps the others out of the average.
        "restraint": (valid and not calls) if not gold else None,
    }


def summarise(scores: list[dict]) -> dict:
    """Each measure as a rate over the examples it applies to, with a 95% bootstrap interval."""
    out: dict = {"examples": len(scores)}
    for measure in ("valid", "right_tools", "exact", "restraint"):
        values = [float(score[measure]) for score in scores if score[measure] is not None]
        if not values:
            continue
        low, high = bootstrap_interval(values, draws=2000)
        out[measure] = {"rate": round(sum(values) / len(values), 4), "n": len(values), "interval_95": [round(low, 4), round(high, 4)]}
    return out
