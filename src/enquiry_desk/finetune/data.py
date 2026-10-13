"""Training and test data for the tool-calling fine-tune: one format, a split, and a leak check.

Every example is one request and the tool calls a good assistant would make for it:

    {"query": "...", "tools": [{"name", "description", "parameters"}], "answers": [{"name", "arguments"}]}

`answers` is empty when the right move is to call nothing. Those examples matter: a model
that calls a tool for every message is as wrong as one that never does.

The split is by TOOL, never by row. If rows were split at random, the test set would hold
new questions about tools the model trained on. It would then measure whether the model
memorised those tools, and call it "tool calling". Holding out whole tools tests the real
skill: reading a schema it has never seen.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from collections.abc import Iterable


def from_xlam(row: dict) -> dict:
    """One row of Salesforce's xlam-function-calling-60k, where tools and answers are JSON strings."""
    tools = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
    answers = json.loads(row["answers"]) if isinstance(row["answers"], str) else row["answers"]
    return {"query": row["query"], "tools": tools, "answers": answers}


def tool_names(example: dict) -> set[str]:
    return {tool["name"] for tool in example["tools"]}


def split_by_tool(examples: list[dict], test_share: float = 0.1, seed: int = 0) -> tuple[list[dict], list[dict], set[str]]:
    """Hold out a share of the tools. Returns (train, test, held-out tool names).

    An example goes to the test set when any of its tools is held out, and to the training set
    only when none is. So no held-out tool is ever seen in training, at the cost of a test set
    somewhat larger than the share asked for.
    """
    names = sorted({name for example in examples for name in tool_names(example)})
    random.Random(seed).shuffle(names)
    held_out = set(names[: max(1, round(len(names) * test_share))])
    train = [e for e in examples if not (tool_names(e) & held_out)]
    test = [e for e in examples if tool_names(e) & held_out]
    return train, test, held_out


def fingerprint(text: str) -> str:
    """The same for two queries that differ only in case, spacing, punctuation or digits.

    "Weather in Paris on 5 May?" and "weather in paris on 12 may" are one template with the
    blanks filled in differently. Digits are replaced with a marker for that reason.
    """
    text = re.sub(r"\d+", "#", text.lower())
    text = re.sub(r"[^\w#]+", " ", text).strip()
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def leak_report(train: list[dict], test: list[dict]) -> dict:
    """What the test set shares with the training set. All three counts should be zero or near it.

    This is the check the ISOT audit taught: before trusting a score, look for what lets the
    model pass without the skill.
    """
    train_queries = {fingerprint(e["query"]) for e in train}
    train_tools = {name for e in train for name in tool_names(e)}
    same_query = [e["query"] for e in test if fingerprint(e["query"]) in train_queries]
    seen_tools = sorted({name for e in test for name in tool_names(e)} & train_tools)
    test_prints = [fingerprint(e["query"]) for e in test]
    return {
        "train": len(train),
        "test": len(test),
        "test_queries_also_in_train": len(same_query),
        "examples_of_shared_queries": same_query[:5],
        # A tool can still appear on both sides when a test example pairs a held-out tool with
        # a common one. The held-out tool is new to the model. The common one is not.
        "tools_on_both_sides": len(seen_tools),
        "duplicate_queries_inside_test": len(test_prints) - len(set(test_prints)),
    }


def drop_leaks(train: list[dict], test: list[dict]) -> list[dict]:
    """The training set without the queries that also appear, as templates, in the test set."""
    test_queries = {fingerprint(e["query"]) for e in test}
    return [e for e in train if fingerprint(e["query"]) not in test_queries]


SYSTEM = (
    "You can call the tools described below. To call one, reply with a JSON object on its own, "
    'like {"name": "tool_name", "arguments": {...}}, wrapped in <tool_call> tags. Use one pair of tags per call. '
    "If no tool fits the request, answer in plain words and call nothing.\n\nTools:\n"
)


def to_chat(example: dict) -> list[dict]:
    """The example as the chat messages a model is trained on: system, user, assistant."""
    calls = "\n".join(f"<tool_call>\n{json.dumps(answer, ensure_ascii=False)}\n</tool_call>" for answer in example["answers"])
    return [
        {"role": "system", "content": SYSTEM + json.dumps(example["tools"], ensure_ascii=False)},
        {"role": "user", "content": example["query"]},
        {"role": "assistant", "content": calls or "No tool fits this request."},
    ]


def write_jsonl(path, rows: Iterable[dict]) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def by_tool_count(examples: list[dict]) -> dict[str, int]:
    """How many examples call zero, one, or several tools. The three are scored separately."""
    counts: dict[str, int] = defaultdict(int)
    for example in examples:
        n = len(example["answers"])
        counts["none" if n == 0 else "one" if n == 1 else "several"] += 1
    return dict(counts)
