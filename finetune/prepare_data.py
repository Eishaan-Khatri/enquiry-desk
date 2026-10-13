"""Step 1. Build train.jsonl and test.jsonl, and print the leak report.

    uv run python finetune/prepare_data.py --public xlam.jsonl --out finetune/data

`--public` is a JSONL export of a public function-calling set in the xLAM format (fields
query, tools, answers). Without it, only the desk's own examples are written, which is enough
to try the evaluation script and far too little to train on.

The desk's own examples are GENERATED from templates. They are kept in separate files and
labelled, so a score on them is never mistaken for a score on real requests.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from enquiry_desk.finetune.data import (
    by_tool_count,
    drop_leaks,
    from_xlam,
    leak_report,
    read_jsonl,
    split_by_tool,
    write_jsonl,
)

DESK_TOOLS = [
    {"name": "search_knowledge", "description": "Search the institute's documents: courses, fees, timings, admission, policies.",
     "parameters": {"query": {"type": "string", "description": "What to look up, in English."}}},
    {"name": "save_lead", "description": "Save or update the caller's details in the CRM.",
     "parameters": {"phone": {"type": "string"}, "name": {"type": "string"}, "student_class": {"type": "string"},
                    "target_exam": {"type": "string"}, "city": {"type": "string"}, "mode": {"type": "string"}}},
    {"name": "list_slots", "description": "List the next open slots for a counselling call.", "parameters": {}},
    {"name": "book_slot", "description": "Book a counselling call in the slot the caller chose.",
     "parameters": {"slot_id": {"type": "string"}}},
    {"name": "request_handoff", "description": "Pass the conversation to a human counsellor.",
     "parameters": {"reason": {"type": "string"}, "summary": {"type": "string"}}},
]

NAMES = ["Asha Verma", "Imran Khan", "Sunita Rao", "Vikram Singh", "Neha Gupta", "Rohit Mehra", "Kavita Joshi", "Pooja Sharma"]
CITIES = ["Noida", "Delhi", "Lucknow", "Agra", "Meerut", "Ghaziabad", "Jaipur", "Patna"]
SMALL_TALK = ["Hello", "Thank you so much", "Okay, got it", "Are you a real person?", "Good morning", "That is all, bye"]


def desk_examples(count: int, seed: int = 0) -> list[dict]:
    """Requests whose right answer has no free text in it, so an exact match is a fair test."""
    rng = random.Random(seed)
    examples = []
    for _ in range(count):
        kind = rng.choice(["lead", "lead", "slots", "book", "none"])
        if kind == "lead":
            name, city = rng.choice(NAMES), rng.choice(CITIES)
            phone = "9" + "".join(rng.choice("0123456789") for _ in range(9))
            grade, exam, mode = rng.choice(["9", "10", "11", "12"]), rng.choice(["JEE", "NEET"]), rng.choice(["classroom", "online"])
            query = f"I am {name}. My child is in class {grade} and wants {exam} coaching, {mode}. We live in {city}. My number is {phone}."
            answers = [{"name": "save_lead", "arguments": {"phone": phone, "name": name, "student_class": grade, "target_exam": exam, "city": city, "mode": mode}}]
        elif kind == "slots":
            query = rng.choice(["When can I speak to a counsellor?", "What slots do you have for a counselling call?", "I would like to book a call. What times are free?"])
            answers = [{"name": "list_slots", "arguments": {}}]
        elif kind == "book":
            day, hour = rng.randint(14, 28), rng.choice(["11:00", "16:00", "18:00"])
            slot = f"2026-10-{day}T{hour}"
            query = f"The open slots are {slot} and 2026-10-{day}T09:00. The caller says: please book the {hour} one."
            answers = [{"name": "book_slot", "arguments": {"slot_id": slot}}]
        else:
            query, answers = rng.choice(SMALL_TALK), []
        examples.append({"query": query, "tools": DESK_TOOLS, "answers": answers, "generated": True})
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, help="JSONL in the xLAM format")
    parser.add_argument("--out", type=Path, default=Path("finetune/data"))
    parser.add_argument("--test-share", type=float, default=0.1)
    parser.add_argument("--desk-examples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.public:
        examples = [from_xlam(row) for row in read_jsonl(args.public)]
        train, test, held_out = split_by_tool(examples, args.test_share, args.seed)
        before = len(train)
        train = drop_leaks(train, test)
        report = leak_report(train, test)
        report |= {"tools_held_out": len(held_out), "train_rows_dropped_as_leaks": before - len(train),
                   "train_by_calls": by_tool_count(train), "test_by_calls": by_tool_count(test)}
        write_jsonl(args.out / "train.jsonl", train)
        write_jsonl(args.out / "test.jsonl", test)
        (args.out / "leak_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("Public data:", json.dumps(report, indent=1))

    desk = desk_examples(args.desk_examples, args.seed)
    write_jsonl(args.out / "desk_test.generated.jsonl", desk)
    print(f"Desk examples (generated): {len(desk)} written, by number of calls {by_tool_count(desk)}")


if __name__ == "__main__":
    main()
