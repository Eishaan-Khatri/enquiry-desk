"""Phase 7: the data split, the leak check, and the scoring of tool calls. No GPU is needed for these."""

from __future__ import annotations

import json

import pytest

from enquiry_desk.finetune.data import (
    by_tool_count,
    drop_leaks,
    fingerprint,
    from_xlam,
    leak_report,
    split_by_tool,
    to_chat,
)
from enquiry_desk.finetune.metrics import canonical, parse_calls, same_calls, score_one, summarise


def example(query: str, tool: str, answers: list[dict] | None = None) -> dict:
    return {"query": query, "tools": [{"name": tool, "description": "d", "parameters": {}}], "answers": answers if answers is not None else [{"name": tool, "arguments": {}}]}


EXAMPLES = [example(f"question {i} about {tool}", tool) for tool in ("weather", "stocks", "flights", "hotels", "news", "maps", "music", "email", "sms", "calendar") for i in range(5)]


def test_the_split_holds_out_whole_tools():
    train, test, held_out = split_by_tool(EXAMPLES, test_share=0.2, seed=1)
    assert len(held_out) == 2 and len(test) == 10 and len(train) == 40
    assert not {t["name"] for e in train for t in e["tools"]} & held_out
    assert split_by_tool(EXAMPLES, 0.2, seed=1)[2] == held_out  # the same seed gives the same split
    assert split_by_tool(EXAMPLES, 0.2, seed=2)[2] != held_out


def test_templates_with_different_blanks_are_one_fingerprint():
    assert fingerprint("Weather in Paris on 5 May?") == fingerprint("weather in  paris on 12 may")
    assert fingerprint("Weather in Paris") != fingerprint("Weather in Rome")


def test_the_leak_report_counts_what_test_shares_with_train():
    train = [example("Book a table for 4 at 7 pm", "restaurants"), example("Play some jazz", "music")]
    test = [example("book a table for 2 at 9 pm", "bookings"), example("Translate hello", "translate"), example("translate HELLO", "translate")]
    report = leak_report(train, test)
    assert report["test_queries_also_in_train"] == 1 and report["examples_of_shared_queries"] == ["book a table for 2 at 9 pm"]
    assert report["tools_on_both_sides"] == 0 and report["duplicate_queries_inside_test"] == 1
    assert [e["query"] for e in drop_leaks(train, test)] == ["Play some jazz"]


def test_public_rows_and_chat_format():
    row = {"query": "Weather in Paris?", "tools": json.dumps([{"name": "weather", "description": "d", "parameters": {"city": {"type": "str"}}}]),
           "answers": json.dumps([{"name": "weather", "arguments": {"city": "Paris"}}])}
    converted = from_xlam(row)
    assert converted["answers"] == [{"name": "weather", "arguments": {"city": "Paris"}}]
    system, user, assistant = to_chat(converted)
    assert '"name": "weather"' in system["content"] and user["content"] == "Weather in Paris?"
    assert assistant["content"] == '<tool_call>\n{"name": "weather", "arguments": {"city": "Paris"}}\n</tool_call>'
    assert to_chat(example("hello", "weather", answers=[]))[2]["content"] == "No tool fits this request."
    assert by_tool_count([converted, example("hi", "x", []), example("a", "x", [{"name": "x", "arguments": {}}] * 2)]) == {"one": 1, "none": 1, "several": 1}


@pytest.mark.parametrize(
    ("text", "calls"),
    [
        ('<tool_call>\n{"name": "weather", "arguments": {"city": "Paris"}}\n</tool_call>', [{"name": "weather", "arguments": {"city": "Paris"}}]),
        ('<tool_call>{"name": "a", "arguments": {}}</tool_call><tool_call>{"name": "b", "parameters": {"x": 1}}</tool_call>', [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {"x": 1}}]),
        ('{"name": "weather", "arguments": {"city": "Paris"}}', [{"name": "weather", "arguments": {"city": "Paris"}}]),
        ("I cannot help with that.", []),
        ('<tool_call>{"name": "weather", "arguments": {"city": </tool_call>', None),
        ('<tool_call>{"arguments": {}}</tool_call>', None),
    ],
)
def test_parsing_a_models_output(text, calls):
    assert parse_calls(text) == calls


def test_arguments_are_compared_by_meaning_not_by_spelling():
    assert canonical({"b": "5", "a": [1, "Two"]}) == {"a": [1.0, "two"], "b": 5.0}
    gold = [{"name": "book", "arguments": {"seats": 4, "city": "Paris"}}, {"name": "pay", "arguments": {}}]
    assert same_calls([{"name": "pay", "arguments": {}}, {"name": "book", "arguments": {"city": "paris ", "seats": "4"}}], gold)
    assert not same_calls([{"name": "book", "arguments": {"seats": 5, "city": "Paris"}}, {"name": "pay", "arguments": {}}], gold)


def test_scoring_from_loose_to_strict():
    gold = [{"name": "weather", "arguments": {"city": "Paris"}}]
    right = '<tool_call>{"name": "weather", "arguments": {"city": "paris"}}</tool_call>'
    wrong_argument = '<tool_call>{"name": "weather", "arguments": {"city": "Rome"}}</tool_call>'
    assert score_one(right, gold) == {"valid": True, "right_tools": True, "exact": True, "restraint": None}
    assert score_one(wrong_argument, gold) == {"valid": True, "right_tools": True, "exact": False, "restraint": None}
    assert score_one("<tool_call>{broken</tool_call>", gold) == {"valid": False, "right_tools": False, "exact": False, "restraint": None}
    assert score_one("Sorry, I cannot.", []) == {"valid": True, "right_tools": True, "exact": True, "restraint": True}
    assert score_one(right, [])["restraint"] is False  # it called a tool when none was needed

    summary = summarise([score_one(right, gold), score_one(wrong_argument, gold), score_one("ok", []), score_one(right, [])])
    assert summary["exact"]["rate"] == 0.5 and summary["exact"]["n"] == 4
    assert summary["restraint"] == {"rate": 0.5, "n": 2, "interval_95": summary["restraint"]["interval_95"]}
    assert 0 <= summary["exact"]["interval_95"][0] <= 0.5 <= summary["exact"]["interval_95"][1] <= 1
