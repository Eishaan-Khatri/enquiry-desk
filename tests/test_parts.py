"""Tools, pricing, metering, the knowledge base, the CRM, slots and the grounding check."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from enquiry_desk.agent import grounding
from enquiry_desk.booking import Slots, SlotTaken, spoken
from enquiry_desk.crm import LocalCRM, normalise_phone
from enquiry_desk.db import Database
from enquiry_desk.kb.chunking import chunk_markdown
from enquiry_desk.llm.pricing import Price, cost_usd
from enquiry_desk.llm.tools import Tool, ToolError, ToolRegistry
from enquiry_desk.llm.wire import Completion, ToolCall, Usage
from enquiry_desk.metering import Meter
from tests.conftest import NOW

# ---- tools ----------------------------------------------------------------------------------


class EchoArgs(BaseModel):
    word: str = Field(min_length=2, description="A word.")
    times: int = 1
    note: str | None = None


async def echo(args: EchoArgs, context) -> dict:
    if args.word == "boom":
        raise RuntimeError("secret internal detail")
    if args.word == "no":
        raise ToolError("That word is not allowed.")
    return {"said": " ".join([args.word] * args.times)}


REGISTRY = ToolRegistry([Tool("echo", "Repeat a word.", EchoArgs, echo)])


def test_schema_is_reduced_to_what_every_server_accepts():
    parameters = REGISTRY.specs()[0]["function"]["parameters"]
    assert parameters["required"] == ["word"]
    assert parameters["properties"]["note"] == {"type": "string"}  # was "string or null"
    assert parameters["properties"]["word"] == {"type": "string", "minLength": 2, "description": "A word."}
    assert "title" not in json.dumps(parameters) and "default" not in json.dumps(parameters)


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ('{"word": "hi", "times": 2}', None),
        ('{"word": "hi", "times": ', "invalid_json"),
        ("[1, 2]", "invalid_json"),
        ('{"times": 2}', "invalid_arguments"),
        ('{"word": "hi", "times": "many"}', "invalid_arguments"),
        ('{"word": "no"}', "tool_error"),
        ('{"word": "boom"}', "internal_error"),
    ],
)
async def test_every_bad_call_becomes_a_result_the_model_can_read(arguments, error):
    result = await REGISTRY.execute(ToolCall("c1", "echo", arguments))
    assert result.error == error and result.ok is (error is None)
    assert result.as_message()["tool_call_id"] == "c1"
    assert "secret internal detail" not in result.content
    if error == "invalid_arguments":
        assert json.loads(result.content)["fields"][0]["field"] in ("word", "times")


async def test_unknown_tool_names_the_real_ones():
    result = await REGISTRY.execute(ToolCall("c1", "delete_everything", "{}"))
    assert result.error == "unknown_tool" and "echo" in result.content


@pytest.mark.parametrize(
    ("text", "recovered"),
    [
        ('echo {"word": "hi"}', '{"word": "hi"}'),
        ('{"name": "echo", "arguments": {"word": "hi"}}', '{"word": "hi"}'),
        ('<tool_call>\n{"name": "echo", "arguments": {"word": "hi"}}\n</tool_call>', '{"word": "hi"}'),
        ('```json\n{"name": "echo", "parameters": {"word": "hi"}}\n```', '{"word": "hi"}'),
        ('I will call echo {"word": "hi"} now.', None),  # a sentence, not a call
        ('delete_everything {"word": "hi"}', None),  # not one of the tools
        ('echo {"word": ', None),  # not valid JSON
        ("The fee is 1,35,000.", None),
    ],
)
def test_a_tool_call_written_as_text_is_recovered(text, recovered):
    from enquiry_desk.llm.tools import recover_text_calls

    done = recover_text_calls(Completion(text, [], "stop", Usage(1, 1, 0), "m", None, 1.0), REGISTRY.names())
    if recovered is None:
        assert done.tool_calls == [] and done.content == text
    else:
        assert done.content == "" and done.finish_reason == "tool_calls"
        assert [(c.name, c.arguments) for c in done.tool_calls] == [("echo", recovered)]


# ---- cost and metering ----------------------------------------------------------------------


def test_cost_counts_cached_tokens_at_their_own_rate():
    price = Price(input=1.0, output=4.0, cached_input=0.1)
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=500_000, cached_tokens=400_000)
    assert cost_usd(price, usage) == pytest.approx(0.6 * 1.0 + 0.4 * 0.1 + 0.5 * 4.0)
    assert cost_usd(None, usage) is None and cost_usd(price, None) is None


def test_turn_totals_and_unknown_cost():
    meter = Meter(Database(), {"priced": Price(input=1.0, output=2.0)})
    done = Completion("x", [], "stop", Usage(1000, 100, 0), "priced", ttft_ms=40.0, latency_ms=300.0)
    meter.record_llm("s1", 1, done)
    meter.record_llm("s1", 1, done)
    turn = meter.summary("s1", 1)
    assert turn["model_calls"] == 2 and turn["prompt_tokens"] == 2000 and turn["first_token_ms"] == 40.0
    assert turn["cost_usd"] == pytest.approx(2 * (1000 * 1.0 + 100 * 2.0) / 1e6)

    # One call on a model with no price makes the session total unknown, not smaller.
    meter.record_llm("s1", 2, Completion("x", [], "stop", Usage(10, 1, 0), "unpriced", None, 5.0))
    assert meter.summary("s1")["cost_usd"] is None
    assert meter.summary("s1", 1)["cost_usd"] is not None


# ---- knowledge base -------------------------------------------------------------------------


def test_chunks_stay_inside_a_section_and_keep_tables_whole():
    text = "# Fees\n\nIntro.\n\n## Classroom\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n## Online\n\n" + "word " * 200 + "\n\nlast paragraph"
    chunks = chunk_markdown("t", "fees.md", text, max_words=50)
    assert [c.section for c in chunks] == ["Fees", "Fees > Classroom", "Fees > Online", "Fees > Online"]
    assert "| 1 | 2 |" in chunks[1].text and chunks[1].source == "fees.md > Fees > Classroom"
    assert [c.id for c in chunks] == [c.id for c in chunk_markdown("t", "fees.md", text, max_words=50)]


def test_search_finds_the_right_section_and_respects_the_tenant(kb, tenant):
    sources = [hit.source for hit in kb.search(tenant.id, "refund if a student leaves the batch")]
    assert sources[0] == "policies.md > Policies > Refunds"
    assert kb.search("another_tenant", "refund") == []


def test_indexing_twice_adds_nothing(kb, tenant):
    before = kb.count(tenant.id)
    chunks = chunk_markdown(tenant.id, "extra.md", "# Extra\n\nOne paragraph about parking.")
    assert kb.replace(tenant.id, chunks) == 1 and kb.count(tenant.id) == 1
    assert kb.replace(tenant.id, chunks) == 1 and kb.count(tenant.id) == 1
    assert before > 10


# ---- CRM and slots --------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["9876543210", "98765 43210", "09876543210", "+91-98765-43210", "919876543210"])
def test_one_number_in_any_form_is_one_key(raw):
    assert normalise_phone(raw) == "+919876543210"


@pytest.mark.parametrize("raw", ["12345", "1234567890", "98765432101", ""])
def test_bad_numbers_are_refused(raw):
    with pytest.raises(ValueError):
        normalise_phone(raw)


async def test_saving_a_lead_twice_merges_and_never_blanks_a_field():
    crm = LocalCRM(Database())
    first = await crm.upsert_lead("+919876543210", "Asha", {"city": "Noida", "target_exam": ""})
    second = await crm.upsert_lead("+919876543210", "", {"city": "", "target_exam": "JEE"})
    assert second.id == first.id and second.name == "Asha"
    assert second.fields == {"city": "Noida", "target_exam": "JEE"}


def test_slots_skip_sunday_and_cannot_be_booked_twice():
    slots = Slots(Database())
    assert slots.seed(NOW, ["11:00", "16:00"]) == 14  # eight days, one of them a Sunday
    assert slots.seed(NOW, ["11:00", "16:00"]) == 0
    free = slots.open_slots(NOW, limit=50)
    assert free[0] == {"slot_id": "2026-10-13T11:00", "when": "Tuesday 13 October, 11:00 am"}
    assert not any(slot["slot_id"].startswith("2026-10-18") for slot in free)

    slots.book("2026-10-13T11:00", "lead-1")
    with pytest.raises(SlotTaken):
        slots.book("2026-10-13T11:00", "lead-2")
    with pytest.raises(SlotTaken):
        slots.book("2030-01-01T11:00", "lead-2")
    assert spoken("2026-10-14T16:00") == "Wednesday 14 October, 4:00 pm"


# ---- grounding check ------------------------------------------------------------------------

EVIDENCE = ["my number is 9876543210, she is in class 11", '{"text": "JEE Dropper Programme | 1,35,000. GST is 18%."}']


def test_figures_from_the_sources_pass_in_any_format():
    reply = "The JEE Dropper Programme costs Rs 135,000 including 18% GST [K1]. I have your number as +91 98765 43210."
    assert grounding.check(reply, EVIDENCE, {"K1"}) == []


def test_an_invented_figure_and_an_invented_reference_are_caught():
    problems = grounding.check("The fee is 1,20,000 and the batch has 25 seats [K7].", EVIDENCE, {"K1"})
    assert problems == [
        "the reference [K7] is not one of the search results",
        "the figure 25 is not in any source",
        "the figure 120000 is not in any source",
    ]


def test_a_phone_number_the_caller_never_gave_is_caught():
    assert grounding.check("I will call 9123456780.", EVIDENCE, set()) == [
        "the phone number 9123456780 was not given by the caller"
    ]


def test_single_digits_are_let_through_and_marks_can_be_stripped():
    assert grounding.check("I have 3 options for you.", EVIDENCE, set()) == []
    assert grounding.strip_refs("It costs 1,35,000 [K1]. Classes are daily [K2].") == "It costs 1,35,000. Classes are daily."


@pytest.mark.parametrize(
    ("reply", "claims"),
    [
        ("I have booked a counselling call for you.", True),
        ("Your call is booked for Tuesday.", True),
        ("Your slot has been confirmed.", True),
        ("Aapki call book kar di hai.", True),
        ("Shall I book it for you?", False),
        ("Would you like to book a counselling call?", False),
        ("A call can be booked once you pick a slot.", False),
    ],
)
def test_a_booking_may_be_announced_only_after_it_happened(reply, claims):
    assert bool(grounding.check(reply, [], set(), booked=False)) is claims
    assert grounding.check(reply, [], set(), booked=True) == []
