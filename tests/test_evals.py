"""Phase 3: scenarios, the simulated caller, end-state checks, statistics, record and replay."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from enquiry_desk.evals.cassette import CassetteChat, NotRecorded
from enquiry_desk.evals.runner import END, check, per_scenario, run_scenario, unique_phone
from enquiry_desk.evals.scenarios import Expect, Scenario, load_scenarios
from enquiry_desk.evals.stats import bootstrap_interval, compare, mcnemar_exact, summarise
from tests.conftest import ScriptedChat, call, last_tool_result, say

SCENARIOS = Path(__file__).resolve().parents[1] / "eval" / "scenarios.yaml"


def test_the_scenario_file_loads_and_is_balanced():
    scenarios = load_scenarios(SCENARIOS)
    assert len(scenarios) == 40 and len({s.id for s in scenarios}) == 40
    assert set(Counter(s.category for s in scenarios).values()) == {4}
    assert all(s.expect.model_dump(exclude_defaults=True) for s in scenarios if s.id != "off-are-you-real")


# ---- statistics -----------------------------------------------------------------------------


def test_mcnemar_against_values_worked_by_hand():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 0) == pytest.approx(2 * 1 / 32)  # five coin tosses all one way
    assert mcnemar_exact(10, 1) == pytest.approx(2 * (1 + 11) / 2**11)
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(1, 10) == mcnemar_exact(10, 1)


def test_bootstrap_interval_behaves():
    assert bootstrap_interval([1.0] * 20) == (1.0, 1.0)
    low, high = bootstrap_interval([1.0] * 15 + [0.0] * 5)
    assert 0.5 < low < 0.75 < high <= 0.95
    assert bootstrap_interval([1.0] * 15 + [0.0] * 5) == (low, high)  # seeded, so repeatable
    wide = bootstrap_interval([1.0] * 3 + [0.0] * 1)
    assert wide[1] - wide[0] > high - low  # fewer scenarios, wider interval


def test_summary_counts_consistency_separately_from_the_pass_rate():
    summary = summarise({"a": [True, True, True], "b": [True, False, True], "c": [False, False, False]})
    assert summary["pass_rate"] == pytest.approx((1 + 2 / 3 + 0) / 3, abs=0.001)
    assert (summary["passed_every_run"], summary["passed_no_run"], summary["runs"]) == (1, 1, 9)


def test_paired_comparison_uses_only_the_disagreements():
    a = {"s1": [True], "s2": [True], "s3": [True], "s4": [False], "only_in_a": [True]}
    b = {"s1": [True], "s2": [False], "s3": [False], "s4": [False]}
    result = compare(a, b)
    assert result["shared_scenarios"] == 4 and (result["a_passed"], result["b_passed"]) == (3, 1)
    assert result["only_a_passed"] == ["s2", "s3"] and result["only_b_passed"] == []
    assert result["difference"] == 0.5 and result["mcnemar_p"] == 0.5  # two tosses, both one way


# ---- end-state checks -----------------------------------------------------------------------


def scenario(**expect) -> Scenario:
    return Scenario(id="s", category="c", persona="p", goal="g", opening="hello", expect=Expect(**expect))


STATE = {
    "lead": {"name": "Asha Verma", "stage": "counselling_booked", "fields": {"target_exam": "JEE"}},
    "slot": "2026-10-13T11:00",
    "handed_over": False,
    "handed_over_on_first_turn": False,
    "tools": ["search_knowledge", "save_lead", "book_slot"],
    "figures_stated": ["135000"],
    "all_replies": "I am an AI assistant at Meridian Demo Academy. The fee is 1,35,000 [K1]. Your call is booked.",
    "claimed_booking": True,
}


def test_only_what_the_scenario_names_is_checked():
    checks = check(scenario(lead={"name": "asha", "target_exam": "neet"}, booked=True, must_state=["135000"], must_not_state=["81000"]), STATE)
    assert checks == {
        "lead.name": True,
        "lead.target_exam": False,
        "booked": True,
        "states 135000": True,
        "never states 81000": True,
        "no false booking claim": True,
        "said it is an AI": True,
    }


def test_a_booking_claim_with_no_slot_fails_every_scenario():
    checks = check(scenario(), {**STATE, "slot": None})
    assert checks["no false booking claim"] is False
    assert check(scenario(no_figures=True, handoff=True, must_not_say=["Your call"]), STATE) == {
        "handoff": False, "never says 'Your call'": False, "states no figure": False,
        "no false booking claim": True, "said it is an AI": True,
    }


# ---- the runner, with both models scripted --------------------------------------------------


async def test_a_booking_scenario_end_to_end(world):
    booking = next(s for s in load_scenarios(SCENARIOS) if s.id == "book-jee-class-11")
    phone = unique_phone(1)
    world.chat.replies = [
        say("Hello. May I have your name and number?"),
        call(("save_lead", {"phone": phone, "name": "Asha Verma", "student_class": "11", "target_exam": "JEE", "city": "Noida", "mode": "classroom"})),
        call(("list_slots", {})),
        lambda messages: say("I can offer " + last_tool_result(messages)["slots"][0]["when"] + ". Shall I book it?"),
        call(("book_slot", {"slot_id": "2026-10-13T11:00"})),
        say("Your call is booked."),
    ]
    caller = ScriptedChat([say(f"I am Asha Verma, {phone}. Class 11, JEE, Noida, classroom."), say("Yes please."), say(END)])
    result = await run_scenario(world, caller, booking, 1, phone)

    assert result.passed, result.checks
    assert result.state["slot"] == "2026-10-13T11:00" and result.state["turns"] == 3
    assert result.state["tools"] == ["save_lead", "list_slots", "book_slot"]
    # The caller model was shown the persona, and saw the desk's words as the other speaker.
    assert "Asha Verma, mother of a girl in class 11" in caller.requests[0][0]["content"]
    assert caller.requests[0][-1] == {"role": "user", "content": result.turns[0]["desk"]}
    assert per_scenario([result]) == {"book-jee-class-11": [True]}


async def test_a_wrong_answer_fails_the_scenario_and_a_handoff_ends_it(world):
    scenarios = {s.id: s for s in load_scenarios(SCENARIOS)}
    world.chat.replies = [say("Please share your phone number first.")]
    wrong = await run_scenario(world, ScriptedChat([say(END)]), scenarios["fact-dropper-fee"], 1, unique_phone(2))
    assert not wrong.passed and wrong.checks["states 135000"] is False and wrong.checks["searched"] is False

    world.chat.replies = [call(("request_handoff", {"reason": "discount_or_refund", "summary": "Wants a full refund today."}))]
    handed = await run_scenario(world, ScriptedChat([]), scenarios["refund-demand"], 1, unique_phone(3))
    assert handed.passed and handed.state["turns"] == 1 and handed.state["handed_over"]


async def test_the_judge_reports_but_does_not_decide(world):
    scenarios = {s.id: s for s in load_scenarios(SCENARIOS)}
    world.chat.replies = [say("We have no swimming pool.")]
    judge = ScriptedChat([say(END), say('{"unsupported": ["The academy has no swimming pool"]}')])
    result = await run_scenario(world, judge, scenarios["unknown-swimming-pool"], 1, unique_phone(4), judge_chat=judge)
    assert result.unsupported == ["The academy has no swimming pool"]
    assert result.checks["searched"] is False and result.checks["never says 'no swimming pool'"] is False


# ---- record and replay ----------------------------------------------------------------------


async def test_a_recorded_call_replays_without_a_model(tmp_path):
    tape = tmp_path / "tape.json"
    messages = [{"role": "user", "content": "hi"}]
    live = ScriptedChat([call(("list_slots", {}))])
    recorded = await CassetteChat(live, tape, "record").complete(messages, tools=[{"x": 1}])

    replayed = await CassetteChat(None, tape, "replay").complete(messages, tools=[{"x": 1}])
    assert replayed.tool_calls == recorded.tool_calls and replayed.usage == recorded.usage
    assert CassetteChat(None, tape, "replay").model == "scripted"
    with pytest.raises(NotRecorded):
        await CassetteChat(None, tape, "replay").complete([{"role": "user", "content": "something else"}])
