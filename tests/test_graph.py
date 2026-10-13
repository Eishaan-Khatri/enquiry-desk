"""Whole turns through the graph, with the model scripted."""

from __future__ import annotations

from fastapi.testclient import TestClient

from enquiry_desk.agent import prompts
from enquiry_desk.api import create_app
from enquiry_desk.llm.wire import LLMError
from tests.conftest import NOW, call, last_tool_result, make_world, say


def cite_fee(messages: list[dict]):
    """Answer with the reference of whichever search result holds the dropper fee."""
    results = last_tool_result(messages)["results"]
    ref = next(r["ref"] for r in results if "1,35,000" in r["text"])
    return say(f"The JEE Dropper Programme costs 1,35,000 in the classroom [{ref}].")


async def history(world, session_id: str) -> list[dict]:
    snapshot = await world.graph.aget_state({"configurable": {"thread_id": session_id}})
    return snapshot.values["messages"]


async def test_search_then_an_answer_with_a_citation(world):
    world.chat.replies = [call(("search_knowledge", {"query": "JEE Dropper Programme classroom fee"})), cite_fee]
    result = await world.desk.handle("s1", "What is the fee for the JEE dropper batch?")

    assert result.status == "ok" and "1,35,000" in result.reply and result.problems == []
    assert result.citations[0]["source"] == "fees.md > Fees > Classroom fees"
    assert (result.usage["model_calls"], result.usage["tool_calls"], result.usage["errors"]) == (2, 1, 0)
    assert result.usage["prompt_tokens"] == 200 and result.usage["cost_usd"] is None  # "scripted" has no price
    assert [m["role"] for m in await history(world, "s1")] == ["user", "assistant", "tool", "assistant"]
    # The second request shows the model its own tool call and the result, in wire format.
    assert world.chat.requests[1][0]["role"] == "system"
    assert world.chat.requests[1][-1]["role"] == "tool" and world.chat.requests[1][-2]["tool_calls"]


async def test_an_invented_figure_is_sent_back_once_then_replaced(world):
    world.chat.replies = [say("The dropper batch costs 99,999."), say("Sorry, it is 88,888.")]
    result = await world.desk.handle("s1", "What is the fee?")

    assert result.reply == prompts.disclosure(world.tenant) + prompts.NOT_GROUNDED
    assert result.problems == ["the figure 88888 is not in any source"]
    # The second request carried the failed draft and the checker's note.
    assert "CHECKER NOTE" in world.chat.requests[1][-1]["content"]
    assert world.chat.requests[1][-2] == {"role": "assistant", "content": "The dropper batch costs 99,999."}
    # Neither failed draft is in the saved history. The caller never saw them.
    assert [m["content"] for m in await history(world, "s1")] == ["What is the fee?", result.reply]


async def test_the_retry_may_search_and_then_pass(world):
    world.chat.replies = [
        say("It costs 99,999."),
        call(("search_knowledge", {"query": "JEE Dropper Programme classroom fee"})),
        cite_fee,
    ]
    result = await world.desk.handle("s1", "What is the dropper fee?")

    assert "1,35,000" in result.reply and result.problems == [] and result.citations
    # On the third request the note sits where the draft was written, before the new tool call.
    roles = [m["role"] for m in world.chat.requests[2]]
    assert roles == ["system", "user", "assistant", "user", "assistant", "tool"]
    assert [m["role"] for m in await history(world, "s1")] == ["user", "assistant", "tool", "assistant"]


async def test_a_bad_phone_number_goes_back_to_the_model_which_corrects_it(world):
    world.chat.replies = [
        call(("save_lead", {"phone": "12345", "name": "Asha"})),
        call(("save_lead", {"phone": "98765 43210", "name": "Asha", "target_exam": "JEE", "grade": "11"})),
        call(("save_lead", {"phone": "98765 43210", "name": "Asha", "target_exam": "JEE"})),
        lambda messages: say("Thank you Asha. Which class is the student in?"),
    ]
    result = await world.desk.handle("s1", "I am Asha, my number is 98765 43210, for JEE")

    assert result.reply.endswith("Which class is the student in?") and result.usage["errors"] == 2
    lead = await world.crm.find_by_phone("+919876543210")
    assert lead.name == "Asha" and lead.fields == {"target_exam": "JEE"}
    saved = last_tool_result(world.chat.requests[3])
    assert saved["still_missing"] == ["student_class", "city", "mode"]


async def test_a_booking_spread_over_three_turns(world):
    world.chat.replies = [
        call(("save_lead", {"phone": "9876543210", "name": "Asha", "student_class": "11"})),
        say("Thank you. Would you like a counselling call?"),
        call(("list_slots", {})),
        lambda messages: say("I can offer " + last_tool_result(messages)["slots"][0]["when"] + ". Shall I book it?"),
        call(("book_slot", {"slot_id": "2026-10-13T11:00"})),
        lambda messages: say("Done. Your call is on " + last_tool_result(messages)["when"] + "."),
    ]
    first = await world.desk.handle("s1", "Asha here, 9876543210, my son is in class 11")
    second = await world.desk.handle("s1", "yes please")
    third = await world.desk.handle("s1", "ok book it")

    assert (first.turn, second.turn, third.turn) == (1, 2, 3)
    assert second.reply == "I can offer Tuesday 13 October, 11:00 am. Shall I book it?" and second.problems == []
    assert third.reply == "Done. Your call is on Tuesday 13 October, 11:00 am."
    # The lead id was saved in turn 1 and is still in the state in turn 3, so the booking works.
    lead = await world.crm.find_by_phone("+919876543210")
    assert lead.stage == "counselling_booked"
    assert world.crm.notes(lead.id) == ["Counselling call booked for Tuesday 13 October, 11:00 am."]
    assert world.slots.open_slots(NOW)[0]["slot_id"] == "2026-10-13T16:00"
    assert world.meter.summary("s1")["model_calls"] == 6


async def test_booking_without_a_saved_lead_is_refused_by_the_tool(world):
    world.chat.replies = [call(("book_slot", {"slot_id": "2026-10-13T11:00"})), say("May I have your phone number first?")]
    result = await world.desk.handle("s1", "book me the first slot")

    assert result.usage["errors"] == 1 and "phone number" in result.reply
    assert world.slots.open_slots(NOW)[0]["slot_id"] == "2026-10-13T11:00"


async def test_saying_booked_before_booking_is_sent_back(world):
    world.chat.replies = [
        call(("save_lead", {"phone": "9876543210", "name": "Asha"})),
        say("I have booked your counselling call."),
        call(("book_slot", {"slot_id": "2026-10-13T11:00"})),
        say("I have booked your counselling call."),
    ]
    result = await world.desk.handle("s1", "Asha here, 9876543210, book me the first slot")

    assert result.reply.endswith("I have booked your counselling call.") and result.problems == []
    assert "book_slot has not succeeded" in world.chat.requests[2][-1]["content"]
    assert world.slots.open_slots(NOW)[0]["slot_id"] == "2026-10-13T16:00"  # the 11:00 slot is now taken

    # In a later turn the call really is booked, so saying so is allowed.
    world.chat.replies = [say("Yes, your call is booked.")]
    assert (await world.desk.handle("s1", "is it confirmed?")).problems == []


async def test_a_phone_number_the_caller_never_gave_cannot_be_saved(world):
    world.chat.replies = [
        call(("save_lead", {"phone": "9123456780", "name": "Unknown"})),
        say("May I have your phone number, please?"),
    ]
    result = await world.desk.handle("s1", "I want a refund, my son studies with you")

    assert result.usage["errors"] == 1 and await world.crm.find_by_phone("+919123456780") is None
    assert "has not given that phone number" in world.chat.requests[1][-1]["content"]

    # Once the caller gives it, in whatever spacing, the same call goes through.
    world.chat.replies = [call(("save_lead", {"phone": "9123456780", "name": "Ravi"})), say("Thank you Ravi.")]
    await world.desk.handle("s1", "ok it is 91234 56780")
    assert (await world.crm.find_by_phone("+919123456780")).name == "Ravi"


async def test_a_tool_call_that_arrives_as_text_still_runs(world):
    world.chat.replies = [say('search_knowledge {"query": "JEE Dropper Programme classroom fee"}'), cite_fee]
    result = await world.desk.handle("s1", "What is the dropper fee?")

    assert "1,35,000" in result.reply and "search_knowledge" not in result.reply
    assert (result.usage["model_calls"], result.usage["tool_calls"]) == (2, 1)


async def test_handoff_pauses_the_graph_until_a_person_replies(world):
    world.chat.replies = [
        call(("request_handoff", {"reason": "discount_or_refund", "summary": "Parent wants a refund decision."})),
        say("Welcome back. How else can I help?"),
    ]
    paused = await world.desk.handle("s1", "I want my fee refunded, let me talk to someone")
    assert paused.status == "awaiting_operator" and paused.reply == world.tenant.handoff_line

    # While a person has the conversation, the model is not called at all.
    waiting = await world.desk.handle("s1", "hello? anyone?")
    assert waiting.status == "awaiting_operator" and len(world.chat.requests) == 1

    answered = await world.desk.operator_reply("s1", "This is Neha from the centre. I will call you at 5 pm.")
    assert answered.status == "ok" and answered.reply.startswith("This is Neha")

    after = await world.desk.handle("s1", "thank you")
    assert after.status == "ok" and after.reply == "Welcome back. How else can I help?" and after.turn == 2
    roles = [m["role"] for m in await history(world, "s1")]
    assert roles == ["user", "assistant", "tool", "assistant", "user", "assistant"]


async def test_a_model_outage_gives_the_caller_a_plain_apology(world):
    world.chat.replies = [LLMError("model server returned 503", status=503)]
    result = await world.desk.handle("s1", "hello")

    assert result.reply.endswith(prompts.MODEL_FAILED) and result.usage["errors"] == 1


async def test_a_model_that_never_stops_calling_tools_is_cut_off(world):
    world.chat.replies = [call(("list_slots", {}))] * 10
    result = await world.desk.handle("s1", "hello")

    assert result.reply.endswith(prompts.TOO_MANY_STEPS)
    assert result.usage["model_calls"] == 4 and len(world.chat.replies) == 6


async def test_the_first_reply_always_says_it_is_an_ai(world):
    world.chat.replies = [say("Hello, how can I help?"), say("Is there anything else?"), say("I am an AI assistant. Hello.")]
    first = await world.desk.handle("s1", "hi")
    second = await world.desk.handle("s1", "ok")
    already = await world.desk.handle("s2", "hi")

    assert first.reply == "I am an AI assistant at Meridian Demo Academy. Hello, how can I help?"
    assert second.reply == "Is there anything else?"  # only the first reply of a conversation
    assert already.reply == "I am an AI assistant. Hello."  # not said twice


async def test_search_first_puts_the_results_in_front_of_the_model(tenant, kb):
    world = make_world(tenant, kb, search_first=True)
    world.chat.replies = [cite_fee, say("Anything else?")]
    result = await world.desk.handle("s1", "What is the JEE Dropper Programme classroom fee?")

    assert "1,35,000" in result.reply and result.citations and result.problems == []
    assert (result.usage["model_calls"], result.usage["tool_calls"]) == (1, 1)
    # The model's very first request already holds a search it did not ask for.
    first_request = world.chat.requests[0]
    assert [m["role"] for m in first_request] == ["system", "user", "assistant", "tool"]
    assert first_request[2]["tool_calls"][0]["id"] == "auto_1"

    await world.desk.handle("s1", "ok")  # too short to be a question, so no search
    assert world.chat.requests[1][-1] == {"role": "user", "content": "ok"}


def test_the_http_api(world):
    world.chat.replies = [say("Hello, I am an AI assistant for Meridian Demo Academy.")]
    with TestClient(create_app(world, run_worker=False)) as client:
        assert client.get("/healthz").json() == {"ok": True}
        body = client.post("/v1/chat", json={"message": "hi"}).json()
        assert body["status"] == "ok" and body["turn"] == 1 and len(body["session_id"]) == 32
        assert client.get(f"/v1/sessions/{body['session_id']}/usage").json()["model_calls"] == 1
        assert client.post("/v1/chat", json={"message": ""}).status_code == 422
        assert client.post("/v1/chat", json={"message": "hi", "session_id": "../etc"}).status_code == 422
        assert client.post(f"/v1/sessions/{body['session_id']}/operator", json={"message": "hi"}).status_code == 409
