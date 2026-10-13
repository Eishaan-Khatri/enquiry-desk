"""Phase 4: metrics, traces, masking, and the weekly refusal report."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from enquiry_desk.agent import prompts
from enquiry_desk.api import create_app
from enquiry_desk.db import Database
from enquiry_desk.llm.wire import LLMError
from enquiry_desk.metering import Meter
from enquiry_desk.observability import JsonLines, MaskPhones, mask_phones, rejection_reason
from enquiry_desk.reports import run_report
from tests.conftest import call, say


def metric(world, name: str, **labels) -> float:
    return world.telemetry.registry.get_sample_value(name, labels) or 0.0


async def test_one_turn_shows_up_in_metrics_and_as_a_trace(world):
    world.chat.replies = [call(("search_knowledge", {"query": "library seats"})), say("The library seats 80 students [K1].")]
    await world.desk.handle("s1", "How many seats does the library have?")

    assert metric(world, "desk_turns_total", outcome="ok") == 1
    assert metric(world, "desk_llm_calls_total", model="scripted", outcome="ok") == 2
    assert metric(world, "desk_llm_tokens_total", model="scripted", kind="input") == 200
    assert metric(world, "desk_llm_tokens_total", model="scripted", kind="output") == 35
    assert metric(world, "desk_tool_calls_total", tool="search_knowledge", outcome="ok") == 1
    assert metric(world, "desk_turn_seconds_count") == 1 and metric(world, "desk_first_token_seconds_count") == 2

    spans = {span.name: span for span in world.spans.get_finished_spans()}
    assert set(spans) == {"turn", "chat scripted", "execute_tool search_knowledge"}
    turn = spans["turn"]
    assert turn.attributes["session.id"] == "s1" and turn.attributes["desk.outcome"] == "ok" and turn.attributes["desk.turn"] == 1
    # The model call and the tool call are children of the turn, so one trace shows the whole turn.
    for name in ("chat scripted", "execute_tool search_knowledge"):
        assert spans[name].parent.span_id == turn.context.span_id
    chat = spans["chat scripted"]
    assert chat.attributes["gen_ai.usage.input_tokens"] == 100 and chat.attributes["gen_ai.request.model"] == "scripted"
    assert chat.end_time - chat.start_time == 10_000_000  # the 10 ms the scripted model reported


async def test_every_other_way_a_turn_can_end_is_counted(world):
    world.chat.replies = [say("It costs 99,999."), say("It costs 88,888.")]
    await world.desk.handle("refused", "fee?")
    world.chat.replies = [LLMError("model server returned 503", status=503)]
    await world.desk.handle("failed", "hello there")
    world.chat.replies = [call(("request_handoff", {"reason": "upset", "summary": "Caller is upset about a teacher."}))]
    await world.desk.handle("handoff", "I am furious")
    world.chat.replies = [call(("save_lead", {"phone": "123"})), say("May I have your number again?")]
    await world.desk.handle("toolerror", "my number is 123")

    assert {o: metric(world, "desk_turns_total", outcome=o) for o in ("ok", "refused", "failed", "handoff")} == {
        "ok": 1, "refused": 1, "failed": 1, "handoff": 1,
    }
    assert metric(world, "desk_reply_rejections_total", reason="figure") == 2  # sent back once, then refused
    assert metric(world, "desk_llm_calls_total", model="scripted", outcome="error") == 1
    assert metric(world, "desk_tool_calls_total", tool="save_lead", outcome="tool_error") == 1
    checks = world.db.query("SELECT outcome, reasons FROM reply_checks ORDER BY id")
    assert [row["outcome"] for row in checks] == ["retried", "refused"] and "88888" in checks[1]["reasons"]
    failed = next(s for s in world.spans.get_finished_spans() if s.name == "chat scripted" and not s.status.is_ok)
    assert "503" in failed.status.description


async def test_no_span_and_no_metric_carries_a_phone_number_or_the_callers_words(world):
    world.chat.replies = [
        call(("save_lead", {"phone": "9876543210", "name": "Asha Verma"})),
        say("Thank you Asha, I have your number as 98765 43210."),
    ]
    await world.desk.handle("s1", "I am Asha Verma and my number is 9876543210, please call me")

    exported = " ".join(
        f"{span.name} {dict(span.attributes)} {[dict(e.attributes) for e in span.events]}" for span in world.spans.get_finished_spans()
    ) + world.telemetry.render().decode()
    assert "9876543210" not in exported and "43210" not in exported
    assert "Asha" not in exported and "please call me" not in exported


def test_phone_numbers_are_masked_in_logs():
    assert mask_phones("lead +91 98765 43210 saved") == "lead 98******10 saved"
    assert mask_phones("call 9876543210 or 09123456789") == "call 98******10 or 91******89"
    assert mask_phones("fee is 135000 and 2,60,000") == "fee is 135000 and 2,60,000"  # a fee is not a phone

    record = logging.LogRecord("desk", logging.WARNING, __file__, 1, "lead %s could not be saved", ("9876543210",), None)
    assert MaskPhones().filter(record) and record.getMessage() == "lead 98******10 could not be saved"
    assert '"message": "lead 98******10 could not be saved"' in JsonLines().format(record)


def test_rejection_reasons_are_a_small_fixed_set():
    problems = [
        "the figure 88888 is not in any source",
        "the reference [K9] is not one of the search results",
        "the phone number 9123456780 was not given by the caller",
        "you told the caller a call is booked, but book_slot has not succeeded. Call book_slot first",
        "something new",
    ]
    assert [rejection_reason(p) for p in problems] == ["figure", "reference", "phone", "booking", "other"]


def test_the_metrics_endpoint_serves_prometheus_text(world):
    world.chat.replies = [say("Hello.")]
    world.jobs.enqueue("lead", {"event_id": "e1", "phone": "12345"})
    with TestClient(create_app(world, run_worker=False)) as client:
        client.post("/v1/chat", json={"message": "hi"})
        client.post("/webhooks/lead", content=b"{}")
        answer = client.get("/metrics")
    assert answer.headers["content-type"].startswith("text/plain")
    text = answer.text
    assert 'desk_turns_total{outcome="ok"} 1.0' in text
    assert 'desk_jobs{status="queued"} 1.0' in text and 'desk_jobs{status="dead"} 0.0' in text
    assert 'desk_webhooks_total{outcome="refused",source="lead"} 1.0' in text
    assert "# HELP desk_turn_seconds Time to answer one caller turn." in text


def test_weekly_refusal_rate_and_its_change():
    """Week 41 has 4 turns and 1 refused. Week 42 has 2 turns and 1 refused, so the rate doubles."""
    db = Database()
    Meter(db)
    monday_41 = datetime(2026, 10, 12, 6, tzinfo=UTC).timestamp()
    monday_42 = monday_41 + 7 * 86_400
    turns = [("a", 1, monday_41), ("a", 2, monday_41 + 60), ("b", 1, monday_41 + 86_400), ("c", 1, monday_41 + 2 * 86_400),
             ("d", 1, monday_42), ("e", 1, monday_42 + 3_600)]
    for session, turn, ts in turns:
        for offset in (0, 1):  # two usage rows per turn, to prove the join does not double count
            db.run("INSERT INTO usage_events (ts, session_id, turn, kind, name, latency_ms, ok) VALUES (?, ?, ?, 'llm', 'm', 10, 1)", (ts + offset, session, turn))
    for session, turn, ts, outcome in [("a", 2, monday_41 + 61, "retried"), ("b", 1, monday_41 + 86_401, "retried"),
                                       ("b", 1, monday_41 + 86_402, "refused"), ("d", 1, monday_42 + 1, "retried"),
                                       ("d", 1, monday_42 + 2, "refused")]:
        db.run("INSERT INTO reply_checks (ts, session_id, turn, outcome, reasons) VALUES (?, ?, ?, ?, 'x')", (ts, session, turn, outcome))

    assert run_report(db, "refusals_weekly", tz=0) == [
        {"week": "2026-W41", "turns": 4, "sent_back": 2, "refused": 1, "refused_pct": 25.0, "change_from_week_before": None},
        {"week": "2026-W42", "turns": 2, "sent_back": 1, "refused": 1, "refused_pct": 50.0, "change_from_week_before": 25.0},
    ]


async def test_the_fixed_lines_are_what_outcomes_are_read_from(world):
    # If someone rewords a fixed line, the outcome metric must not silently become "ok".
    world.chat.replies = [call(("list_slots", {}))] * 10
    await world.desk.handle("s1", "hello there")
    assert metric(world, "desk_turns_total", outcome="failed") == 1 and prompts.TOO_MANY_STEPS
