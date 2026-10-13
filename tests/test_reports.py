"""The SQL reports, on a small seeded history whose answers were worked out by hand.

The history, in UTC (tz=0 in these tests):

    Mon 12 Oct  session A  turn 1: model 400, tool 20, model 600     = 1020 ms, cost 0.003
                           turn 2: model 500                          =  500 ms, cost 0.002
                session B  turn 1: model 300, tool 5 (failed), model 300, tool 5, model 200
                                                                      =  810 ms, cost 0.003
    Tue 13 Oct  session C  turn 1: model 2000                         = 2000 ms, cost 0.004
                           turn 2: model 100, tool 10 (failed), model 150 = 260 ms, cost 0.002
    Wed 14 Oct  nothing
    Thu 15 Oct  session D  turn 1: model 900                          =  900 ms, cost 0.010
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from enquiry_desk.crm import LocalCRM
from enquiry_desk.db import Database
from enquiry_desk.metering import Meter
from enquiry_desk.reports import REPORTS, SQL_DIR, UnknownReport, as_table, run_report

D12 = datetime(2026, 10, 12, tzinfo=UTC).timestamp()
D13, D15 = D12 + 86_400, D12 + 3 * 86_400

# (session, turn, kind, name, prompt tokens, completion tokens, cost, latency ms, ok, time)
EVENTS = [
    ("A", 1, "llm", "m1", 1000, 50, 0.001, 400, 1, D12 + 100),
    ("A", 1, "tool", "search_knowledge", None, None, None, 20, 1, D12 + 101),
    ("A", 1, "llm", "m1", 1300, 80, 0.002, 600, 1, D12 + 102),
    ("A", 2, "llm", "m1", 1500, 40, 0.002, 500, 1, D12 + 160),
    ("B", 1, "llm", "m1", 1000, 30, 0.001, 300, 1, D12 + 500),
    ("B", 1, "tool", "save_lead", None, None, None, 5, 0, D12 + 501),
    ("B", 1, "llm", "m1", 1100, 30, 0.001, 300, 1, D12 + 502),
    ("B", 1, "tool", "save_lead", None, None, None, 5, 1, D12 + 503),
    ("B", 1, "llm", "m1", 1200, 20, 0.001, 200, 1, D12 + 504),
    ("C", 1, "llm", "m1", 1000, 50, 0.004, 2000, 1, D13 + 50),
    ("C", 2, "llm", "m1", 1100, 10, 0.001, 100, 1, D13 + 90),
    ("C", 2, "tool", "book_slot", None, None, None, 10, 0, D13 + 91),
    ("C", 2, "llm", "m1", 1200, 10, 0.001, 150, 1, D13 + 92),
    ("D", 1, "llm", "m1", 1000, 50, 0.010, 900, 1, D15 + 10),
]


@pytest.fixture
def db() -> Database:
    database = Database()
    Meter(database)  # creates usage_events
    for event in EVENTS:
        database.run(
            "INSERT INTO usage_events (session_id, turn, kind, name, prompt_tokens, completion_tokens,"
            " cost_usd, latency_ms, ok, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            event,
        )
    return database


def report(db: Database, name: str, **params) -> list[dict]:
    return run_report(db, name, tz=0, **params)


def test_every_report_has_a_file_and_every_file_is_a_report():
    assert {path.stem for path in SQL_DIR.glob("*.sql")} == set(REPORTS)


def test_percentiles_by_rank(db):
    assert report(db, "turn_latency") == [
        {"day": "2026-10-12", "turns": 3, "p50_ms": 810.0, "p95_ms": 1020.0, "slowest_ms": 1020.0},
        {"day": "2026-10-13", "turns": 2, "p50_ms": 260.0, "p95_ms": 2000.0, "slowest_ms": 2000.0},
        {"day": "2026-10-15", "turns": 1, "p50_ms": 900.0, "p95_ms": 900.0, "slowest_ms": 900.0},
    ]


def test_top_n_per_day(db):
    rows = report(db, "slowest_turns", n=1)
    assert [(r["day"], r["session_id"], r["turn"], r["turn_ms"]) for r in rows] == [
        ("2026-10-12", "A", 1, 1020.0),
        ("2026-10-13", "C", 1, 2000.0),
        ("2026-10-15", "D", 1, 900.0),
    ]
    assert rows[0]["model_calls"] == 2 and rows[0]["tool_calls"] == 1
    assert len(report(db, "slowest_turns", n=2)) == 5


def test_running_totals_and_the_gap_between_turns(db):
    first, second = (r for r in report(db, "session_costs") if r["session_id"] == "A")
    assert (first["tokens"], first["running_tokens"], first["seconds_since_last_turn"]) == (2430, 2430, None)
    assert first["running_cost_usd"] == pytest.approx(0.003) and first["share_of_session_pct"] == 60.0
    assert (second["tokens"], second["running_tokens"], second["seconds_since_last_turn"]) == (1540, 3970, 58.0)
    assert second["running_cost_usd"] == pytest.approx(0.005) and second["share_of_session_pct"] == 40.0


def test_daily_cost_fills_the_empty_day(db):
    rows = report(db, "daily_cost")
    assert [r["day"] for r in rows] == ["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15"]
    assert [r["sessions"] for r in rows] == [2, 1, 0, 1]
    assert [r["cost_usd"] for r in rows] == pytest.approx([0.008, 0.006, 0, 0.010])
    assert [r["avg_7_days"] for r in rows] == pytest.approx([0.008, 0.007, 0.014 / 3, 0.006])
    assert rows[0]["change_from_day_before"] is None
    assert [r["change_from_day_before"] for r in rows[1:]] == pytest.approx([-0.002, -0.006, 0.010])
    assert [r["total_so_far"] for r in rows] == pytest.approx([0.008, 0.014, 0.014, 0.024])


def test_tools_ranked_by_error_rate(db):
    rows = report(db, "tool_errors")
    assert [(r["name"], r["calls"], r["errors"], r["error_rate_pct"], r["rank_by_error_rate"]) for r in rows] == [
        ("book_slot", 1, 1, 100.0, 1),
        ("save_lead", 2, 1, 50.0, 2),
        ("search_knowledge", 1, 0, 0.0, 3),
    ]
    assert [r["share_of_all_errors_pct"] for r in rows] == [50.0, 50.0, 0.0]


def test_latest_row_of_each_session(db):
    rows = {r["session_id"]: r for r in report(db, "sessions_overview")}
    assert list(rows) == ["D", "C", "B", "A"]  # newest first
    assert (rows["A"]["turns"], rows["A"]["model_calls"], rows["A"]["tool_calls"]) == (2, 3, 1)
    assert rows["A"]["last_activity"] == "2026-10-12 00:02:40" and rows["A"]["cost_usd"] == pytest.approx(0.005)
    assert (rows["C"]["errors"], rows["C"]["last_event"], rows["C"]["last_event_name"]) == (1, "llm", "m1")


async def test_funnel_and_time_in_stage():
    """Four leads. L1 goes through every stage, L2 books without being qualified,
    L3 stops at qualified, L4 never moves."""
    clock = iter([0, 0, 0, 0, 600, 1800, 300, 120, 999])
    database = Database()
    crm = LocalCRM(database, clock=lambda: next(clock))
    leads = [await crm.upsert_lead(f"+91987654321{i}", f"L{i}", {}) for i in range(1, 5)]
    await crm.set_stage(leads[0].id, "qualified")  # at 600 s
    await crm.set_stage(leads[0].id, "counselling_booked")  # at 1800 s
    await crm.set_stage(leads[1].id, "counselling_booked")  # at 300 s
    await crm.set_stage(leads[2].id, "qualified")  # at 120 s
    await crm.set_stage(leads[2].id, "qualified")  # a repeat adds no event

    assert report(database, "lead_funnel") == [
        {"stage": "new", "leads": 4, "pct_of_all_leads": 100.0, "pct_of_stage_before": None},
        {"stage": "qualified", "leads": 3, "pct_of_all_leads": 75.0, "pct_of_stage_before": 75.0},
        {"stage": "counselling_booked", "leads": 2, "pct_of_all_leads": 50.0, "pct_of_stage_before": 66.7},
    ]
    assert report(database, "stage_durations") == [
        {"stage": "new", "entered": 4, "still_here": 1, "avg_minutes_before_moving_on": 5.67},
        {"stage": "qualified", "entered": 2, "still_here": 1, "avg_minutes_before_moving_on": 20.0},
        {"stage": "counselling_booked", "entered": 2, "still_here": 2, "avg_minutes_before_moving_on": None},
    ]


def test_empty_tables_and_unknown_names(db):
    empty = Database()
    Meter(empty)
    LocalCRM(empty)
    assert run_report(empty, "daily_cost") == [] and run_report(empty, "turn_latency") == []
    assert [r["leads"] for r in run_report(empty, "lead_funnel")] == [0, 0, 0]
    with pytest.raises(UnknownReport):
        run_report(db, "drop_tables")
    assert as_table([]) == "(no rows)" and "p95_ms" in as_table(report(db, "turn_latency"))
