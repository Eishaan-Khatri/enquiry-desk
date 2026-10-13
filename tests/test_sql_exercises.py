"""SQL exercises. Each one is skipped until you write its file.

Write `src/enquiry_desk/sql/exercises/<name>.sql`. The test then runs it on the seeded history
from test_reports.py and compares your rows with the answer, which is computed here in Python
so that no SQL solution is given away. Use :tz for the time zone offset, as the reports do.

The answers assume the column names and the row order written in each docstring.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta

import pytest

from enquiry_desk.reports import SQL_DIR
from tests.test_reports import EVENTS, db  # noqa: F401  (db is a fixture)

EXERCISES = SQL_DIR / "exercises"


def day_of(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).date().isoformat()


def turns() -> list[dict]:
    """One dict per turn: session, turn, day, total ms, total cost."""
    grouped: dict[tuple, dict] = {}
    for session, turn, _kind, _name, _p, _c, cost, ms, _ok, ts in EVENTS:
        row = grouped.setdefault((session, turn), {"session_id": session, "turn": turn, "ts": ts, "ms": 0.0, "cost": 0.0})
        row["ts"] = min(row["ts"], ts)
        row["ms"] += ms
        row["cost"] += cost or 0.0
    return [{**row, "day": day_of(row["ts"])} for row in grouped.values()]


def second_slowest_turn_per_day() -> list[dict]:
    """The second slowest turn of each day that has at least two turns.
    Columns: day, session_id, turn, turn_ms. Order: day."""
    by_day = defaultdict(list)
    for row in turns():
        by_day[row["day"]].append(row)
    out = []
    for day in sorted(by_day):
        ordered = sorted(by_day[day], key=lambda r: -r["ms"])
        if len(ordered) >= 2:
            second = ordered[1]
            out.append({"day": day, "session_id": second["session_id"], "turn": second["turn"], "turn_ms": second["ms"]})
    return out


def busy_streaks() -> list[dict]:
    """Runs of consecutive days that each had at least one session ("gaps and islands").
    Columns: first_day, last_day, days. Order: first_day."""
    days = sorted({row["day"] for row in turns()})
    streaks: list[list[str]] = []
    for day in days:
        previous = (datetime.fromisoformat(day) - timedelta(days=1)).date().isoformat()
        if streaks and streaks[-1][-1] == previous:
            streaks[-1].append(day)
        else:
            streaks.append([day])
    return [{"first_day": s[0], "last_day": s[-1], "days": len(s)} for s in streaks]


def costliest_quarter_of_sessions() -> list[dict]:
    """Split sessions into four equal groups by total cost and return the costliest group.
    Columns: session_id, cost_usd. Order: cost_usd descending."""
    totals: dict[str, float] = defaultdict(float)
    for row in turns():
        totals[row["session_id"]] += row["cost"]
    ordered = sorted(totals.items(), key=lambda item: -item[1])
    return [{"session_id": s, "cost_usd": c} for s, c in ordered[: max(1, len(ordered) // 4)]]


def turn_faster_than_the_one_before() -> list[dict]:
    """Turns that were faster than the previous turn of the same session.
    Columns: session_id, turn, turn_ms, previous_ms. Order: session_id, turn."""
    by_session = defaultdict(list)
    for row in sorted(turns(), key=lambda r: (r["session_id"], r["turn"])):
        by_session[row["session_id"]].append(row)
    out = []
    for session in sorted(by_session):
        for before, after in zip(by_session[session], by_session[session][1:], strict=False):
            if after["ms"] < before["ms"]:
                out.append({"session_id": session, "turn": after["turn"], "turn_ms": after["ms"], "previous_ms": before["ms"]})
    return out


ANSWERS = [second_slowest_turn_per_day, busy_streaks, costliest_quarter_of_sessions, turn_faster_than_the_one_before]


@pytest.mark.parametrize("answer", ANSWERS, ids=lambda f: f.__name__)
def test_exercise(answer, db):  # noqa: F811
    path = EXERCISES / f"{answer.__name__}.sql"
    if not path.exists():
        pytest.skip(f"write {path.relative_to(SQL_DIR.parents[2])}")
    sql = path.read_text(encoding="utf-8")
    rows = [dict(row) for row in db.query(sql, {"tz": 0} if ":tz" in sql else {})]
    expected = answer()
    assert len(rows) == len(expected)
    for got, want in zip(rows, expected, strict=True):
        assert set(got) == set(want)
        for column, value in want.items():
            assert got[column] == (pytest.approx(value) if isinstance(value, float) else value)
