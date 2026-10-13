"""Reports over the desk's own tables. Each one is a SQL file in sql/, with CTEs and window
functions, and the reasoning for the query written at the top of the file.

    uv run python -m enquiry_desk.cli report turn_latency
"""

from __future__ import annotations

from pathlib import Path

from enquiry_desk.db import Database

SQL_DIR = Path(__file__).parent / "sql"
IST_OFFSET_SECONDS = 19_800  # timestamps are stored in UTC, and days are reported in Indian time

REPORTS = {
    "turn_latency": "Median and 95th percentile turn time per day",
    "slowest_turns": "The slowest turns of each day",
    "session_costs": "Each turn of each session, with running cost and tokens",
    "daily_cost": "Cost per day, seven-day average and running total",
    "tool_errors": "Tools ranked by error rate",
    "sessions_overview": "One row per session, with its last event",
    "lead_funnel": "How many leads reached each stage",
    "stage_durations": "How long leads stay in each stage",
    "refusals_weekly": "Share of turns the reply check refused, week by week",
    "voice_latency": "Voice response time per day, stage by stage",
}


class UnknownReport(KeyError):
    pass


def run_report(db: Database, name: str, *, tz: int = IST_OFFSET_SECONDS, n: int = 3) -> list[dict]:
    if name not in REPORTS:
        raise UnknownReport(name)
    sql = (SQL_DIR / f"{name}.sql").read_text(encoding="utf-8")
    # Pass only the parameters this query names.
    params = {key: value for key, value in {"tz": tz, "n": n}.items() if f":{key}" in sql}
    try:
        return [dict(row) for row in db.query(sql, params)]
    except Exception as error:
        # A table is created by the part of the desk that writes to it. If that part has never
        # run (no voice call yet, say), the table is not there, and the honest answer is no rows.
        # SQLite says "no such table". PostgreSQL says the relation "does not exist".
        if "no such table" in str(error) or ("relation" in str(error) and "does not exist" in str(error)):
            return []
        raise


def as_table(rows: list[dict]) -> str:
    """Rows as fixed-width text, for the terminal."""
    if not rows:
        return "(no rows)"

    def show(value) -> str:
        if value is None:
            return "-"
        return f"{value:.6g}" if isinstance(value, float) else str(value)

    columns = list(rows[0])
    cells = [[show(row[column]) for column in columns] for row in rows]
    widths = [max(len(column), *(len(line[i]) for line in cells)) for i, column in enumerate(columns)]
    lines = ["  ".join(column.ljust(width) for column, width in zip(columns, widths, strict=True))]
    lines.append("  ".join("-" * width for width in widths))
    lines += ["  ".join(cell.ljust(width) for cell, width in zip(line, widths, strict=True)) for line in cells]
    return "\n".join(lines)
