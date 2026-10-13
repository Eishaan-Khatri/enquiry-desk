"""One row for every model call and every tool call, so cost and latency add up per turn.

Kept in the desk's own table and not only in a tracing product, so a figure such as "cost per
conversation" comes from a query anyone can rerun.
"""

from __future__ import annotations

import time

from enquiry_desk.db import Database
from enquiry_desk.llm.pricing import Price, cost_usd
from enquiry_desk.llm.tools import ToolResult
from enquiry_desk.llm.wire import Completion

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_events (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    session_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    kind TEXT NOT NULL,              -- 'llm' or 'tool'
    name TEXT NOT NULL,              -- model name or tool name
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cached_tokens INTEGER,
    cost_usd REAL,                   -- NULL when the price or the token count is unknown
    ttft_ms REAL,
    latency_ms REAL NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 1,
    ok INTEGER NOT NULL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS usage_by_session ON usage_events (session_id, turn);
-- One row each time the reply check objects to a draft. 'retried' means the draft was sent
-- back to the model. 'refused' means the second draft failed too and the caller got the fixed
-- line. A rising share of refused turns is the first sign that the model or the documents changed.
CREATE TABLE IF NOT EXISTS reply_checks (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    session_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    outcome TEXT NOT NULL,           -- 'retried' or 'refused'
    reasons TEXT NOT NULL
);
"""


class Meter:
    def __init__(self, db: Database, prices: dict[str, Price] | None = None, observer=None):
        self._db = db
        self._prices = prices or {}
        # Anything recorded here is also reported to the observer (metrics and traces), so the
        # usage table, the dashboards and the traces can never disagree about what happened.
        self._observer = observer
        db.script(SCHEMA)

    def record_llm(self, session_id: str, turn: int, completion: Completion) -> float | None:
        usage = completion.usage
        cost = cost_usd(self._prices.get(completion.model), usage)
        self._db.run(
            "INSERT INTO usage_events (ts, session_id, turn, kind, name, prompt_tokens,"
            " completion_tokens, cached_tokens, cost_usd, ttft_ms, latency_ms, attempts, ok)"
            " VALUES (?, ?, ?, 'llm', ?, ?, ?, ?, ?, ?, ?, ?, 1)",
            (
                time.time(),
                session_id,
                turn,
                completion.model,
                usage.prompt_tokens if usage else None,
                usage.completion_tokens if usage else None,
                usage.cached_tokens if usage else None,
                cost,
                completion.ttft_ms,
                completion.latency_ms,
                completion.attempts,
            ),
        )
        if self._observer:
            self._observer.llm(completion, cost)
        return cost

    def record_llm_failure(self, session_id: str, turn: int, model: str, error: str, latency_ms: float) -> None:
        self._db.run(
            "INSERT INTO usage_events (ts, session_id, turn, kind, name, latency_ms, ok, error)"
            " VALUES (?, ?, ?, 'llm', ?, ?, 0, ?)",
            (time.time(), session_id, turn, model, latency_ms, error[:300]),
        )
        if self._observer:
            self._observer.llm_failed(model, error, latency_ms)

    def record_tool(self, session_id: str, turn: int, result: ToolResult) -> None:
        self._db.run(
            "INSERT INTO usage_events (ts, session_id, turn, kind, name, latency_ms, ok, error)"
            " VALUES (?, ?, ?, 'tool', ?, ?, ?, ?)",
            (time.time(), session_id, turn, result.name, result.latency_ms, int(result.ok), result.error),
        )
        if self._observer:
            self._observer.tool(result)

    def record_check(self, session_id: str, turn: int, problems: list[str], outcome: str) -> None:
        self._db.run(
            "INSERT INTO reply_checks (ts, session_id, turn, outcome, reasons) VALUES (?, ?, ?, ?, ?)",
            (time.time(), session_id, turn, outcome, "; ".join(problems)[:500]),
        )
        if self._observer:
            self._observer.rejected(problems)

    def summary(self, session_id: str, turn: int | None = None) -> dict:
        """Totals for one turn, or for the whole session when turn is None."""
        where, params = "session_id = ?", [session_id]
        if turn is not None:
            where += " AND turn = ?"
            params.append(turn)
        row = self._db.query(
            f"""
            SELECT
                SUM(kind = 'llm') AS model_calls,
                SUM(kind = 'tool') AS tool_calls,
                SUM(prompt_tokens) AS prompt_tokens,
                SUM(completion_tokens) AS completion_tokens,
                SUM(cached_tokens) AS cached_tokens,
                SUM(cost_usd) AS cost_usd,
                SUM(kind = 'llm' AND ok = 1 AND cost_usd IS NULL) AS unpriced_calls,
                SUM(CASE WHEN kind = 'llm' THEN latency_ms END) AS model_ms,
                SUM(CASE WHEN kind = 'tool' THEN latency_ms END) AS tool_ms,
                SUM(ok = 0) AS errors
            FROM usage_events WHERE {where}
            """,
            tuple(params),
        )[0]
        first = self._db.query(
            f"SELECT ttft_ms FROM usage_events WHERE {where} AND kind = 'llm' AND ok = 1"
            " ORDER BY id LIMIT 1",
            tuple(params),
        )
        totals = {key: row[key] or 0 for key in row.keys()}
        # A total that leaves out an unpriced call would understate the cost, so report unknown.
        totals["cost_usd"] = None if totals["unpriced_calls"] else round(row["cost_usd"] or 0.0, 6)
        ttft = first[0]["ttft_ms"] if first else None
        totals["first_token_ms"] = None if ttft is None else round(ttft, 1)
        for key in ("model_ms", "tool_ms"):
            totals[key] = round(totals[key], 1)
        return totals
