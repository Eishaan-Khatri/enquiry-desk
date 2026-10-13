"""A small job queue in the desk's own database.

A webhook must be answered within a second or two, or the sender gives up and sends it again.
The work it triggers (a model call, a CRM write) can take far longer and can fail. So the
webhook only writes a job row and returns. A worker picks the row up afterwards.

The queue is a table, not an in-memory list, so a restart loses nothing.

Delivery is "at least once": a job that crashed halfway is run again. Handlers must therefore
be safe to run twice. The lead handler is, because saving a lead is keyed on the phone number.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from enquiry_desk.db import Database

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    dedupe_key TEXT UNIQUE,          -- the sender's event id. The same event twice makes one job
    payload TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',   -- queued, running, done, dead
    attempts INTEGER NOT NULL DEFAULT 0,
    run_after REAL NOT NULL,
    claimed_at REAL,
    last_error TEXT,
    created REAL NOT NULL,
    finished REAL
);
CREATE INDEX IF NOT EXISTS jobs_due ON jobs (status, run_after);
"""


class Permanent(Exception):
    """A failure that a retry cannot fix, such as a phone number with five digits."""


@dataclass
class Job:
    id: int
    kind: str
    payload: dict
    attempts: int


def backoff_seconds(attempt: int) -> float:
    """30 s, 1 min, 2 min, 4 min ... capped at an hour. `attempt` is the one that just failed."""
    return min(3600.0, 30.0 * 2 ** (attempt - 1))


class Jobs:
    def __init__(self, db: Database, clock: Callable[[], float] = time.time, max_attempts: int = 5):
        self._db = db
        self._clock = clock
        self._max_attempts = max_attempts
        db.script(SCHEMA)

    def enqueue(self, kind: str, payload: dict, dedupe_key: str | None = None, delay: float = 0.0) -> bool:
        """Add a job. Returns False when a job with this dedupe key already exists."""
        now = self._clock()
        added = self._db.run(
            "INSERT INTO jobs (kind, dedupe_key, payload, run_after, created) VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT (dedupe_key) DO NOTHING",
            (kind, dedupe_key, json.dumps(payload, ensure_ascii=False), now + delay, now),
        )
        return added == 1

    def claim(self) -> Job | None:
        """Take the oldest job that is due, and mark it running, in one statement.

        The inner SELECT picks the job and the outer UPDATE takes it only if it is still
        queued. Two workers running this at the same moment cannot both get the same job.
        """
        now = self._clock()
        rows = self._db.query(
            "UPDATE jobs SET status = 'running', attempts = attempts + 1, claimed_at = ?"
            " WHERE id = (SELECT id FROM jobs WHERE status = 'queued' AND run_after <= ?"
            "             ORDER BY run_after, id LIMIT 1)"
            "   AND status = 'queued'"
            " RETURNING id, kind, payload, attempts",
            (now, now),
        )
        if not rows:
            return None
        row = rows[0]
        return Job(row["id"], row["kind"], json.loads(row["payload"]), row["attempts"])

    def done(self, job: Job) -> None:
        self._db.run("UPDATE jobs SET status = 'done', finished = ?, last_error = NULL WHERE id = ?", (self._clock(), job.id))

    def failed(self, job: Job, error: str, *, permanent: bool = False) -> str:
        """Queue the job again for later, or mark it dead. Returns the new status."""
        now = self._clock()
        if permanent or job.attempts >= self._max_attempts:
            self._db.run(
                "UPDATE jobs SET status = 'dead', finished = ?, last_error = ? WHERE id = ?", (now, error[:500], job.id)
            )
            return "dead"
        self._db.run(
            "UPDATE jobs SET status = 'queued', run_after = ?, last_error = ? WHERE id = ?",
            (now + backoff_seconds(job.attempts), error[:500], job.id),
        )
        return "queued"

    def recover(self, stale_after: float = 300.0) -> int:
        """Put back jobs left 'running' by a process that died. Call at start-up."""
        return self._db.run(
            "UPDATE jobs SET status = 'queued' WHERE status = 'running' AND claimed_at < ?",
            (self._clock() - stale_after,),
        )

    def counts(self) -> dict[str, int]:
        found = {row["status"]: row["n"] for row in self._db.query("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")}
        return {status: found.get(status, 0) for status in ("queued", "running", "done", "dead")}

    def dead(self) -> list[dict]:
        rows = self._db.query("SELECT id, kind, attempts, last_error FROM jobs WHERE status = 'dead' ORDER BY id")
        return [dict(row) for row in rows]


Handler = Callable[[dict], Awaitable[None]]


class Worker:
    def __init__(self, jobs: Jobs, handlers: dict[str, Handler], poll_seconds: float = 1.0):
        self._jobs = jobs
        self._handlers = handlers
        self._poll = poll_seconds

    async def run_one(self) -> bool:
        """Run one due job. Returns False when nothing was due."""
        job = self._jobs.claim()
        if job is None:
            return False
        handler = self._handlers.get(job.kind)
        try:
            if handler is None:
                raise Permanent(f"no handler for job kind {job.kind!r}")
            await handler(job.payload)
        except Permanent as error:
            log.warning("job %s (%s) is dead: %s", job.id, job.kind, error)
            self._jobs.failed(job, str(error), permanent=True)
        except Exception as error:
            status = self._jobs.failed(job, f"{type(error).__name__}: {error}")
            log.warning("job %s (%s) failed on attempt %s, now %s: %s", job.id, job.kind, job.attempts, status, error)
        else:
            self._jobs.done(job)
        return True

    async def run_forever(self, stop: asyncio.Event) -> None:
        self._jobs.recover()
        while not stop.is_set():
            if not await self.run_one():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self._poll)
                except TimeoutError:
                    pass
