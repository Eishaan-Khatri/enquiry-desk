"""Counselling call slots. One slot can be booked by one lead."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from enquiry_desk.db import Database

IST = ZoneInfo("Asia/Kolkata")

SCHEMA = """
CREATE TABLE IF NOT EXISTS slots (
    id TEXT PRIMARY KEY,             -- the start time, e.g. 2026-10-14T16:00
    start TEXT NOT NULL,
    lead_id TEXT
);
"""


class SlotTaken(Exception):
    pass


class Slots:
    def __init__(self, db: Database):
        self._db = db
        db.script(SCHEMA)

    def seed(self, now: datetime, hours: list[str], days: int = 7, closed_weekdays: tuple[int, ...] = (6,)) -> int:
        """Make sure slots exist for the coming days. Safe to call on every start."""
        added = 0
        for offset in range(days + 1):
            day = (now + timedelta(days=offset)).date()
            if day.weekday() in closed_weekdays:
                continue
            for hour in hours:
                start = f"{day.isoformat()}T{hour}"
                added += self._db.run(
                    "INSERT INTO slots (id, start) VALUES (?, ?) ON CONFLICT (id) DO NOTHING", (start, start)
                )
        return added

    def open_slots(self, now: datetime, limit: int = 3) -> list[dict]:
        rows = self._db.query(
            "SELECT id, start FROM slots WHERE lead_id IS NULL AND start > ? ORDER BY start LIMIT ?",
            (now.strftime("%Y-%m-%dT%H:%M"), limit),
        )
        return [{"slot_id": r["id"], "when": spoken(r["start"])} for r in rows]

    def book(self, slot_id: str, lead_id: str) -> dict:
        """Book the slot if it is still free.

        The check and the write are one UPDATE statement, so two callers who pick the same slot
        at the same moment cannot both get it.
        """
        changed = self._db.run("UPDATE slots SET lead_id = ? WHERE id = ? AND lead_id IS NULL", (lead_id, slot_id))
        if changed == 0:
            exists = self._db.query("SELECT 1 FROM slots WHERE id = ?", (slot_id,))
            raise SlotTaken("That slot has just been taken." if exists else "There is no such slot.")
        return {"slot_id": slot_id, "when": spoken(slot_id)}


def spoken(start: str) -> str:
    """2026-10-14T16:00 -> 'Wednesday 14 October, 4:00 pm'."""
    moment = datetime.strptime(start, "%Y-%m-%dT%H:%M")
    hour = moment.hour % 12 or 12
    return f"{moment:%A} {moment.day} {moment:%B}, {hour}:{moment:%M} {'am' if moment.hour < 12 else 'pm'}"
