"""The CRM as the agent sees it, and a local one that needs no account.

The agent talks to the `CRM` interface. `LocalCRM` keeps leads in SQLite so the whole system
runs and is tested offline. A HubSpot adapter with the same four methods is phase 2.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel, Field

from enquiry_desk.db import Database


class Lead(BaseModel):
    id: str
    phone: str
    name: str = ""
    fields: dict[str, str] = Field(default_factory=dict)
    stage: str = "new"


def normalise_phone(raw: str) -> str:
    """Indian mobile numbers in one form, +91 followed by ten digits.

    The phone number is the key that stops one caller becoming two leads, so "98765 43210",
    "09876543210" and "+91-9876543210" must all come out the same.
    """
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    if len(digits) != 10 or digits[0] not in "6789":
        raise ValueError("a mobile number needs ten digits and starts with 6, 7, 8 or 9")
    return "+91" + digits


class CRM(Protocol):
    async def find_by_phone(self, phone: str) -> Lead | None: ...

    async def upsert_lead(self, phone: str, name: str, fields: dict[str, str]) -> Lead: ...

    async def add_note(self, lead_id: str, text: str) -> None: ...

    async def set_stage(self, lead_id: str, stage: str) -> None: ...


SCHEMA = """
CREATE TABLE IF NOT EXISTS leads (
    id TEXT PRIMARY KEY,
    phone TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL DEFAULT '',
    fields TEXT NOT NULL DEFAULT '{}',
    stage TEXT NOT NULL DEFAULT 'new',
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS lead_notes (
    id INTEGER PRIMARY KEY,
    lead_id TEXT NOT NULL REFERENCES leads (id),
    ts REAL NOT NULL,
    text TEXT NOT NULL
);
-- One row each time a lead enters a stage. `leads.stage` is only the latest. The history is
-- what the funnel and time-in-stage reports are computed from.
CREATE TABLE IF NOT EXISTS lead_events (
    id INTEGER PRIMARY KEY,
    lead_id TEXT NOT NULL REFERENCES leads (id),
    ts REAL NOT NULL,
    stage TEXT NOT NULL
);
"""

STAGES = ("new", "qualified", "counselling_booked")


class LocalCRM:
    def __init__(self, db: Database, clock: Callable[[], float] = time.time):
        self._db = db
        self._clock = clock
        db.script(SCHEMA)

    async def find_by_phone(self, phone: str) -> Lead | None:
        rows = self._db.query("SELECT * FROM leads WHERE phone = ?", (phone,))
        return _lead(rows[0]) if rows else None

    async def upsert_lead(self, phone: str, name: str, fields: dict[str, str]) -> Lead:
        """Create the lead, or merge into the one that already has this phone number.

        An empty value never overwrites a filled one, so saving twice loses nothing.
        """
        now = self._clock()
        existing = await self.find_by_phone(phone)
        if existing is None:
            lead = Lead(id=uuid.uuid4().hex[:12], phone=phone, name=name, fields=_filled(fields))
            # ON CONFLICT covers the race where another process saved this caller a moment ago.
            # The insert then changes nothing, and the lead is merged like any existing one.
            inserted = self._db.run(
                "INSERT INTO leads (id, phone, name, fields, created, updated) VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (phone) DO NOTHING",
                (lead.id, phone, name, json.dumps(lead.fields, ensure_ascii=False), now, now),
            )
            if inserted:
                self._db.run("INSERT INTO lead_events (lead_id, ts, stage) VALUES (?, ?, 'new')", (lead.id, now))
                return lead
            existing = await self.find_by_phone(phone)
        merged = {**existing.fields, **_filled(fields)}
        self._db.run(
            "UPDATE leads SET name = ?, fields = ?, updated = ? WHERE id = ?",
            (name or existing.name, json.dumps(merged, ensure_ascii=False), now, existing.id),
        )
        return existing.model_copy(update={"name": name or existing.name, "fields": merged})

    async def add_note(self, lead_id: str, text: str) -> None:
        self._db.run("INSERT INTO lead_notes (lead_id, ts, text) VALUES (?, ?, ?)", (lead_id, self._clock(), text))

    async def set_stage(self, lead_id: str, stage: str) -> None:
        now = self._clock()
        # The WHERE clause makes a repeat call a no-op, so the history gets no duplicate rows.
        changed = self._db.run(
            "UPDATE leads SET stage = ?, updated = ? WHERE id = ? AND stage <> ?", (stage, now, lead_id, stage)
        )
        if changed:
            self._db.run("INSERT INTO lead_events (lead_id, ts, stage) VALUES (?, ?, ?)", (lead_id, now, stage))

    def notes(self, lead_id: str) -> list[str]:
        return [r["text"] for r in self._db.query("SELECT text FROM lead_notes WHERE lead_id = ? ORDER BY id", (lead_id,))]


def _filled(fields: dict[str, str]) -> dict[str, str]:
    return {key: value.strip() for key, value in fields.items() if value and value.strip()}


def _lead(row) -> Lead:
    return Lead(id=row["id"], phone=row["phone"], name=row["name"], fields=json.loads(row["fields"]), stage=row["stage"])
