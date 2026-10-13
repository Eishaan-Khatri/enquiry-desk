"""Events the CRM sends back to the desk, such as a counsellor changing a lead's status."""

from __future__ import annotations

from enquiry_desk.db import Database

SCHEMA = """
CREATE TABLE IF NOT EXISTS crm_events (
    event_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,              -- for example contact.propertyChange
    object_id TEXT,
    property TEXT,
    value TEXT,
    occurred_at REAL
);
"""


class CrmEvents:
    def __init__(self, db: Database):
        self._db = db
        db.script(SCHEMA)

    async def handle(self, payload: dict) -> None:
        """The job handler. The primary key makes a second delivery of the same event a no-op."""
        self._db.run(
            "INSERT INTO crm_events (event_id, kind, object_id, property, value, occurred_at)"
            " VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (event_id) DO NOTHING",
            (
                str(payload["eventId"]),
                payload.get("subscriptionType", "unknown"),
                str(payload.get("objectId", "")),
                payload.get("propertyName"),
                payload.get("propertyValue"),
                (payload.get("occurredAt") or 0) / 1000,  # HubSpot sends milliseconds
            ),
        )

    def recent(self, limit: int = 20) -> list[dict]:
        rows = self._db.query("SELECT * FROM crm_events ORDER BY occurred_at DESC LIMIT ?", (limit,))
        return [dict(row) for row in rows]
