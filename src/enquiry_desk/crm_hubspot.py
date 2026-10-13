"""The same four CRM methods, against HubSpot's API.

NOT YET RUN AGAINST A LIVE ACCOUNT. It is written from HubSpot's public API documentation
and tested against mocked responses of the documented shape. Expect to fix details on the
first live run.

Setup, once, in a free HubSpot account:
  1. Settings > Integrations > Private Apps > create an app with the scopes
     crm.objects.contacts.read, crm.objects.contacts.write and crm.schemas.contacts.write.
  2. Put its token in .env as DESK_HUBSPOT_TOKEN and set DESK_CRM=hubspot.
  3. Run `uv run python -m enquiry_desk.cli hubspot-setup` to create the custom properties.

How the desk's lead maps onto a HubSpot contact:
  phone            -> phone (always written as +91XXXXXXXXXX, and searched in that form)
  name             -> firstname
  stage            -> hs_lead_status
  each lead field  -> a custom property named desk_<field>
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import httpx

from enquiry_desk.crm import Lead

BASE = "https://api.hubapi.com"
STAGE_TO_STATUS = {"new": "NEW", "qualified": "OPEN", "counselling_booked": "IN_PROGRESS"}
STATUS_TO_STAGE = {status: stage for stage, status in STAGE_TO_STATUS.items()}
NOTE_TO_CONTACT = 202  # HubSpot's association type id for "note belongs to contact"


class HubSpotError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HubSpot answered {status}: {body[:300]}")
        self.status = status


class HubSpotCRM:
    def __init__(
        self,
        token: str,
        lead_fields: list[str],
        *,
        http: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        max_retries: int = 3,
    ):
        self._http = http or httpx.AsyncClient(timeout=15.0)
        self._headers = {"Authorization": f"Bearer {token}"}
        self._fields = lead_fields
        self._sleep = sleep
        self._max_retries = max_retries

    async def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        """One API call. A 429 or a 5xx is retried, waiting as long as HubSpot asks."""
        for attempt in range(1, self._max_retries + 2):
            response = await self._http.request(method, BASE + path, json=body, headers=self._headers)
            if response.status_code < 300:
                return response.json() if response.content else {}
            retryable = response.status_code == 429 or response.status_code >= 500
            if not retryable or attempt > self._max_retries:
                raise HubSpotError(response.status_code, response.text)
            try:
                wait = float(response.headers.get("retry-after", ""))
            except ValueError:
                wait = min(10.0, 2.0 ** (attempt - 1))
            await self._sleep(wait)
        raise AssertionError("unreachable")

    def _properties(self) -> list[str]:
        return ["phone", "firstname", "hs_lead_status", *(f"desk_{name}" for name in self._fields)]

    def _lead(self, contact: dict) -> Lead:
        props = contact.get("properties") or {}
        return Lead(
            id=str(contact["id"]),
            phone=props.get("phone") or "",
            name=props.get("firstname") or "",
            fields={name: props[f"desk_{name}"] for name in self._fields if props.get(f"desk_{name}")},
            stage=STATUS_TO_STAGE.get(props.get("hs_lead_status") or "", "new"),
        )

    async def find_by_phone(self, phone: str) -> Lead | None:
        found = await self._request(
            "POST",
            "/crm/v3/objects/contacts/search",
            {
                "filterGroups": [{"filters": [{"propertyName": "phone", "operator": "EQ", "value": phone}]}],
                "properties": self._properties(),
                "limit": 1,
            },
        )
        results = found.get("results") or []
        return self._lead(results[0]) if results else None

    async def upsert_lead(self, phone: str, name: str, fields: dict[str, str]) -> Lead:
        """Search, then create or update. An empty value is never sent, so it cannot blank a field.

        Two requests are not atomic. Two enquiries from one number in the same instant could
        both find nothing and both create a contact. The job queue runs one job at a time,
        which is what prevents that here.
        """
        props = {f"desk_{key}": value.strip() for key, value in fields.items() if key in self._fields and value and value.strip()}
        if name:
            props["firstname"] = name
        existing = await self.find_by_phone(phone)
        if existing is None:
            created = await self._request(
                "POST", "/crm/v3/objects/contacts", {"properties": {"phone": phone, "hs_lead_status": "NEW", **props}}
            )
            return self._lead({"id": created["id"], "properties": {"phone": phone, "hs_lead_status": "NEW", **props}})
        if props:
            await self._request("PATCH", f"/crm/v3/objects/contacts/{existing.id}", {"properties": props})
        merged = {**existing.fields, **{key[5:]: value for key, value in props.items() if key.startswith("desk_")}}
        return existing.model_copy(update={"name": name or existing.name, "fields": merged})

    async def add_note(self, lead_id: str, text: str) -> None:
        await self._request(
            "POST",
            "/crm/v3/objects/notes",
            {
                "properties": {"hs_note_body": text, "hs_timestamp": datetime.now(UTC).isoformat(timespec="seconds")},
                "associations": [
                    {
                        "to": {"id": lead_id},
                        "types": [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": NOTE_TO_CONTACT}],
                    }
                ],
            },
        )

    async def set_stage(self, lead_id: str, stage: str) -> None:
        await self._request(
            "PATCH", f"/crm/v3/objects/contacts/{lead_id}", {"properties": {"hs_lead_status": STAGE_TO_STATUS[stage]}}
        )

    async def ensure_properties(self) -> list[str]:
        """Create the desk_<field> properties that do not exist yet. Returns the ones created."""
        created = []
        for name in self._fields:
            try:
                await self._request(
                    "POST",
                    "/crm/v3/properties/contacts",
                    {
                        "name": f"desk_{name}",
                        "label": f"Desk: {name.replace('_', ' ')}",
                        "type": "string",
                        "fieldType": "text",
                        "groupName": "contactinformation",
                    },
                )
                created.append(name)
            except HubSpotError as error:
                if error.status != 409:  # 409 means it is already there
                    raise
        return created
