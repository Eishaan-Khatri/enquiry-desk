"""A lead that arrives from a website form, turned into a scored CRM record.

Three steps, and only the first uses a model:
  1. Extract: one model call reads the free-text message and fills a typed record.
  2. Score:   a plain function. A score a person can recompute by hand can be argued with.
  3. Save:    write the lead and a note to the CRM.

The lead is saved even when extraction fails. A form submission must never be lost because a
model had a bad moment.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from enquiry_desk.crm import CRM, normalise_phone
from enquiry_desk.jobs import Jobs, Permanent
from enquiry_desk.llm.tools import _portable
from enquiry_desk.llm.wire import ChatClient, Completion
from enquiry_desk.metering import Meter
from enquiry_desk.tenant import Tenant
from enquiry_desk.webhooks.signing import sign

log = logging.getLogger(__name__)

URGENCY_POINTS = {"this_week": 30, "this_month": 20, "later": 5, "unknown": 0}

EXTRACT_INSTRUCTIONS = (
    "Read the enquiry and call record_lead once. Fill a field only with what the text states. "
    "When the text does not state it, use an empty string, and 'unknown' for urgency. Do not guess."
)


class LeadForm(BaseModel):
    """What the website sends."""

    model_config = ConfigDict(extra="ignore")
    event_id: str = Field(min_length=1, max_length=100)
    phone: str = Field(max_length=30)
    name: str = Field("", max_length=100)
    email: str = Field("", max_length=200)
    message: str = Field("", max_length=4000)
    source: str = Field("website", max_length=50)


def record_model(tenant: Tenant) -> type[BaseModel]:
    """The typed record, built from the tenant's lead fields plus two fixed ones."""
    return create_model(
        "RecordLead",
        __config__=ConfigDict(extra="ignore"),
        wants=(str, Field("", description="A short summary of the request, in your own words.")),
        urgency=(Literal["this_week", "this_month", "later", "unknown"], Field("unknown", description="How soon they want to start.")),
        **{name: (str, Field("", description=meaning)) for name, meaning in tenant.lead_fields.items()},
    )


def score(record: dict | None, tenant: Tenant) -> tuple[int, str]:
    """0 to 100. 30 for a valid phone, up to 40 for known details, up to 30 for urgency."""
    points = 30  # the phone number was already validated, or the job would not have got here
    if record:
        known = sum(1 for name in tenant.lead_fields if record.get(name))
        points += round(40 * known / max(1, len(tenant.lead_fields)))
        points += URGENCY_POINTS.get(record.get("urgency", "unknown"), 0)
    band = "hot" if points >= 70 else "warm" if points >= 45 else "cold"
    return points, band


class Intake:
    def __init__(self, tenant: Tenant, chat: ChatClient, crm: CRM, meter: Meter, jobs: Jobs, notify_bands: tuple[str, ...] = ("hot",)):
        self._tenant = tenant
        self._chat = chat
        self._crm = crm
        self._meter = meter
        self._jobs = jobs
        self._notify_bands = notify_bands
        self._record = record_model(tenant)
        self._spec = {
            "type": "function",
            "function": {
                "name": "record_lead",
                "description": "Record what the enquiry says.",
                "parameters": _portable(self._record.model_json_schema()),
            },
        }

    async def extract(self, message: str, session_id: str) -> dict | None:
        """One model call, with one retry when the answer does not validate."""
        if not message.strip():
            return None
        messages = [{"role": "system", "content": EXTRACT_INSTRUCTIONS}, {"role": "user", "content": message}]
        # tool_choice names the function, which forces a call on servers that honour it. Some
        # ignore it, so a JSON object in the text of the reply is accepted as well.
        forced = {"type": "function", "function": {"name": "record_lead"}}
        for attempt in (1, 2):
            completion = await self._chat.complete(messages, tools=[self._spec], tool_choice=forced, temperature=0.0)
            self._meter.record_llm(session_id, attempt, completion)
            raw = _arguments(completion)
            try:
                record = self._record.model_validate(json.loads(raw)).model_dump()
                # A small model sometimes copies a field's description into the field.
                descriptions = {field.description for field in self._record.model_fields.values()}
                return {name: "" if value in descriptions else value for name, value in record.items()}
            except (ValidationError, json.JSONDecodeError, TypeError) as error:
                problem = error.errors()[0]["msg"] if isinstance(error, ValidationError) else "the arguments were not a JSON object"
                messages += [
                    completion.as_message() if completion.tool_calls else {"role": "assistant", "content": completion.content},
                    *({"role": "tool", "tool_call_id": c.id, "content": json.dumps({"error": problem})} for c in completion.tool_calls),
                    {"role": "user", "content": f"That was not accepted: {problem}. Call record_lead again with valid arguments."},
                ]
        return None

    async def handle(self, payload: dict) -> None:
        """The job handler. Safe to run twice: the lead is keyed on the phone number."""
        form = LeadForm.model_validate(payload)
        try:
            phone = normalise_phone(form.phone)
        except ValueError as error:
            raise Permanent(f"lead {form.event_id}: {error}") from error

        record = await self.extract(form.message, session_id=f"lead:{form.event_id}")
        points, band = score(record, self._tenant)
        fields = {name: (record or {}).get(name, "") for name in self._tenant.lead_fields}
        fields.update(source=form.source, email=form.email, lead_score=str(points), lead_band=band)
        lead = await self._crm.upsert_lead(phone, form.name, fields)
        if lead.stage == "new" and all(fields[name] for name in self._tenant.lead_fields):
            await self._crm.set_stage(lead.id, "qualified")

        wants = (record or {}).get("wants") or "(not extracted)"
        urgency = (record or {}).get("urgency", "unknown")
        await self._crm.add_note(
            lead.id,
            f"Enquiry from {form.source} [{form.event_id}]: {form.message or '(no message)'}\n"
            f"Wants: {wants}. Urgency: {urgency}. Score: {points} ({band}).",
        )
        if band in self._notify_bands:
            self._jobs.enqueue(
                "notify",
                {"text": f"{band.capitalize()} lead: {lead.name or 'unnamed'}, {lead.phone}. {wants} (score {points})"},
                dedupe_key=f"notify:lead:{form.event_id}",
            )


def _arguments(completion: Completion) -> str | None:
    for call in completion.tool_calls:
        if call.name == "record_lead":
            return call.arguments
    text = completion.content.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return text or None


class Notifier:
    """Sends an alert to an incoming-webhook URL, such as Slack's or Discord's."""

    def __init__(
        self, url: str, secret: str = "", http: httpx.AsyncClient | None = None, clock: Callable[[], float] = time.time
    ):
        self._url = url
        self._secret = secret
        self._http = http or httpx.AsyncClient(timeout=10.0)
        self._clock = clock

    async def handle(self, payload: dict) -> None:
        """The job handler. Any answer other than 2xx raises, so the queue retries it."""
        # Slack reads "text" and Discord reads "content", so both are sent.
        body = json.dumps({"text": payload["text"], "content": payload["text"]}, ensure_ascii=False).encode()
        headers = {"Content-Type": "application/json"}
        if self._secret:
            timestamp = str(int(self._clock()))
            headers |= {"X-Desk-Timestamp": timestamp, "X-Desk-Signature": sign(self._secret, timestamp, body)}
        response = await self._http.post(self._url, content=body, headers=headers)
        if response.status_code >= 300:
            raise RuntimeError(f"the alert endpoint answered {response.status_code}")
