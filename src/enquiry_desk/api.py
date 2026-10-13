"""The HTTP surface. Run with: uv run uvicorn enquiry_desk.api:app"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field, ValidationError

from enquiry_desk.agent.service import NotWaiting
from enquiry_desk.config import get_settings
from enquiry_desk.intake import LeadForm
from enquiry_desk.reports import REPORTS, UnknownReport, run_report
from enquiry_desk.runtime import Runtime, open_desk
from enquiry_desk.webhooks.signing import BadSignature, verify, verify_hubspot

log = logging.getLogger(__name__)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class OperatorIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


def create_app(runtime: Runtime | None = None, *, run_worker: bool = True) -> FastAPI:
    """With no arguments the real desk is built at startup. Tests pass their own."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with AsyncExitStackOrGiven(runtime) as parts:
            app.state.rt = parts
            stop = asyncio.Event()
            task = asyncio.create_task(parts.worker.run_forever(stop)) if run_worker else None
            try:
                yield
            finally:
                stop.set()
                if task:
                    await task

    app = FastAPI(title="Enquiry Desk", lifespan=lifespan)

    def rt() -> Runtime:
        return app.state.rt

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True}

    @app.post("/v1/chat")
    async def chat(body: ChatIn) -> dict:
        session_id = body.session_id or uuid.uuid4().hex
        result = await rt().desk.handle(session_id, body.message)
        return result.as_dict()

    @app.post("/v1/sessions/{session_id}/operator")
    async def operator(session_id: str, body: OperatorIn, x_desk_secret: str | None = Header(default=None)) -> dict:
        """A person answers a conversation that the agent handed over."""
        secret = rt().settings.operator_secret
        if secret and not hmac.compare_digest(x_desk_secret or "", secret):
            raise HTTPException(401, "operator secret not accepted")
        try:
            result = await rt().desk.operator_reply(session_id, body.message)
        except NotWaiting:
            raise HTTPException(409, "This session is not waiting for an operator.") from None
        return result.as_dict()

    @app.get("/v1/sessions/{session_id}/usage")
    async def usage(session_id: str) -> dict:
        return rt().meter.summary(session_id)

    @app.get("/v1/reports")
    async def reports() -> dict:
        return REPORTS

    @app.get("/v1/reports/{name}")
    async def report(name: str, n: int = 3) -> list[dict]:
        try:
            return run_report(rt().db, name, n=n)
        except UnknownReport:
            raise HTTPException(404, f"No report named {name!r}.") from None

    @app.get("/metrics")
    async def metrics() -> Response:
        """What Prometheus scrapes."""
        body = rt().telemetry.render(rt().jobs.counts())
        return Response(body, media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.get("/v1/jobs")
    async def jobs() -> dict:
        return {"counts": rt().jobs.counts(), "dead": rt().jobs.dead()}

    @app.post("/webhooks/lead", status_code=202)
    async def lead_webhook(request: Request) -> dict:
        """A website form posts a lead here. The work is queued, and the answer is immediate."""
        body = await request.body()  # the raw bytes, because the signature covers exactly these
        try:
            verify(
                rt().settings.webhook_secret,
                request.headers.get("x-desk-timestamp"),
                request.headers.get("x-desk-signature"),
                body,
                time.time(),
            )
        except BadSignature as error:
            # The reason goes to the log only. Telling the sender which check failed would
            # help someone who is guessing.
            log.warning("lead webhook refused: %s", error)
            rt().telemetry.webhook("lead", "refused")
            raise HTTPException(401, "signature not accepted") from None
        try:
            form = LeadForm.model_validate_json(body)
        except ValidationError as error:
            raise HTTPException(422, error.errors(include_url=False, include_input=False)) from None
        is_new = rt().jobs.enqueue("lead", form.model_dump(), dedupe_key=f"lead:{form.event_id}")
        rt().telemetry.webhook("lead", "accepted" if is_new else "duplicate")
        # A repeat delivery is answered with the same success, or the sender would retry forever.
        return {"accepted": True, "duplicate": not is_new}

    @app.post("/webhooks/hubspot")
    async def hubspot_webhook(request: Request) -> dict:
        """HubSpot posts a batch of events here when something changes in the CRM."""
        body = await request.body()
        url = rt().settings.public_base_url.rstrip("/") + request.url.path
        try:
            verify_hubspot(
                rt().settings.hubspot_client_secret,
                "POST",
                url,
                body,
                request.headers.get("x-hubspot-request-timestamp"),
                request.headers.get("x-hubspot-signature-v3"),
                time.time(),
            )
        except BadSignature as error:
            log.warning("hubspot webhook refused: %s", error)
            rt().telemetry.webhook("hubspot", "refused")
            raise HTTPException(401, "signature not accepted") from None
        events = json.loads(body)
        if not isinstance(events, list):
            raise HTTPException(422, "expected a JSON array of events")
        new = sum(
            rt().jobs.enqueue("crm_event", event, dedupe_key=f"hubspot:{event['eventId']}")
            for event in events
            if isinstance(event, dict) and "eventId" in event
        )
        rt().telemetry.webhook("hubspot", "accepted")
        return {"received": len(events), "new": new}

    try:
        from enquiry_desk.voice.server import register as register_voice
    except ImportError:
        # The speech libraries are an optional extra. Without them the desk is chat only.
        log.info("voice endpoints are off: install the 'voice' extra to enable them")
    else:
        register_voice(app, rt)

    return app


class AsyncExitStackOrGiven:
    """Yield the runtime a test passed in, or open the real one."""

    def __init__(self, given: Runtime | None):
        self._given = given
        self._opened = None

    async def __aenter__(self) -> Runtime:
        if self._given is not None:
            return self._given
        self._opened = open_desk(get_settings())
        return await self._opened.__aenter__()

    async def __aexit__(self, *exc) -> None:
        if self._opened is not None:
            await self._opened.__aexit__(*exc)


app = create_app()
