"""A phone number in front of the same voice session, using Twilio's Media Streams.

NOT YET RUN AGAINST A REAL CALL. It is written from Twilio's documented message formats and
tested against messages of that shape. A first real call needs a Twilio account, a number,
and this server reachable at a public https address (DESK_PUBLIC_BASE_URL).

How a call flows:
  1. Someone dials the number. Twilio POSTs to /webhooks/twilio/voice and asks what to do.
  2. The desk answers with TwiML, a few lines of XML, telling Twilio to open a WebSocket to
     /ws/twilio and stream the call's audio over it.
  3. Twilio sends JSON messages: "start" once, then "media" with 20 ms of audio each, then
     "stop". The audio is 8,000 Hz mu-law, base64 encoded.
  4. The desk sends "media" messages back with its own speech in the same format, and
     "clear" when the caller interrupts, which empties what Twilio has queued to play.
  5. When the call ends Twilio POSTs to /webhooks/twilio/status.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from urllib.parse import parse_qsl
from xml.sax.saxutils import quoteattr

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect

from enquiry_desk.crm import normalise_phone
from enquiry_desk.db import Database
from enquiry_desk.voice import audio
from enquiry_desk.voice.endpointing import EndpointConfig
from enquiry_desk.voice.session import VoiceSession
from enquiry_desk.webhooks.signing import BadSignature, verify_twilio

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    call_sid TEXT PRIMARY KEY,       -- Twilio's id for the call. Also what makes a replayed webhook harmless
    from_number TEXT,
    started REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'ringing',
    duration_s INTEGER
);
"""


def twiml_stream(socket_url: str, caller: str) -> str:
    """The XML that tells Twilio to stream the call to the desk.

    <Connect><Stream> is two-way: the desk both hears the caller and speaks. The caller's
    number is passed along as a parameter, because the audio stream itself does not carry it.
    """
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response><Connect><Stream url={quoteattr(socket_url)}>"
        f'<Parameter name="from" value={quoteattr(caller)}/>'
        "</Stream></Connect></Response>"
    )


async def twilio_form(request: Request, auth_token: str, public_base_url: str) -> dict[str, str]:
    """Read a Twilio webhook's form fields, after checking that Twilio sent it."""
    body = await request.body()
    params = dict(parse_qsl(body.decode(), keep_blank_values=True))
    url = public_base_url.rstrip("/") + request.url.path + (f"?{request.url.query}" if request.url.query else "")
    try:
        verify_twilio(auth_token, url, params, request.headers.get("x-twilio-signature"))
    except BadSignature as error:
        log.warning("twilio webhook refused: %s", error)
        raise HTTPException(401, "signature not accepted") from None
    return params


def caller_note(number: str, lead) -> str | None:
    """What the model is told about who is calling, from the phone network's caller ID."""
    if not number:
        return None
    if lead is None:
        return f"[Caller ID: this call is from {number}. No lead is saved under this number yet.]"
    known = ", ".join(f"{key} {value}" for key, value in lead.fields.items() if key in ("student_class", "target_exam", "city", "mode"))
    return f"[Caller ID: this call is from {number}, saved as {lead.name or 'an unnamed lead'}" + (f" ({known})" if known else "") + ".]"


def register(app: FastAPI, get_runtime, get_engines) -> None:
    def calls() -> Database:
        db = get_runtime().db
        db.script(SCHEMA)
        return db

    @app.post("/webhooks/twilio/voice")
    async def incoming_call(request: Request) -> Response:
        settings = get_runtime().settings
        params = await twilio_form(request, settings.twilio_auth_token, settings.public_base_url)
        calls().run(
            "INSERT INTO calls (call_sid, from_number, started) VALUES (?, ?, ?) ON CONFLICT (call_sid) DO NOTHING",
            (params.get("CallSid", ""), params.get("From", ""), time.time()),
        )
        socket_url = settings.public_base_url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://") + "/ws/twilio"
        return Response(twiml_stream(socket_url, params.get("From", "")), media_type="application/xml")

    @app.post("/webhooks/twilio/status")
    async def call_status(request: Request) -> dict:
        settings = get_runtime().settings
        params = await twilio_form(request, settings.twilio_auth_token, settings.public_base_url)
        status, duration = params.get("CallStatus", "unknown"), int(params.get("CallDuration") or 0)
        changed = calls().run(
            "UPDATE calls SET status = ?, duration_s = ? WHERE call_sid = ? AND status <> ?",
            (status, duration, params.get("CallSid", ""), status),
        )
        # Only the first delivery of a status writes the note, so a retry does not add a second one.
        if changed and status == "completed":
            try:
                lead = await get_runtime().crm.find_by_phone(normalise_phone(params.get("From", "")))
            except ValueError:
                lead = None
            if lead:
                await get_runtime().crm.add_note(lead.id, f"Phone call with the desk, {duration} seconds.")
        return {"ok": True}

    @app.websocket("/ws/twilio")
    async def media_stream(socket: WebSocket) -> None:
        await socket.accept()
        engines = await get_engines()
        session: VoiceSession | None = None
        stream_sid = ""

        async def send_audio(chunk: bytes) -> None:
            payload = base64.b64encode(audio.desk_to_phone(chunk, engines.tts.rate)).decode()
            await socket.send_text(json.dumps({"event": "media", "streamSid": stream_sid, "media": {"payload": payload}}))

        async def send_event(event: dict) -> None:
            if event["type"] == "clear":  # the caller interrupted: drop what Twilio has not played yet
                await socket.send_text(json.dumps({"event": "clear", "streamSid": stream_sid}))

        try:
            while True:
                message = json.loads(await socket.receive_text())
                kind = message.get("event")
                if kind == "start":
                    stream_sid = message["start"]["streamSid"]
                    number = (message["start"].get("customParameters") or {}).get("from", "")
                    try:
                        phone = normalise_phone(number)
                        lead = await get_runtime().crm.find_by_phone(phone)
                    except ValueError:
                        phone, lead = "", None
                    session = VoiceSession(
                        session_id="call-" + message["start"].get("callSid", stream_sid),
                        desk=get_runtime().desk,
                        vad=engines.make_vad(),
                        stt=engines.stt,
                        tts=engines.tts,
                        send_audio=send_audio,
                        send_event=send_event,
                        config=EndpointConfig(silence_ms=engines.silence_ms),
                        log=engines.log,
                        first_turn_note=caller_note(phone, lead),
                    )
                    await session.start()
                elif kind == "media" and session is not None:
                    await session.feed(audio.phone_to_desk(base64.b64decode(message["media"]["payload"])))
                elif kind == "stop":
                    break
        except (WebSocketDisconnect, RuntimeError):
            pass
        if session is not None:
            await session.wait()
