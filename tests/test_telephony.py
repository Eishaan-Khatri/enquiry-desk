"""Phase 6: the phone webhooks and the audio stream, against messages shaped like Twilio's."""

from __future__ import annotations

import base64
import json
import time
from urllib.parse import urlencode

from fastapi.testclient import TestClient

from enquiry_desk.api import create_app
from enquiry_desk.crm import Lead
from enquiry_desk.voice import audio
from enquiry_desk.voice.engines import EnergyVAD
from enquiry_desk.voice.server import VoiceEngines
from enquiry_desk.voice.session import VoiceLog
from enquiry_desk.voice.telephony import caller_note, twiml_stream
from enquiry_desk.webhooks import signing
from tests.conftest import say
from tests.test_voice import BeepTTS, ScriptedSTT, frames

TOKEN = "tw-token"
CALL = {"CallSid": "CA123", "From": "+919876543210", "To": "+911204567890", "CallStatus": "ringing"}


def twilio_post(client: TestClient, path: str, params: dict, token: str = TOKEN):
    signature = signing.twilio(token, "https://desk.test" + path, params)
    headers = {"X-Twilio-Signature": signature, "Content-Type": "application/x-www-form-urlencoded"}
    return client.post(path, content=urlencode(params), headers=headers)


def phone_messages(pcm16_at_16k: bytes) -> list[str]:
    """Audio as Twilio would send it: mu-law at 8 kHz, 20 ms (160 bytes) per message."""
    line = audio.desk_to_phone(pcm16_at_16k, 16_000)
    return [
        json.dumps({"event": "media", "media": {"payload": base64.b64encode(line[i : i + 160]).decode()}})
        for i in range(0, len(line), 160)
    ]


def test_the_stream_instruction_is_valid_xml_and_escapes_its_values():
    xml = twiml_stream("wss://desk.test/ws/twilio?a=1&b=2", '+91"98765')
    assert xml.startswith('<?xml version="1.0" encoding="UTF-8"?><Response><Connect><Stream url="wss://desk.test/ws/twilio?a=1&amp;b=2">')
    assert "<Parameter name=\"from\" value='+91\"98765'/>" in xml and xml.endswith("</Stream></Connect></Response>")


def test_caller_id_becomes_a_note_for_the_model():
    assert caller_note("", None) is None
    assert caller_note("+919876543210", None) == "[Caller ID: this call is from +919876543210. No lead is saved under this number yet.]"
    lead = Lead(id="1", phone="+919876543210", name="Asha Verma", fields={"student_class": "11", "target_exam": "JEE", "lead_score": "80"})
    assert caller_note("+919876543210", lead) == "[Caller ID: this call is from +919876543210, saved as Asha Verma (student_class 11, target_exam JEE).]"


def test_an_incoming_call_is_checked_logged_once_and_told_to_stream(world):
    world.settings.twilio_auth_token = TOKEN
    with TestClient(create_app(world, run_worker=False)) as client:
        assert twilio_post(client, "/webhooks/twilio/voice", CALL, token="guess").status_code == 401
        assert client.post("/webhooks/twilio/voice", content=urlencode(CALL)).status_code == 401

        answer = twilio_post(client, "/webhooks/twilio/voice", CALL)
        assert answer.status_code == 200 and answer.headers["content-type"].startswith("application/xml")
        assert '<Stream url="wss://desk.test/ws/twilio">' in answer.text
        assert '<Parameter name="from" value="+919876543210"/>' in answer.text
        twilio_post(client, "/webhooks/twilio/voice", CALL)  # the same webhook again

    rows = world.db.query("SELECT call_sid, from_number, status FROM calls")
    assert [tuple(dict(row).values()) for row in rows] == [("CA123", "+919876543210", "ringing")]


async def test_the_end_of_a_call_is_noted_on_the_lead_once(world):
    world.settings.twilio_auth_token = TOKEN
    lead = await world.crm.upsert_lead("+919876543210", "Asha", {})
    done = {**CALL, "CallStatus": "completed", "CallDuration": "74"}
    with TestClient(create_app(world, run_worker=False)) as client:
        twilio_post(client, "/webhooks/twilio/voice", CALL)
        assert twilio_post(client, "/webhooks/twilio/status", done).json() == {"ok": True}
        twilio_post(client, "/webhooks/twilio/status", done)  # Twilio retries. No second note.

    row = world.db.query("SELECT status, duration_s FROM calls")[0]
    assert (row["status"], row["duration_s"]) == ("completed", 74)
    assert world.crm.notes(lead.id) == ["Phone call with the desk, 74 seconds."]


def test_a_phone_call_is_heard_and_answered_in_phone_audio(world):
    world.voice_engines = VoiceEngines(ScriptedSTT(["What is the fee?"]), BeepTTS(), VoiceLog(world.db), EnergyVAD, silence_ms=320)
    world.chat.replies = [say("Let me check that for you.")]
    start = {"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA123", "customParameters": {"from": "+919876543210"},
                                         "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}}}
    with TestClient(create_app(world, run_worker=False)) as client, client.websocket_connect("/ws/twilio") as socket:
        socket.send_text(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        socket.send_text(json.dumps(start))

        greeting = json.loads(socket.receive_text())
        assert greeting["event"] == "media" and greeting["streamSid"] == "MZ1"
        on_the_line = base64.b64decode(greeting["media"]["payload"])
        assert len(on_the_line) == 4_000  # half a second, 8,000 one-byte samples a second
        time.sleep(0.6)  # the greeting finishes playing

        for message in phone_messages(frames("speech", 25) + frames("silence", 14)):
            socket.send_text(message)
        reply = json.loads(socket.receive_text())
        assert reply["event"] == "media" and len(base64.b64decode(reply["media"]["payload"])) == 4_000
        # Wait for the turn to be recorded while the call is still up. Once the test client
        # closes the socket it also cancels the server's handler, which a real server does not.
        deadline = time.time() + 5
        while not (rows := world.db.query("SELECT session_id FROM voice_turns")) and time.time() < deadline:
            time.sleep(0.05)
        socket.send_text(json.dumps({"event": "stop"}))

    # The model was told who is calling, from caller ID, before the caller's own words.
    first_words = world.chat.requests[0][-1]["content"]
    assert first_words == "[Caller ID: this call is from +919876543210. No lead is saved under this number yet.]\nWhat is the fee?"
    assert rows[0]["session_id"] == "call-CA123"
