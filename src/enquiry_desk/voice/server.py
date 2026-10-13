"""The voice endpoints: a page to talk from, and the WebSocket the audio travels over.

What goes over the WebSocket:
  browser -> desk   binary messages: 16-bit PCM, 16,000 Hz, mono
  desk -> browser   binary messages: 16-bit PCM at the rate announced in "ready"
                    text messages (JSON): ready, heard, reply, clear
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from enquiry_desk.config import ROOT, Settings
from enquiry_desk.db import Database
from enquiry_desk.voice import telephony
from enquiry_desk.voice.endpointing import EndpointConfig
from enquiry_desk.voice.engines import (
    PiperTTS,
    SileroVAD,
    SpeechToText,
    TextToSpeech,
    VoiceActivity,
    WhisperSTT,
)
from enquiry_desk.voice.session import VoiceLog, VoiceSession


@dataclass
class VoiceEngines:
    stt: SpeechToText
    tts: TextToSpeech
    log: VoiceLog
    make_vad: type[VoiceActivity] | None = None  # one per call, because it keeps state between frames
    silence_ms: int = 700


class OneAtATime:
    """Wraps a voice so that two calls never synthesise at the same moment.

    The synthesiser is not documented as safe to use from two threads. A lock is the plain
    fix. It also means two simultaneous callers take turns, which a real deployment would
    solve with one voice instance per worker.
    """

    def __init__(self, inner: TextToSpeech):
        self._inner = inner
        self._lock = threading.Lock()
        self.rate = inner.rate

    def synthesize(self, text: str) -> Iterator[bytes]:
        with self._lock:
            yield from list(self._inner.synthesize(text))


def load_engines(settings: Settings, db: Database, business_name: str) -> VoiceEngines:
    """Load the models. Slow the first time: the files are read from disk, or downloaded."""
    stt = WhisperSTT(
        settings.voice_stt_model,
        download_root=str(settings.data_dir / "whisper"),
        language=settings.voice_language or None,
        hints=f"{business_name}. JEE, NEET, dropper, counselling, scholarship, classroom, online batch.",
    )
    tts = OneAtATime(PiperTTS(Path(settings.voice_file)))
    return VoiceEngines(stt, tts, VoiceLog(db), SileroVAD, settings.voice_silence_ms)


def register(app: FastAPI, get_runtime) -> None:
    lock = asyncio.Lock()

    async def engines() -> VoiceEngines:
        runtime = get_runtime()
        async with lock:  # two calls arriving together must not load the models twice
            if getattr(runtime, "voice_engines", None) is None:
                runtime.voice_engines = await asyncio.to_thread(
                    load_engines, runtime.settings, runtime.db, runtime.tenant.business_name
                )
        return runtime.voice_engines

    telephony.register(app, get_runtime, engines)

    @app.get("/voice")
    async def voice_page() -> FileResponse:
        return FileResponse(ROOT / "web" / "voice.html")

    @app.websocket("/ws/voice")
    async def voice_socket(socket: WebSocket) -> None:
        await socket.accept()
        loaded = await engines()
        session = VoiceSession(
            session_id="voice-" + uuid.uuid4().hex[:12],
            desk=get_runtime().desk,
            vad=loaded.make_vad(),
            stt=loaded.stt,
            tts=loaded.tts,
            send_audio=socket.send_bytes,
            send_event=socket.send_json,
            config=EndpointConfig(silence_ms=loaded.silence_ms),
            log=loaded.log,
        )
        await socket.send_json({"type": "ready", "rate": loaded.tts.rate, "session_id": session.session_id})
        try:
            await session.start()
            while True:
                message = await socket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes"):
                    await session.feed(message["bytes"])
        except (WebSocketDisconnect, RuntimeError):
            pass  # the caller hung up. A send after that raises, and there is nobody to tell.
