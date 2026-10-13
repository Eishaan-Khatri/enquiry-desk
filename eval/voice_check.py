"""The voice path with the real models, on audio made by the desk's own voice.

    uv run python eval/voice_check.py

For each test sentence it makes caller audio with the text-to-speech voice, plays it into a
voice session frame by frame, and records what the session heard and how long each stage took.

READ THIS BEFORE QUOTING ANY NUMBER FROM IT. The "caller" is a synthetic voice in a silent
room. That is the easiest audio a speech recogniser will ever get: no accent, no noise, no
phone line. The error rate here is a floor, not an estimate for real callers. What the run
does show honestly is the timing of each stage on this machine, and that the parts connect.

The desk's model is replaced by a stub that answers after a fixed delay, so the timings
isolate speech to text and text to speech.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from enquiry_desk.agent.service import TurnResult
from enquiry_desk.config import get_settings
from enquiry_desk.tenant import load_tenant
from enquiry_desk.voice import audio
from enquiry_desk.voice.endpointing import EndpointConfig
from enquiry_desk.voice.engines import FRAME_SAMPLES, PiperTTS, SileroVAD, WhisperSTT
from enquiry_desk.voice.session import VoiceSession
from enquiry_desk.voice.wer import word_error_rate

HERE = Path(__file__).parent
MODEL_DELAY_S = 0.4
REPLY = ["The JEE Dropper Programme costs 1,35,000 [K1].", "Shall I book a counselling call for you?"]

ENGLISH = [
    "What is the fee for the JEE dropper classroom batch?",
    "When is the scholarship test held?",
    "My daughter is in class eleven and wants to prepare for NEET.",
    "Do you have a hostel for students from outside Noida?",
    "I want to book a counselling call for tomorrow evening.",
    "My phone number is nine eight seven six five four three two one zero.",
    "Can the fee be paid in instalments?",
    "I want to talk to a counsellor, not a machine.",
]
HINDI = [
    "मुझे फीस के बारे में जानकारी चाहिए",
    "क्या आपके यहाँ हॉस्टल की सुविधा है",
    "नया बैच कब से शुरू होगा",
]


class StubDesk:
    """Answers every turn with two fixed sentences, the first after a fixed delay."""

    def __init__(self, tenant):
        self.tenant = tenant

    async def handle(self, session_id: str, text: str, on_sentence=None) -> TurnResult:
        for sentence in REPLY:
            await asyncio.sleep(MODEL_DELAY_S if sentence is REPLY[0] else 0.15)
            on_sentence(sentence)
        return TurnResult(session_id, 1, " ".join(REPLY), "ok")


def caller_audio(voice: PiperTTS, sentence: str) -> bytes:
    """A caller saying the sentence: half a second of quiet, the speech at 16 kHz, a second of quiet."""
    speech = np.concatenate([audio.pcm16_to_float(chunk) for chunk in voice.synthesize(sentence)])
    speech = audio.resample(speech, voice.rate, audio.DESK_RATE)
    quiet = np.zeros(audio.DESK_RATE // 2, dtype=np.float32)
    samples = np.concatenate([quiet, speech, quiet, quiet])
    samples = samples[: len(samples) // FRAME_SAMPLES * FRAME_SAMPLES]
    return audio.float_to_pcm16(samples)


async def run(label: str, sentences: list[str], caller: PiperTTS, stt: WhisperSTT, desk_voice: PiperTTS, tenant) -> dict:
    rows = []
    for sentence in sentences:
        events: list[dict] = []
        out: list[bytes] = []

        async def send_audio(chunk: bytes, out=out) -> None:
            out.append(chunk)

        async def send_event(event: dict, events=events) -> None:
            events.append(event)

        session = VoiceSession(
            session_id="check", desk=StubDesk(tenant), vad=SileroVAD(), stt=stt, tts=desk_voice,
            send_audio=send_audio, send_event=send_event, config=EndpointConfig(silence_ms=700),
        )
        pcm = caller_audio(caller, sentence)
        frame = FRAME_SAMPLES * 2
        for start in range(0, len(pcm), frame):
            await session.feed(pcm[start : start + frame])
            await asyncio.sleep(0)  # audio arrives as fast as we can feed it. A real call paces itself
        await session.wait()
        heard = next((e["text"] for e in events if e["type"] == "heard"), "")
        timings = next((e["timings"] for e in events if e["type"] == "reply"), {})
        rows.append({"said": sentence, "heard": heard, "wer": round(word_error_rate(sentence, heard), 3), **timings})
        print(f"  said : {sentence}\n  heard: {heard}\n         WER {rows[-1]['wer']:.2f}, speech to text {timings.get('stt_ms', 0):.0f} ms, "
              f"first audio {timings.get('first_audio_ms', 0):.0f} ms after end of speech, caller audio {timings.get('audio_in_s', 0):.1f} s")
    done = [r for r in rows if "first_audio_ms" in r]
    summary = {
        "sentences": len(rows),
        "turn_detected": len(done),
        "mean_wer": round(statistics.mean(r["wer"] for r in rows), 3),
        "median_stt_ms": round(statistics.median(r["stt_ms"] for r in done)) if done else None,
        "median_first_audio_ms": round(statistics.median(r["first_audio_ms"] for r in done)) if done else None,
        "median_tts_ms": round(statistics.median(r["first_audio_ms"] - r["first_sentence_ms"] for r in done)) if done else None,
    }
    print(f"  {label}: {summary}\n")
    return {"label": label, "summary": summary, "rows": rows}


async def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # a Windows console cannot print Devanagari otherwise
    settings = get_settings()
    tenant = load_tenant(settings.tenant_dir)
    voices = settings.voice_file.parent
    started = time.perf_counter()
    english_voice = PiperTTS(voices / "en_US-lessac-medium.onnx")
    stt = WhisperSTT(settings.voice_stt_model, download_root=str(settings.data_dir / "whisper"))
    print(f"Models loaded in {time.perf_counter() - started:.1f} s. Stub model delay {MODEL_DELAY_S} s. End-of-turn silence 700 ms.\n")

    results = [await run("English, synthetic voice", ENGLISH, english_voice, stt, english_voice, tenant)]
    hindi_path = voices / "hi_IN-pratham-medium.onnx"
    if hindi_path.exists():
        hindi_voice = PiperTTS(hindi_path)
        results.append(await run("Hindi, language detected by the model", HINDI, hindi_voice, stt, english_voice, tenant))
        # Hindi and Urdu sound the same and are written in different scripts. Left to detect the
        # language, the model may pick Urdu and write Arabic script. Pinning the language fixes the script.
        pinned = WhisperSTT(settings.voice_stt_model, download_root=str(settings.data_dir / "whisper"), language="hi")
        results.append(await run("Hindi, language pinned to hi", HINDI, hindi_voice, pinned, english_voice, tenant))

    out = HERE / "results" / "voice_check.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps({"stt_model": settings.voice_stt_model, "note": "Synthetic caller audio. Error rates are a floor.",
                    "model_delay_s": MODEL_DELAY_S, "results": results}, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
