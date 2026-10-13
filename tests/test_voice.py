"""Phase 5 and 6: speech text, audio formats, turn detection, the sentence gate, a spoken turn."""

from __future__ import annotations

import audioop  # the standard library's reference G.711, used here only to check our own
import time
import warnings

import numpy as np
import pytest
from fastapi.testclient import TestClient

from enquiry_desk.agent import prompts
from enquiry_desk.api import create_app
from enquiry_desk.reports import run_report
from enquiry_desk.voice import audio
from enquiry_desk.voice.endpointing import EndpointConfig, Endpointer
from enquiry_desk.voice.engines import FRAME_SAMPLES, EnergyVAD
from enquiry_desk.voice.server import VoiceEngines
from enquiry_desk.voice.session import VoiceLog, VoiceSession
from enquiry_desk.voice.speech_text import SentenceSplitter, number_to_words, speakable
from tests.conftest import call, say

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---- text for speech ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("number", "words"),
    [
        (0, "zero"), (15, "fifteen"), (48, "forty eight"), (105, "one hundred and five"), (2000, "two thousand"),
        (48000, "forty eight thousand"), (135000, "one lakh thirty five thousand"), (260000, "two lakh sixty thousand"),
        (156000, "one lakh fifty six thousand"), (10000000, "one crore"), (12345678, "one crore twenty three lakh forty five thousand six hundred seventy eight"),
    ],
)
def test_numbers_are_read_in_lakh_and_crore(number, words):
    assert number_to_words(number) == words


def test_a_screen_reply_becomes_something_a_voice_can_say():
    assert speakable("The JEE Dropper Programme costs Rs 1,35,000 [K1].") == "The JEE Dropper Programme costs one lakh thirty five thousand rupees."
    assert speakable("Classes run from 4:00 pm to 8:30 pm [K2].") == "Classes run from four pm to eight thirty pm."
    assert speakable("Online is 60% of the fee.") == "Online is sixty percent of the fee."
    assert speakable("I have your number as +91 98765 43210.") == "I have your number as nine eight seven six five, four three two one zero."
    assert speakable("**Slots:**\n- Tuesday 13 October, 11:00 am") == "Slots: Tuesday thirteen October, eleven am"


def test_sentences_come_out_as_soon_as_they_are_complete():
    splitter = SentenceSplitter()
    assert splitter.feed("The fee is Rs. 1,35") == []
    assert splitter.feed(",000. It includes 18") == ["The fee is Rs. 1,35,000."]  # "Rs." did not end a sentence
    assert splitter.feed("% GST. Shall I book a call") == ["It includes 18% GST."]
    assert splitter.feed("? Version 1.5 is next") == ["Shall I book a call?"]
    assert splitter.flush() == ["Version 1.5 is next"] and splitter.flush() == []
    assert SentenceSplitter().feed("Fees ye hain। Aur kuch? ") == ["Fees ye hain।", "Aur kuch?"]


# ---- audio formats --------------------------------------------------------------------------


def tone(seconds: float, hz: float = 440.0, rate: int = 16_000, level: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (level * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_mulaw_matches_the_standard_library_for_every_sample_value():
    every_value = np.arange(-32768, 32768, dtype="<i2").tobytes()
    assert audio.mulaw_encode(every_value) == audioop.lin2ulaw(every_value, 2)
    every_code = bytes(range(256))
    assert audio.mulaw_decode(every_code) == audioop.ulaw2lin(every_code, 2)
    assert audio.mulaw_encode(b"\x00\x00") == b"\xff"  # silence is 0xFF on a phone line, not zero


def test_resampling_keeps_the_length_and_the_pitch():
    one_second = tone(1.0, hz=440, rate=16_000)
    down = audio.resample(one_second, 16_000, 8_000)
    assert len(down) == 8_000
    crossings = int(np.sum(np.diff(np.sign(down)) != 0))
    assert 870 <= crossings <= 890  # a 440 Hz tone crosses zero 880 times a second at any sample rate
    assert len(audio.resample(down, 8_000, 16_000)) == 16_000


def test_a_phone_call_round_trip_keeps_speech_shaped():
    spoken = audio.float_to_pcm16(tone(0.5, hz=300, rate=22_050))
    on_the_line = audio.desk_to_phone(spoken, 22_050)
    assert len(on_the_line) == 4_000  # half a second at 8,000 one-byte samples
    back = audio.pcm16_to_float(audio.phone_to_desk(on_the_line))
    assert len(back) == 8_000 and 0.2 < float(np.sqrt(np.mean(back**2))) < 0.45
    assert audio.seconds(spoken, 22_050) == pytest.approx(0.5, abs=0.001)


# ---- turn detection -------------------------------------------------------------------------


def events(endpointer: Endpointer, probabilities: list[float]) -> list[tuple[int, str]]:
    return [(i, e) for i, p in enumerate(probabilities) if (e := endpointer.feed(p))]


def test_a_turn_starts_after_real_speech_and_ends_after_a_long_enough_silence():
    config = EndpointConfig(frame_ms=32, min_speech_ms=160, silence_ms=640)
    stream = [0.0] * 5 + [0.9] * 20 + [0.1] * 25
    # Speech is confirmed on its 5th frame (160 ms), and ended on the 20th silent frame (640 ms).
    assert events(Endpointer(config), stream) == [(9, "speech_start"), (44, "speech_end")]


def test_a_cough_is_ignored_and_a_pause_does_not_end_the_turn():
    config = EndpointConfig(frame_ms=32, min_speech_ms=160, silence_ms=640)
    cough = [0.9] * 3 + [0.0] * 30
    assert events(Endpointer(config), cough) == []
    thinking = [0.9] * 10 + [0.1] * 15 + [0.9] * 10 + [0.1] * 20  # a 480 ms pause in the middle
    assert [e for _, e in events(Endpointer(config), thinking)] == ["speech_start", "speech_end"]
    # Between the two thresholds counts as still speaking, so a fading word does not flicker.
    fading = [0.9] * 10 + [0.4] * 40
    assert [e for _, e in events(Endpointer(config), fading)] == ["speech_start"]


# ---- the sentence gate ----------------------------------------------------------------------


async def test_voice_gets_each_sentence_as_it_is_written(world):
    spoken: list[str] = []
    world.chat.replies = [say("Hello there. The centre is in Noida. How can I help you?")]
    result = await world.desk.handle("v1", "hello", on_sentence=spoken.append)

    assert spoken == ["Hello there.", "The centre is in Noida.", "How can I help you?"]
    assert result.reply == "Hello there. The centre is in Noida. How can I help you?"  # no AI line: the greeting said it
    assert "v1" not in world.desk._sinks  # the sink is removed when the turn ends


async def test_a_sentence_with_an_invented_figure_stops_the_speech(world):
    spoken: list[str] = []
    world.chat.replies = [say("Let me tell you. The dropper fee is 99,999 rupees. We also have a library.")]
    result = await world.desk.handle("v1", "what is the fee", on_sentence=spoken.append)

    # The first sentence was already spoken. The bad one and everything after it never is.
    assert spoken == ["Let me tell you.", prompts.NOT_GROUNDED]
    assert result.reply == "Let me tell you. " + prompts.NOT_GROUNDED
    assert result.problems == ["the figure 99999 is not in any source"]
    assert len(world.chat.requests) == 1  # no second attempt in voice
    assert world.db.query("SELECT outcome FROM reply_checks")[0]["outcome"] == "refused"


async def test_a_tool_call_written_as_text_is_never_spoken(world):
    spoken: list[str] = []
    world.chat.replies = [say('search_knowledge {"query": "library seats"}'), say("The library seats 80 students [K1].")]
    await world.desk.handle("v1", "how many seats in the library", on_sentence=spoken.append)
    assert spoken == ["The library seats 80 students [K1]."]


# ---- a whole spoken turn --------------------------------------------------------------------


class ScriptedSTT:
    def __init__(self, texts: list[str]):
        self.texts = list(texts)
        self.heard_seconds: list[float] = []

    def transcribe(self, pcm16: bytes) -> str:
        self.heard_seconds.append(audio.seconds(pcm16, 16_000))
        return self.texts.pop(0)


class BeepTTS:
    """Half a second of tone for every sentence, and a record of what it was asked to say."""

    rate = 16_000

    def __init__(self):
        self.said: list[str] = []

    def synthesize(self, text: str):
        self.said.append(text)
        yield audio.float_to_pcm16(tone(0.5))


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def frames(kind: str, count: int) -> bytes:
    samples = tone(count * FRAME_SAMPLES / 16_000) if kind == "speech" else np.zeros(count * FRAME_SAMPLES, dtype=np.float32)
    return audio.float_to_pcm16(samples)


def voice_session(world, stt_texts: list[str]):
    out: list[bytes] = []
    sent: list[dict] = []
    clock = Clock()

    async def send_audio(chunk: bytes) -> None:
        out.append(chunk)

    async def send_event(event: dict) -> None:
        sent.append(event)

    stt, tts = ScriptedSTT(stt_texts), BeepTTS()
    session = VoiceSession(
        session_id="call-1", desk=world.desk, vad=EnergyVAD(), stt=stt, tts=tts, send_audio=send_audio, send_event=send_event,
        config=EndpointConfig(silence_ms=320), log=VoiceLog(world.db), clock=clock,
    )
    return session, stt, tts, out, sent, clock


async def test_a_spoken_question_is_answered_in_speech(world):
    session, stt, tts, out, sent, clock = voice_session(world, ["What is the dropper fee?"])
    world.chat.replies = [
        call(("search_knowledge", {"query": "JEE Dropper Programme classroom fee"})),
        say("The JEE Dropper Programme costs 1,35,000 [K1]. Shall I book a call?"),
    ]
    await session.start()
    assert tts.said == ["I am an AI assistant at Meridian Demo Academy. How can I help you?"]
    greeting_chunks = len(out)
    clock.now += 2  # the greeting has finished playing before the caller speaks

    await session.feed(frames("silence", 10) + frames("speech", 30) + frames("silence", 15))
    await session.wait()

    # The utterance given to speech-to-text is the speech plus the pre-roll, not the long silences.
    assert 0.9 < stt.heard_seconds[0] < 1.6
    assert tts.said[-2:] == ["The JEE Dropper Programme costs one lakh thirty five thousand.", "Shall I book a call?"]
    assert len(out) == greeting_chunks + 2
    assert [e["type"] for e in sent] == ["heard", "reply"] and sent[0]["text"] == "What is the dropper fee?"
    assert sent[1]["interrupted"] is False and "first_audio_ms" in sent[1]["timings"]
    row = world.db.query("SELECT * FROM voice_turns")[0]
    assert row["audio_out_s"] == pytest.approx(1.0) and row["interrupted"] == 0 and row["first_audio_ms"] is not None
    assert run_report(world.db, "voice_latency")[0]["turns"] == 1


async def test_speaking_over_the_desk_stops_it_and_the_model_is_told_what_was_heard(world):
    session, _stt, _tts, _out, sent, clock = voice_session(world, ["Tell me about the academy.", "Stop, just the fee please."])
    world.chat.replies = [say("We teach for JEE and NEET. We have one centre in Noida. Classes are in the evening."), say("Of course.")]
    await session.feed(frames("speech", 20) + frames("silence", 12))
    await session.wait()
    # Three half-second sentences were sent at once. On the caller's speaker, 0.7 s have played.
    clock.now += 0.7
    await session.feed(frames("speech", 20) + frames("silence", 12))
    await session.wait()

    assert [e["type"] for e in sent] == ["heard", "reply", "clear", "heard", "reply"]
    second_request = world.chat.requests[1][-1]["content"]
    assert second_request.startswith('[The caller interrupted you. Of your last reply they heard only: "We teach for JEE and NEET. We have one')
    assert second_request.endswith("Stop, just the fee please.") and "Classes are in the evening" not in second_request


async def test_noise_that_transcribes_to_nothing_gets_no_reply(world):
    session, _stt, tts, _out, sent, _ = voice_session(world, ["  "])
    await session.feed(frames("speech", 20) + frames("silence", 12))
    await session.wait()
    assert sent == [{"type": "heard", "text": ""}] and tts.said == [] and world.chat.requests == []


def test_the_voice_socket_carries_a_call(world):
    stt, tts = ScriptedSTT(["hello"]), BeepTTS()
    world.voice_engines = VoiceEngines(stt, tts, VoiceLog(world.db), EnergyVAD, silence_ms=320)
    world.chat.replies = [say("Hello. How can I help?")]
    with TestClient(create_app(world, run_worker=False)) as client:
        assert "Enquiry Desk" in client.get("/voice").text
        with client.websocket_connect("/ws/voice") as socket:
            ready = socket.receive_json()
            assert ready["type"] == "ready" and ready["rate"] == 16_000
            assert len(socket.receive_bytes()) == 16_000  # the greeting: half a second of 16-bit audio
            time.sleep(0.6)  # let the greeting finish, or speaking now would count as interrupting it
            socket.send_bytes(frames("speech", 20) + frames("silence", 12))
            assert socket.receive_json() == {"type": "heard", "text": "hello"}
            assert len(socket.receive_bytes()) == 16_000 and len(socket.receive_bytes()) == 16_000
            assert socket.receive_json()["text"] == "Hello. How can I help?"
