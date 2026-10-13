"""One voice call, from audio in to audio out.

    microphone or phone
        -> 32 ms frames -> voice activity -> has the caller started? stopped?
        -> the finished utterance -> speech to text
        -> the desk (the same graph the chat uses), which hands back one sentence at a time
        -> each sentence rewritten for speech -> text to speech -> audio out

Three things here have no equivalent in chat:

Turn-taking. Nobody presses Enter. The end of a turn is a guess made from silence.

Interruption. If the caller speaks while the desk is speaking, the desk stops at once. It also
notes what the caller had heard by then, because the model wrote a whole reply and believes
all of it was said.

Latency by stage. One number, "it feels slow", is no use. Every turn records how long each
stage took, so the slow stage can be found.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from enquiry_desk.agent import prompts
from enquiry_desk.agent.service import Desk
from enquiry_desk.db import Database
from enquiry_desk.voice.audio import DESK_RATE, pcm16_to_float, seconds
from enquiry_desk.voice.endpointing import EndpointConfig, Endpointer
from enquiry_desk.voice.engines import FRAME_SAMPLES, SpeechToText, TextToSpeech, VoiceActivity
from enquiry_desk.voice.speech_text import SentenceSplitter, speakable

FRAME_BYTES = FRAME_SAMPLES * 2

SCHEMA = """
-- One row per spoken turn. Timings only: what was said lives in the conversation's own state.
CREATE TABLE IF NOT EXISTS voice_turns (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    session_id TEXT NOT NULL,
    audio_in_s REAL NOT NULL,        -- length of what the caller said
    stt_ms REAL NOT NULL,            -- speech to text
    first_sentence_ms REAL,          -- end of speech -> the desk's first sentence exists
    first_audio_ms REAL,             -- end of speech -> first audio sent back. The number that matters
    total_ms REAL NOT NULL,          -- end of speech -> last audio sent
    audio_out_s REAL NOT NULL,       -- length of what the desk said
    interrupted INTEGER NOT NULL
);
"""


@dataclass
class Spoken:
    sentence: str
    starts: float  # when the caller starts to hear it, on the session's clock
    ends: float


class VoiceLog:
    def __init__(self, db: Database):
        self._db = db
        db.script(SCHEMA)

    def record(self, session_id: str, **t: float) -> None:
        self._db.run(
            "INSERT INTO voice_turns (ts, session_id, audio_in_s, stt_ms, first_sentence_ms, first_audio_ms,"
            " total_ms, audio_out_s, interrupted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), session_id, t["audio_in_s"], t["stt_ms"], t.get("first_sentence_ms"), t.get("first_audio_ms"),
             t["total_ms"], t["audio_out_s"], int(t["interrupted"])),
        )


class VoiceSession:
    def __init__(
        self,
        *,
        session_id: str,
        desk: Desk,
        vad: VoiceActivity,
        stt: SpeechToText,
        tts: TextToSpeech,
        send_audio: Callable[[bytes], Awaitable[None]],
        send_event: Callable[[dict], Awaitable[None]],
        config: EndpointConfig | None = None,
        log: VoiceLog | None = None,
        clock: Callable[[], float] = time.perf_counter,
        first_turn_note: str | None = None,
    ):
        self.session_id = session_id
        self._desk = desk
        self._vad = vad
        self._stt = stt
        self._tts = tts
        self._raw_send_audio = send_audio
        self._raw_send_event = send_event
        self._gone = False  # the caller has hung up, so there is nobody to send to
        self._endpointer = Endpointer(config)
        self._log = log
        self._clock = clock
        self._buffer = b""
        pre_roll_frames = max(1, self._endpointer.config.pre_roll_ms // self._endpointer.config.frame_ms)
        self._pre_roll: deque[bytes] = deque(maxlen=pre_roll_frames)
        self._utterance: list[bytes] = []
        self._turn_lock = asyncio.Lock()  # one desk turn at a time, in the order the caller spoke
        self._tasks: set[asyncio.Task] = set()
        self._playing: list[Spoken] = []  # what has been sent for the current reply, with its timing
        self._generation = 0  # goes up when the caller interrupts. Speech from an older one is dropped
        self._cut_off: str | None = None  # what the caller had heard when they interrupted
        self._first_turn_note = first_turn_note  # put in front of the caller's first words, once

    async def _send_audio(self, chunk: bytes) -> None:
        await self._send(self._raw_send_audio, chunk)

    async def _send_event(self, event: dict) -> None:
        await self._send(self._raw_send_event, event)

    async def _send(self, send, item) -> None:
        # A caller can hang up in the middle of a reply. The send then fails, and that must
        # not stop the turn from being finished and its timings recorded.
        if self._gone:
            return
        try:
            await send(item)
        except Exception:
            self._gone = True

    # ---- audio in -------------------------------------------------------------------------

    async def start(self) -> None:
        """Speak first. The greeting tells the caller they are talking to a machine."""
        greeting = prompts.disclosure(self._desk.tenant) + "How can I help you?"
        await self._say([greeting], generation=self._generation, timings={})

    async def feed(self, pcm16: bytes) -> None:
        """Take audio of any length at 16,000 Hz and work through it one 32 ms frame at a time."""
        self._buffer += pcm16
        while len(self._buffer) >= FRAME_BYTES:
            frame, self._buffer = self._buffer[:FRAME_BYTES], self._buffer[FRAME_BYTES:]
            await self._frame(frame)

    async def _frame(self, frame: bytes) -> None:
        event = self._endpointer.feed(self._vad.probability(pcm16_to_float(frame)))
        if event == "speech_start":
            # The frames that came before speech was confirmed hold the start of the first word.
            self._utterance = [*self._pre_roll, frame]
            self._pre_roll.clear()
            if self._desk_is_audible():
                await self._interrupt()
        elif self._endpointer.speaking:
            self._utterance.append(frame)
        elif event == "speech_end":
            audio, self._utterance = b"".join(self._utterance), []
            task = asyncio.create_task(self._respond(audio))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        else:
            self._pre_roll.append(frame)

    # ---- interruption ---------------------------------------------------------------------

    def _desk_is_audible(self) -> bool:
        # Audio is sent faster than it plays, so "still being sent" is the wrong test. The
        # question is whether the caller's speaker is still playing what was sent.
        return bool(self._playing) and self._clock() < self._playing[-1].ends

    async def _interrupt(self) -> None:
        now = self._clock()
        heard = [s.sentence for s in self._playing if s.ends <= now]
        cut = next((s.sentence for s in self._playing if s.starts < now < s.ends), None)
        self._cut_off = " ".join(heard) + (f" {cut[: len(cut) // 2]}..." if cut else "")
        self._generation += 1  # anything still queued for the old reply is now ignored
        self._playing = []
        await self._send_event({"type": "clear"})  # the client throws away audio it has not played yet

    # ---- one turn -------------------------------------------------------------------------

    async def _respond(self, audio: bytes) -> None:
        end_of_speech = self._clock()
        async with self._turn_lock:
            text = (await asyncio.to_thread(self._stt.transcribe, audio)).strip()
            timings: dict = {"audio_in_s": seconds(audio, DESK_RATE), "stt_ms": (self._clock() - end_of_speech) * 1000,
                             "audio_out_s": 0.0, "end_of_speech": end_of_speech,
                             # The clock above starts when the desk decides the caller has stopped,
                             # which is already this long after their last word. What the caller
                             # sits through is this plus first_audio_ms.
                             "endpoint_silence_ms": float(self._endpointer.config.silence_ms)}
            await self._send_event({"type": "heard", "text": text})
            if not text:
                return  # noise that passed for speech. Say nothing and keep listening.

            if self._cut_off is not None:
                # The model thinks its whole last reply was heard. Tell it what really was.
                text = f'[The caller interrupted you. Of your last reply they heard only: "{self._cut_off.strip()}"]\n{text}'
                self._cut_off = None
            if self._first_turn_note:
                text, self._first_turn_note = f"{self._first_turn_note}\n{text}", None
            self._playing = []
            generation = self._generation
            queue: asyncio.Queue[str | None] = asyncio.Queue()
            speaker = asyncio.create_task(self._speak_queue(queue, generation, timings))
            streamed = 0

            def on_sentence(sentence: str) -> None:
                nonlocal streamed
                streamed += 1
                timings.setdefault("first_sentence_ms", (self._clock() - end_of_speech) * 1000)
                queue.put_nowait(sentence)

            result = await self._desk.handle(self.session_id, text, on_sentence=on_sentence)
            if streamed == 0:
                # A fixed line or a handoff. It did not come through the model's stream.
                splitter = SentenceSplitter()
                for sentence in [*splitter.feed(result.reply), *splitter.flush()]:
                    queue.put_nowait(sentence)
            queue.put_nowait(None)
            await speaker

            interrupted = generation != self._generation
            timings["total_ms"] = (self._clock() - end_of_speech) * 1000
            await self._send_event({"type": "reply", "text": result.reply, "status": result.status, "interrupted": interrupted,
                                    "timings": {k: round(v, 1) for k, v in timings.items() if k.endswith(("_ms", "_s"))}})
            if self._log:
                self._log.record(self.session_id, interrupted=interrupted, **timings)

    async def _speak_queue(self, queue: asyncio.Queue, generation: int, timings: dict) -> None:
        while (sentence := await queue.get()) is not None:
            await self._say([sentence], generation, timings)

    async def _say(self, sentences: list[str], generation: int, timings: dict) -> None:
        for sentence in sentences:
            words = speakable(sentence)
            if not words or generation != self._generation:
                continue
            # Synthesis is CPU work, so it runs in a thread and the event loop keeps listening.
            chunks = await asyncio.to_thread(lambda text=words: list(self._tts.synthesize(text)))
            for chunk in chunks:
                if generation != self._generation:
                    return  # the caller interrupted while this sentence was being made
                now = self._clock()
                if "end_of_speech" in timings:
                    timings.setdefault("first_audio_ms", (now - timings["end_of_speech"]) * 1000)
                length = seconds(chunk, self._tts.rate)
                starts = max(now, self._playing[-1].ends if self._playing else now)
                self._playing.append(Spoken(sentence, starts, starts + length))
                timings["audio_out_s"] = timings.get("audio_out_s", 0.0) + length
                await self._send_audio(chunk)

    async def wait(self) -> None:
        """Wait for every turn in progress to finish. Used at hang-up and in tests."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
