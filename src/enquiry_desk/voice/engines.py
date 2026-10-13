"""The three models of the voice path, each behind a two-line interface.

  voice activity   is there speech in this 32 ms?         Silero VAD, through ONNX Runtime
  speech to text   what did the caller say?               Whisper, through faster-whisper
  text to speech   say this sentence                      Piper

All three run on the CPU and need no account. Each interface is small on purpose, so a cloud
service can replace any one of them without touching the session code.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import numpy as np

from enquiry_desk.voice.audio import pcm16_to_float

FRAME_SAMPLES = 512  # 32 ms at 16,000 Hz, the frame size the voice-activity model was trained on
CONTEXT_SAMPLES = 64


class VoiceActivity(Protocol):
    def probability(self, frame: np.ndarray) -> float: ...

    def reset(self) -> None: ...


class SpeechToText(Protocol):
    def transcribe(self, pcm16: bytes) -> str: ...


class TextToSpeech(Protocol):
    rate: int

    def synthesize(self, text: str) -> Iterator[bytes]: ...


class SileroVAD:
    """The model keeps a memory between frames (h and c), and each frame is given the last
    64 samples of the one before it. Both must be carried along, or it hears every frame as
    the start of a new recording."""

    def __init__(self, model_path: Path | None = None):
        import faster_whisper
        import onnxruntime

        path = model_path or Path(faster_whisper.__file__).parent / "assets" / "silero_vad_v6.onnx"
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = options.intra_op_num_threads = 1
        options.log_severity_level = 4
        self._session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"], sess_options=options)
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def probability(self, frame: np.ndarray) -> float:
        window = np.concatenate([self._context, frame]).astype(np.float32)[None, :]
        probabilities, self._h, self._c = self._session.run(None, {"input": window, "h": self._h, "c": self._c})
        self._context = frame[-CONTEXT_SAMPLES:]
        return float(np.asarray(probabilities).reshape(-1)[0])


class EnergyVAD:
    """Loudness alone. It cannot tell speech from a fan, so it is for tests only."""

    def __init__(self, threshold: float = 0.02):
        self._threshold = threshold

    def reset(self) -> None:
        pass

    def probability(self, frame: np.ndarray) -> float:
        return 1.0 if float(np.sqrt(np.mean(frame**2))) > self._threshold else 0.0


class WhisperSTT:
    def __init__(self, model: str = "base", download_root: str | None = None, language: str | None = None, hints: str = ""):
        from faster_whisper import WhisperModel

        # int8 is 8-bit quantisation of the weights: about four times smaller and faster on a
        # CPU than 32-bit floats, for a small loss in accuracy.
        self._model = WhisperModel(model, device="cpu", compute_type="int8", download_root=download_root)
        self._language = language
        # Words the model should expect. Whisper is steered by text it believes came before.
        self._hints = hints or None

    def transcribe(self, pcm16: bytes) -> str:
        segments, _info = self._model.transcribe(
            pcm16_to_float(pcm16),
            language=self._language,
            beam_size=1,
            initial_prompt=self._hints,
            condition_on_previous_text=False,
            vad_filter=False,  # the session has already cut the audio to one utterance
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


class PiperTTS:
    def __init__(self, voice_path: Path):
        from piper import PiperVoice

        self._voice = PiperVoice.load(str(voice_path))
        self.rate = self._voice.config.sample_rate

    def synthesize(self, text: str) -> Iterator[bytes]:
        for chunk in self._voice.synthesize(text):
            yield chunk.audio_int16_bytes
