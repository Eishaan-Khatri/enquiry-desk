"""Audio as numbers: sample formats, resampling, and the telephone's mu-law encoding.

Sound is a list of samples, each the air pressure at one instant. Three facts fix a stream:
  sample rate   how many samples per second. The desk works at 16,000. A phone line carries 8,000.
  sample width  here 16 bits per sample ("PCM16"), a whole number from -32768 to 32767.
  channels      one. Speech does not need stereo.
"""

from __future__ import annotations

import numpy as np

DESK_RATE = 16_000  # what the speech-to-text and voice-activity models expect
PHONE_RATE = 8_000


def pcm16_to_float(data: bytes) -> np.ndarray:
    """Bytes of 16-bit samples to floats between -1 and 1, which is what the models take."""
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Change the sample rate by linear interpolation.

    Going down (16,000 to 8,000) first averages neighbouring samples. Without that, any sound
    above half the new rate folds back as a false lower tone ("aliasing"). This is a simple
    filter, good enough for speech on a phone line and not for music.
    """
    if from_rate == to_rate or len(samples) == 0:
        return samples.astype(np.float32)
    if to_rate < from_rate:
        width = int(round(from_rate / to_rate))
        if width > 1:
            samples = np.convolve(samples, np.ones(width, dtype=np.float32) / width, mode="same")
    length = int(round(len(samples) * to_rate / from_rate))
    positions = np.linspace(0, len(samples) - 1, num=length)
    return np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)


# ---- mu-law (G.711), the telephone network's 8-bit encoding --------------------------------
#
# A phone line sends one byte per sample. Eight bits spread evenly over the loudness range
# would leave quiet speech with too few steps. Mu-law spends its steps on a logarithmic scale:
# fine steps for quiet sounds and coarse ones for loud sounds, as the ear does.

_BIAS = 0x84
_BIAS_14 = 0x21
_CLIP_14 = 8159
_SEGMENT_END = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF])


def mulaw_encode(pcm16: bytes) -> bytes:
    # G.711 is defined on 14-bit samples, so the two lowest bits are dropped first. The shift
    # rounds a negative sample away from zero, which is why this is done before taking the
    # magnitude: doing it after gives a byte that is off by one for some negative samples.
    samples = np.frombuffer(pcm16, dtype="<i2").astype(np.int32) >> 2
    sign = np.where(samples < 0, 0x80, 0)
    magnitude = np.minimum(np.abs(samples), _CLIP_14) + _BIAS_14
    segment = np.searchsorted(_SEGMENT_END, magnitude, side="left")  # which of the 8 loudness bands
    mantissa = (magnitude >> (segment + 1)) & 0x0F  # 4 bits of position inside the band
    # The very loudest samples land past the last band once the bias is added. They take the
    # top code of the top band. Without this the band number would spill into the sign bit.
    loudest = segment > 7
    segment, mantissa = np.where(loudest, 7, segment), np.where(loudest, 0x0F, mantissa)
    return (~(sign | (segment << 4) | mantissa) & 0xFF).astype(np.uint8).tobytes()


def mulaw_decode(data: bytes) -> bytes:
    codes = ~np.frombuffer(data, dtype=np.uint8).astype(np.int32) & 0xFF
    segment = (codes >> 4) & 0x07
    mantissa = codes & 0x0F
    magnitude = (((mantissa << 3) + _BIAS) << segment) - _BIAS
    return np.where(codes & 0x80, -magnitude, magnitude).astype("<i2").tobytes()


def phone_to_desk(mulaw_8k: bytes) -> bytes:
    """What arrives from a phone call, as the PCM16 at 16,000 Hz that the desk works with."""
    return float_to_pcm16(resample(pcm16_to_float(mulaw_decode(mulaw_8k)), PHONE_RATE, DESK_RATE))


def desk_to_phone(pcm16: bytes, rate: int) -> bytes:
    """Speech from the desk at any rate, as mu-law at 8,000 Hz for a phone call."""
    return mulaw_encode(float_to_pcm16(resample(pcm16_to_float(pcm16), rate, PHONE_RATE)))


def seconds(pcm16: bytes, rate: int) -> float:
    return len(pcm16) / 2 / rate
