"""Deciding when the caller has started and stopped speaking.

A voice-activity model looks at 32 ms of audio and returns the probability that it holds
speech. This turns that stream of probabilities into two events:

  speech_start  the caller began to speak. If the desk is talking, this is an interruption.
  speech_end    the caller has been silent long enough that their turn is taken to be over.

The silence length is the main trade-off in a voice agent. Shorter, and the desk answers
sooner but cuts in when the caller pauses to think. Longer, and it never interrupts but
always feels slow. It adds directly to every response time, so it is a setting to measure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EndpointConfig:
    frame_ms: int = 32
    start_threshold: float = 0.5
    # Lower than the start threshold. Once someone is speaking, the trailing quiet part of a
    # word should not count as silence. Two thresholds stop the state flickering.
    end_threshold: float = 0.35
    min_speech_ms: int = 160  # shorter bursts are a cough or a door, not a turn
    silence_ms: int = 700
    pre_roll_ms: int = 320  # audio kept from before speech was confirmed, so the first word is whole


class Endpointer:
    def __init__(self, config: EndpointConfig | None = None):
        self.config = config or EndpointConfig()
        self.speaking = False
        self._loud_ms = 0
        self._quiet_ms = 0

    def feed(self, probability: float) -> str | None:
        """Take the probability for the next frame. Returns an event name, or None."""
        c = self.config
        if not self.speaking:
            self._loud_ms = self._loud_ms + c.frame_ms if probability >= c.start_threshold else 0
            if self._loud_ms >= c.min_speech_ms:
                self.speaking, self._quiet_ms = True, 0
                return "speech_start"
            return None
        self._quiet_ms = self._quiet_ms + c.frame_ms if probability < c.end_threshold else 0
        if self._quiet_ms >= c.silence_ms:
            self.speaking, self._loud_ms = False, 0
            return "speech_end"
        return None

    def reset(self) -> None:
        self.speaking, self._loud_ms, self._quiet_ms = False, 0, 0
