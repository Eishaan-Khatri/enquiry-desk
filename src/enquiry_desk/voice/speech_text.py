"""Text on its way to being spoken: cut into sentences, and rewritten so a voice can say it.

A reply written for a screen cannot be read out as it is. "[K2]" is noise, "1,35,000" must
become "one lakh thirty five thousand", and a phone number is read digit by digit.
"""

from __future__ import annotations

import re

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

# A full stop ends a sentence only when a space or the end follows it, so "1.5" and "Rs.500"
# are not cut. The abbreviations are ones a reply about an institute actually uses.
ABBREVIATIONS = ("rs.", "mr.", "mrs.", "ms.", "dr.", "no.", "e.g.", "i.e.", "a.m.", "p.m.")


def below_hundred(n: int) -> str:
    if n < 20:
        return ONES[n]
    return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])


def number_to_words(n: int) -> str:
    """A whole number in the Indian system: thousand, lakh, crore.

    135000 is "one lakh thirty five thousand", which is what a caller in India expects to
    hear. "One hundred thirty five thousand" is correct English and sounds foreign.
    """
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 100:
        return below_hundred(n)
    parts = []
    for size, name in ((10_000_000, "crore"), (100_000, "lakh"), (1_000, "thousand"), (100, "hundred")):
        count, n = divmod(n, size)
        if count:
            parts.append(number_to_words(count) + " " + name)
    if n:
        parts.append(("and " if parts and len(parts) == 1 and parts[0].endswith("hundred") else "") + below_hundred(n))
    return " ".join(parts)


def speakable(text: str) -> str:
    """Rewrite a reply so that a text-to-speech voice reads it the way a person would."""
    text = re.sub(r"\s*\[K\d+\]", "", text)  # references are for the screen
    text = re.sub(r"[*_`#|]+", " ", text)  # Markdown marks
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)  # list bullets

    def phone(match: re.Match) -> str:
        digits = re.sub(r"\D", "", match.group())[-10:]
        return " ".join(ONES[int(d)] for d in digits[:5]) + ", " + " ".join(ONES[int(d)] for d in digits[5:])

    text = re.sub(r"(?<![\d,.])(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)", phone, text)

    def clock(match: re.Match) -> str:
        hour, minute, half = int(match.group(1)), int(match.group(2)), match.group(3) or ""
        spoken_minute = "" if minute == 0 else (" oh " + ONES[minute] if minute < 10 else " " + below_hundred(minute))
        return f"{number_to_words(hour)}{spoken_minute} {half.replace('.', '').lower()}".strip()

    text = re.sub(r"\b(\d{1,2}):(\d{2})\s*(a\.m\.|p\.m\.|am\b|pm\b)?", clock, text, flags=re.IGNORECASE)
    text = re.sub(r"(?:Rs\.?|INR|₹)\s*(\d[\d,]*)", r"\1 rupees", text)
    text = re.sub(r"(\d[\d,]*(?:\.\d+)?)\s*%", r"\1 percent", text)

    def figure(match: re.Match) -> str:
        whole, _, fraction = match.group().replace(",", "").partition(".")
        words = number_to_words(int(whole))
        return words + (" point " + " ".join(ONES[int(d)] for d in fraction) if fraction else "")

    text = re.sub(r"\d[\d,]*(?:\.\d+)?", figure, text)
    return re.sub(r"\s+", " ", text).strip()


class SentenceSplitter:
    """Collects text as it streams in and hands back each sentence as soon as it is complete.

    This is what lets the desk start speaking before the model has finished writing: the first
    sentence goes to the voice while the second is still being generated.
    """

    def __init__(self):
        self._pending = ""

    def feed(self, text: str) -> list[str]:
        self._pending += text
        sentences = []
        while True:
            cut = self._find_end()
            if cut is None:
                return sentences
            sentence, self._pending = self._pending[:cut].strip(), self._pending[cut:]
            if sentence:
                sentences.append(sentence)

    def flush(self) -> list[str]:
        rest, self._pending = self._pending.strip(), ""
        return [rest] if rest else []

    def _find_end(self) -> int | None:
        for match in re.finditer(r"[.!?।]+(?=\s)|\n+", self._pending):
            before = self._pending[: match.end()].lower()
            if match.group().startswith(".") and before.endswith(ABBREVIATIONS):
                continue
            return match.end()
        return None
