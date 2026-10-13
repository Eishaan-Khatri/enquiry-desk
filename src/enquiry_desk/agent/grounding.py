"""A check on every reply before the caller sees it. No model is involved.

Rule 1: a reference such as [K2] must be one the search tool actually returned.
Rule 2: every figure in the reply must appear in a retrieved document, a tool result, or the
        caller's own words. A fee, a date or a batch size the model made up fails here.

Rule 3: the reply may say a call is booked only if book_slot has succeeded.

What it does not catch, both seen in real runs on 13 Oct 2026:
  - a wrong claim with no number in it ("the academy has no swimming pool")
  - a real figure attached to the wrong thing. The two-year course's online fee is in the
    documents, and a model quoted it as the dropper course's fee. The figure exists in a
    source, so rule 2 passes it.
Those need the offline evaluation, where a second model reads each reply against the sources.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from enquiry_desk.voice.speech_text import SentenceSplitter

REF = re.compile(r"\[(K\d+)\]")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
# "I have booked", "is booked", "booking is confirmed", and the Hinglish "book kar diya",
# "book ho gaya". Not "shall I book" or "to book", which are offers.
SAYS_BOOKED = re.compile(
    r"\b(have|has|is|are|been|got)\s+(now\s+|successfully\s+|already\s+)?(booked|scheduled|confirmed)\b"
    r"|\bbook(ing)?\s+(is\s+)?(confirmed|done|kar\s+di|ho\s+ga)",
    re.IGNORECASE,
)
PHONE = re.compile(r"(?<![\d,.])(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")


def numbers(text: str) -> set[str]:
    """The figures in a text, in one form: '45,000' and '45000.0' both become '45000'."""
    found = set()
    for token in NUMBER.findall(text):
        token = token.replace(",", "")
        if "." in token:
            token = token.rstrip("0").rstrip(".")
        found.add(token.lstrip("0") or "0")
    return found


def strip_refs(text: str) -> str:
    """The reply without its [K1] marks, for text that will be spoken."""
    return re.sub(r"\s*\[K\d+\]", "", text)


def check(reply: str, evidence: list[str], known_refs: set[str], booked: bool = True) -> list[str]:
    """Return what is wrong with the reply. An empty list means it may be sent.

    `booked` says whether book_slot has succeeded in this conversation. A reply that tells the
    caller a call is booked when it has not is the worst thing the desk can say, because the
    caller then waits for a call that nobody will make.
    """
    problems = [f"the reference [{ref}] is not one of the search results" for ref in sorted(set(REF.findall(reply)) - known_refs)]
    if not booked and SAYS_BOOKED.search(reply):
        problems.append("you told the caller a call is booked, but book_slot has not succeeded. Call book_slot first")
    body = REF.sub(" ", reply)

    # A phone number may be written with spaces or a country code, so compare digits only.
    digit_runs = [re.sub(r"\D", "", item) for item in evidence]
    for match in PHONE.finditer(body):
        last_ten = re.sub(r"\D", "", match.group())[-10:]
        if not any(last_ten in run for run in digit_runs):
            problems.append(f"the phone number {match.group().strip()} was not given by the caller")
    body = PHONE.sub(" ", body)

    allowed: set[str] = set()
    for item in evidence:
        allowed |= numbers(item)
    for figure in sorted(numbers(body) - allowed, key=lambda f: (len(f), f)):
        # Single digits are skipped: "3 options" or "step 2" are not claims about the business.
        if len(figure) > 1:
            problems.append(f"the figure {figure} is not in any source")
    return problems


class SentenceGate:
    """For voice. Checks each sentence as the model writes it, and lets it be spoken only if
    it passes.

    In chat the whole reply is checked before the caller sees any of it. A voice cannot wait
    that long: the first sentence has to be spoken while the model is still writing the rest.
    So the check moves from the reply to the sentence. The first sentence that fails stops the
    stream, and nothing after it is spoken.
    """

    def __init__(self, check_sentence: Callable[[str], list[str]], emit: Callable[[str], None]):
        self._check = check_sentence
        self._emit = emit
        self._splitter = SentenceSplitter()
        self.passed: list[str] = []
        self.problems: list[str] = []

    def feed(self, text: str) -> None:
        for sentence in self._splitter.feed(text):
            self._take(sentence)

    def finish(self) -> None:
        for sentence in self._splitter.flush():
            self._take(sentence)

    def discard(self) -> None:
        """Drop text that turned out not to be speech, such as a tool call written as text."""
        self._splitter = SentenceSplitter()

    def _take(self, sentence: str) -> None:
        if self.problems:
            return
        self.problems = self._check(sentence)
        if not self.problems:
            self.passed.append(sentence)
            self._emit(sentence)


def feedback(problems: list[str]) -> str:
    return (
        "CHECKER NOTE (this is not from the caller). Your last reply was not sent because "
        + "; ".join(problems)
        + ". Write the reply again. Use only what the search results, the tool results and the "
        "caller have said. If the information is not there, say you do not have it and offer a "
        "call with a counsellor."
    )
