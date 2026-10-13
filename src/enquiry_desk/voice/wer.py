"""Word error rate: the standard measure of how wrong a transcript is.

    WER = (substitutions + deletions + insertions) / words in the reference

It is the edit distance between the two word lists, divided by the reference length. It can
exceed 1 when the transcript has many extra words.

Three traps:
  1. Case and punctuation are not errors. Both texts are normalised first.
  2. "11" against "eleven" counts as an error although the recogniser heard correctly.
     A fair comparison writes numbers one way on both sides before scoring.
  3. Hindi can come back in Devanagari or in Roman letters. Compared directly, a perfect
     transcript in the other script scores as all wrong. Transliterate to one script first.
"""

from __future__ import annotations

import re


def normalise(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # punctuation out. \w keeps letters of every script
    return text.split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = normalise(reference), normalise(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    # The classic dynamic programme. previous[j] is the fewest edits that turn the first i-1
    # reference words into the first j hypothesis words.
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, 1):
        current = [i]
        for j, hyp_word in enumerate(hyp, 1):
            current.append(min(
                previous[j] + 1,  # the reference word is missing from the transcript (deletion)
                current[j - 1] + 1,  # the transcript has an extra word (insertion)
                previous[j - 1] + (ref_word != hyp_word),  # same word, or one swapped for another
            ))
        previous = current
    return previous[-1] / len(ref)
