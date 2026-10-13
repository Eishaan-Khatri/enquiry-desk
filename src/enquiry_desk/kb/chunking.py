"""Split a Markdown document into chunks that each stay inside one section."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

HEADING = re.compile(r"^(#{1,4})\s+(.*)$")
NAMESPACE = uuid.UUID("6f1c1b1e-0c5e-4a55-9d0b-2a4f1f1f8d10")


@dataclass(frozen=True)
class Chunk:
    id: str
    tenant: str
    doc: str
    section: str
    text: str

    @property
    def source(self) -> str:
        return f"{self.doc} > {self.section}" if self.section else self.doc

    @property
    def embed_text(self) -> str:
        # The section path goes in front, so a chunk that only says "Rs 4,000 per month"
        # still carries the words "Hostel fees" when it is embedded.
        return f"{self.source}\n{self.text}"


def chunk_markdown(tenant: str, doc: str, text: str, max_words: int = 160) -> list[Chunk]:
    """Cut at headings first, then pack whole paragraphs up to max_words.

    A paragraph, list or table is never split, so a fee table stays in one piece. The id is a
    hash of the tenant, document and position, so indexing the same file twice overwrites the
    same points and adds none.
    """
    chunks: list[Chunk] = []
    path: list[str] = []
    paragraphs: list[str] = []

    def flush() -> None:
        packed: list[str] = []
        words = 0
        for paragraph in paragraphs + [None]:
            size = len(paragraph.split()) if paragraph else 0
            if packed and (paragraph is None or words + size > max_words):
                number = len(chunks)
                chunks.append(
                    Chunk(
                        id=str(uuid.uuid5(NAMESPACE, f"{tenant}/{doc}/{number}")),
                        tenant=tenant,
                        doc=doc,
                        section=" > ".join(path),
                        text="\n\n".join(packed),
                    )
                )
                packed, words = [], 0
            if paragraph:
                packed.append(paragraph)
                words += size
        paragraphs.clear()

    for block in re.split(r"\n\s*\n", text.strip()):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        heading = HEADING.match(lines[0])
        if heading:
            flush()
            level = len(heading.group(1))
            path[:] = path[: level - 1] + [heading.group(2).strip()]
            block = "\n".join(lines[1:]).strip()
            if not block:
                continue
        paragraphs.append(block)
    flush()
    return chunks
