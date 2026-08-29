"""Split extracted page text into ranked-retrieval units.

Chunking is paragraph-aware: we never cut mid-sentence if we can avoid it, and
CJK text has no spaces so the splitter works on characters, not words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PARA_RE = re.compile(r"\n\s*\n")
_SENT_END_RE = re.compile(r"(?<=[.!?。！？；;])\s*")


@dataclass
class Chunk:
    text: str
    title: str
    url: str
    domain: str
    published: str | None = None
    position: int = 0          # index within its source document
    doc_rank: int = 0          # rank of the source document in search results


@dataclass
class Evidence:
    """A chunk selected for the prompt, with its relevance score."""

    text: str
    title: str
    url: str
    domain: str
    published: str | None
    score: float


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    """Break an oversized block on sentence ends, with a little overlap."""
    pieces: list[str] = []
    sentences = [s for s in _SENT_END_RE.split(text) if s.strip()]
    current = ""
    for sentence in sentences:
        # A single sentence longer than the budget gets hard-cut.
        while len(sentence) > size:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:size])
            sentence = sentence[size - overlap :]
        if len(current) + len(sentence) <= size:
            current += sentence
        else:
            pieces.append(current)
            current = (current[-overlap:] if overlap else "") + sentence
    if current.strip():
        pieces.append(current)
    return [p.strip() for p in pieces if p.strip()]


def chunk_text(
    text: str,
    *,
    title: str,
    url: str,
    domain: str,
    published: str | None = None,
    doc_rank: int = 0,
    size: int = 1200,
    overlap: int = 150,
) -> list[Chunk]:
    if not text or not text.strip():
        return []

    blocks: list[str] = []
    buffer = ""
    for para in _PARA_RE.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) > size:
            if buffer:
                blocks.append(buffer)
                buffer = ""
            blocks.extend(_split_long(para, size, overlap))
        elif len(buffer) + len(para) + 1 <= size:
            buffer = f"{buffer}\n{para}" if buffer else para
        else:
            blocks.append(buffer)
            buffer = para
    if buffer.strip():
        blocks.append(buffer)

    return [
        Chunk(
            text=block,
            title=title,
            url=url,
            domain=domain,
            published=published,
            position=i,
            doc_rank=doc_rank,
        )
        for i, block in enumerate(blocks)
    ]
