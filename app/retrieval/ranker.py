"""Rank chunks against the query with BM25 plus a couple of cheap priors.

No embeddings. On 3-5 freshly fetched pages a lexical score is competitive, and
skipping embeddings saves a model load, ~500MB of RAM and a second of latency on
every web question. The one thing that genuinely matters for this user is CJK
tokenisation: Cantonese has no spaces, so whitespace splitting would produce a
single useless token per sentence.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from app.retrieval.chunker import Chunk, Evidence

_LATIN_RE = re.compile(r"[a-z0-9][a-z0-9.+#_-]*")
_CJK_RE = re.compile(r"[㐀-鿿豈-﫿]")

_STOPWORDS = {
    "the", "a", "an", "of", "to", "is", "are", "was", "were", "be", "in", "on",
    "for", "and", "or", "it", "this", "that", "with", "as", "at", "by", "from",
    "what", "which", "who", "how", "do", "does", "did", "i", "you", "my",
}

# BM25 constants; the standard defaults work fine at this corpus size.
_K1 = 1.5
_B = 0.75


def tokenize(text: str) -> list[str]:
    """Latin words plus CJK character unigrams and bigrams."""
    lowered = text.lower()
    tokens = [t for t in _LATIN_RE.findall(lowered) if t not in _STOPWORDS]

    # Bigrams over runs of CJK characters: 最新版本 -> 最新, 新版, 版本 (+ unigrams).
    for run in re.findall(r"[㐀-鿿豈-﫿]+", lowered):
        tokens.extend(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def _has_cjk(text: str) -> bool:
    return _CJK_RE.search(text) is not None


def rank(
    query: str,
    chunks: list[Chunk],
    *,
    top_k: int,
    budget_chars: int,
) -> list[Evidence]:
    """Return the best chunks, best first, within the character budget."""
    if not chunks:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        query_tokens = list(query.lower())

    docs = [tokenize(f"{c.title} {c.text}") for c in chunks]
    lengths = [len(d) or 1 for d in docs]
    avg_len = sum(lengths) / len(lengths)
    n_docs = len(docs)

    doc_freq: Counter[str] = Counter()
    for doc in docs:
        doc_freq.update(set(doc))

    scored: list[tuple[float, Chunk]] = []
    for chunk, doc, length in zip(chunks, docs, lengths):
        counts = Counter(doc)
        score = 0.0
        for term in set(query_tokens):
            tf = counts.get(term, 0)
            if not tf:
                continue
            # BM25 IDF, floored so a term present in every document is worth ~0
            # rather than negative.
            idf = max(
                math.log((n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5) + 1.0),
                0.01,
            )
            score += idf * (tf * (_K1 + 1)) / (tf + _K1 * (1 - _B + _B * length / avg_len))

        # Priors: trust the search engine's ordering a little, and prefer the top
        # of a page, where the answer usually is.
        score *= 1.0 / (1.0 + 0.12 * chunk.doc_rank)
        score *= 1.0 / (1.0 + 0.06 * chunk.position)
        scored.append((score, chunk))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    evidence: list[Evidence] = []
    used = 0
    seen_domains: Counter[str] = Counter()
    for score, chunk in scored:
        if len(evidence) >= top_k or used >= budget_chars:
            break
        if score <= 0:
            continue
        # Keep at most 3 chunks from one domain so a single verbose page cannot
        # crowd out corroborating sources.
        if seen_domains[chunk.domain] >= 3:
            continue
        if used + len(chunk.text) > budget_chars and evidence:
            continue
        seen_domains[chunk.domain] += 1
        used += len(chunk.text)
        evidence.append(
            Evidence(
                text=chunk.text,
                title=chunk.title,
                url=chunk.url,
                domain=chunk.domain,
                published=chunk.published,
                score=round(score, 4),
            )
        )
    return evidence
