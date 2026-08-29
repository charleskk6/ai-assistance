"""Web Search RAG: query in, grounded answer plus sources out.

Deliberately query-scoped and stateless. Nothing is persisted, no vector store,
no background indexing - the laptop goes back to idle the moment the request
finishes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from app.config import Settings
from app.llm.base import LLMProvider
from app.rag.prompt import build_web_user_prompt, web_system_prompt
from app.retrieval.chunker import Chunk, Evidence, chunk_text
from app.retrieval.extractor import extract
from app.retrieval.fetcher import PageFetcher
from app.retrieval.ranker import rank
from app.search.base import SearchError, SearchProvider, SearchResult
from app.utils.logging import get_logger

log = get_logger("assistant.rag")


@dataclass
class WebAnswer:
    answer: str
    evidence: list[Evidence]
    stats: dict = field(default_factory=dict)

    @property
    def sources(self) -> list[Evidence]:
        """One entry per URL, in the order the evidence was ranked."""
        seen: set[str] = set()
        out: list[Evidence] = []
        for ev in self.evidence:
            if ev.url not in seen:
                seen.add(ev.url)
                out.append(ev)
        return out


class InsufficientEvidence(RuntimeError):
    """Raised when the web produced nothing worth reasoning over."""

    code = "web_search_failed"


class WebRagPipeline:
    def __init__(
        self,
        settings: Settings,
        search: SearchProvider,
        fetcher: PageFetcher,
        llm: LLMProvider,
    ) -> None:
        self.settings = settings
        self.search = search
        self.fetcher = fetcher
        self.llm = llm

    def _select(self, results: list[SearchResult]) -> list[SearchResult]:
        """Pick which results are worth spending a page fetch on."""
        chosen: list[SearchResult] = []
        per_domain: dict[str, int] = {}
        for result in results:
            if len(chosen) >= self.settings.fetch_max_pages:
                break
            # Two pages from one domain is plenty; we want corroboration.
            if per_domain.get(result.domain, 0) >= 2:
                continue
            per_domain[result.domain] = per_domain.get(result.domain, 0) + 1
            chosen.append(result)
        return chosen

    async def _gather_chunks(self, selected: list[SearchResult]) -> list[Chunk]:
        pages = await self.fetcher.fetch_many([r.url for r in selected])
        by_url = {r.url: (i, r) for i, r in enumerate(selected)}

        chunks: list[Chunk] = []
        for page in pages:
            doc_rank, result = by_url.get(page.url, (99, None))
            # Extraction is CPU-bound lxml work; keep the event loop responsive.
            extracted = await asyncio.to_thread(extract, page.html, page.url)
            if not extracted.text:
                log.debug("no extractable text at %s", page.url)
                continue
            chunks.extend(
                chunk_text(
                    extracted.text,
                    title=extracted.title or (result.title if result else page.url),
                    url=page.url,
                    domain=(result.domain if result else ""),
                    published=extracted.published or (result.published if result else None),
                    doc_rank=doc_rank,
                    size=self.settings.chunk_chars,
                    overlap=self.settings.chunk_overlap_chars,
                )
            )
        return chunks

    def _top_up_with_snippets(
        self, evidence: list[Evidence], results: list[SearchResult]
    ) -> list[Evidence]:
        """Add search snippets when the fetched pages yielded too little.

        Snippets are thin evidence, so they top the real page content up rather
        than replacing it - a fetch that half-worked is still better than a
        summary line. The grounded prompt makes the model admit when what it has
        is too weak to answer.
        """
        have = {e.url for e in evidence}
        room = self.settings.rag_top_chunks - len(evidence)
        extra = [
            Evidence(
                text=r.snippet,
                title=r.title,
                url=r.url,
                domain=r.domain,
                published=r.published,
                score=0.0,
            )
            for r in results
            if len(r.snippet) > 40 and r.url not in have
        ][: max(room, 1)]
        return evidence + extra

    async def run(self, query: str, source: str, max_tokens: int) -> WebAnswer:
        stats = {"results": 0, "fetched": 0, "chunks": 0, "used": 0, "fallback": False}

        try:
            results = await self.search.search(query, self.settings.search_results)
        except SearchError as exc:
            log.warning("search failed: %s", exc.message)
            raise InsufficientEvidence(exc.message) from exc

        stats["results"] = len(results)
        if not results:
            raise InsufficientEvidence("The search returned no results.")

        selected = self._select(results)
        chunks = await self._gather_chunks(selected)
        stats["fetched"] = len({c.url for c in chunks})
        stats["chunks"] = len(chunks)

        evidence = rank_chunks(query, chunks, self.settings)
        if sum(len(e.text) for e in evidence) < self.settings.rag_min_evidence_chars:
            evidence = self._top_up_with_snippets(evidence, results)
            stats["fallback"] = True

        if sum(len(e.text) for e in evidence) < 80:
            raise InsufficientEvidence("Could not extract usable content from the results.")

        stats["used"] = len(evidence)

        answer = await self.llm.complete(
            web_system_prompt(source),
            build_web_user_prompt(query, evidence),
            max_tokens=max_tokens,
            temperature=self.settings.llm_temperature,
        )
        return WebAnswer(answer=answer, evidence=evidence, stats=stats)


def rank_chunks(query: str, chunks: list[Chunk], settings: Settings) -> list[Evidence]:
    return rank(
        query,
        chunks,
        top_k=settings.rag_top_chunks,
        budget_chars=settings.rag_context_budget_chars,
    )
