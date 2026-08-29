"""HTTP surface: POST /ask and GET /health."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends, Request

from app.api.deps import require_token
from app.api.errors import error_response
from app.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    LLMHealthInfo,
    SourceRef,
)
from app.llm.base import LLMError
from app.rag.pipeline import InsufficientEvidence
from app.rag.prompt import system_prompt
from app.router.classifier import classify
from app.utils.logging import get_logger
from app.utils.text import to_speech_text

router = APIRouter()
log = get_logger("assistant.api")


@router.post("/ask", response_model=AskResponse, dependencies=[Depends(require_token)])
async def ask(req: AskRequest, request: Request):
    started = time.perf_counter()
    settings = request.app.state.settings
    llm = request.app.state.llm

    decision = classify(req.query, req.mode)
    max_tokens = (
        settings.llm_max_tokens_siri
        if req.source == "siri"
        else settings.llm_max_tokens_default
    )

    stats: dict = {}
    sources: list[SourceRef] = []
    try:
        # Bounded concurrency and one overall deadline: a request either answers
        # or fails, it never hangs the Shortcut indefinitely. The deadline wraps
        # the semaphore, so time spent queueing behind another inference counts
        # against it too.
        async with asyncio.timeout(settings.request_timeout_s):
            async with request.app.state.limiter:
                answer, stats, sources, truncated = await _answer(
                    req, decision, request, max_tokens
                )
    except TimeoutError:
        _log_failure(req, decision, "request_timeout", started)
        return error_response(
            "request_timeout", "The request took too long and was cancelled.", req.source, 504
        )
    except InsufficientEvidence as exc:
        _log_failure(req, decision, "web_search_failed", started)
        return error_response(exc.code, str(exc), req.source)
    except LLMError as exc:
        _log_failure(req, decision, exc.code, started)
        return error_response(exc.code, exc.message, req.source)

    if req.source == "siri":
        answer = to_speech_text(answer, drop_trailing_fragment=truncated)
        # An answer that was entirely code or markup shapes down to nothing, and
        # a silent Shortcut looks like a crash. Say something instead.
        if not answer:
            _log_failure(req, decision, "llm_empty", started)
            return error_response("llm_empty", "The answer was empty.", req.source)

    latency_ms = int((time.perf_counter() - started) * 1000)
    # Note the query is truncated and page bodies are never logged.
    log.info(
        "ask route=%s source=%s model=%s reason=%r results=%d fetched=%d "
        "chunks=%d used=%d fallback=%s context_chars=%d "
        "search_ms=%d fetch_ms=%d rank_ms=%d llm_ms=%d latency_ms=%d query=%r",
        decision.route,
        req.source,
        llm.model,
        decision.reason,
        stats.get("results", 0),
        stats.get("fetched", 0),
        stats.get("chunks", 0),
        stats.get("used", 0),
        stats.get("fallback", False),
        stats.get("context_chars", 0),
        stats.get("search_ms", 0),
        stats.get("fetch_ms", 0),
        stats.get("rank_ms", 0),
        stats.get("llm_ms", 0),
        latency_ms,
        req.query[:120],
    )
    return AskResponse(
        answer=answer,
        route=decision.route,
        sources=sources,
        latency_ms=latency_ms,
        timings={
            k: stats[k]
            for k in ("search_ms", "fetch_ms", "rank_ms", "llm_ms", "context_chars")
            if k in stats
        },
    )


async def _answer(
    req: AskRequest, decision, request: Request, max_tokens: int
) -> tuple[str, dict, list[SourceRef], bool]:
    """Run the chosen route. Raises; the caller turns errors into responses."""
    settings = request.app.state.settings
    llm = request.app.state.llm
    stats: dict = {}
    sources: list[SourceRef] = []

    if decision.route == "web":
        result = await request.app.state.rag.run(decision.query, req.source, max_tokens)
        answer, stats, truncated = result.answer, result.stats, result.truncated
        sources = [
            SourceRef(title=s.title or s.domain, url=s.url, domain=s.domain)
            for s in result.sources
        ]
    else:
        completion = await llm.complete(
            system_prompt(req.source),
            decision.query,
            max_tokens=max_tokens,
            temperature=settings.llm_temperature,
        )
        answer, truncated = completion.text, completion.truncated
    return answer, stats, sources, truncated


def _log_failure(req: AskRequest, decision, code: str, started: float) -> None:
    log.warning(
        "ask failed route=%s source=%s error=%s latency_ms=%d query=%r",
        decision.route,
        req.source,
        code,
        int((time.perf_counter() - started) * 1000),
        req.query[:120],
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Checks the backend, the LLM runtime and the model. Never hits the web."""
    settings = request.app.state.settings
    llm_health = await request.app.state.llm.health()
    return HealthResponse(
        status="ok" if (llm_health.runtime_ok and llm_health.model_ok) else "degraded",
        backend=True,
        llm=LLMHealthInfo(
            provider=llm_health.provider,
            model=llm_health.model,
            runtime_ok=llm_health.runtime_ok,
            model_ok=llm_health.model_ok,
            detail=llm_health.detail,
        ),
        search_provider=settings.search_provider,
    )
