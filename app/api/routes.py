"""HTTP surface: POST /ask and GET /health."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request

from app.api.deps import require_token
from app.api.schemas import AskRequest, AskResponse, HealthResponse, LLMHealthInfo
from app.llm.base import LLMProvider
from app.rag.prompt import system_prompt
from app.utils.logging import get_logger
from app.utils.text import to_speech_text

router = APIRouter()
log = get_logger("assistant.api")


@router.post("/ask", response_model=AskResponse, dependencies=[Depends(require_token)])
async def ask(req: AskRequest, request: Request) -> AskResponse:
    started = time.perf_counter()
    settings = request.app.state.settings
    llm: LLMProvider = request.app.state.llm

    max_tokens = (
        settings.llm_max_tokens_siri
        if req.source == "siri"
        else settings.llm_max_tokens_default
    )
    answer = await llm.complete(
        system_prompt(req.source),
        req.query,
        max_tokens=max_tokens,
        temperature=settings.llm_temperature,
    )
    if req.source == "siri":
        answer = to_speech_text(answer)

    latency_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "ask route=local source=%s model=%s latency_ms=%d query=%r",
        req.source,
        llm.model,
        latency_ms,
        req.query[:120],
    )
    return AskResponse(answer=answer, route="local", sources=[], latency_ms=latency_ms)


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
