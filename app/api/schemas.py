from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Mode = Literal["auto", "local", "web"]
Source = Literal["siri", "api", "cli"]
Route = Literal["local", "web"]


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    mode: Mode = "auto"
    source: Source = "api"

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be empty")
        return v


class SourceRef(BaseModel):
    title: str
    url: str
    domain: str


class AskResponse(BaseModel):
    answer: str
    route: Route
    sources: list[SourceRef] = []
    latency_ms: int
    # Per-stage breakdown, so latency can be tuned without reading server logs.
    # The Shortcut only ever reads "answer" and ignores this.
    timings: dict[str, int] = {}


class ErrorResponse(BaseModel):
    error: str
    message: str


class LLMHealthInfo(BaseModel):
    provider: str
    model: str
    runtime_ok: bool
    model_ok: bool
    detail: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    backend: bool
    llm: LLMHealthInfo
    search_provider: str
