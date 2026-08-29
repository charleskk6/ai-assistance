from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import Settings, get_settings
from app.llm.base import LLMError
from app.llm.factory import build_llm
from app.utils.logging import get_logger, setup_logging

log = get_logger("assistant")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)

    if not settings.local_assistant_token:
        raise RuntimeError(
            "LOCAL_ASSISTANT_TOKEN is not set. Copy .env.example to .env and set a "
            "token; the assistant refuses to run unauthenticated."
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.llm = build_llm(settings)
        log.info(
            "assistant ready provider=%s model=%s search=%s listen=%s:%d",
            settings.llm_provider,
            settings.llm_model,
            settings.search_provider,
            settings.host,
            settings.port,
        )
        try:
            yield
        finally:
            await app.state.llm.aclose()

    app = FastAPI(title="Local Assistant", version="0.1.0", lifespan=lifespan)
    app.include_router(router)

    @app.exception_handler(LLMError)
    async def _llm_error(request: Request, exc: LLMError) -> JSONResponse:
        log.warning("llm error code=%s: %s", exc.code, exc.message)
        return JSONResponse(status_code=503, content={"error": exc.code, "message": exc.message})

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException) -> JSONResponse:
        codes = {401: "unauthorized", 404: "not_found", 405: "method_not_allowed"}
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": codes.get(exc.status_code, "error"), "message": str(exc.detail)},
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "invalid_request", "message": "The request body was not valid."},
        )

    return app


def __getattr__(name: str):
    """Lets `uvicorn app.main:app` work without building the app at import time
    (which would require a token to be configured just to run the tests)."""
    if name == "app":
        return create_app()
    raise AttributeError(name)
