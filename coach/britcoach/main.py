"""FastAPI application: REST API, the voice WebSocket, and the built React UI."""

from __future__ import annotations

import asyncio
import json
import logging
import platform
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .cloud.router import CloudRouter
from .config import Settings, get_settings, load_catalog
from .conversation.modes import DIFFICULTIES, MODES, SCENARIOS
from .llm.factory import mlx_supported
from .pipeline.engines import Engines
from .pipeline.metrics import memory_snapshot, summarise
from .pipeline.session import Session
from .storage.db import Store

log = logging.getLogger("britcoach")


def create_app(settings: Settings | None = None, engines: Engines | None = None, warmup: bool = True) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    state: dict = {"sessions": {}, "metrics": []}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        eng = engines or Engines(settings)
        await eng.start(warmup=warmup)
        app.state.engines = eng
        app.state.store = Store(settings.path(settings.db_path)) if settings.save_history else None
        log.info("Engines: %s", json.dumps(eng.status))
        yield
        for sess in list(state["sessions"].values()):
            await sess.close()
        eng.shutdown()

    app = FastAPI(title="British Conversation Coach", version=__version__, lifespan=lifespan)

    # ------------------------------------------------------------------ REST
    @app.get("/api/health")
    async def health():
        eng: Engines = app.state.engines
        ok, why = eng.ready()
        return {"status": "ok" if ok else "degraded", "problems": why, "engines": eng.status, "version": __version__}

    @app.get("/api/options")
    async def options():
        eng: Engines = app.state.engines
        cat = load_catalog(str(settings.path(settings.models_toml)))
        profiles = {k: {kk: vv for kk, vv in v.items() if not kk.endswith("alternatives")} for k, v in cat["llm"]["profiles"].items()}
        voices = eng.tts.voices() if eng.tts else []
        return {
            "modes": [{"id": m.id, "title": m.title, "description": m.description, "sensitivity": m.default_sensitivity} for m in MODES.values()],
            "scenarios": [{"id": s.id, "title": s.title, "modes": list(s.modes)} for s in SCENARIOS.values()],
            "difficulties": list(DIFFICULTIES),
            "llm_profiles": profiles,
            "llm_backends": ["mlx", "ollama", "openai_compat", "mock"],
            "mlx_supported": mlx_supported(),
            "current_llm": eng.llm.info() if eng.llm else None,
            "voices": voices,
            "default_voice": eng.default_voice(),
            "tts_backend": eng.tts.name if eng.tts else None,
            "settings": settings.public_dict(),
            "platform": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version()},
        }

    class SelectLLM(BaseModel):
        backend: str
        profile: str | None = None
        model: str | None = None

    @app.post("/api/llm/select")
    async def select_llm(req: SelectLLM):
        eng: Engines = app.state.engines
        try:
            return await eng.switch_llm(req.backend, req.model, req.profile)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"{type(e).__name__}: {e}") from e

    @app.get("/api/metrics")
    async def metrics():
        rows = [r for s in state["sessions"].values() for r in s.metrics_log] + state["metrics"]
        return {"summary": summarise(rows), "turns": rows[-50:], "memory": memory_snapshot()}

    @app.get("/api/sessions")
    async def sessions():
        store: Store | None = app.state.store
        return await store.list_sessions() if store else []

    @app.get("/api/sessions/{sid}")
    async def session_detail(sid: str):
        store: Store | None = app.state.store
        data = await store.get_session(sid) if store else None
        if not data:
            raise HTTPException(404, "No such session")
        return data

    @app.get("/api/progress")
    async def progress():
        store: Store | None = app.state.store
        return await store.progress() if store else {"sessions": [], "top_categories": []}

    @app.get("/api/cloud/calls")
    async def cloud_calls():
        store: Store | None = app.state.store
        return await store.cloud_calls() if store else []

    # ------------------------------------------------------------------ WebSocket
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        eng: Engines = app.state.engines
        store: Store | None = app.state.store
        send_lock = asyncio.Lock()

        async def send_json(msg: dict) -> None:
            async with send_lock:
                await ws.send_text(json.dumps(msg))

        async def send_bytes(data: bytes) -> None:
            async with send_lock:
                await ws.send_bytes(data)

        session_ref: dict = {}

        async def audit(info: dict) -> None:
            if store and "s" in session_ref:
                await store.log_cloud_call(session_ref["s"].id, info)

        cloud = CloudRouter(settings, on_event=send_json, audit=audit)
        sess = Session(eng, settings, send_json, send_bytes, store=store, cloud=cloud)
        session_ref["s"] = sess
        state["sessions"][sess.id] = sess
        await send_json({"type": "hello", "session_id": sess.id, "config": sess.public_config(), "engines": eng.status})
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    await sess.handle_audio(msg["bytes"])
                elif msg.get("text") is not None:
                    try:
                        data = json.loads(msg["text"])
                    except json.JSONDecodeError:
                        continue
                    await sess.handle_json(data)
        except WebSocketDisconnect:
            pass
        finally:
            state["metrics"].extend(sess.metrics_log)
            state["metrics"] = state["metrics"][-500:]
            state["sessions"].pop(sess.id, None)
            await sess.close()

    # ------------------------------------------------------------------ frontend
    dist = settings.path(settings.frontend_dist)
    if (dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str):
            f = dist / path
            if path and f.is_file() and dist in f.resolve().parents:
                return FileResponse(f)
            return FileResponse(dist / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        async def no_ui():
            return HTMLResponse("<h1>British Conversation Coach</h1><p>The UI isn't built yet. Run "
                                "<code>./scripts/install.sh</code> (or <code>cd frontend && npm install && npm run build</code>).</p>"
                                "<p>API: <a href='/api/health'>/api/health</a></p>")

    @app.exception_handler(ValueError)
    async def value_error(_, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    return app


app = create_app()
