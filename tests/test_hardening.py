"""Phase 1G: nothing hangs, nothing leaks, failures speak Cantonese."""

from __future__ import annotations

import asyncio
import logging

import httpx
import respx
from fastapi.testclient import TestClient

from app.llm.base import LLMError, LLMHealth
from app.main import create_app
from tests.conftest import AUTH, TOKEN, make_settings


class SlowLLM:
    name, model = "slow", "slow"

    async def complete(self, *a, **kw):
        await asyncio.sleep(10)
        return "too late"

    async def health(self):
        return LLMHealth(True, True, self.name, self.model)

    async def aclose(self):
        return None


class BrokenLLM:
    name, model = "broken", "broken"

    def __init__(self, code: str, message: str = "boom") -> None:
        self.code, self.message = code, message

    async def complete(self, *a, **kw):
        raise LLMError(self.code, self.message)

    async def health(self):
        return LLMHealth(False, False, self.name, self.model, "Ollama unreachable")

    async def aclose(self):
        return None


def app_with(llm, **overrides):
    app = create_app(make_settings(**overrides))
    client = TestClient(app)
    client.__enter__()
    app.state.llm = llm
    app.state.rag.llm = llm
    return client


def test_request_timeout_does_not_hang_the_shortcut():
    client = app_with(SlowLLM(), request_timeout_s=0.2)
    r = client.post("/ask", json={"query": "hi", "source": "siri"}, headers=AUTH)
    assert r.status_code == 504
    assert r.json()["error"] == "request_timeout"
    assert r.json()["message"] == "今次搞得太耐，唔該再問一次。"


def test_llm_down_gives_siri_a_spoken_hint():
    client = app_with(BrokenLLM("llm_unavailable", "connection refused"))
    body = client.post("/ask", json={"query": "hi", "source": "siri"}, headers=AUTH).json()
    assert body["error"] == "llm_unavailable"
    assert "Ollama" in body["message"] and "connection refused" not in body["message"]


def test_missing_model_message_is_actionable_for_api_callers():
    client = app_with(BrokenLLM("model_not_found", "Run: ollama pull qwen3:8b"))
    body = client.post("/ask", json={"query": "hi", "source": "api"}, headers=AUTH).json()
    assert body["error"] == "model_not_found"
    assert "ollama pull qwen3:8b" in body["message"]


def test_health_is_degraded_when_the_runtime_is_down():
    client = app_with(BrokenLLM("llm_unavailable"))
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["backend"] is True          # the backend itself is fine
    assert body["llm"]["runtime_ok"] is False
    assert body["llm"]["detail"]


@respx.mock
def test_health_never_performs_a_web_search():
    route = respx.post("https://html.duckduckgo.com/html/").mock(
        return_value=httpx.Response(200, text="")
    )
    client = app_with(BrokenLLM("x"))
    client.get("/health")
    assert not route.called


def test_concurrency_is_bounded():
    """Two slots configured; a third caller must wait rather than pile on."""
    settings = make_settings(max_concurrent_requests=2)
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.app.state.limiter._value == 2


def test_token_is_never_logged(caplog):
    client = app_with(BrokenLLM("llm_error"))
    with caplog.at_level(logging.DEBUG):
        client.post("/ask", json={"query": "hi"}, headers=AUTH)
        client.post("/ask", json={"query": "hi"}, headers={"Authorization": f"Bearer {TOKEN}"})
    assert TOKEN not in caplog.text


def test_long_query_is_truncated_in_logs(caplog):
    client = app_with(BrokenLLM("llm_error"))
    long_query = "x" * 3000
    with caplog.at_level(logging.INFO):
        client.post("/ask", json={"query": long_query}, headers=AUTH)
    assert long_query not in caplog.text


def test_oversized_query_is_rejected(client):
    r = client.post("/ask", json={"query": "x" * 5000}, headers=AUTH)
    assert r.status_code == 422


def test_server_refuses_to_start_without_a_token():
    import pytest

    with pytest.raises(RuntimeError, match="LOCAL_ASSISTANT_TOKEN"):
        create_app(make_settings(local_assistant_token=""))


def test_answer_that_shapes_down_to_nothing_is_reported_not_spoken_silently():
    """A reply that is entirely a code block leaves no speech behind."""
    from tests.conftest import ScriptedLLM

    client = app_with(ScriptedLLM("```python\nprint('hi')\n```"))
    r = client.post("/ask", json={"query": "show me code", "source": "siri"}, headers=AUTH)
    assert r.status_code == 503
    assert r.json()["error"] == "llm_empty"
    assert r.json()["message"] == "今次答唔到你，唔該再問一次。"

    # The same answer is fine for a caller that is not speaking it.
    body = client.post(
        "/ask", json={"query": "show me code", "source": "api"}, headers=AUTH
    ).json()
    assert "print('hi')" in body["answer"]


def test_deadline_covers_time_spent_queueing():
    """With one slot and a slow model, the second caller must time out rather
    than wait for the first to finish plus its own full deadline."""
    import threading

    client = app_with(SlowLLM(), max_concurrent_requests=1, request_timeout_s=0.3)
    results: list[int] = []

    def call():
        results.append(
            client.post("/ask", json={"query": "hi"}, headers=AUTH).status_code
        )

    threads = [threading.Thread(target=call) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert results == [504, 504]
