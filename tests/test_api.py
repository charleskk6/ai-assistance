from __future__ import annotations

from tests.conftest import AUTH


def test_health_reports_llm(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["backend"] is True
    assert body["llm"]["provider"] == "echo"
    assert body["llm"]["runtime_ok"] and body["llm"]["model_ok"]


def test_health_needs_no_auth(client):
    assert client.get("/health").status_code == 200


def test_ask_local(client):
    r = client.post("/ask", json={"query": "hello", "mode": "local"}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "local"
    assert "hello" in body["answer"]
    assert body["sources"] == []
    assert isinstance(body["latency_ms"], int)


def test_ask_requires_token(client):
    r = client.post("/ask", json={"query": "hello"})
    assert r.status_code == 401
    assert r.json()["error"] == "unauthorized"


def test_ask_rejects_wrong_token(client):
    r = client.post("/ask", json={"query": "hi"}, headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_ask_rejects_empty_query(client):
    r = client.post("/ask", json={"query": "   "}, headers=AUTH)
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_request"


def test_ask_rejects_invalid_mode(client):
    r = client.post("/ask", json={"query": "hi", "mode": "quantum"}, headers=AUTH)
    assert r.status_code == 422


def test_health_reports_the_settings_actually_in_effect(client):
    """A .env copied from an older .env.example silently pins old defaults; the
    only symptom is that a tuning change appears to do nothing."""
    cfg = client.get("/health").json()["config"]
    for key in (
        "rag_context_budget_chars", "rag_top_chunks", "chunk_chars",
        "fetch_max_pages", "llm_num_ctx", "llm_max_tokens_siri",
    ):
        assert key in cfg, key


def test_health_config_reflects_an_override(scripted_client):
    client, _ = scripted_client("ok", rag_context_budget_chars=1234)
    assert client.get("/health").json()["config"]["rag_context_budget_chars"] == 1234


def test_local_route_reports_token_counters(scripted_client):
    client, _ = scripted_client("ok")
    timings = client.post(
        "/ask", json={"query": "hi", "source": "cli"}, headers=AUTH
    ).json()["timings"]
    # ScriptedLLM reports zeros, but the keys must be plumbed through so a real
    # runtime's counters reach the caller.
    for key in ("prompt_tokens", "output_tokens", "done_reason"):
        assert key in timings, key
