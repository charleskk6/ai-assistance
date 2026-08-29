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
