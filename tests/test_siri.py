"""Phase 1B: the response shape Apple's TTS actually has to read."""

from __future__ import annotations

from tests.conftest import AUTH

MARKDOWN_ANSWER = (
    "## Dependency Injection\n\n"
    "**Dependency injection** 係一種 design pattern：\n"
    "- 唔係喺 class 入面自己 `new` 個 dependency\n"
    "- 而係由外面 inject 入嚟\n\n"
    "詳情見 [官方文件](https://example.com/di)。\n"
)


def test_siri_answer_is_speech_shaped(scripted_client):
    client, _ = scripted_client(MARKDOWN_ANSWER)
    body = client.post(
        "/ask",
        json={"query": "解釋下 dependency injection", "mode": "auto", "source": "siri"},
        headers=AUTH,
    ).json()

    answer = body["answer"]
    assert body["route"] == "local"
    for bad in ("#", "**", "- ", "`", "http", "[", "]"):
        assert bad not in answer, f"{bad!r} would be read aloud"
    assert "design pattern" in answer  # English technical term preserved
    assert "dependency injection" in answer.lower()


def test_api_source_keeps_markdown(scripted_client):
    client, _ = scripted_client(MARKDOWN_ANSWER)
    body = client.post(
        "/ask", json={"query": "explain DI", "source": "api"}, headers=AUTH
    ).json()
    assert "##" in body["answer"]  # untouched for non-speech callers


def test_siri_gets_speech_system_prompt_and_smaller_budget(scripted_client):
    client, llm = scripted_client("ok", llm_max_tokens_siri=222)
    client.post("/ask", json={"query": "hi", "source": "siri"}, headers=AUTH)
    call = llm.calls[0]
    assert call["max_tokens"] == 222
    assert "read aloud" in call["system"]
    assert "廣東話" in call["system"]


def test_api_source_gets_larger_budget(scripted_client):
    client, llm = scripted_client("ok", llm_max_tokens_default=777)
    client.post("/ask", json={"query": "hi", "source": "api"}, headers=AUTH)
    assert llm.calls[0]["max_tokens"] == 777
    assert "read aloud" not in llm.calls[0]["system"]


def test_source_defaults_to_api(client):
    r = client.post("/ask", json={"query": "hi"}, headers=AUTH)
    assert r.status_code == 200
