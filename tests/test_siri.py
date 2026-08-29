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


def test_truncated_siri_answer_loses_its_half_sentence(scripted_client):
    """End to end: the runtime says it hit the token ceiling, so the dangling
    clause never reaches Apple's TTS."""
    client, llm = scripted_client(
        "Dependency injection 係一種 design pattern。例如你有個 server object，佢需要一個 databa"
    )
    llm.truncated = True
    body = client.post(
        "/ask", json={"query": "解釋下 DI", "source": "siri"}, headers=AUTH
    ).json()
    assert body["answer"] == "Dependency injection 係一種 design pattern。"


def test_untruncated_answer_is_left_alone(scripted_client):
    text = "Dependency injection 係一種 design pattern，由外面 inject 入嚟"
    client, llm = scripted_client(text)
    body = client.post(
        "/ask", json={"query": "解釋下 DI", "source": "siri"}, headers=AUTH
    ).json()
    assert body["answer"] == text


def test_cantonese_prompt_names_the_actual_substitutions(scripted_client):
    """The old prompt just said 'natural Cantonese' and the model drifted into
    Standard Written Chinese after the first clause."""
    client, llm = scripted_client("ok")
    client.post("/ask", json={"query": "解釋下 DI", "source": "siri"}, headers=AUTH)
    system = llm.calls[0]["system"]
    for pair in ("嘅 not 的", "係 not 是", "唔 not 不", "佢 not 它"):
        assert pair in system
    assert "design pattern" in system and "設計模式" in system  # keep-in-English list
