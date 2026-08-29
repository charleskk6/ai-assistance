from __future__ import annotations

import httpx
import pytest
import respx

from app.llm.base import LLMError
from app.llm.local import OllamaProvider, strip_thinking

HOST = "http://127.0.0.1:11434"


def provider(**kw) -> OllamaProvider:
    return OllamaProvider(model="qwen3:8b", host=HOST, **kw)


@respx.mock
async def test_complete_sends_expected_payload_and_strips_thinking():
    route = respx.post(f"{HOST}/api/chat").mock(
        return_value=httpx.Response(
            200, json={"message": {"content": "<think>hmm</think>The answer is 42."}}
        )
    )
    out = await provider().complete("sys", "q", max_tokens=100, temperature=0.4)
    assert out == "The answer is 42."

    body = route.calls[0].request.read().decode()
    assert '"think":false' in body
    assert '"num_predict":100' in body
    assert '"keep_alive"' in body
    # Qwen3's soft switch is appended to the system prompt.
    assert "/no_think" in body


@respx.mock
async def test_missing_model_gives_actionable_error():
    respx.post(f"{HOST}/api/chat").mock(return_value=httpx.Response(404, json={}))
    with pytest.raises(LLMError) as exc:
        await provider().complete("sys", "q", max_tokens=10, temperature=0.1)
    assert exc.value.code == "model_not_found"
    assert "ollama pull qwen3:8b" in exc.value.message


@respx.mock
async def test_runtime_down_is_reported_not_raised_by_health():
    respx.get(f"{HOST}/api/tags").mock(side_effect=httpx.ConnectError("refused"))
    h = await provider().health()
    assert h.runtime_ok is False and h.model_ok is False
    assert "unreachable" in h.detail


@respx.mock
async def test_health_detects_missing_model():
    respx.get(f"{HOST}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "llama3:8b"}]})
    )
    h = await provider().health()
    assert h.runtime_ok is True and h.model_ok is False
    assert "ollama pull" in h.detail


@respx.mock
async def test_timeout_maps_to_llm_timeout():
    respx.post(f"{HOST}/api/chat").mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(LLMError) as exc:
        await provider().complete("s", "q", max_tokens=10, temperature=0.1)
    assert exc.value.code == "llm_timeout"


@respx.mock
async def test_thinking_enabled_keeps_block_out_of_answer_but_sends_flag():
    respx.post(f"{HOST}/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": "plain"}})
    )
    p = provider(enable_thinking=True)
    assert await p.complete("s", "q", max_tokens=10, temperature=0.1) == "plain"


def test_strip_thinking_handles_unterminated_block():
    assert strip_thinking("<think>ran out of tokens") == ""
    assert strip_thinking("hello") == "hello"
