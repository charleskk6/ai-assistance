import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from britcoach.llm.base import CancelToken, GenParams, iterate_in_thread
from britcoach.llm.http_backends import OllamaBackend, OpenAICompatBackend
from britcoach.tts.text import SentenceSplitter, ThinkFilter, clean_for_speech


def feed(splitter: SentenceSplitter, text: str, step: int = 3) -> list[str]:
    out = []
    for i in range(0, len(text), step):
        out += splitter.push(text[i:i + step])
    return out + splitter.flush()


def test_sentence_splitter_streams_sentences():
    s = SentenceSplitter(first_chunk_min_chars=200)
    out = feed(s, "Oh, lovely! Mr. Smith said it costs 3.50 pounds. Did you go? Yes.")
    assert out == ["Oh, lovely!", "Mr. Smith said it costs 3.50 pounds.", "Did you go?", "Yes."]


def test_first_chunk_flushes_early_at_clause():
    s = SentenceSplitter(first_chunk_min_chars=30)
    out = feed(s, "Well, that sounds absolutely brilliant, to be honest, what did the kids think of it all? Fab.")
    assert out[0].endswith(",") and len(out[0]) >= 15
    assert "".join(out).replace(" ", "") == "Well,thatsoundsabsolutelybrilliant,tobehonest,whatdidthekidsthinkofitall?Fab."


def test_splitter_emits_sentence_before_stream_ends():
    s = SentenceSplitter(first_chunk_min_chars=500)
    assert s.push("How did he get on? ") == ["How did he get on?"]
    assert s.push("Was it") == []


def test_think_filter_across_chunks():
    f = ThinkFilter()
    out = "".join(f.push(p) for p in ["Hi <thi", "nk>secret plan</th", "ink>there", " mate"]) + f.flush()
    assert out == "Hi there mate"


def test_clean_for_speech():
    assert clean_for_speech("**Great!** 😀 [Coach note: x] Fancy a *cuppa*? (laughs)") == "Great! Fancy a cuppa?"
    assert clean_for_speech("Lovely — how was it?") == "Lovely — how was it?"


async def test_iterate_in_thread_cancels_quickly():
    produced = []
    stop_seen = threading.Event()

    def gen():
        for i in range(10_000):
            produced.append(i)
            time.sleep(0.002)
            yield i
        stop_seen.set()

    ex = ThreadPoolExecutor(1)
    tok = CancelToken()
    got = []
    async for x in iterate_in_thread(gen, ex, tok):
        got.append(x)
        if len(got) == 5:
            tok.cancel("test")
    # Producer stopped within a couple of items after cancellation, not at 10k.
    assert len(produced) < 20 and not stop_seen.is_set()


async def test_iterate_in_thread_normal_end_does_not_cancel_shared_token():
    tok = CancelToken()
    ex = ThreadPoolExecutor(1)
    assert [x async for x in iterate_in_thread(lambda: iter([1, 2, 3]), ex, tok)] == [1, 2, 3]
    assert not tok.cancelled
    # but abandoning the iterator early does cancel it
    agen = iterate_in_thread(lambda: iter(range(100)), ex, tok)
    await agen.__anext__()
    await agen.aclose()
    assert tok.cancelled


async def test_iterate_in_thread_propagates_errors():
    def bad():
        yield 1
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        async for _ in iterate_in_thread(bad, ThreadPoolExecutor(1), CancelToken()):
            pass


def _ollama_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert body["stream"] is True and body["think"] is False
    lines = [json.dumps({"message": {"content": w}, "done": False}) for w in ["Oh, ", "lovely. ", "How ", "was ", "it?"]]
    lines.append(json.dumps({"message": {"content": ""}, "done": True, "eval_count": 5, "eval_duration": 250_000_000,
                             "prompt_eval_count": 42}))
    return httpx.Response(200, content="\n".join(lines).encode())


async def test_ollama_streaming():
    be = OllamaBackend("qwen3:4b", "http://x", transport=httpx.MockTransport(_ollama_handler))
    chunks = [c async for c in be.stream([{"role": "user", "content": "hi"}], GenParams(), CancelToken())]
    assert "".join(c.text for c in chunks) == "Oh, lovely. How was it?"
    assert chunks[-1].done and chunks[-1].stats["tokens_per_s"] == pytest.approx(20.0)
    assert chunks[-1].stats["prompt_tokens"] == 42


async def test_ollama_cancel_stops_reading():
    be = OllamaBackend("m", "http://x", transport=httpx.MockTransport(_ollama_handler))
    tok = CancelToken()
    texts = []
    async for c in be.stream([], GenParams(), tok):
        texts.append(c.text)
        if len(texts) == 2:
            tok.cancel()
    assert texts[:2] == ["Oh, ", "lovely. "] and len(texts) == 3 and texts[-1] == ""  # final stats chunk only


async def test_openai_compat_streaming():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        events = [f"data: {json.dumps({'choices': [{'delta': {'content': w}}]})}" for w in ["Right, ", "cheers!"]]
        return httpx.Response(200, content=("\n\n".join(events + ["data: [DONE]"]) + "\n\n").encode())

    be = OpenAICompatBackend("m", "http://x/v1", transport=httpx.MockTransport(handler))
    out = await be.complete([{"role": "user", "content": "hi"}], GenParams())
    assert out == "Right, cheers!"
