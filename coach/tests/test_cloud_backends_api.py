import json
import platform

import pytest
from fastapi.testclient import TestClient

from britcoach.cloud.providers import AnthropicProvider, CloudError, CloudProvider, OpenAIProvider
from britcoach.cloud.router import CloudPrefs, CloudRouter
from britcoach.config import resolve_llm_model
from britcoach.llm.base import LLMBackend
from britcoach.llm.factory import create_llm
from britcoach.llm.mock import MockLLM
from britcoach.main import create_app
from britcoach.pipeline.engines import Engines
from britcoach.stt.engines import MockSTT
from britcoach.audio.vad import EnergyVAD
from britcoach.tts.engines import MockTTS

from .conftest import make_settings


# ----------------------------------------------------------------- cloud routing
def router(**kw) -> CloudRouter:
    s = make_settings(**kw)
    s.anthropic_api_key = "sk-test"
    s.openai_api_key = None
    return CloudRouter(s)


def test_cloud_disabled_by_default():
    r = router()
    d = r.decide("review", CloudPrefs(), explicit_request=True, consent=True)
    assert not d.allowed and "disabled" in d.reason


def test_conversation_never_goes_to_cloud():
    r = router(cloud_enabled=True)
    assert not r.decide("conversation", CloudPrefs(enabled=True)).allowed


def test_review_requires_explicit_request_and_consent():
    r = router()
    prefs = CloudPrefs(enabled=True)
    assert not r.decide("review", prefs, explicit_request=True, consent=False).allowed
    assert r.decide("review", prefs, explicit_request=True, consent=True).allowed
    assert not r.decide("review", prefs, explicit_request=False, consent=True).allowed


def test_escalation_policy():
    prefs = CloudPrefs(enabled=True)
    assert not router().decide("review", prefs, consent=True, local_failed=True).allowed
    assert router(cloud_escalation="local_failure").decide("review", prefs, consent=True, local_failed=True).allowed


def test_missing_key_blocks():
    d = router().decide("review", CloudPrefs(enabled=True, provider="openai"), explicit_request=True, consent=True)
    assert not d.allowed and "no API key" in d.reason


def test_turn_corrections_need_opt_in():
    r = router()
    assert not r.decide("turn_correction", CloudPrefs(enabled=True)).allowed
    assert r.decide("turn_correction", CloudPrefs(enabled=True, allow_turn_corrections=True)).allowed


async def test_cloud_call_is_announced_and_audited():
    events, audits = [], []

    class Fake(CloudProvider):
        name, model = "anthropic", "fake"

        async def complete(self, system, user, max_tokens=2000):
            return "{}"

    async def on_event(e):
        events.append(e)

    async def audit(i):
        audits.append(i)

    s = make_settings()
    s.anthropic_api_key = "k"
    r = CloudRouter(s, provider_factory=lambda name: Fake(), on_event=on_event, audit=audit)
    d = r.decide("review", CloudPrefs(enabled=True), explicit_request=True, consent=True)
    assert await r.complete(d, "review", "sys", "user") == "{}"
    assert [e["active"] for e in events] == [True, False]
    assert audits[0]["purpose"] == "review" and audits[0]["chars_sent"] == 7 and audits[0]["ok"]
    with pytest.raises(CloudError):
        await r.complete(r.decide("conversation", CloudPrefs(enabled=True)), "conversation", "s", "u")


async def test_providers_call_sdks_correctly():
    calls = {}

    class Block:
        type, text = "text", '{"mistakes": []}'

    class Resp:
        stop_reason, content = "end_turn", [Block()]

    class FakeAnthropic:
        class beta:
            class messages:
                @staticmethod
                async def create(**kw):
                    calls["anthropic"] = kw
                    return Resp()

    out = await AnthropicProvider("claude-opus-5-5", client=FakeAnthropic()).complete("sys", "hi", 100)
    assert out == '{"mistakes": []}'
    kw = calls["anthropic"]
    assert kw["model"] == "claude-opus-5-5" and kw["system"] == "sys" and kw["max_tokens"] == 100
    assert kw["fallbacks"] == "default" and "temperature" not in kw

    class Msg:
        content = "ok"

    class Choice:
        message = Msg()

    class OResp:
        choices = [Choice()]

    class FakeOpenAI:
        class chat:
            class completions:
                @staticmethod
                async def create(**kw):
                    calls["openai"] = kw
                    return OResp()

    assert await OpenAIProvider("gpt-x", client=FakeOpenAI()).complete("s", "u", 50) == "ok"
    assert calls["openai"]["max_completion_tokens"] == 50


# ----------------------------------------------------------------- backend switching
def test_profiles_resolve_per_backend():
    s = make_settings(llm_backend="mlx", llm_model="")
    assert resolve_llm_model(s, "mlx", "small") == "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    assert resolve_llm_model(s, "ollama", "medium") == "qwen3:4b"
    assert resolve_llm_model(s, "mlx", "large").startswith("mlx-community/")
    s2 = make_settings(llm_backend="ollama", llm_model="my-custom:latest")
    assert resolve_llm_model(s2) == "my-custom:latest"


def test_factory_backends():
    s = make_settings()
    assert create_llm(s, "mock").name == "mock"
    assert create_llm(s, "ollama", profile="small").model_id == "qwen2.5:1.5b"
    assert create_llm(s, "openai_compat", model="x").name == "openai_compat"
    with pytest.raises(ValueError):
        create_llm(s, "nope", model="x")
    if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
        with pytest.raises(RuntimeError, match="Apple Silicon"):
            create_llm(s, "mlx", profile="medium")


async def test_engines_switch_llm_unloads_old(monkeypatch):
    class Tracked(MockLLM):
        unloaded = False

        async def unload(self):
            self.unloaded = True

    s = make_settings()
    old = Tracked()
    eng = Engines(s, llm=old, stt=MockSTT(), tts=MockTTS(), vad=EnergyVAD())
    await eng.start(warmup=False)
    new = Tracked()
    monkeypatch.setattr("britcoach.pipeline.engines.create_llm", lambda *a, **k: new)
    info = await eng.switch_llm("mock", profile="small")
    assert eng.llm is new and old.unloaded and not new.unloaded and info["ok"]


async def test_engine_load_failure_is_reported_not_fatal():
    class Broken(LLMBackend):
        name = "broken"

        async def load(self):
            raise RuntimeError("weights missing")

        async def stream(self, *a, **k):  # pragma: no cover
            yield

    eng = Engines(make_settings(), llm=Broken(), stt=MockSTT(), tts=MockTTS(), vad=EnergyVAD())
    await eng.start(warmup=True)
    ok, why = eng.ready()
    assert not ok and "weights missing" in why and eng.status["stt"]["ok"]


# ----------------------------------------------------------------- API + WebSocket
def test_api_and_websocket_roundtrip(tmp_path):
    s = make_settings(save_history=True, db_path=str(tmp_path / "t.sqlite3"))
    eng = Engines(s, llm=MockLLM(replies=["Hello! Lovely to meet you. How's your week been?", "Oh, you took him. How did he get on?"]),
                  stt=MockSTT(), tts=MockTTS(), vad=EnergyVAD())
    app = create_app(s, engines=eng, warmup=False)
    with TestClient(app) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        opts = client.get("/api/options").json()
        assert {m["id"] for m in opts["modes"]} == {"free", "social", "roleplay", "tutor"}
        assert "anthropic_api_key" not in json.dumps(opts["settings"])
        with client.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello"
            ws.send_text(json.dumps({"type": "start", "config": {"mode": "social", "scenario": "school_gate"}}))
            got_audio = False
            types = []
            while True:
                msg = ws.receive()
                if msg.get("bytes"):
                    got_audio = True
                    continue
                data = json.loads(msg["text"])
                types.append(data["type"])
                if data["type"] == "assistant_done":
                    break
            assert got_audio and "state" in types
            ws.send_text(json.dumps({"type": "text_input", "text": "Yesterday I bring my son to school"}))
            while True:
                msg = ws.receive()
                if msg.get("text") and json.loads(msg["text"])["type"] == "assistant_done":
                    break
            ws.send_text(json.dumps({"type": "end_session"}))
            while True:
                msg = ws.receive()
                if msg.get("text") and json.loads(msg["text"])["type"] == "review":
                    review = json.loads(msg["text"])["review"]
                    break
            assert review["mistakes"][0]["better"] == "I took"
            sid = review["session_id"]
        stored = client.get(f"/api/sessions/{sid}").json()
        assert stored["review"]["session_id"] == sid and len(stored["turns"]) >= 3
        assert client.get("/api/progress").json()["sessions"]
        assert "summary" in client.get("/api/metrics").json()
