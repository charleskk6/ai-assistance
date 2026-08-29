from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

TOKEN = "test-token-123"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def make_settings(**overrides) -> Settings:
    base = dict(
        local_assistant_token=TOKEN,
        llm_provider="echo",
        llm_model="echo",
        _env_file=None,
    )
    base.update(overrides)
    env_file = base.pop("_env_file", None)
    return Settings(_env_file=env_file, **base)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def client(settings) -> TestClient:
    with TestClient(create_app(settings)) as c:
        yield c


class ScriptedLLM:
    """LLM stand-in that returns a canned answer and records what it was sent."""

    name = "scripted"

    def __init__(
        self, answer: str = "ok", model: str = "scripted", truncated: bool = False
    ) -> None:
        self.model = model
        self.answer = answer
        self.truncated = truncated
        self.calls: list[dict] = []

    async def complete(self, system, user, *, max_tokens, temperature):
        from app.llm.base import Completion

        self.calls.append(
            {"system": system, "user": user, "max_tokens": max_tokens,
             "temperature": temperature}
        )
        return Completion(self.answer, truncated=self.truncated)

    async def health(self):
        from app.llm.base import LLMHealth

        return LLMHealth(True, True, self.name, self.model)

    async def aclose(self):
        return None


@pytest.fixture
def scripted_client():
    """Returns (TestClient, ScriptedLLM) so tests can assert on the prompt."""

    def _make(answer: str = "ok", **setting_overrides):
        app = create_app(make_settings(**setting_overrides))
        llm = ScriptedLLM(answer)
        client = TestClient(app)
        client.__enter__()
        app.state.llm = llm  # replace the one lifespan built
        return client, llm

    return _make
