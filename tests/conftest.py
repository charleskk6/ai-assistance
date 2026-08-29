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
