"""테스트 공통 — 가짜 LLM 과, 의존성을 바꾼 TestClient."""

import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.config import load_settings
from app.llm import LLMResult
from app.main import app, get_link_llm, get_llm, get_match_llm, get_route_llm, get_settings


class FakeLLM:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = 0

    async def generate_json(self, system, user, schema, deadline):
        self.calls += 1
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
        return LLMResult(text=text, model="fake-model")


@pytest.fixture
def client_with():
    def make(llm, **settings):
        app.dependency_overrides[get_llm] = lambda: llm
        app.dependency_overrides[get_match_llm] = lambda: llm
        app.dependency_overrides[get_route_llm] = lambda: llm
        app.dependency_overrides[get_link_llm] = lambda: llm
        app.dependency_overrides[get_settings] = lambda: replace(load_settings(), **settings)
        return TestClient(app)

    yield make
    app.dependency_overrides.clear()
