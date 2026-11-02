"""모델 폴백 · 제공사 — 「지금 못 씀」이면 다음 모델로, 키 · 요청 문제면 바로 실패(네트워크 없음)."""

import asyncio
import time

import pytest

from app.config import parse_models
from app.errors import LLMError, LLMTimeout
from app.extract import LLM_JSON_SCHEMA
from app.llm import FallbackLLM, Unavailable, _strict


class ScriptedProvider:
    """모델마다 정한 결과를 돌려준다 — 예외면 던지고, float 면 그만큼 잠든다."""

    def __init__(self, name, script, available=True):
        self.name = name
        self.available = available
        self.script = script
        self.called: list[str] = []

    async def call(self, model, system, user, schema):
        self.called.append(model)
        out = self.script[model]
        if isinstance(out, Exception):
            raise out
        if isinstance(out, float):
            await asyncio.sleep(out)
            return "{}"
        return out


def make(models, *providers, call_timeout=5.0):
    return FallbackLLM(parse_models(models), {p.name: p for p in providers}, call_timeout)


def run(llm, budget=10.0):
    return asyncio.run(llm.generate_json("s", "u", {}, time.monotonic() + budget))


def test_first_model_answers():
    oa = ScriptedProvider("openai", {"a": "{}", "b": "{}"})
    result = run(make("openai:a,openai:b", oa))
    assert (result.model, oa.called) == ("openai:a", ["a"])


def test_unavailable_falls_back_across_providers():
    oa = ScriptedProvider("openai", {"a": Unavailable("429 RateLimitError")})
    gm = ScriptedProvider("gemini", {"g": '{"x": 1}'})
    result = run(make("openai:a,gemini:g", oa, gm))
    assert result.model == "gemini:g"
    assert result.text == '{"x": 1}'


def test_provider_without_key_is_skipped():
    oa = ScriptedProvider("openai", {"a": "{}"})
    gm = ScriptedProvider("gemini", {"g": "{}"}, available=False)
    llm = make("gemini:g,openai:a", oa, gm)
    assert run(llm).model == "openai:a"
    assert gm.called == []


def test_no_keys_is_llm_error():
    oa = ScriptedProvider("openai", {}, available=False)
    gm = ScriptedProvider("gemini", {}, available=False)
    with pytest.raises(LLMError, match="키가 없습니다"):
        run(make("openai:a,gemini:g", oa, gm))


def test_key_or_request_error_stops():
    oa = ScriptedProvider("openai", {"a": LLMError("401"), "b": "{}"})
    with pytest.raises(LLMError):
        run(make("openai:a,openai:b", oa))
    assert oa.called == ["a"]


def test_unexpected_exception_falls_back():
    oa = ScriptedProvider("openai", {"a": ConnectionError("x"), "b": "{}"})
    assert run(make("openai:a,openai:b", oa)).model == "openai:b"


def test_slow_model_falls_back():
    oa = ScriptedProvider("openai", {"a": 1.0, "b": "{}"})
    assert run(make("openai:a,openai:b", oa, call_timeout=0.05)).model == "openai:b"


def test_all_unavailable_is_llm_error():
    oa = ScriptedProvider("openai", {"a": Unavailable("503"), "b": Unavailable("429")})
    with pytest.raises(LLMError, match="openai:a: 503.*openai:b: 429"):
        run(make("openai:a,openai:b", oa))


def test_all_slow_is_timeout():
    oa = ScriptedProvider("openai", {"a": 1.0, "b": 1.0})
    with pytest.raises(LLMTimeout):
        run(make("openai:a,openai:b", oa, call_timeout=0.05))


def test_budget_runs_out_is_timeout():
    oa = ScriptedProvider("openai", {"a": 1.0, "b": "{}"})
    with pytest.raises(LLMTimeout):
        run(make("openai:a,openai:b", oa), budget=0.05)
    assert oa.called == ["a"]


@pytest.mark.parametrize("raw", ["gpt-x", "claude:x", "openai:", ""])
def test_parse_models_rejects_bad_entries(raw):
    with pytest.raises(ValueError):
        parse_models(raw)


def test_strict_schema_closes_every_object():
    strict = _strict(LLM_JSON_SCHEMA)

    def objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                yield node
            for v in node.values():
                yield from objects(v)
        elif isinstance(node, list):
            for v in node:
                yield from objects(v)

    found = list(objects(strict))
    assert len(found) == 4  # 최상위 · 활동 · 기간 · 경험
    for obj in found:
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])
    assert "additionalProperties" not in LLM_JSON_SCHEMA  # 원본은 그대로
