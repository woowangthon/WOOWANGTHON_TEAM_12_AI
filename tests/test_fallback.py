"""모델 폴백 — 「지금 못 씀」이면 다음 모델로, 키 · 요청 문제면 바로 실패(네트워크 없음)."""

import asyncio
import time

import pytest
from google.genai import errors as genai_errors

from app.errors import LLMError, LLMTimeout
from app.llm import GeminiLLM


def api_error(code: int) -> genai_errors.APIError:
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, {"error": {"code": code, "message": "m", "status": "S"}})


class ScriptedGemini(GeminiLLM):
    """모델마다 정한 결과를 돌려준다 — 값이 예외면 던지고, 숫자면 그만큼 잠든다."""

    def __init__(self, script: dict, models: tuple[str, ...], call_timeout: float = 5):
        super().__init__(api_key="unused", models=models, call_timeout=call_timeout)
        self.script = script
        self.called: list[str] = []

    async def _call(self, model, system, user, schema):
        self.called.append(model)
        out = self.script[model]
        if isinstance(out, Exception):
            raise out
        if isinstance(out, float):
            await asyncio.sleep(out)
            return "{}"
        return out


def run(llm: GeminiLLM, budget: float = 10):
    return asyncio.run(llm.generate_json("s", "u", {}, time.monotonic() + budget))


def test_first_model_answers():
    llm = ScriptedGemini({"a": "{}", "b": "{}"}, ("a", "b"))
    result = run(llm)
    assert (result.model, llm.called) == ("a", ["a"])


@pytest.mark.parametrize("code", [404, 429, 500, 503])
def test_unavailable_falls_back(code):
    llm = ScriptedGemini({"a": api_error(code), "b": '{"x": 1}'}, ("a", "b"))
    result = run(llm)
    assert result.model == "b"
    assert result.text == '{"x": 1}'
    assert llm.called == ["a", "b"]


@pytest.mark.parametrize("code", [400, 401, 403])
def test_key_or_request_error_stops(code):
    llm = ScriptedGemini({"a": api_error(code), "b": "{}"}, ("a", "b"))
    with pytest.raises(LLMError):
        run(llm)
    assert llm.called == ["a"]


def test_slow_model_falls_back():
    llm = ScriptedGemini({"a": 1.0, "b": "{}"}, ("a", "b"), call_timeout=0.05)
    assert run(llm).model == "b"


def test_all_unavailable_is_llm_error():
    llm = ScriptedGemini({"a": api_error(503), "b": api_error(429)}, ("a", "b"))
    with pytest.raises(LLMError, match="a: 503.*b: 429"):
        run(llm)


def test_all_slow_is_timeout():
    llm = ScriptedGemini({"a": 1.0, "b": 1.0}, ("a", "b"), call_timeout=0.05)
    with pytest.raises(LLMTimeout):
        run(llm)


def test_budget_runs_out_is_timeout():
    # 전체 시간이 한 번 부를 시간보다 짧으면 다음 모델로 못 넘어간다
    llm = ScriptedGemini({"a": 1.0, "b": "{}"}, ("a", "b"), call_timeout=5)
    with pytest.raises(LLMTimeout):
        run(llm, budget=0.05)
    assert llm.called == ["a"]
