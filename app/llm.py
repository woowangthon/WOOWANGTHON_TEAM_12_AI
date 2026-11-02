"""LLM 호출 — Gemini(google-genai). 모델 · 제공사를 바꾸면 ETC docs/ai-scope.md 를 같은 PR 에서.

모델 목록을 앞에서부터 쓰고, 그 모델을 「지금 못 쓰면」 다음 모델로 넘어간다(폴백).
넘어가도 진짜 LLM 의 답이다 — 응답 `model` 에 실제로 답한 모델이 찍힌다.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.errors import LLMError, LLMTimeout

log = logging.getLogger(__name__)

# 다음 모델로 넘어갈 HTTP 코드 — 모델이 없어짐(404) · 한도(429) · 제공사 서버(5xx).
# 400 · 401 · 403(키 · 요청 문제)은 다른 모델도 같으니 바로 실패로 돌려준다.
FALLBACK_CODES = {404, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str


class JsonLLM(Protocol):
    async def generate_json(
        self, system: str, user: str, schema: dict[str, Any], deadline: float
    ) -> LLMResult:
        """deadline(time.monotonic 기준) 안에 JSON 문자열을 돌려준다. 실패하면 LLMError · LLMTimeout."""
        ...


class GeminiLLM:
    def __init__(self, api_key: str | None, models: tuple[str, ...], call_timeout: float):
        self.models = models
        self.call_timeout = call_timeout
        self._api_key = api_key
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            if not self._api_key:
                raise LLMError("GEMINI_API_KEY 가 설정되지 않았습니다")
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def _call(self, model: str, system: str, user: str, schema: dict[str, Any]) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema,
            temperature=0,
        )
        response = await self._get_client().aio.models.generate_content(
            model=model, contents=user, config=config
        )
        # 안전 필터 등으로 비어 오면 빈 문자열 — 호출한 쪽에서 출력 규칙 위반으로 다룬다
        return response.text or ""

    async def generate_json(
        self, system: str, user: str, schema: dict[str, Any], deadline: float
    ) -> LLMResult:
        failures: list[str] = []
        ran_out_of_time = False  # 마지막으로 멈춘 이유가 시간인가
        for model in self.models:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                ran_out_of_time = True
                break
            timeout = min(self.call_timeout, remaining)
            try:
                text = await asyncio.wait_for(self._call(model, system, user, schema), timeout)
                return LLMResult(text=text, model=model)
            except TimeoutError:
                ran_out_of_time = True
                failures.append(f"{model}: {timeout:.0f}초 초과")
            except genai_errors.APIError as e:
                ran_out_of_time = False
                if e.code not in FALLBACK_CODES:
                    raise LLMError(f"LLM 호출 실패({model}): {e.code} {e.message}") from e
                failures.append(f"{model}: {e.code} {e.status}")
            except LLMError:
                raise
            except Exception as e:  # 네트워크 등
                ran_out_of_time = False
                failures.append(f"{model}: {type(e).__name__}")
            log.warning("LLM 폴백 — %s", failures[-1])

        summary = " · ".join(failures) or "부를 시간이 남지 않음"
        if ran_out_of_time:
            raise LLMTimeout(f"LLM 이 제한 시간 안에 답하지 않았습니다 — {summary}")
        raise LLMError(f"쓸 수 있는 모델이 없습니다 — {summary}")
