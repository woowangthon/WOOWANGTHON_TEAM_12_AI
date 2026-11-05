"""LLM 호출 — OpenAI(기본, 팀 크레딧) · Gemini(선택). 모델 · 제공사를 바꾸면 ETC docs/ai-scope.md 를 같은 PR 에서.

모델 목록(`제공사:모델`)을 앞에서부터 쓰고, 그 모델을 「지금 못 쓰면」 다음 모델로 넘어간다(폴백).
키가 없는 제공사는 건너뛴다. 넘어가도 진짜 LLM 의 답이다 — 응답 `model` 에 실제로 답한 모델이 찍힌다.
"""

import asyncio
import copy
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import openai
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import ValidationError

from app.config import Settings
from app.errors import LLMError, LLMInvalidOutput, LLMTimeout

log = logging.getLogger(__name__)

# 다음 모델로 넘어갈 HTTP 코드 — 모델이 없어짐(404) · 한도 · 크레딧(429) · 제공사 서버(5xx).
# 400 · 401 · 403(키 · 요청 문제)은 다른 모델로 가도 같으니 바로 실패로 돌려준다.
FALLBACK_CODES = {404, 429, 500, 502, 503, 504}


class Unavailable(Exception):
    """이 모델을 지금 못 쓴다 — 다음 모델로 넘어간다."""


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


class Provider(Protocol):
    name: str
    available: bool

    async def call(self, model: str, system: str, user: str, schema: dict[str, Any]) -> str:
        """JSON 문자열. 모델을 지금 못 쓰면 Unavailable, 키 · 요청 문제면 LLMError."""
        ...


def _strict(schema: dict[str, Any]) -> dict[str, Any]:
    """OpenAI strict 모드 — 모든 객체에 additionalProperties: false, 모든 칸을 required 로."""
    schema = copy.deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    return schema


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str | None):
        self.available = bool(api_key)
        # 다시 시도 · 시간 제한은 FallbackLLM 이 맡는다 — SDK 자체 재시도는 끈다
        self._client = openai.AsyncOpenAI(api_key=api_key, max_retries=0) if api_key else None

    async def call(self, model: str, system: str, user: str, schema: dict[str, Any]) -> str:
        assert self._client is not None
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction",
                        "schema": _strict(schema),
                        "strict": True,
                    },
                },
            )
        except openai.APIStatusError as e:
            if e.status_code in FALLBACK_CODES:
                raise Unavailable(f"{e.status_code} {type(e).__name__}") from e
            raise LLMError(f"LLM 호출 실패(openai:{model}): {e.status_code} {e.message}") from e
        except openai.APIConnectionError as e:
            raise Unavailable(type(e).__name__) from e
        message = response.choices[0].message
        # 거절 · 빈 답은 빈 문자열 — 호출한 쪽에서 출력 규칙 위반으로 다룬다
        return "" if message.refusal else (message.content or "")


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str | None):
        self.available = bool(api_key)
        self._client = genai.Client(api_key=api_key) if api_key else None

    async def call(self, model: str, system: str, user: str, schema: dict[str, Any]) -> str:
        assert self._client is not None
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema,
            temperature=0,
        )
        try:
            response = await self._client.aio.models.generate_content(
                model=model, contents=user, config=config
            )
        except genai_errors.APIError as e:
            if e.code in FALLBACK_CODES:
                raise Unavailable(f"{e.code} {e.status}") from e
            raise LLMError(f"LLM 호출 실패(gemini:{model}): {e.code} {e.message}") from e
        return response.text or ""


class FallbackLLM:
    def __init__(
        self,
        models: tuple[tuple[str, str], ...],
        providers: dict[str, Provider],
        call_timeout: float,
        min_call_seconds: float = 1.0,
    ):
        self.models = models
        self.providers = providers
        self.call_timeout = call_timeout
        # 남은 시간이 이보다 짧으면 새 호출을 시작하지 않는다 — 답을 받을 수 없는 호출이다
        self.min_call_seconds = min_call_seconds

    async def generate_json(
        self, system: str, user: str, schema: dict[str, Any], deadline: float
    ) -> LLMResult:
        usable = [(p, m) for p, m in self.models if self.providers[p].available]
        if not usable:
            raise LLMError("쓸 수 있는 LLM 키가 없습니다(OPENAI_API_KEY · GEMINI_API_KEY)")

        failures: list[str] = []
        ran_out_of_time = False  # 마지막으로 멈춘 이유가 시간인가
        for provider_name, model in usable:
            name = f"{provider_name}:{model}"
            remaining = deadline - time.monotonic()
            if remaining < self.min_call_seconds:
                ran_out_of_time = True
                break
            timeout = min(self.call_timeout, remaining)
            try:
                text = await asyncio.wait_for(
                    self.providers[provider_name].call(model, system, user, schema), timeout
                )
                return LLMResult(text=text, model=name)
            except TimeoutError:
                ran_out_of_time = True
                failures.append(f"{name}: {timeout:.0f}초 초과")
            except Unavailable as e:
                ran_out_of_time = False
                failures.append(f"{name}: {e}")
            except LLMError:
                raise
            except Exception as e:  # 네트워크 등 예상 못 한 호출 오류 — 다음 모델로
                ran_out_of_time = False
                failures.append(f"{name}: {type(e).__name__}")
            log.warning("LLM 폴백 — %s", failures[-1])

        summary = " · ".join(failures) or "부를 시간이 남지 않음"
        if ran_out_of_time:
            raise LLMTimeout(f"LLM 이 제한 시간 안에 답하지 않았습니다 — {summary}")
        raise LLMError(f"쓸 수 있는 모델이 없습니다 — {summary}")


def build_llm(
    settings: Settings,
    models: tuple[tuple[str, str], ...] | None = None,
    call_timeout: float | None = None,
) -> FallbackLLM:
    """models 를 안 주면 settings.llm_models(추출) 순서로, call_timeout 을 안 주면 settings.llm_call_timeout_seconds."""
    providers: dict[str, Provider] = {
        "openai": OpenAIProvider(settings.openai_api_key),
        "gemini": GeminiProvider(settings.gemini_api_key),
    }
    return FallbackLLM(
        models or settings.llm_models,
        providers,
        settings.llm_call_timeout_seconds if call_timeout is None else call_timeout,
    )


@dataclass(frozen=True)
class Valid[T]:
    value: T
    model: str
    attempt: int
    seconds: float


async def generate_valid[T](
    llm: JsonLLM,
    system: str,
    user: str,
    schema: dict[str, Any],
    parse: Callable[[str], T],
    settings: Settings,
    timeout: float | None = None,
) -> Valid[T]:
    """LLM 출력을 parse 로 검사하고, 규칙을 벗어나면 다시 부른다(최대 settings.max_attempts 번).

    parse 는 규칙 위반이면 ValueError(pydantic ValidationError 포함)를 던진다. 출력을 그럴듯하게
    고치지 않는다 — 끝까지 벗어나면 LLMInvalidOutput(계약 ai-api 「응답 규칙」).
    timeout 은 폴백 · 다시 시도까지 합친 전체 시간 — 안 주면 settings.llm_timeout_seconds(작업 호출).
    """
    deadline = time.monotonic() + (settings.llm_timeout_seconds if timeout is None else timeout)
    last_error = ""
    for attempt in range(1, settings.max_attempts + 1):
        if deadline - time.monotonic() <= 0:
            raise LLMTimeout(f"다시 시도할 시간이 남지 않았습니다(이전 오류: {last_error})")
        started = time.monotonic()
        result = await llm.generate_json(system, user, schema, deadline)
        elapsed = time.monotonic() - started
        try:
            value = parse(result.text)
        except ValidationError as e:
            first = e.errors()[0]
            last_error = f"{e.error_count()}개 규칙 위반: {first['loc']} {first['msg']}"
        except ValueError as e:
            last_error = str(e)
        else:
            return Valid(value=value, model=result.model, attempt=attempt, seconds=elapsed)
        log.warning(
            "LLM 출력 규칙 위반 attempt=%d model=%s elapsed=%.1fs %s",
            attempt,
            result.model,
            elapsed,
            last_error,
        )
    raise LLMInvalidOutput(
        f"LLM 출력이 {settings.max_attempts}번 모두 규칙을 벗어났습니다 — {last_error}"
    )
