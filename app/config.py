"""환경 변수에서 읽는 설정. 값의 뜻은 `.env.example`."""

import os
from dataclasses import dataclass

DEFAULT_MODELS = "gemini-3.5-flash,gemini-3.8-flash,gemini-3.5-flash-lite"


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None
    # 앞에서부터 쓰고, 「지금 못 씀」(503 · 429 · 5xx · 404 · 시간 초과)이면 다음 모델로 넘어간다
    llm_models: tuple[str, ...]
    # BE 제한 시간(제안 120초)보다 짧게 — 폴백 · 다시 시도까지 합친 전체 시간
    llm_timeout_seconds: float
    # 모델 하나를 한 번 부를 때 제한 시간 — 한 모델이 전체 시간을 다 먹지 않게
    llm_call_timeout_seconds: float
    # BE 한도(experience-api, 제안 50000자)보다 작으면 안 된다
    max_text_chars: int
    # LLM 출력이 규칙을 벗어날 때 몇 번까지 부르나(처음 포함)
    max_attempts: int


def _models(raw: str) -> tuple[str, ...]:
    models = tuple(m.strip() for m in raw.split(",") if m.strip())
    if not models:
        raise ValueError("LLM_MODELS 가 비었습니다")
    return models


def load_settings() -> Settings:
    return Settings(
        gemini_api_key=os.environ.get("GEMINI_API_KEY") or None,
        llm_models=_models(os.environ.get("LLM_MODELS", DEFAULT_MODELS)),
        llm_timeout_seconds=float(os.environ.get("LLM_TIMEOUT_SECONDS", "100")),
        llm_call_timeout_seconds=float(os.environ.get("LLM_CALL_TIMEOUT_SECONDS", "45")),
        max_text_chars=int(os.environ.get("MAX_TEXT_CHARS", "50000")),
        max_attempts=int(os.environ.get("LLM_MAX_ATTEMPTS", "2")),
    )
