"""환경 변수에서 읽는 설정. 값의 뜻은 `.env.example`."""

import os
from dataclasses import dataclass

PROVIDERS = ("openai", "gemini")
# 기본은 팀 OpenAI 크레딧. Gemini 는 GEMINI_API_KEY 가 있을 때만 쓰는 선택 폴백.
# 고르는 규칙: 지원 중인 모델을 최근 세대부터, 비싼 등급(astra · pro)과 지원 종료 예정은 뺀다.
# 같은 세대가 함께 막힐 때를 대비해 끝에 다른 세대를 하나 둔다. 바꾸면 ETC ai-scope.md §2 도
# 첫 모델은 시험 자료 채점(scripts/eval_extract.py, AI#5)으로 골랐다 — luna 는 활동을 빠뜨리지 않고 가장 싸다.
# terra 는 짧은 메모에서 활동을 통째로 빠뜨렸다(2회 중 2회)
DEFAULT_MODELS = (
    "openai:gpt-6-luna,openai:gpt-6.1-terra,openai:gpt-5.4-mini,"
    "gemini:gemini-3.5-flash,gemini:gemini-3.5-flash-lite"
)
# 매칭(/match)은 순서만 다르다 — 채점(scripts/eval_match.py, AI#20)에서 terra 가 판정이 같거나 낫고 빨랐다.
# luna 는 문항당 17 ~ 25초에 모델 한 번 제한 45초를 넘긴 적이 있다(19회 중 1회)
DEFAULT_MATCH_MODELS = (
    "openai:gpt-6.1-terra,openai:gpt-6-luna,openai:gpt-5.4-mini,"
    "gemini:gemini-3.5-flash,gemini:gemini-3.5-flash-lite"
)
# 홈 입력 가르기(/route) — 사용자가 화면에서 기다리는 동기 호출. 순서는 시험 입력 40개 채점(scripts/eval_route.py,
# AI#33)으로 골랐다 — 세 모델 다 40/40(terra 최대 3초 · mini 1.5초). terra 는 범용 프롬프트로도 40/40 이라 문구에
# 덜 흔들리고, mini 는 범용 프롬프트로 6건을 틀린 곳으로 보냈다 — 처음 보는 실제 입력엔 terra 가 안전하다.
# 지원 나누기(/parse-application)도 이 순서다 — 시험 글 21개(scripts/eval_parse.py, AI#35)에서 terra · luna 는 2회 다 만점
DEFAULT_ROUTE_MODELS = (
    "openai:gpt-6.1-terra,openai:gpt-6-luna,openai:gpt-5.4-mini,"
    "gemini:gemini-3.5-flash,gemini:gemini-3.5-flash-lite"
)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    gemini_api_key: str | None
    # (제공사, 모델). 앞에서부터 쓰고, 「지금 못 씀」(404 · 429 · 5xx · 시간 초과)이면 다음으로.
    # 키가 없는 제공사는 건너뛴다
    llm_models: tuple[tuple[str, str], ...]
    # /match · /draft 에 쓰는 순서 — 규칙은 llm_models 와 같다
    match_models: tuple[tuple[str, str], ...]
    # /route · /parse-application 에 쓰는 순서 — 규칙은 llm_models 와 같다
    route_models: tuple[tuple[str, str], ...]
    # BE 제한 시간(제안 120초)보다 짧게 — 폴백 · 다시 시도까지 합친 전체 시간
    llm_timeout_seconds: float
    # 입구 셋(/route …) — 사용자가 기다리는 동기 호출. BE 제한 시간(AI_SYNC_TIMEOUT_SECONDS 60초)보다 짧게
    llm_sync_timeout_seconds: float
    # 입구 셋에서 모델 하나를 한 번 부를 때 제한 — 한 모델이 멈춰도 50초 안에 다음 모델로 넘어가게 짧게
    llm_sync_call_timeout_seconds: float
    # 모델 하나를 한 번 부를 때 제한 시간 — 한 모델이 전체 시간을 다 먹지 않게
    llm_call_timeout_seconds: float
    # BE 한도(experience-api, 제안 50000자)보다 작으면 안 된다
    max_text_chars: int
    # /match 의 문항 · 경험 본문 · 근거 · 보완 글자 수 합 한도(계약 ai-api 「POST /match」 413)
    max_match_chars: int
    # LLM 출력이 규칙을 벗어날 때 몇 번까지 부르나(처음 포함)
    max_attempts: int


def parse_models(raw: str) -> tuple[tuple[str, str], ...]:
    models = []
    for item in (m.strip() for m in raw.split(",")):
        if not item:
            continue
        provider, sep, model = item.partition(":")
        if not sep or provider not in PROVIDERS or not model:
            raise ValueError(f"LLM_MODELS 항목은 「제공사:모델」(제공사 {PROVIDERS}) — {item!r}")
        models.append((provider, model))
    if not models:
        raise ValueError("LLM_MODELS 가 비었습니다")
    return tuple(models)


def load_settings() -> Settings:
    return Settings(
        openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
        gemini_api_key=os.environ.get("GEMINI_API_KEY") or None,
        llm_models=parse_models(os.environ.get("LLM_MODELS", DEFAULT_MODELS)),
        match_models=parse_models(os.environ.get("MATCH_LLM_MODELS", DEFAULT_MATCH_MODELS)),
        route_models=parse_models(os.environ.get("ROUTE_LLM_MODELS", DEFAULT_ROUTE_MODELS)),
        llm_timeout_seconds=float(os.environ.get("LLM_TIMEOUT_SECONDS", "100")),
        llm_sync_timeout_seconds=float(os.environ.get("LLM_SYNC_TIMEOUT_SECONDS", "50")),
        llm_sync_call_timeout_seconds=float(os.environ.get("LLM_SYNC_CALL_TIMEOUT_SECONDS", "20")),
        llm_call_timeout_seconds=float(os.environ.get("LLM_CALL_TIMEOUT_SECONDS", "45")),
        max_text_chars=int(os.environ.get("MAX_TEXT_CHARS", "50000")),
        max_match_chars=int(os.environ.get("MAX_MATCH_CHARS", "100000")),
        max_attempts=int(os.environ.get("LLM_MAX_ATTEMPTS", "2")),
    )
