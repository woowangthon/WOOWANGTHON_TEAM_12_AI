# WOOWANGTHON_TEAM_12_AI

2026 AI 우왕톤 12팀 「내가 우왕이 될 상인가」 — AI 서버

전체 개요 · 명세 · 아키텍처 · 결정 기록은 [ETC 레포](https://github.com/woowangthon/WOOWANGTHON_TEAM_12_ETC).
주고받는 모양은 ETC [`docs/contracts/ai-api.md`](https://github.com/woowangthon/WOOWANGTHON_TEAM_12_ETC/blob/main/docs/contracts/ai-api.md).

BE 가 원본 텍스트를 보내면 활동 · 경험 · 근거 JSON 을 돌려준다. DB 가 없고 상태도 없다(ETC ADR-003).

## 스택

Python 3.12 · FastAPI · [uv](https://docs.astral.sh/uv/) · OpenAI(기본, 팀 크레딧) · Gemini(선택) — 모델은 `LLM_MODELS` 순서대로 쓰고 지금 못 쓰면 다음으로 넘어간다. 키가 없는 제공사는 건너뛴다

## 실행 방법

```bash
uv sync
cp .env.example .env        # OPENAI_API_KEY(필수) · GEMINI_API_KEY(선택)는 사람이 넣는다
uv run uvicorn app.main:app --env-file .env --port 8000
```

- `GET /health` → `{"status":"ok"}` (LLM 을 부르지 않는다)
- `POST /extract` → `{"sourceName": "회고.md", "text": "..."}`

진짜 LLM 으로 시험 자료를 돌려 결과를 눈으로 보기(파일마다 LLM 1회, 규칙 위반이면 1회 더):

```bash
uv run python scripts/try_extract.py samples/*.md
```

## 검증

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest -q
```

테스트는 LLM 을 가짜로 바꿔 계약 모양 · 오류 처리만 본다(키 · 네트워크 없이 돈다). 추출 품질은 `samples/` 로 손으로 본다 — [`samples/README.md`](samples/README.md).

## 코드 지도

| 파일 | 하는 일 |
| --- | --- |
| `app/main.py` | 엔드포인트 · 오류 응답 모양 |
| `app/extract.py` | 프롬프트 · LLM 출력 검사 · 다시 시도 · 응답 만들기 |
| `app/llm.py` | OpenAI · Gemini 호출과 폴백 — 제공사를 더하면 여기 |
| `app/schemas.py` | 요청 · 응답 · LLM 출력 모양과 규칙 |
| `app/config.py` | 환경 변수 — 뜻은 `.env.example` |
