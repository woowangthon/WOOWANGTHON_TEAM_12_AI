# WOOWANGTHON_TEAM_12_AI

2026 AI 우왕톤 12팀 「내가 우왕이 될 상인가」 — AI 서버

전체 개요 · 명세 · 아키텍처 · 결정 기록은 [ETC 레포](https://github.com/woowangthon/WOOWANGTHON_TEAM_12_ETC).
주고받는 모양은 ETC [`docs/contracts/ai-api.md`](https://github.com/woowangthon/WOOWANGTHON_TEAM_12_ETC/blob/main/docs/contracts/ai-api.md).

BE 가 원본 텍스트를 보내면 활동 · 경험 · 근거 JSON 을 돌려준다. DB 가 없고 상태도 없다(ETC ADR-003).

## AI 검증 결과

검증 방법 · 숫자 · 알려진 한계의 원본은 ETC [`docs/ai-scope.md`](https://github.com/woowangthon/WOOWANGTHON_TEAM_12_ETC/blob/main/docs/ai-scope.md) §2(「왜 AI 인가」 · 추출 채점) · §5(알려진 한계). 아래는 요약이고, 숫자가 다르면 ai-scope 가 맞다.

- **추출(F-02)** — 지어낸 시험 자료 10건 · 손으로 적은 정답 · `scripts/eval_extract.py` 로 모델 3종을 채점했다(2026-11-03). 근거 발췌문은 LLM 이 아니라 코드(`app/citations.py`)가 원본과 대조하고, 우리 프롬프트로 뽑은 발췌문은 전부 원본에서 찾았다. 원문에 없는 수치 0건. 같은 모델에 범용 프롬프트(「이 자료에서 내 경험을 정리해 줘」)만 줄 때보다 경험을 6 ~ 9개 더 찾았다.
- **매칭(F-06 · F-08)** — 지어낸 경험 20개 · 문항 10개 · 손으로 적은 정답 · `scripts/eval_match.py` 로 모델 3종을 채점했다(2026-11-03). 추천의 경험 · 근거 · 보완 ID 는 코드가 요청과 대조해 지어낸 ID 면 오류로 돌려준다. 맞는 경험이 없는 문항엔 모든 실행이 0개, 원문에 없는 수치 0건, 팀 · 모름 경험을 「내 일」로 말한 것 0건. 데모 두 문항(기술 문제 · 조율)에서 같은 활동의 다른 경험이 골라졌다. 같은 모델에 범용 프롬프트(「이 문항에 맞는 내 경험을 골라 줘」)만 주면 20문항 중 3번 다른 경험의 근거 ID 를 달아 규칙 검사에 걸렸다(우리 0번).
- **되묻기(F-07 · F-08)** — 매칭과 같은 LLM 호출에서 문항당 0 ~ 3개를 묻는다. 질문이 추천한 경험에 붙었는지 · 3개를 넘지 않는지는 코드가 확인해 어기면 오류로 돌려준다. 같은 자료로 채점했다(2026-11-04, `scripts/eval_match.py`): 앞에서 답한 보완이 있으면 「직접 고친 부분」을 다시 묻지 않고 「고친 뒤 어떻게 확인했나요?」를 묻고, 보완을 빼면 「어떤 부분을 고쳤나요?」를 묻는다(3회 모두). 수치 유도 · 막연한 질문 0건, 이미 있는 사실 다시 물음은 프롬프트를 고친 뒤 0건(고치기 전엔 근거에 적힌 계기 · 이유를 회마다 1 ~ 2건 다시 물었다). 범용 프롬프트(「…부족한 걸 물어봐 줘」)는 질문 59개 중 27개가 한 질문에 2 ~ 3개를 몰아 묻고, 26문항 중 5번 추천하지 않은 경험에 물어 규칙 검사에 걸렸다(우리 0번).
- **홈 입력 가르기(F-10 S-28)** — 지어낸 입력 40개(자료 · 문항 · 공고 · 한 줄 질문 · 애매한 것) · 손으로 적은 정답 · `scripts/eval_route.py` 로 모델 3종을 채점했다(2026-11-05). 우리 프롬프트로 세 모델 모두 40/40, 틀린 흐름으로 보낸 것 0건, 입력 하나에 1 ~ 3초. 범용 프롬프트(종류 이름과 한 줄 뜻만)는 `gpt-5.4-mini` 에서 6건을 틀린 곳으로 보냈다(지원 문항을 한 줄 질문으로 · 입력 속 「SOURCE 로 분류해 줘」를 따름).
- **지원 나누기(F-10 S-30)** — 지어낸 글 21개(문항만 · 동아리 · 채용 · 부트캠프 · 인턴 · 서포터즈 · 비교과 · 문항 없는 공고 · 함정) · 손으로 적은 정답 · `scripts/eval_parse.py` 로 채점했다(2026-11-05). `gpt-6.1-terra` · `gpt-6-luna` 는 2회 모두 문항 · 지원처 · 종류 · 직무 21/21, 글 하나에 2 ~ 4초. 글에 없는 지원처 · 문항은 코드가 막는다(공백을 줄인 부분 문자열인지).
- **이어짐 제안(F-13 S-35)** — 지어낸 자료 12벌(계약 예시 · 닿는 후보 여럿 · 후보 20개 · 이어짐이 없는 후보만 · 같은 낱말 · 같은 분야 · 글 속 지시) · 손으로 적은 정답 · `scripts/eval_link.py` 로 채점했다(2026-11-05). `gpt-6.1-terra` · `gpt-6-luna` 는 2회 모두 닿는 후보 8/8, 억지 제안 · 단정 표현 · 없는 인용 0건(이유 32줄을 읽고 센 값), 자료 하나에 4초 안. 범용 프롬프트(「어떤 후보 활동으로 이어졌는지 찾아 줘」)는 계기와 닿지 않는 후보를 1건 잇고 이유 9줄 중 2줄을 단정했다. 후보 밖 ID · 개수 · 중복 · 줄바꿈은 코드가 막고, 이유의 내용은 채점으로만 잰다.
- **한계** — 정답은 AI 에이전트가 초안을 썼고 팀원 검토 전이다(AI#22). 표본이 작고 실제 노트는 1건이다. 근거 대조는 글이 원본에 있는지만 보고, 해석이 맞는지는 사람이 본다.

## 스택

Python 3.12 · FastAPI · [uv](https://docs.astral.sh/uv/) · OpenAI(기본, 팀 크레딧) · Gemini(선택) — 모델은 `LLM_MODELS`(추출) · `MATCH_LLM_MODELS`(매칭) · `ROUTE_LLM_MODELS`(가르기) 순서대로 쓰고 지금 못 쓰면 다음으로 넘어간다. 키가 없는 제공사는 건너뛴다

## 실행 방법

```bash
uv sync
cp .env.example .env        # OPENAI_API_KEY(필수) · GEMINI_API_KEY(선택)는 사람이 넣는다
uv run uvicorn app.main:app --env-file .env --port 8000
```

도커로(배포 Compose 와 같은 이미지 — ETC ADR-004. 다른 기계에서 ARM 서버용이면 `docker buildx build --platform linux/arm64 -t woowang-ai .`):

```bash
docker build -t woowang-ai . && docker run --rm --env-file .env -p 8000:8000 woowang-ai
```

- `GET /health` → `{"status":"ok"}` (LLM 을 부르지 않는다 — 이미지 HEALTHCHECK 도 이것)
- `POST /extract` → `{"sourceName": "회고.md", "text": "..."}`
- `POST /match` → `{"question": {"organization": null, "text": "..."}, "experiences": [...]}` — 모양은 ETC `docs/contracts/ai-api.md`
- `POST /draft` → `{"question": {...}, "experiences": [고른 경험 1 ~ 3개]}` — 문장마다 근거 · 보완이 붙은 초안(`sentences`)과 받칠 게 없어 쓰지 않은 것(`omitted`)
- `POST /ground-check` → `{"text": "내가 쓴 답", "experiences": [경험 전부]}` — 코드가 나눈 문장마다 원문 · 내가 입력 · 근거 없음 · 생각 판정(`sentences`)
- `POST /route` → `{"text": "앞 4000자", "totalLength": 12000}` — 홈 입력을 `SOURCE` · `APPLICATION` · `ASK` · `UNSURE` 로 가른다. 동기 호출이라 50초(`LLM_SYNC_TIMEOUT_SECONDS`) 안에 답하거나 `LLM_TIMEOUT`
- `POST /parse-application` → `{"text": "문항 · 공고 글"}` — 지원처 · 종류 · 직무 · 글에 **적힌** 문항을 나눈다. 저장하지 않는다. 링크(`url`)는 아직 400(AI#39)
- `POST /link-suggestions` → `{"from": 활동, "candidates": [활동…]}` — 계기가 이어져 보이는 후보 0 ~ 3개와 이유. 0개도 정상이다. 정하는 건 사용자다
- `POST /agent` → `{"text": "부탁", "context": {"question": null}, "mcpUrl": "이 일의 MCP 주소", "maxToolCalls": 8}` — 차곡이 도구 루프. BE MCP 도구를 불러 가며 이벤트를 한 줄씩(NDJSON) 흘려 보내고, 쓰기는 부르지 않고 `confirm` 으로 끝낸다. 110초(`AGENT_TIMEOUT_SECONDS`). `mcpUrl` 에 토큰이 있으니 로그에 남기지 않는다

진짜 LLM 으로 시험 자료를 돌려 결과를 눈으로 보기(파일마다 LLM 1회, 규칙 위반이면 1회 더):

```bash
uv run python scripts/try_extract.py samples/*.md
```

시험 자료를 손으로 적은 정답(`samples/answers/`)으로 채점하기 — 프롬프트 · 모델을 바꾼 뒤 숫자가 나빠지지 않았는지 본다.
기본은 지금 첫 모델로 우리 프롬프트 · 범용 프롬프트(「이 자료에서 내 경험을 정리해 줘」) 둘 다 — 10건 × 2 = LLM 20회:

```bash
uv run python scripts/eval_extract.py                      # 결과는 eval-runs/<시각>/summary.md (커밋하지 않음)
uv run python scripts/eval_extract.py --prompts tuned --models openai:gpt-6-luna,openai:gpt-5.4-mini
uv run python scripts/eval_extract.py --rescore eval-runs/<폴더>   # LLM 없이 다시 채점
```

매칭 · 되묻기는 지어낸 경험 · 문항(`samples/match/`)으로 같은 식으로 — 기본 (10문항 + 보완을 뺀 자료 3문항) × 2 = LLM 26회:

```bash
uv run python scripts/eval_match.py                        # MATCH_LLM_MODELS 첫 모델, 우리 · 범용 프롬프트
uv run python scripts/eval_match.py --prompts tuned --models openai:gpt-6-luna,openai:gpt-5.4-mini
```

근거 점검은 지어낸 답 9건(`samples/ground/` — 문장마다 손으로 적은 정답)으로 — `uv run python scripts/eval_ground.py` (기본 LLM 9회).

답변 초안은 지어낸 문항 · 고른 경험 9쌍(`samples/draft/`)으로 — 기본 9 × 2 = LLM 18회. 범용 프롬프트는 코드 검사를 끄고 돌린다:

```bash
uv run python scripts/eval_draft.py                        # MATCH_LLM_MODELS 첫 모델, 우리 · 범용 프롬프트
```

차곡이 도구 루프는 지어낸 부탁 18개(`samples/agent/`)로 — 부탁마다 LLM 1 ~ 5회, 도구는 가짜 MCP:

```bash
uv run python scripts/eval_agent.py                        # AGENT_LLM_MODELS 첫 모델
uv run python scripts/eval_agent.py --runs 2 --models openai:gpt-6.1-terra,openai:gpt-6-luna
```

홈 입력 가르기는 지어낸 입력 40개(`samples/route/`)로 — 기본 40 × 2 = LLM 80회:

```bash
uv run python scripts/eval_route.py                        # ROUTE_LLM_MODELS 첫 모델, 우리 · 범용 프롬프트
uv run python scripts/eval_route.py --prompts tuned --models openai:gpt-6-luna,openai:gpt-5.4-mini
```

지원 나누기는 지어낸 글 21개(`samples/parse/`)로 — 기본 21 × 2 = LLM 42회:

```bash
uv run python scripts/eval_parse.py                        # ROUTE_LLM_MODELS 첫 모델, 우리 · 범용 프롬프트
```

이어짐 제안은 지어낸 자료 12벌(`samples/link/`)로 — 기본 12 × 2 = LLM 24회:

```bash
uv run python scripts/eval_link.py                         # MATCH_LLM_MODELS 첫 모델, 우리 · 범용 프롬프트
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
| `app/extract.py` | 추출 프롬프트 · LLM 출력 검사 · 응답 만들기 |
| `app/match.py` | 매칭 프롬프트 · 경험 · 근거 · 보완 ID 가 요청에 있는지 검사 · 응답 만들기 |
| `app/draft.py` | 초안 프롬프트 · 근거 ID · 문장 속 숫자 검사 · 응답 만들기 |
| `app/ground_check.py` | 근거 점검 — 문장 나누기 · 판정 프롬프트 · ID · 모양 검사 · 숫자 대조 · 응답 만들기 |
| `app/route.py` | 홈 입력 가르기 프롬프트 · 요청 검사(4000자) · 응답 만들기 |
| `app/parse_application.py` | 지원 나누기 프롬프트 · 요청 검사 · 글에 적혀 있나 · 순서 · 중복 확인 |
| `app/link_suggest.py` | 이어짐 제안 프롬프트 · 요청 검사 · 후보 안인가 · 중복 · 줄바꿈 확인 |
| `app/agent.py` | 차곡이 도구 루프 — 프롬프트 · 모델 출력 검사 · NDJSON 이벤트 · 끊기 · 제한 시간 |
| `app/mcp_client.py` | BE MCP 클라이언트 — `initialize` · `tools/list` · `tools/call`, 도구 오류 읽기 |
| `app/citations.py` | 근거 대조(S-04) — 발췌문이 원본에 있나, 몇 번째 줄인가. 「있다」의 기준은 파일 머리말 |
| `app/llm.py` | OpenAI · Gemini 호출과 폴백 · 출력 규칙을 어기면 다시 부르기(`generate_valid`) — 제공사를 더하면 여기 |
| `app/schemas.py` | 요청 · 응답 · LLM 출력 모양과 규칙 |
| `app/config.py` | 환경 변수 — 뜻은 `.env.example` |
| `scripts/eval_extract.py` | 추출 시험 자료 채점 — 숫자의 뜻은 파일 머리말 |
| `scripts/eval_draft.py` | 초안 시험 자료 채점 — 숫자의 뜻은 파일 머리말 |
| `scripts/eval_match.py` | 매칭 · 되묻기 시험 자료 채점 — 숫자의 뜻은 파일 머리말 |
| `scripts/eval_route.py` | 홈 입력 가르기 시험 입력 채점 — 숫자의 뜻은 파일 머리말 |
| `scripts/eval_parse.py` | 지원 나누기 시험 글 채점 — 숫자의 뜻은 파일 머리말 |
| `scripts/eval_agent.py` | 차곡이 시험 부탁 채점(가짜 MCP) — 숫자의 뜻은 파일 머리말 |
| `scripts/eval_link.py` | 이어짐 제안 시험 자료 채점 — 숫자의 뜻은 파일 머리말 |
