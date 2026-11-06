# 같은 활동 제안 시험 자료

SAMPLE(same-samples): 전부 **지어낸** 자료 · 활동이다. 실제 사람 · 노트 · 단체가 아니다(공개 레포). `contract-example` 만 계약 예시(회고 → 이력서) 그대로다.

- 채점: `uv run python scripts/eval_same.py` — 숫자의 뜻은 스크립트 머리말
- 목록을 실어도 추출이 나빠지지 않나: `uv run python scripts/eval_extract.py --prompts tuned --existing samples/same/existing-100.json`
- `inputs.json` — 자료마다 손으로 적은 정답

| 칸 | 뜻 |
| --- | --- |
| `request` | `/extract` 요청(`sourceName` · `text`) |
| `textFrom` | 원본을 `samples/` 의 그 파일로 채운다 |
| `existing` | 기존 활동 목록 — `base`(`existing-base.json` 7개) · `long`(기본 7개 + `existing-100.json` 100개) · `none`(`[]`). 없으면 `request.existingActivities` 그대로 |
| `target` | 있으면 `targetActivityId`(S-60 「자료 더하기」) — 자료 id 가 `target-` 로 시작한다 |
| `answers[].keys` | 새 활동을 LLM 활동 이름에서 찾을 이름 조각(원본에 있는 말) |
| `answers[].whole` | `true` 면 이름과 상관없이 LLM 활동 전부가 이 정답이다 — 활동 이름이 없는 자료(`keys` 는 `[]`) |
| `answers[].same` | 정답 — 같은 기존 활동 `id` 또는 `null` |
| `answers[].maybe` | 달아도 틀렸다고 하지 않는 값(`null` 포함 — 해석이 갈린다) |
| `why` | 이 자료로 무엇을 보나 |

`existing-100.json` 은 2019 ~ 2022 년의 무관한 활동 100개(스크립트로 만들어 한 번 저장) — 시험 원본과 겹치지 않는다.

보는 것: ① 이름이 다르게 적힌 같은 활동(줄임말 「멋사」 · 연도만 있는 이력서 항목 · 서비스 이름만) ② 다른 활동을 같다고 하지 않나(같은 이름의 다른 기간 스터디 · 같은 학기 다른 프로젝트 · 다음 회차 해커톤 · 같은 학기 다른 캡스톤 · 글 속 지시) ③ 빈 목록이면 전부 `null` · 100개 목록에서도 잡나 · 시간. ④ 대상이 있을 때(S-60) — 이름 없는 README · 회고 · 메모를 대상에 붙이나 · 활동 여럿인 이력서에서 대상에만 붙이나 · 대상과 다른 자료(같은 학기 비슷한 주제 · 대상 전에 끝난 같은 이름 · 글 속 지시)를 붙이지 않나.

정답을 프롬프트와 같은 AI 에이전트가 썼다 — 팀원이 보고 고친다. 실제 사용자 자료로 잰 숫자가 아니다.
