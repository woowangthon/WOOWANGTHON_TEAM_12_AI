# 공고 기준 경험 찾기 시험 자료(`/posting-match`, S-93)

SAMPLE(posting-match-samples): 공고 · 경험 전부 **지어낸** 것이다. 실제 공고 · 사람이 아니다(공개 레포).

- 채점: `uv run python scripts/eval_posting_match.py` — 숫자의 뜻은 스크립트 머리말
- 경험 더미 2종(AI#122 완료 조건)
  - `dev` 개발 경험만 — `../match/experiences.json` 중 개발 활동(바꿔요 · 알고리즘 스터디 · 모여봄 · 가계부 웹앱)
  - `mixed` 섞임 — `../match/experiences.json` 전부(카페 아르바이트 · 학생회 포함) + `extra-experiences.json`(연합 동아리 홍보팀)
- `postings.json` — 백엔드 · 마케팅 공고 각 1건(지어낸 줄). 줄마다 더미별 사람 판정

| 칸 | 뜻 |
| --- | --- |
| `dev` · `mixed` 의 `expect` | 사람이 본 판정 하나(`SUPPORTED` · `PARTIAL` · `NOT_FOUND`) |
| `dev` · `mixed` 의 `ok` | 그렇게 판정해도 틀렸다고 하기 어려운 판정 — 「허용 포함」으로 따로 센다 |
| `bad` | 그 줄의 근거로 고르면 틀린 경험(다른 분야 · 그 일과 상관없는 경험). 더미에 없는 ID 는 건너뛴다 |

보는 것: 개발 경험만 있는 사람의 마케팅 공고에서 `NOT_FOUND` 가 제대로 나오나(다른 서비스처럼 개발 경험을 마케팅 줄에 억지로 맞추지 않나 — ETC#323) · 그래도 분야와 상관없는 태도 줄(2005 일정 조율)은 받치나 · 팀원이 한 일(103 AWS · 162 포스터)을 `SUPPORTED` 로 올리지 않나 · 되묻기가 공고의 조건을 겪었다고 전제하지 않나.

정답은 AI 에이전트가 초안을 썼다 — 팀원이 보고 고친다. ID 가 실제 경험을 가리키는지는 `tests/test_eval_posting_match.py` 가 본다.
