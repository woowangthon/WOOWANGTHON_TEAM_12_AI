# 이미 낸 되묻기 · 「딱히 없음」 보완 시험 자료(AI#138 · ETC#354)

SAMPLE(asked-samples): 경험 · 근거 · 보완 · 문항 전부 **지어낸** 것이다. 실제 사람 · 기록이 아니다(공개 레포).

- 채점: `uv run python scripts/eval_asked.py --runs 2` — 전(이 PR 전 동작) · 후를 같은 회차로. 숫자의 뜻은 스크립트 머리말
- `cases.json`
  - `experiences` — 501(대관 일정 조율, TEAM) · 502(같은 일 + 「맡은 일」에 「딱히 없음」) · 503(카페 주문 순서, ME + 「어떻게 확인」에 「기억이 안 나요」) · 504(스터디 문제 고르기, ME + 「왜」에 「딱히 이유는 없었어요」)
  - `cases` — `dup-*` 는 ETC#354 의 중복 쌍 그대로(다른 문항에서 「…조율하면서 맡은 일」을 이미 물음), `none-*` 는 /match, `draft-none-*` 는 목표 글자 수가 큰 /draft

| 칸 | 뜻 |
| --- | --- |
| `endpoint` | `match` 또는 `draft` |
| `experienceIds` | 요청에 싣는 경험 |
| `asked` | 경험 ID → `askedQuestions`(후에만 싣는다) |
| `target` | 채점하는 경험 |
| `reask` | 다시 물었는지 거르는 정규식(되묻기 · omitted 질문) |
| `claim` | 「없다」고 답한 사실을 있었다고 썼는지 거르는 정규식(추천 이유 · 강조점, 초안 문장) |
