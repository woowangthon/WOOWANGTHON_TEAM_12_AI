# 근거 점검 시험 자료

SAMPLE(ground-samples): 전부 **지어낸** 답이다. 경험은 `../match/experiences.json` 을 그대로 쓴다(지어낸 사용자 한 명). 요청엔 경험 전부를 싣는다(제품과 같다).

- 채점: `uv run python scripts/eval_ground.py` — 숫자의 뜻은 스크립트 머리말
- `answers.json` — 답 하나를 문장으로 적고, 문장마다 손으로 적은 정답. 글은 문장을 공백으로 이어 만든다. 번호 목록 답(`list-*`)은 문장마다 `mark`(줄 맨 앞 번호)를 붙여 줄로 잇는다 — `text` 엔 번호가 없다

| 정답 칸 | 뜻 |
| --- | --- |
| `want` | 맞는 판정 — `SOURCE` · `USER_INPUT` · `UNSUPPORTED` · `OPINION` |
| `kind` | `UNSUPPORTED` 인 까닭 — `exaggerate` 자료에 없는 결과 · 평가 · 직책, `mine` 팀 일을 내 일로, `number` 코드가 잡는 숫자, `word-number` 한글 수사, `none` 자료에 없는 일 |
| `evidence` | 받쳐진 문장에 붙어야 할 것 — 이 중 하나는 있어야 맞다 |

정답은 AI 에이전트가 초안을 썼다 — 팀원이 보고 고친다. ID 가 실제 경험의 것인지는 `tests/test_eval_ground.py` 가 본다.
