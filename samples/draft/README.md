# 초안 시험 자료

SAMPLE(draft-samples): 전부 **지어낸** 문항 · 경험이다. 경험은 `../match/experiences.json` 을 그대로 쓴다(지어낸 사용자 한 명).

- 채점: `uv run python scripts/eval_draft.py` — 숫자의 뜻은 스크립트 머리말
- `pairs.json` — 문항 하나 + 고른 경험, 손으로 적은 정답

| 정답 칸 | 뜻 |
| --- | --- |
| `experienceIds` | 고른 경험(1 ~ 3개, `/draft` 요청에 이것만 싣는다) |
| `missing[].fact` | 문항에 필요한데 고른 경험의 근거 · 보완에 없는 사실 |
| `missing[].pattern` | 문장이 그 사실을 지어냈는지 거르는 정규식(걸린 것은 후보 — 사람이 읽는다) |
| `missing[].omitPattern` | `omitted` 에 그 사실을 적었는지 거르는 정규식 |
| `missing[].mustOmit` | 문항이 대놓고 묻는 것이라 `omitted` 에 나와야 맞는 사실 |
| `expect[]` | 문장이 `pattern` 의 사실을 쓰면 `evidence` 중 하나를 붙여야 맞다(「직접 고친 부분」은 보완 301) |

정답은 AI 에이전트가 초안을 썼다 — 팀원이 보고 고친다. ID 가 실제 경험의 것인지는 `tests/test_eval_draft.py` 가 본다.
