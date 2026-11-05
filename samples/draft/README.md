# 초안 시험 자료

SAMPLE(draft-samples): 전부 **지어낸** 문항 · 경험이다. 경험은 `../match/experiences.json` 을 그대로 쓴다(지어낸 사용자 한 명).

- 채점: `uv run python scripts/eval_draft.py` — 숫자의 뜻은 스크립트 머리말
- `pairs.json` — 문항 하나 + 고른 경험, 손으로 적은 정답
- `experiences.json` — 근거 · 보완이 풍부한 지어낸 경험 151 · 152(152 는 결과 칸이 비어 있다). 근거는 일부러 글의 순서와 다르게 놓았다 — 시간순이면 입력을 따라 쓰기만 해도 구성 순서를 맞힌다. 매칭 채점 숫자가 흔들리지 않게 `../match/experiences.json` 에 넣지 않고 따로 둔다

| 정답 칸 | 뜻 |
| --- | --- |
| `experienceIds` | 고른 경험(1 ~ 3개, `/draft` 요청에 이것만 싣는다) |
| `missing[].fact` | 문항에 필요한데 고른 경험의 근거 · 보완에 없는 사실 |
| `missing[].pattern` | 문장이 그 사실을 지어냈는지 거르는 정규식(걸린 것은 후보 — 사람이 읽는다) |
| `missing[].omitPattern` | `omitted` 에 그 사실을 적었는지 거르는 정규식 |
| `missing[].mustOmit` | 문항이 대놓고 묻는 것이라 `omitted` 에 나와야 맞는 사실 |
| `beats[]` | 글이 이 순서로 나와야 맞는 사실의 정규식 — 처음 나온 문장 위치로 본다(근거가 풍부한 쌍만) |
| `once[]` | 한 문장에만 나와야 맞는 사실의 정규식 — 두 문장 이상에 나오면 되풀이(후보) |
| `expect[]` | 문장이 `pattern` 의 사실을 쓰면 `evidence` 중 하나를 붙여야 맞다(「직접 고친 부분」은 보완 301) |

정답은 AI 에이전트가 초안을 썼다 — 팀원이 보고 고친다. ID 가 실제 경험의 것인지는 `tests/test_eval_draft.py` 가 본다.
