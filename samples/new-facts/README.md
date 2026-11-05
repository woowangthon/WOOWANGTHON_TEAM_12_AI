# 새 경험 후보 시험 자료

SAMPLE(new-facts-samples): 전부 **지어낸** 확정 답이다. 경험은 `../match/experiences.json` 을 그대로 쓴다(지어낸 사용자 한 명). 활동 `id` 는 `activities.json` — 경험의 활동 이름과 같은 이름에 붙인 지어낸 번호다. 요청엔 경험 · 활동 전부를 싣는다(제품과 같다).

- 채점: `uv run python scripts/eval_new_facts.py` — 숫자의 뜻은 스크립트 머리말
- `answers.json` — 확정 답 하나를 문장으로 적고, 문장마다 손으로 적은 정답. 글은 문장을 공백으로 이어 만든다. `"excluded": true` 인 문장은 요청의 `excludedSentences`(저장된 초안 문장)로 싣는다

| 정답 칸 | 뜻 |
| --- | --- |
| `want` | `NEW` 후보로 골라야 한다 · `SKIP` 고르면 안 된다 |
| `kind` | `NEW` — `new` 자료에 없는 지난 일 · 결과, `mixed` 받치는 사실 + 더한 결과(더한 부분만 옮겨야 한다). `SKIP` — `draft` 초안 복사(코드가 뺀다), `backed` 자료가 이미 받침, `opinion` 생각 · 포부, `puff` 자기 칭찬 · 포장, `mine` 팀 · 모르는 사람의 일을 「제가」로, `number` 자료와 다른 수치 |
| `actor` · `activityId` | `NEW` 의 맞는 수행 주체(둘 다 맞으면 목록) · 붙일 활동(없으면 `null`) |
| `notIn` | `mixed` 의 `content` 에 다시 나오면 안 되는 받치던 사실의 낱말 |

정답은 AI 에이전트가 초안을 썼다 — 팀원이 보고 고친다. 문장 나누기 · 활동 이름이 맞는지는 `tests/test_eval_new_facts.py` 가 본다.
