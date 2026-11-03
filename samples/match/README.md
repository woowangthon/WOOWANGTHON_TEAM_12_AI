# 매칭 시험 자료

SAMPLE(match-samples): 전부 **지어낸** 사용자 한 명의 경험이다. 실제 사람 · 노트 · 지원 문항이 아니다(공개 레포).

- 채점: `uv run python scripts/eval_match.py` — 숫자의 뜻은 스크립트 머리말
- `experiences.json` — `/match` 요청의 `experiences` 모양 그대로(계약 ETC `docs/contracts/ai-api.md` 「POST /match」). 「바꿔요」 · 알고리즘 스터디는 계약 예시를 넓힌 것, 「모여봄」은 `../team-retro.md` 에서 옮긴 것
- `questions.json` — 문항과 손으로 적은 정답

| 정답 칸 | 뜻 |
| --- | --- |
| `kind` | `match` 맞는 경험이 있다 · `partial` 일부만 받친다 · `none` 맞는 경험이 없다(0개여야 한다) |
| `good` | 첫 추천으로 맞는 경험 |
| `ok` | 추천해도 되는 경험 |
| `bad` | 추천하면 틀린 경험(팀원이 한 일을 「내 역할」 문항에 등) |
| `allowZero` | 0개도 맞음 |

보는 것: 데모 두 문항(`tech-problem` · `team-coordination`)에서 같은 활동의 **다른** 경험이 골라지나(S-25 · 데모 6) · 보완 301 · 321 을 근거로 쓰나 · `TEAM` · `UNKNOWN` 경험을 「내가 한 일」로 말하나 · 맞는 게 없으면 0개인가 · 예산 문항(회장이 짰다)처럼 팀 일로 「직접 한 경험」을 채우지 않나.

정답은 AI 에이전트가 초안을 썼다 — 팀원이 보고 고친다. ID 가 실제 경험을 가리키는지는 `tests/test_eval_match.py` 가 본다.
