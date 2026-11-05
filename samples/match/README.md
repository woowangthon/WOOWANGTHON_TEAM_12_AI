# 매칭 시험 자료

SAMPLE(match-samples): 전부 **지어낸** 사용자 한 명의 경험이다. 실제 사람 · 노트 · 지원 문항이 아니다(공개 레포).

- 채점: `uv run python scripts/eval_match.py` — 숫자의 뜻은 스크립트 머리말
- `experiences.json` — `/match` 요청의 `experiences` 모양 그대로(계약 ETC `docs/contracts/ai-api.md` 「POST /match」). 「바꿔요」 · 알고리즘 스터디는 계약 예시를 넓힌 것, 「모여봄」은 `../team-retro.md` 에서 옮긴 것
- `questions.json` — 문항과 손으로 적은 정답
- `question-types.json` — 문항 유형(S-41)만 채점하는 문항. 「자주 나오는 문항」(FE `src/lib/common-questions.ts`) 꼴을 본떠 **지어낸** 문항이다(그 파일은 실제 지원서 문항이라 옮기지 않았다). `questions.json` 과 합쳐 `OTHER` 를 뺀 10 유형이 다 들어 있다
- `clarify.json` — 되묻기 채점용. 경험마다 근거 · 보완에 **이미 있는 사실**과 그걸 묻는 질문을 거르는 정규식(걸린 것은 후보 — 사람이 읽는다). 보완 사실의 `askIfMissing` 은 보완을 뺀 자료에서 그 빈칸을 물어야 맞는 문항(데모 4)

| 정답 칸 | 뜻 |
| --- | --- |
| `kind` | `match` 맞는 경험이 있다 · `partial` 일부만 받친다 · `none` 맞는 경험이 없다(0개여야 한다) |
| `good` | 첫 추천으로 맞는 경험 |
| `ok` | 추천해도 되는 경험 |
| `bad` | 추천하면 틀린 경험(팀원이 한 일을 「내 역할」 문항에 등) |
| `allowZero` | 0개도 맞음 |
| `asksMyRole` | 문항이 내 역할 · 내 행동을 본다 — `TEAM` · `UNKNOWN` 경험을 추천하면 내 몫을 물어야 맞다(S-20) |
| `questionType` | 문항이 **주로** 묻는 유형 하나(계약 README `questionType`). 겹치는 문항(「실패와 그 뒤 바꾼 것」)은 문항이 경험을 고르게 하는 기준 쪽 |
| `questionTypeOk` | 겹치는 문항이라 그렇게 붙여도 틀렸다고 하기 어려운 유형 — 「허용 포함」으로 따로 센다 |

보는 것: 데모 두 문항(`tech-problem` · `team-coordination`)에서 같은 활동의 **다른** 경험이 골라지나(S-25 · 데모 6) · 보완 301 · 321 을 근거로 쓰나 · `TEAM` · `UNKNOWN` 경험을 「내가 한 일」로 말하나 · 맞는 게 없으면 0개인가 · 예산 문항(회장이 짰다)처럼 팀 일로 「직접 한 경험」을 채우지 않나. 되묻기: 보완 301 이 있으면 「직접 고친 부분」을 다시 묻지 않고, 빼면 묻나(데모 4 · 6) · 근거에 적힌 계기를 다시 묻지 않나 · 한 질문에 하나만 묻나.

정답은 AI 에이전트가 초안을 썼다 — 팀원이 보고 고친다. ID 가 실제 경험을 가리키는지는 `tests/test_eval_match.py` 가 본다.
