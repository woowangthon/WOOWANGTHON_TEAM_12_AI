# 초안 시험 자료

SAMPLE(draft-samples): 전부 **지어낸** 문항 · 경험이다. 경험은 `../match/experiences.json` 을 그대로 쓴다(지어낸 사용자 한 명).

- 채점: `uv run python scripts/eval_draft.py` — 숫자의 뜻은 스크립트 머리말
- `pairs.json` — 문항 하나 + 고른 경험, 손으로 적은 정답
- `clarify.json` — 경험 151 · 152 에 이미 있는 사실과 그걸 묻는 질문을 거르는 정규식. 나머지 경험은 `../match/clarify.json` 을 그대로 쓴다
- `guided.json` — 기본 쌍(`base`)에 구성 틀(`frame`) · 작성 지침(`guidelines`)을 얹은 요청(S-49 · S-50). 채점은 `uv run python scripts/eval_draft_guided.py` 가 기본 쌍과 나란히 한다. 지침의 `expect: "not"` 은 근거와 부딪치거나(「수치로」 · 「리더였다고」) 서로 부딪쳐 `applied: false` 가 맞는 것, `frameBeats` 는 틀의 단계 순서대로 나와야 맞는 사실의 정규식 — 「첫 문장에 무엇을 했고 어떻게 됐는지」 지침(901)이 만드는 요약 첫 문장에 걸리지 않는 낱말로 적어 본문의 순서만 본다
- `cards.json` — 기본 쌍(`base`)에 구조 카드(`frame` — contracts README 「구성 틀」 제안 표의 카드, S-82)를 얹은 요청 8쌍. 채점은 `uv run python scripts/eval_frame_cards.py` 가 같은 쌍에서 `frame` 만 뺀 요청과 나란히 한다(나열 · 경험 전환 · 줄기 비중 — 숫자의 뜻은 스크립트 머리말)
- `experiences.json` — 근거 · 보완이 풍부한 지어낸 경험 151 · 152(152 는 결과 칸이 비어 있다). 근거는 일부러 글의 순서와 다르게 놓았다 — 시간순이면 입력을 따라 쓰기만 해도 구성 순서를 맞힌다. 매칭 채점 숫자가 흔들리지 않게 `../match/experiences.json` 에 넣지 않고 따로 둔다. 161 ~ 165 는 한 활동(스프링 읽기 스터디)의 한 줄 색인 경험이고 163 · 165 는 근거 원문이 수동형(「발표가 진행됐다」 · 「회의록은 … 전달됐다」)인 TEAM 경험이다 — ADR-014 배경의 「활동마다 한 줄씩 · 수동형」 초안을 재현하는 쌍 `study-lines` · `study-lines-all`(AI#87)

| 정답 칸 | 뜻 |
| --- | --- |
| `experienceIds` | 고른 경험(1 ~ 3개, `/draft` 요청에 이것만 싣는다) |
| `missing[].fact` | 문항에 필요한데 고른 경험의 근거 · 보완에 없는 사실 |
| `missing[].pattern` | 문장이 그 사실을 지어냈는지 거르는 정규식(걸린 것은 후보 — 사람이 읽는다) |
| `missing[].omitPattern` | `omitted` 에 그 사실을 적었는지 거르는 정규식 |
| `missing[].mustOmit` | 문항이 대놓고 묻는 것이라 `omitted` 에 나와야 맞는 사실 |
| `missing[].noQuestion` | 경험에 붙지 않는 빈칸 — `experienceId` · `question` 이 `null` 이어야 맞다(지원 동기 · 고른 경험에서 썼는지 모르는 기술) |
| `beats[]` | 글이 이 순서로 나와야 맞는 사실의 정규식 — 처음 나온 문장 위치로 본다(근거가 풍부한 쌍만) |
| `once[]` | 한 문장에만 나와야 맞는 사실의 정규식 — 두 문장 이상에 나오면 되풀이(후보) |
| `expect[]` | 문장이 `pattern` 의 사실을 쓰면 `evidence` 중 하나를 붙여야 맞다(「직접 고친 부분」은 보완 301) |
| `jobRole` · `requirements[]` · `hints` | 매칭 결과 · 직무를 선택 칸으로 싣는 쌍(ETC#141). `hints` 는 경험 ID → `reason` · `emphasis` · `gaps`. 근거가 아니다 — 강조점에만 있는 판단 · 숫자 · 직무 경력은 `missing` 에 적는다 |
| `lead` · `leadWithin` | 강조점의 받칠 수 있는 사실 정규식 — FACT 앞 `leadWithin`(기본 2)문장 안에 나와야 「강조 앞」. 같은 경험 151 을 `rich-problem` · `emph-*` 로 견준다 |
| `staleGaps[]` | 매칭 때 `gaps` 에 있었지만 지금 보완이 채운 사실 — `omitted` 에 다시 적으면 「지난 빈칸 또 물음」 |
| `addSupplements[]` | 빠진 사실에 답한 뒤 다시 만드는 쌍 — 그 경험의 보완에 더해 부른다. `pattern` 은 문장이 그 답을 쓴 것, `omitPattern` 은 `omitted` 에 다시 적은 것을 거른다 |

정답은 AI 에이전트가 초안을 썼다 — 팀원이 보고 고친다. ID 가 실제 경험의 것인지는 `tests/test_eval_draft.py` 가 본다.
