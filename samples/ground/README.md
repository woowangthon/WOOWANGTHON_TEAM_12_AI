# 근거 점검 시험 자료

SAMPLE(ground-samples): 전부 **지어낸** 답이다. 경험은 `../match/experiences.json` 에 `experiences.json` 을 더해 쓴다(지어낸 사용자 한 명). 요청엔 경험 전부를 싣는다(제품과 같다).

- `experiences.json` — 점검에만 쓰는 경험. 되묻기에 **짧게 답한** 보완(「힙 덤프 떠 봄」)과 질문을 받아 주지 않은 답(「공유는 못 했다」)이 있다 — 질문과 이어 읽어야 받쳐지는 문장, 질문에만 있는 일을 쓴 문장을 가른다(`supp-short` · AI#100). 다른 채점(매칭 · 초안 …)에 섞이지 않게 따로 뒀다

- 채점: `uv run python scripts/eval_ground.py` — 숫자의 뜻은 스크립트 머리말
- `answers.json` — 답 하나를 문장으로 적고, 문장마다 손으로 적은 정답. 글은 문장을 공백으로 이어 만든다. 번호 목록 답(`list-*`)은 문장마다 `mark`(줄 맨 앞 번호)를 붙여 줄로 잇는다 — `text` 엔 번호가 없다

| 정답 칸 | 뜻 |
| --- | --- |
| `want` | 맞는 판정 — `SOURCE` · `USER_INPUT` · `UNSUPPORTED` · `OPINION` |
| `kind` | `UNSUPPORTED` 인 까닭 — `exaggerate` 자료에 없는 결과 · 평가 · 직책, `mine` 팀 일을 내 일로, `number` 코드가 잡는 숫자, `word-number` 한글 수사, `none` 자료에 없는 일 |
| `evidence` | 받쳐진 문장에 붙어야 할 것 — 이 중 하나는 있어야 맞다 |

정답은 AI 에이전트가 초안을 썼다 — 팀원이 보고 고친다. ID 가 실제 경험의 것인지는 `tests/test_eval_ground.py` 가 본다.
