# 시험 자료

SAMPLE(extract-samples): 전부 **지어낸** 자료다. 실제 사람 · 노트 · 자소서가 아니다(공개 레포). 실제 노트로는 로컬에서만 돌려 본다.

- 눈으로 보기: `uv run python scripts/try_extract.py samples/<파일>.md`
- 채점: `uv run python scripts/eval_extract.py` — 정답은 `answers/<같은 이름>.json`, 숫자의 뜻은 스크립트 머리말
- 매칭(`/match`) 시험 자료는 [`match/`](match/README.md) · 홈 입력 가르기(`/route`)는 [`route/`](route/README.md) · 지원 나누기(`/parse-application`)는 [`parse/`](parse/README.md) · 이어짐 제안(`/link-suggestions`)은 [`link/`](link/README.md) · 같은 활동 제안(`/extract` 의 `sameAsActivityId`)은 [`same/`](same/README.md)

| 파일 | 사용자 | 보는 것 |
| --- | --- | --- |
| `mixed-notes.md` | 대충 정리하는 사람 — 목록형 노트 | S-03 활동 4개로 나뉘나 · 기간이 원문만큼만(「2023년 2학기」→ `2023`) · 진행 중 스터디가 `ongoing` |
| `team-retro.md` | 팀 회고를 쓴 사람 | S-06 팀원 · 「우리는」이 한 일이 `ME` 가 아니나 · 주어 없는 문장이 `UNKNOWN` 이나 |
| `readme-no-numbers.md` | 개인 프로젝트 README | S-05 경험에 수치 · 성과(「N% 개선」「사용자 N명」)가 생기지 않나 |
| `organized-obsidian.md` | 잘 정리하는 사람 — 옵시디언처럼 활동별 노트 | 수치가 있는 문장은 원문 수치 그대로 · 「부장이 제안해서」 옮긴 일을 `ME` 로 올리지 않나 · 「배포는 사수가」 |
| `team-readme.md` | 정리 안 하는 사람 — 팀 프로젝트 README 만 | 기능 목록 · 팀원 표를 내 일(`ME`)로 만들지 않나 · 실행 명령을 경험으로 뽑지 않나 |
| `commit-memo.md` | 정리 안 하는 사람 — 커밋 메모만 | 혼자 한 프로젝트라 `ME` · 「방문자 거의 없음」에 수치를 붙이지 않나 |
| `meeting-notes.md` | 정리 안 하는 사람 — 노션 회의록만 | 이름이 나온 사람의 제안 · 담당은 `TEAM` · 안 한 할 일(`[ ]`)을 한 일처럼 쓰지 않나 |
| `sparse-memo.md` | 드문드문 기록하는 사람 — 몇 줄 · 날짜 없음 | 활동 3개 · 기간 `null` · 「마감이 빨라짐」「성적 올랐다」에 수치를 붙이지 않나 |
| `resume-cover-letter.md` | 지원서를 써 본 사람 — 이력서 · 자소서 | 과장 문장(「누구보다 헌신하는」)을 경험으로 뽑지 않나 · 「크게 빨라졌습니다」에 없는 수치를 더하지 않나 · `2024.03` → `2024-03` |
| `obsidian-moc.md` | 잘 정리하는 사람 — 옵시디언 MOC(활동 이력이 `[[링크]]` 목록) | 링크만 있는 활동도 경험 0개로 남나(S-03) · 노트 제목 · 파일 이름(「…회고」「수료증.pdf」)을 경험으로 만들지 않나 · 공부 노트 · 스크랩 목록을 활동으로 만들지 않나 |
| `category-list.md` | 이력서 활동 줄만 있는 사람 — 이름 · 기간만 | S-57 경험 0개 활동도 이름으로 분류하나 · 공모전 · 아르바이트 → `PROJECT`, 자격증 · 어학 → `STUDY`, 봉사 · 멘토링 → `CLUB` · 무엇인지 모를 이름(「둥지」)은 억지로 고르지 않고 `null` 이나 |

정답(`answers/*.json`)은 원문을 읽고 손으로 적는다(처음 10건은 AI 에이전트 초안 — 팀원이 보고 고친다). 경험은 원문 구절(`anchor`) · 맞는 수행 주체(`actor`, 판단이 갈리면 여럿)로,
있어도 없어도 되는 것은 `optional`, 경험으로 뽑으면 안 되는 것은 `notExperiences` 로 적는다.
활동은 원문에 있는 이름 조각(`keys`)으로 찾는다. 이름 · 기간만 있는 활동은 `"experiences": []`, 활동으로 만들면 안 되는 것(공부 노트 · 스크랩)은 `notActivities`. 활동 분류는 `category` — 맞는 값 목록(판단이 갈리면 여럿, 가늠이 안 되면 `null` 도 값). 구절이 원본에 있는지는 `tests/test_eval_answers.py` 가 본다.
