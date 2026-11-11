# 문항별 소재 규칙 시험 자료

SAMPLE(match-materials): AI#124 채점용으로 전부 지어낸 경험 · 문항이다. 경험MAP 양식의 문구나 개인 작성본을 옮기지 않았다. 제품 동작에는 쓰지 않는다.

`uv run python scripts/eval_match.py --data samples/match-materials --variants with --prompts tuned --runs 2`

실패 · 협업 · 갈등 각 2문항. `pool`은 각 문항에 보낼 경험 ID로, 서로 다른 상황의 사실이 섞이지 않게 한다. `eventualSuccess`는 실패 뒤 결국 성공한 경험 ID다. 실패로 끝난 경험이 함께 있을 때 그 성공 경험이 1순위에 오는 수를 센다. `materialAskPatterns`는 경험 ID별로 기대한 질문(실패에서 배운 것 · 업무 역할 · 상대 입장이나 양쪽 대안)의 정규식이다. 일치 비율은 낱말로 거른 후보이므로 실제 질문을 함께 읽는다. 질문을 안 내서 비율이 높아 보이지 않도록 「소재 질문 있는 문항」도 함께 센다.

전후는 `--system-prompt-file`로 변경 전 프롬프트를 지정하고 같은 자료 · 같은 모델 · 2회씩 돌린다. 기본 `samples/match`의 전체 매칭 · 보완 없음 · 유형 채점도 따로 전후 2회씩 돌린다. 정답 초안은 AI 에이전트가 작성했고 팀원 검토 전이다.
