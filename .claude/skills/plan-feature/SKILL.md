---
name: plan-feature
description: SPEC 항목(R · F)을 레포별 GitHub Issue 로 쪼갠다. 「F-02 이슈로 나눠 줘」 「R-01 계획 세워 줘」 같은 요청에 쓴다.
---

# plan-feature

SPEC 항목 하나를 **한 레포 · 한 PR** 크기의 이슈들로 나눈다.

1. `../WOOWANGTHON_TEAM_12_ETC/docs/SPEC.md` 의 해당 절과 `docs/contracts/` 를 읽는다. 열린 이슈와 겹치는지 본다.

   ```bash
   for r in ETC FE BE AI; do gh issue list -R woowangthon/WOOWANGTHON_TEAM_12_$r --state open; done
   ```

2. 레포별로 나눈다. 한 이슈 = 한 레포 · 한 PR.
   - FE · BE · AI 에 걸치면 ETC 에 부모 이슈(`epic`)를 두고 각 레포에 하위 이슈.
3. 순서: **계약 먼저.** 레포 사이 API · JSON 이 필요하면 ETC 에 계약 이슈를 맨 앞에 두고, 각 레포 이슈는 계약 이후 병렬(계약 전엔 목으로 시작 가능).
4. 이슈마다 템플릿(`.github/ISSUE_TEMPLATE/feature.yml`) 칸을 채운다: 제목 · SPEC 참조 · **목적**(무엇을 왜 — 지금 상태 · 사용자에게 어떻게 보이나 · 기획 의도) · 읽을 것(문서는 절까지) · 완료 조건 · 범위 제외 · 위험 · 미결정(갈림길엔 기본값) · 선행 이슈 · 라벨(종류 + `P0`/`P1`/`P2`) · Milestone(지금 스프린트).
   - **담당은 비운다** — 시작하는 사람이 건다(ADR-002).
   - 누구의 에이전트든 **본문만 읽고 시작할 수 있게** 쓴다. Notion 내용은 필요한 만큼 본문에 옮기고 링크는 덧붙이기만.
5. **바로 만든다** — 묻지 않는다. 이슈는 그때그때 만들고, 틀리면 고치거나 닫는다.
6. 만든 뒤 연결한다. 본문에 「선행: #3」이라고만 쓰면 GitHub 이 모른다.

   ```bash
   # 하위 이슈: 부모 이슈에 자식을 붙인다 (자식은 다른 레포여도 된다)
   CHILD=$(gh api repos/woowangthon/<자식 레포>/issues/<자식번호> --jq .id)
   gh api --method POST repos/woowangthon/<부모 레포>/issues/<부모번호>/sub_issues -F sub_issue_id="$CHILD"

   # 선행: <번호> 는 <선행번호> 에 막혀 있다
   PRE=$(gh api repos/woowangthon/<선행 레포>/issues/<선행번호> --jq .id)
   gh api --method POST repos/woowangthon/<레포>/issues/<번호>/dependencies/blocked_by -F issue_id="$PRE"
   ```

7. 만든 이슈 목록(레포 · 번호 · 제목 · 선행)을 한 번에 보고한다.
