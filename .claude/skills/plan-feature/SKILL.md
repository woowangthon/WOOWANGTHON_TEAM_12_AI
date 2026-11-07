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
   - **선행은 기술 의존만** — 그 이슈가 끝나야 코드 · 계약 · 데이터상 시작 · 완료할 수 없을 때. 「`P0` 가 아직」 · 「다음 스프린트」 · 담당자 일정은 선행이 아니다 — 라벨(`P0` ~ `P3`) · Milestone 으로만 나타낸다(ADR-009).
4. 이슈마다 템플릿(`.github/ISSUE_TEMPLATE/feature.yml`) 칸을 채운다: 제목 · SPEC 참조 · **목적**(무엇을 왜 — 지금 상태 · 사용자에게 어떻게 보이나 · 기획 의도) · 읽을 것(문서는 절까지) · 완료 조건 · 범위 제외 · 위험 · 미결정(갈림길엔 기본값) · 선행 이슈 · 라벨(종류 + `P0` ~ `P3`) · Milestone(지금 스프린트).
   - **담당은 비운다** — 시작하는 사람이 건다(ADR-002).
   - 누구의 에이전트든 **본문만 읽고 시작할 수 있게** 쓴다. Notion 내용은 필요한 만큼 본문에 옮기고 링크는 덧붙이기만.
   - **FE 화면이 바뀌는 이슈**는 「읽을 것」에 시안의 그 화면(ETC `docs/design/chagok/` 의 `index.html` · `experience.html` · `apply.html`, 창 · 메뉴는 `mock.js` 의 자리)과 `docs/design/DESIGN.md` 「시안과 일부러 다른 곳」을 적는다. 시안에 없는 화면이면 「시안 없음 — DESIGN.md 「반응형」 · 「상태」 기준」이라고 적는다.
5. **바로 만든다** — 묻지 않는다. 이슈는 그때그때 만들고, 틀리면 고치거나 닫는다.
6. **만든 뒤 반드시 GitHub 에 연결한다.** 본문 「선행 이슈」에 적은 것은 **전부** 선행(blocked by)으로 건다 — 본문에만 쓰면 GitHub 도 `next-up` 도 모르고, 추천 순서가 틀린다. 선행이 없으면 본문에 「없음」.

   ```bash
   bash .claude/skills/plan-feature/link.sh blocked FE#7 FE#5 BE#4   # FE#7 은 FE#5 · BE#4 에 막혀 있다
   bash .claude/skills/plan-feature/link.sh sub ETC#5 FE#4 BE#4      # epic 의 하위 이슈 (다른 레포여도 된다)
   bash .claude/skills/plan-feature/link.sh check                    # 본문 「선행 이슈」와 GitHub 선행이 같은지
   ```

   `check` 가 아무 줄도 내지 않아야 끝이다. 다른 레포 이슈는 본문에 `woowangthon/WOOWANGTHON_TEAM_12_BE#4` 처럼 전체 이름으로 쓴다(같은 레포는 `#4`).

7. 만든 이슈 목록(레포 · 번호 · 제목 · 선행)을 한 번에 보고한다.
