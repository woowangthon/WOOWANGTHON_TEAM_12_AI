#!/usr/bin/env bash
# 네 레포의 열린 이슈와 「열린 선행 · 열린 하위 이슈」를 한 줄씩 뽑는다. 읽기만 한다.
# 쓰는 법: bash .claude/skills/next-up/collect.sh
# 출력(탭 구분): 이슈 · 제목 · 라벨 · Milestone · 담당 · 열린 선행 · 열린 하위 이슈
#   이슈 키는 ETC#6 · FE#4 처럼 레포 짧은 이름 + 번호
set -euo pipefail
O=woowangthon
short='sub(".*WOOWANGTHON_TEAM_12_"; "")'

echo "== Milestone (열린 것)"
for r in ETC FE BE AI; do
  gh api "repos/$O/WOOWANGTHON_TEAM_12_$r/milestones?state=open" \
    --jq ".[] | \"$r\t\(.title)\t열림 \(.open_issues) · 닫힘 \(.closed_issues)\t\(.description // \"\")\""
done

echo "== 열린 이슈"
printf '이슈\t제목\t라벨\tMilestone\t담당\t열린 선행\t열린 하위\n'
for r in ETC FE BE AI; do
  R="$O/WOOWANGTHON_TEAM_12_$r"
  gh issue list -R "$R" --state open --limit 100 \
    --json number,title,labels,milestone,assignees \
    --jq '.[] | [.number, .title, ([.labels[].name]|join(",")), (.milestone.title // "-"), ([.assignees[].login]|join(",")|if .=="" then "-" else . end)] | @tsv' |
  while IFS=$'\t' read -r n title labels ms who; do
    blk=$(gh api "repos/$R/issues/$n/dependencies/blocked_by" \
      --jq "[.[] | select(.state==\"open\") | (.repository_url|$short) + \"#\" + (.number|tostring)] | join(\",\")")
    subs=$(gh api "repos/$R/issues/$n/sub_issues" \
      --jq "[.[] | select(.state==\"open\") | (.repository_url|$short) + \"#\" + (.number|tostring)] | join(\",\")")
    printf '%s#%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$r" "$n" "$title" "$labels" "$ms" "$who" "${blk:--}" "${subs:--}"
  done
done
