#!/usr/bin/env bash
# 이슈 사이 연결을 GitHub 에 건다 — 본문에 「선행: #3」이라고만 쓰면 GitHub 도 next-up 도 모른다.
# 이슈 키: ETC#6 · FE#4 처럼 레포 짧은 이름 + 번호
#
#   bash .claude/skills/plan-feature/link.sh blocked FE#7 FE#5 BE#4   # FE#7 은 FE#5 · BE#4 에 막혀 있다
#   bash .claude/skills/plan-feature/link.sh sub ETC#5 FE#4 BE#4      # ETC#5 의 하위 이슈로 FE#4 · BE#4
#   bash .claude/skills/plan-feature/link.sh check                    # 열린 이슈의 본문 「선행 이슈」와 GitHub 선행이 같은지
#
# 이미 걸려 있으면 건너뛴다. check 는 읽기만 한다.
set -euo pipefail
O=woowangthon
short='sub(".*WOOWANGTHON_TEAM_12_"; "")'

repo() { echo "$O/WOOWANGTHON_TEAM_12_${1%%#*}"; }
num() { echo "${1##*#}"; }
id_of() { gh api "repos/$(repo "$1")/issues/$(num "$1")" --jq .id; }
keys() { # <레포 키> <엔드포인트> → 걸려 있는 이슈 키(닫힌 것 포함)
  gh api "repos/$(repo "$1")/issues/$(num "$1")/$2" \
    --jq ".[] | (.repository_url|$short) + \"#\" + (.number|tostring)"
}

cmd=${1:-}; shift || true
case "$cmd" in
  blocked|sub)
    [ $# -ge 2 ] || { echo "쓰는 법: link.sh $cmd <이슈> <대상>..." >&2; exit 2; }
    me=$1; shift
    if [ "$cmd" = blocked ]; then ep=dependencies/blocked_by; field=issue_id; else ep=sub_issues; field=sub_issue_id; fi
    have=$(keys "$me" "$ep")
    for k in "$@"; do
      if grep -qx "$k" <<<"$have"; then echo "이미 있음  $me ← $k"; continue; fi
      gh api --method POST "repos/$(repo "$me")/issues/$(num "$me")/$ep" -F "$field=$(id_of "$k")" >/dev/null
      echo "걸었음    $me ← $k ($cmd)"
    done
    ;;
  check)
    bad=0
    for r in ETC FE BE AI; do
      for n in $(gh issue list -R "$O/WOOWANGTHON_TEAM_12_$r" --state open --limit 100 --json number --jq '.[].number'); do
        me="$r#$n"
        sec=$(gh api "repos/$(repo "$me")/issues/$n" --jq '.body // ""' | tr -d '\r' |
              awk '/^## 선행 이슈/{f=1; next} /^## /{f=0} f')
        want=$( { grep -oE '(ETC|FE|BE|AI)#[0-9]+' <<<"$sec" || true
                  grep -oE '(^|[^A-Za-z0-9_])#[0-9]+' <<<"$sec" | grep -oE '#[0-9]+' | sed "s/^/$r/" || true; } | sort -u)
        have=$(keys "$me" dependencies/blocked_by | sort -u)
        miss=$(comm -23 <(echo "$want" | sed '/^$/d') <(echo "$have" | sed '/^$/d') | paste -sd, -)
        extra=$(comm -13 <(echo "$want" | sed '/^$/d') <(echo "$have" | sed '/^$/d') | paste -sd, -)
        [ -n "$miss" ] && echo "빠짐  $me — 본문엔 있는데 GitHub 선행에 없음: $miss  → link.sh blocked $me ${miss//,/ }"
        [ -n "$extra" ] && echo "더 있음 $me — GitHub 선행엔 있는데 본문에 없음: $extra  (본문을 고치거나 연결을 뺀다)"
        true
      done
    done
    echo "검사 끝 — 위에 줄이 없으면 본문과 GitHub 선행이 같다"
    ;;
  *)
    sed -n '2,9p' "$0"; exit 2 ;;
esac
