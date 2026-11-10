"""구조 카드(S-82)를 켠 초안을, 같은 문항 · 경험의 틀 없는 초안과 나란히 채점한다(AI#82).

    uv run python scripts/eval_frame_cards.py                 # MATCH_LLM_MODELS 첫 모델
    uv run python scripts/eval_frame_cards.py --runs 2        # 같은 요청을 두 번씩 — 흔들림을 본다
    uv run python scripts/eval_frame_cards.py --only c-skill  # 그 쌍만
    uv run python scripts/eval_frame_cards.py --rescore eval-runs/cards-<시각>

LLM 호출 수 = 카드 쌍 수 × 2(틀 없이 · 카드) × 모델 수 × runs — 기본 9 × 2 × 1 × 1 = 18회(규칙 위반이면 쌍마다 최대
LLM_MAX_ATTEMPTS 번). 응답은 --out 폴더(기본 eval-runs/cards-<시각>, 커밋하지 않음)에 저장한다. 키는 --env 파일에서 읽는다.

자료: samples/draft/cards.json — 기본 쌍(samples/draft/pairs.json 의 base)에 contracts README 「구성 틀」 제안 표의
카드(frame)를 얹은 지어낸 요청(카드에 experienceIds · expect 를 적으면 그것으로 덮는다). 「틀 없이」 줄은
같은 쌍에서 frame 만 뺀 것이다.

숫자의 뜻(활동 나열이 줄었나 — 「틀 없이」 줄과 「카드」 줄을 견준다. 경험 = 근거가 속한 경험, 보조 재료 포함):
- 나열: 경험이 둘 이상 쓰였는데 경험마다 한 문장씩인 쌍 수(eval_draft 의 listed) / 경험이 둘 이상인 쌍 수
- 대표 깊음: 첫째로 고른 경험이 두 문장 이상인 쌍 수(eval_draft 의 deep)
- 경험 전환: FACT · ELABORATED · SUPPOSED 문장을 차례로 볼 때 앞 문장과 다른 경험으로 넘어간 횟수(쌍 평균 — 적을수록 한 경험을
  끌고 간다). 가정(SUPPOSED)은 첫 근거, 근거가 없으면 짝 omitted 의 경험에 속한다(AI#84)
- 줄기 비중: 가장 많이 쓰인 경험이 그 문장들에서 차지하는 비율(쌍 평균 — 높을수록 한 경험이 줄기다)
- 이음말: 둘째 문장부터 「그래서 · 그런데 · 그 뒤 · 이 …」처럼 앞 문장을 받는 말로 연 문장 비율(후보 — 이음말 없이도
  잘 이어지는 문장이 있다). 카드가 단계 칸에 끼워 맞춘 뚝뚝 끊기는 글을 만들면 낮아진다(AI#90)
- 지어냄(후보) · 없는 숫자 · 팀 일을 내 일로(후보) · 모름에 주체(후보) · 종류 위반 · 오류 — eval_draft 와 같다.
  카드를 켰다고 늘면 근거 규칙이 밀린 것이다
"""

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.draft import backings  # noqa: E402
from app.schemas import LLMDraftOutput, MatchExperience  # noqa: E402
from scripts.eval_draft import (  # noqa: E402
    Score,
    context_of,
    load_experiences,
    load_facts,
    load_pairs,
    picked,
    run,
    score,
    sentence_owners,
)

CARDS = ROOT / "samples" / "draft" / "cards.json"

# 앞 문장을 받아 여는 말(후보 — 사람이 읽는다)
CONNECTIVE = re.compile(
    r"^(?:그래서|그런데|그러나|하지만|그러자|그리고|따라서|결국|덕분에|다만|또한|또|먼저|처음|이후|이어|당시|"
    r"마지막|끝으로|그 ?뒤|그 ?후|그 ?결과|그때|그중|그런|그렇게|그 |이 |이런|이렇게|이처럼|이를|이에|이때|이로)"
)


def load_cards() -> list[dict]:
    """카드 쌍 — 기본 쌍에 카드의 칸(id · frame, 있으면 experienceIds · expect 등)을 덮어쓴 것."""
    base = {p["id"]: p for p in load_pairs()}
    cards = json.loads(CARDS.read_text(encoding="utf-8"))
    return [{**base[c["base"]], **{k: v for k, v in c.items() if k != "base"}} for c in cards]


def plain_of(card: dict) -> dict:
    """같은 쌍에서 frame 만 뺀 것 — id 뒤에 -plain."""
    return {**{k: v for k, v in card.items() if k != "frame"}, "id": card["id"] + "-plain"}


@dataclass
class Trunk:
    """경험 전환 · 줄기 비중 — 쌍마다 더한 값(나누는 건 보고 때)."""

    pairs: int = 0
    switches: int = 0
    share: float = 0.0
    lines: list[str] = field(default_factory=list)
    joins: int = 0  # 이음말로 연 문장 수(둘째 문장부터)
    gaps: int = 0  # 둘째 문장부터의 문장 수

    def add(self, other: "Trunk") -> None:
        for name, value in vars(other).items():
            setattr(self, name, getattr(self, name) + value)


def joins_of(data: LLMDraftOutput) -> tuple[int, int]:
    """(이음말로 연 문장 수, 둘째 문장부터의 문장 수)."""
    rest = [s.text for s in data.sentences[1:]]
    return sum(bool(CONNECTIVE.match(t)) for t in rest), len(rest)


def score_trunk(
    pair: dict,
    chosen: list[MatchExperience],
    context: list[MatchExperience],
    data: LLMDraftOutput,
) -> Trunk:
    joins, gaps = joins_of(data)
    known = backings([*chosen, *context])
    by_sentence = sentence_owners(data, known)
    owners: list[int] = []
    for i, x in enumerate(data.sentences):
        if x.kind == "OPINION":
            continue
        mine = [known[e.type, e.id].experience.id for e in x.evidence if (e.type, e.id) in known]
        if mine:
            owners.append(mine[0])  # 문장의 첫 근거가 속한 경험
        elif by_sentence[i]:
            owners.append(min(by_sentence[i]))  # 근거 없는 가정 — 짝 omitted 의 경험(AI#84)
    if not owners:
        return Trunk(joins=joins, gaps=gaps)
    switches = sum(a != b for a, b in zip(owners, owners[1:], strict=False))
    share = max(owners.count(o) for o in set(owners)) / len(owners)
    line = f"{pair['id']}: 경험 {owners} 전환 {switches} 줄기 {share:.0%}"
    return Trunk(1, switches, share, [line], joins, gaps)


def _row(name: str, t: Score, k: Trunk, total: int) -> str:
    n = max(k.pairs, 1)
    return (
        f"| {name} | {total} | {len(t.errors)} | {len(t.sentences)} "
        f"| {len(t.listed)}/{t.list_chances} | {t.deep_hit}/{t.deep_chances} "
        f"| {k.switches / n:.1f} | {k.share / n:.0%} | {k.joins / max(k.gaps, 1):.0%} "
        f"| {len(t.invented)} | {len(t.bad_numbers)} | {len(t.team_as_mine)} "
        f"| {len(t.unknown_actor)} | {len(t.kind_broken)} |"
    )


def report(out: Path, experiences: list[dict], cards: list[dict]) -> str:
    facts = load_facts()
    plain = [plain_of(c) for c in cards]
    rows, details = [], []
    # 폴더: <out>/<회차>/<모델>/tuned/<쌍>.json — eval_draft.run 을 회차마다 부른 모양
    for round_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        for model_dir in sorted(p for p in round_dir.iterdir() if p.is_dir()):
            run_dir = model_dir / "tuned"
            name = f"{model_dir.name} · {round_dir.name}"
            lines: list[str] = []
            for label, pairs in (("틀 없이", plain), ("카드", cards)):
                t, k, n = Score(), Trunk(), 0
                for pair in pairs:
                    path = run_dir / f"{pair['id']}.json"
                    if not path.exists():
                        continue
                    n += 1
                    record = json.loads(path.read_text(encoding="utf-8"))
                    if "error" in record:
                        t.errors.append(f"{pair['id']}: {record['error']}")
                        continue
                    exps = [MatchExperience.model_validate(e) for e in picked(pair, experiences)]
                    ctx = [MatchExperience.model_validate(e) for e in context_of(pair, experiences)]
                    data = LLMDraftOutput.model_validate(record["response"])
                    t.add(score(pair, exps, data, facts, ctx))
                    k.add(score_trunk(pair, exps, ctx, data))
                rows.append(_row(f"{name} · {label}", t, k, n))
                lines += [
                    *(f"- {label} 오류 {x}" for x in t.errors),
                    *(f"- {label} 나열 {x}" for x in t.listed),
                    *(f"- {label} 대표 얕음 {x}" for x in t.shallow),
                    *(f"- {label} 전환 {x}" for x in k.lines),
                    *(f"- {label} 지어냄(후보) {x}" for x in t.invented),
                    *(f"- {label} 없는 숫자 {x}" for x in t.bad_numbers),
                    *(f"- {label} 팀 일을 내 일로(후보) {x}" for x in t.team_as_mine),
                    *(f"- {label} 모름에 주체(후보) {x}" for x in t.unknown_actor),
                    *(f"- {label} 종류 위반 {x}" for x in t.kind_broken),
                    *(f"- {label} 생각 {x}" for x in t.opinions),
                    *(f"- {label} 문장 {x}" for x in t.sentences),
                ]
            details.append(f"\n### {name}\n\n" + "\n".join(lines))
    head = (
        "| 모델 · 회차 · 요청 | 쌍 | 오류 | FACT 문장 | 나열 | 대표 깊음 | 경험 전환(평균) "
        "| 줄기 비중(평균) | 이음말 | 지어냄(후보) | 없는 숫자 | 팀 일을 내 일로(후보) "
        "| 모름에 주체(후보) | 종류 위반 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    return "\n".join([head, *rows]) + "\n" + "".join(details) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="쉼표로 「제공사:모델」 — 기본은 MATCH_LLM_MODELS 첫 모델")
    parser.add_argument("--runs", type=int, default=1, help="같은 요청을 몇 번씩")
    parser.add_argument("--out", type=Path, help="응답 저장 폴더")
    parser.add_argument("--env", type=Path, default=ROOT / ".env", help="키를 읽을 .env")
    parser.add_argument("--rescore", type=Path, help="저장한 응답만 다시 채점(LLM 안 부름)")
    parser.add_argument("--only", help="쉼표로 카드 쌍 id — 이것만(예: c-skill,c-skill-one)")
    args = parser.parse_args()

    experiences = load_experiences()
    cards = load_cards()
    if args.only:
        only = {x.strip() for x in args.only.split(",")}
        cards = [c for c in cards if c["id"] in only]
    if args.rescore:
        out = args.rescore
    else:
        if args.models:
            models = [m.strip() for m in args.models.split(",") if m.strip()]
        else:
            from dotenv import load_dotenv

            from app.config import load_settings

            load_dotenv(args.env)
            p, m = load_settings().match_models[0]
            models = [f"{p}:{m}"]
        pairs = [*(plain_of(c) for c in cards), *cards]
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = args.out or ROOT / "eval-runs" / f"cards-{stamp}"
        print(f"LLM 호출 {len(pairs) * len(models) * args.runs}회(규칙 위반 시 더) → {out}")
        for i in range(args.runs):
            asyncio.run(run(models, ["tuned"], experiences, pairs, out / f"run{i + 1}", args.env))

    text = report(out, experiences, cards)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
