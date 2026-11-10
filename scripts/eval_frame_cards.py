"""구조 카드(S-82)를 켠 초안을 판마다 나란히 채점한다(AI#82 · AI#90).

    uv run python scripts/eval_frame_cards.py                          # 제품 판(CARD_MODE)만, MATCH_LLM_MODELS 첫 모델
    uv run python scripts/eval_frame_cards.py --variants off,steps,example --runs 2   # AI#90 세 판
    uv run python scripts/eval_frame_cards.py --variants plain,example                # 틀 없이와 견주기(AI#82)
    uv run python scripts/eval_frame_cards.py --rescore eval-runs/cards-<시각>

판(--variants): plain — 같은 쌍에서 frame 만 뺀 것 · off — 카드 단계 문구만(AI#90 전) · steps — 단계 이름 + 문장마다
step · example — steps + 카드 예시(app.draft.CARD_MODES). LLM 호출 수 = 카드 쌍 수 × 판 수 × 모델 수 × runs —
예: 8 × 3 × 1 × 2 = 48회(규칙 위반이면 쌍마다 최대 LLM_MAX_ATTEMPTS 번). 응답은 --out 폴더(기본
eval-runs/cards-<시각>, 커밋하지 않음)에 <판>/run<n>/<모델>/tuned/<쌍>.json 으로 저장한다. 키는 --env 파일에서 읽는다.

자료: samples/draft/cards.json — 기본 쌍(samples/draft/pairs.json 의 base)에 contracts README 「구성 틀」 제안 표의
카드(frame)를 얹은 지어낸 요청.

숫자의 뜻(경험 = 근거가 속한 경험, 보조 재료 포함):
- 나열: 경험이 둘 이상 쓰였는데 경험마다 한 문장씩인 쌍 수(eval_draft 의 listed) / 경험이 둘 이상인 쌍 수
- 대표 깊음: 첫째로 고른 경험이 두 문장 이상인 쌍 수(eval_draft 의 deep)
- 경험 전환: FACT · ELABORATED · SUPPOSED 문장을 차례로 볼 때 앞 문장과 다른 경험으로 넘어간 횟수(쌍 평균 — 적을수록 한 경험을
  끌고 간다). 가정(SUPPOSED)은 첫 근거, 근거가 없으면 짝 omitted 의 경험에 속한다(AI#84)
- 줄기 비중: 가장 많이 쓰인 경험이 그 문장들에서 차지하는 비율(쌍 평균 — 높을수록 한 경험이 줄기다)
- 지어냄(후보) · 없는 숫자 · 팀 일을 내 일로(후보) · 모름에 주체(후보) · 종류 위반 · 오류 — eval_draft 와 같다.
  카드를 켰다고 늘면 근거 규칙이 밀린 것이다
단계(AI#90 — step 은 steps · example 판에만 있다):
- 단계 순서: 출력 step 이 전부 카드 단계 이름이고 카드 순서대로(뒤로 가지 않음 — SKILL_STEPS 는 첫 단계로 돌아가는 것만
  허용)인 쌍 수 / step 이 있는 쌍 수
- 빠진 단계: 카드 단계 중 어느 문장의 step 에도 없는 수(쌍 평균) — 이웃 단계를 합친 문장도 있으니 0 이 목표는 아니다
- 예시 베끼기: 문장이 그 카드의 예시 문장과 공백을 뺀 8글자 이상을 같이 쓰는 수 — 모든 판에서 센다(off · plain 은 우연히
  겹치는 바닥값). 쌍마다 겹친 토막을 찍는다
- 이음말: 둘째 문장부터 「그래서 · 그런데 · 그 뒤 · 이 …」처럼 앞 문장을 받는 말로 여는 문장 비율 — 모든 판
- 끊긴 전환(후보): step 이 앞 문장과 달라졌는데 이음말 없이 연 문장 수 / step 이 바뀐 수. 이음말 없이도 잘 이어지는
  문장이 있으니 문장을 읽는다
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

from app.draft import (  # noqa: E402
    CARD_EXAMPLES,
    CARD_MODE,
    CARD_MODES,
    CARD_REPEATS,
    CARD_STEPS,
    backings,  # noqa: E402
)
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
    shared_run,
)

CARDS = ROOT / "samples" / "draft" / "cards.json"
VARIANTS = ("plain", *CARD_MODES)

# 앞 문장을 받아 여는 말(후보 — 사람이 읽는다)
CONNECTIVE = re.compile(
    r"^(?:그래서|그런데|그러나|하지만|그러자|그리고|따라서|결국|덕분에|다만|또한|또|먼저|처음|이후|이어|당시|"
    r"마지막|끝으로|그 ?뒤|그 ?후|그 ?결과|그때|그중|그런|그렇게|그 |이 |이런|이렇게|이처럼|이를|이에|이때|이로)"
)


def load_cards() -> list[dict]:
    """카드 쌍 — 기본 쌍에 id · frame 을 덮어쓴 것."""
    base = {p["id"]: p for p in load_pairs()}
    cards = json.loads(CARDS.read_text(encoding="utf-8"))
    return [{**base[c["base"]], "id": c["id"], "frame": c["frame"]} for c in cards]


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

    def add(self, other: "Trunk") -> None:
        self.pairs += other.pairs
        self.switches += other.switches
        self.share += other.share
        self.lines += other.lines


def score_trunk(
    pair: dict,
    chosen: list[MatchExperience],
    context: list[MatchExperience],
    data: LLMDraftOutput,
) -> Trunk:
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
        return Trunk()
    switches = sum(a != b for a, b in zip(owners, owners[1:], strict=False))
    share = max(owners.count(o) for o in set(owners)) / len(owners)
    line = f"{pair['id']}: 경험 {owners} 전환 {switches} 줄기 {share:.0%}"
    return Trunk(1, switches, share, [line])


def in_order(frame: str, steps: list) -> bool:
    """step 이 전부 카드 단계 이름이고 뒤로 가지 않나 — 되풀이 카드는 첫 단계로 돌아가는 것만 허용."""
    names = CARD_STEPS[frame]
    if not steps or any(s not in names for s in steps):
        return False
    idx = [names.index(s) for s in steps]
    return all(
        b >= a or (frame in CARD_REPEATS and b == 0) for a, b in zip(idx, idx[1:], strict=False)
    )


@dataclass
class Steps:
    """단계 순서 · 예시 베끼기 · 이음말 — 쌍마다 더한 값."""

    pairs: int = 0
    ordered: int = 0
    missing: int = 0
    copied: list[str] = field(default_factory=list)
    joins: int = 0
    gaps: int = 0
    bare: list[str] = field(default_factory=list)
    changes: int = 0
    lines: list[str] = field(default_factory=list)

    def add(self, other: "Steps") -> None:
        for name, value in vars(other).items():
            setattr(self, name, getattr(self, name) + value)


def score_steps(card: dict, data: LLMDraftOutput, steps: list | None) -> Steps:
    """card 의 frame 으로 잰다 — plain 판도 같은 카드의 예시와 겹치는지 본다."""
    frame, texts = card["frame"], [s.text for s in data.sentences]
    got = Steps(gaps=max(len(texts) - 1, 0))
    for i, text in enumerate(texts):
        for _, _, example in CARD_EXAMPLES[frame]:
            run_ = shared_run(text, example)
            if run_:
                got.copied.append(f"{card['id']}[{i}] 「{run_}」 {text}")
                break
    opened = [bool(CONNECTIVE.match(t)) for t in texts]
    got.joins = sum(opened[1:])
    if not steps:
        return got
    got.pairs = 1
    ok = len(steps) == len(texts) and in_order(frame, steps)
    got.ordered = int(ok)
    got.missing = len(set(CARD_STEPS[frame]) - set(steps))
    for i in range(1, min(len(steps), len(texts))):
        if steps[i] != steps[i - 1]:
            got.changes += 1
            if not opened[i]:
                got.bare.append(f"{card['id']}[{i}] {steps[i - 1]} → {steps[i]}: {texts[i]}")
    seq = " → ".join(str(s) for s in steps)
    got.lines.append(f"{card['id']}: {'순서 맞음' if ok else '순서 어긋남'} — {seq}")
    return got


def _row(name: str, t: Score, k: Trunk, s: Steps, total: int) -> str:
    n = max(k.pairs, 1)
    order = f"{s.ordered}/{s.pairs}" if s.pairs else "—"
    missing = f"{s.missing / s.pairs:.1f}" if s.pairs else "—"
    bare = f"{len(s.bare)}/{s.changes}" if s.pairs else "—"
    return (
        f"| {name} | {total} | {len(t.errors)} | {len(t.sentences)} "
        f"| {len(t.listed)}/{t.list_chances} | {t.deep_hit}/{t.deep_chances} "
        f"| {k.switches / n:.1f} | {k.share / n:.0%} "
        f"| {order} | {missing} | {len(s.copied)} | {s.joins / max(s.gaps, 1):.0%} | {bare} "
        f"| {len(t.invented)} | {len(t.bad_numbers)} | {len(t.team_as_mine)} "
        f"| {len(t.unknown_actor)} | {len(t.kind_broken)} |"
    )


def _texts(data: LLMDraftOutput, steps: list | None) -> list[str]:
    tags = steps if steps and len(steps) == len(data.sentences) else [None] * len(data.sentences)
    return [
        f"  {i}. [{x.kind}{' · ' + tag if tag else ''}] {x.text}"
        for i, (x, tag) in enumerate(zip(data.sentences, tags, strict=True))
    ]


def report(out: Path, experiences: list[dict], cards: list[dict]) -> str:
    facts = load_facts()
    rows, details = [], []
    # 폴더: <out>/<판>/<회차>/<모델>/tuned/<쌍>.json
    for variant_dir in sorted(p for p in out.iterdir() if p.is_dir() and p.name in VARIANTS):
        variant = variant_dir.name
        for round_dir in sorted(p for p in variant_dir.iterdir() if p.is_dir()):
            for model_dir in sorted(p for p in round_dir.iterdir() if p.is_dir()):
                run_dir = model_dir / "tuned"
                name = f"{model_dir.name} · {round_dir.name} · {variant}"
                t, k, s, n, drafts = Score(), Trunk(), Steps(), 0, []
                for card in cards:
                    pair = plain_of(card) if variant == "plain" else card
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
                    s.add(score_steps(card, data, record.get("steps")))
                    drafts += [
                        f"- {pair['id']} ({card['frame']})",
                        *_texts(data, record.get("steps")),
                    ]
                rows.append(_row(name, t, k, s, n))
                lines = [
                    *(f"- 오류 {x}" for x in t.errors),
                    *(f"- 단계 {x}" for x in s.lines),
                    *(f"- 예시 베끼기 {x}" for x in s.copied),
                    *(f"- 끊긴 전환(후보) {x}" for x in s.bare),
                    *(f"- 나열 {x}" for x in t.listed),
                    *(f"- 대표 얕음 {x}" for x in t.shallow),
                    *(f"- 전환 {x}" for x in k.lines),
                    *(f"- 지어냄(후보) {x}" for x in t.invented),
                    *(f"- 없는 숫자 {x}" for x in t.bad_numbers),
                    *(f"- 팀 일을 내 일로(후보) {x}" for x in t.team_as_mine),
                    *(f"- 모름에 주체(후보) {x}" for x in t.unknown_actor),
                    *(f"- 종류 위반 {x}" for x in t.kind_broken),
                    "",
                    "초안:",
                    *drafts,
                ]
                details.append(f"\n### {name}\n\n" + "\n".join(lines))
    head = (
        "| 모델 · 회차 · 판 | 쌍 | 오류 | FACT 문장 | 나열 | 대표 깊음 | 경험 전환(평균) "
        "| 줄기 비중(평균) | 단계 순서 | 빠진 단계(평균) | 예시 베끼기 | 이음말 | 끊긴 전환(후보) "
        "| 지어냄(후보) | 없는 숫자 | 팀 일을 내 일로(후보) | 모름에 주체(후보) | 종류 위반 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- "
        "| --- | --- | --- | --- | --- |"
    )
    return "\n".join([head, *rows]) + "\n" + "".join(details) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="쉼표로 「제공사:모델」 — 기본은 MATCH_LLM_MODELS 첫 모델")
    parser.add_argument(
        "--variants", default=CARD_MODE, help=f"쉼표로 판 — {list(VARIANTS)} (기본 제품 판)"
    )
    parser.add_argument("--runs", type=int, default=1, help="같은 요청을 몇 번씩")
    parser.add_argument("--out", type=Path, help="응답 저장 폴더")
    parser.add_argument("--env", type=Path, default=ROOT / ".env", help="키를 읽을 .env")
    parser.add_argument("--rescore", type=Path, help="저장한 응답만 다시 채점(LLM 안 부름)")
    args = parser.parse_args()

    experiences = load_experiences()
    cards = load_cards()
    if args.rescore:
        out = args.rescore
    else:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
        for v in variants:
            if v not in VARIANTS:
                parser.error(f"판은 {list(VARIANTS)} 중 — {v!r}")
        if args.models:
            models = [m.strip() for m in args.models.split(",") if m.strip()]
        else:
            from dotenv import load_dotenv

            from app.config import load_settings

            load_dotenv(args.env)
            p, m = load_settings().match_models[0]
            models = [f"{p}:{m}"]
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = args.out or ROOT / "eval-runs" / f"cards-{stamp}"
        calls = len(cards) * len(variants) * len(models) * args.runs
        print(f"LLM 호출 {calls}회(규칙 위반 시 더) → {out}")
        for v in variants:
            pairs = [plain_of(c) for c in cards] if v == "plain" else cards
            mode = None if v == "plain" else v
            for i in range(args.runs):
                folder = out / v / f"run{i + 1}"
                asyncio.run(run(models, ["tuned"], experiences, pairs, folder, args.env, mode))

    text = report(out, experiences, cards)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
