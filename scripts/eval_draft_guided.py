"""구성 틀 · 작성 지침을 켠 초안을, 같은 문항 · 경험의 지침 · 틀 없는 초안과 나란히 채점한다(S-49 · S-50 · AI#55).

    uv run python scripts/eval_draft_guided.py                 # MATCH_LLM_MODELS 첫 모델, 우리 프롬프트만
    uv run python scripts/eval_draft_guided.py --runs 2        # 같은 요청을 두 번씩 — 흔들림을 본다
    uv run python scripts/eval_draft_guided.py --rescore eval-runs/guided-<시각>

LLM 호출 수 = (기본 쌍 수 + 지침 쌍 수) × 모델 수 × runs — 기본 (6 + 7) × 1 × 1 = 13회(규칙 위반이면 쌍마다 최대
LLM_MAX_ATTEMPTS 번). 응답은 --out 폴더(기본 eval-runs/guided-<시각>, 커밋하지 않음)에 저장한다. 키는 --env 파일에서 읽는다.

자료: samples/draft/guided.json — 기본 쌍(samples/draft/pairs.json 의 base)에 frame · guidelines 를 더한 지어낸 요청.
채점은 scripts/eval_draft.py 의 score 를 그대로 쓴다(missing · expect 는 기본 쌍의 정답). 기본 쌍의 beats(기본 순서)는
지침 쌍에서 빼고, 틀의 단계 순서는 frameBeats 로 따로 본다.

숫자의 뜻(「기본」 줄은 지침 쌍이 가리키는 기본 쌍만, 「지침 · 틀」 줄은 지침 쌍 — 같은 문항 · 경험이다):
- 지어냄(후보) · 없는 숫자 · 팀 일을 내 일로(후보) · 모름에 주체(후보) · 없는 사실(후보) · 종류 위반 · 오류 — eval_draft 와 같다.
  지침 · 틀을 켰다고 늘면 근거 규칙이 밀린 것이다(이슈 「위험」)
- 지침 결과 위반: id 개수 · 순서 · applied/note 짝이 틀린 수(코드 검사와 같은 기준 — 0 이어야 한다)
- 반영 못 함 맞음: 근거와 부딪치거나(「수치로」 · 「리더였다고」) 서로 부딪치는 지침(expect: not)에 applied: false 를 돌려준 수 / 그런 지침 수
- 반영함: 그 밖의 지침에 applied: true 를 돌려준 수 / 그런 지침 수 — LLM 의 자기 보고라 문장을 읽어 맞는지 본다
- 기본 순서: 기본 쌍의 beats(AI#46 기본 순서)가 맞은 쌍 수 — 「기본」 줄에만 있다(지침 쌍은 beats 를 뺀다).
  되풀이(후보)도 eval_draft 와 같다
- 틀 순서: frameBeats 의 사실이 전부 나오고 그 순서인 쌍 수 / frameBeats 가 있는 쌍 수
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

from app.schemas import LLMDraftOutput, MatchExperience  # noqa: E402
from scripts.eval_draft import (  # noqa: E402
    Score,
    load_experiences,
    load_facts,
    load_pairs,
    picked,
    run,
    score,
)

GUIDED = ROOT / "samples" / "draft" / "guided.json"


def load_guided() -> list[dict]:
    """지침 쌍 — 기본 쌍에 frame · guidelines · frameBeats 를 얹고 beats(기본 순서)는 뺀다."""
    base = {p["id"]: p for p in load_pairs()}
    out = []
    for g in json.loads(GUIDED.read_text(encoding="utf-8")):
        pair = {k: v for k, v in base[g["base"]].items() if k != "beats"}
        out.append({**pair, **g})
    return out


@dataclass
class Style:
    broken: list[str] = field(default_factory=list)
    refused: int = 0
    refuse_chances: int = 0
    refuse_miss: list[str] = field(default_factory=list)
    applied: int = 0
    apply_chances: int = 0
    notes: list[str] = field(default_factory=list)
    order_hit: int = 0
    order_total: int = 0
    order_miss: list[str] = field(default_factory=list)

    def add(self, other: "Style") -> None:
        for name, value in vars(other).items():
            setattr(self, name, getattr(self, name) + value)


def score_style(pair: dict, data: LLMDraftOutput) -> Style:
    pid = pair["id"]
    s = Style()
    want = pair.get("guidelines", [])
    got = data.guidelines
    if [g.id for g in got] != [g["id"] for g in want]:
        s.broken.append(f"{pid}: id {[g.id for g in got]}")
    for g in got:
        if g.applied == (g.note is not None):
            s.broken.append(f"{pid}: {g.id} applied={g.applied} note={g.note!r}")
    by_id = {g.id: g for g in got}
    for g in want:
        r = by_id.get(g["id"])
        s.notes.append(
            f"{pid}: [{g['id']}] {g['text']} → "
            + ("(없음)" if r is None else f"{'반영' if r.applied else '못 함'} {r.note or ''}")
        )
        if g.get("expect") == "not":
            s.refuse_chances += 1
            if r is not None and not r.applied:
                s.refused += 1
            else:
                s.refuse_miss.append(f"{pid}: [{g['id']}] {g['text']}")
        else:
            s.apply_chances += 1
            s.applied += int(r is not None and r.applied)
    if pair.get("frameBeats"):
        s.order_total = 1
        spots = [
            next((i for i, x in enumerate(data.sentences) if re.search(b, x.text)), None)
            for b in pair["frameBeats"]
        ]
        if None not in spots and spots == sorted(spots):
            s.order_hit = 1
        else:
            s.order_miss.append(f"{pid}: 문장 위치 {spots}")
    return s


def _row(name: str, t: Score, total: int) -> str:
    return (
        f"| {name} | {total} | {len(t.errors)} | {len(t.sentences)} | {len(t.invented)} "
        f"| {len(t.bad_numbers)} | {len(t.team_as_mine)} | {len(t.unknown_actor)} "
        f"| {len(t.opinion_invented)} | {len(t.kind_broken)} | {t.omit_hit}/{t.omit_chances} "
        f"| {t.order_hit}/{t.order_total} | {len(t.repeated)} |"
    )


def report(out: Path, experiences: list[dict], guided: list[dict]) -> str:
    facts = load_facts()
    bases = {p["id"]: p for p in load_pairs()}
    base_pairs = [bases[i] for i in dict.fromkeys(g["base"] for g in guided)]
    rows, style_rows, details = [], [], []
    # 폴더: <out>/<회차>/<모델>/tuned/<쌍>.json — eval_draft.run 을 회차마다 부른 모양
    for round_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        for model_dir in sorted(p for p in round_dir.iterdir() if p.is_dir()):
            run_dir = model_dir / "tuned"
            name = f"{model_dir.name} · {round_dir.name}"
            style = Style()
            lines: list[str] = []
            for label, pairs in (("기본", base_pairs), ("지침 · 틀", guided)):
                t, n = Score(), 0
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
                    data = LLMDraftOutput.model_validate(record["response"])
                    t.add(score(pair, exps, data, facts))
                    if label != "기본":
                        style.add(score_style(pair, data))
                rows.append(_row(f"{name} · {label}", t, n))
                lines += [
                    *(f"- {label} 오류 {x}" for x in t.errors),
                    *(f"- {label} 지어냄(후보) {x}" for x in t.invented),
                    *(f"- {label} 없는 숫자 {x}" for x in t.bad_numbers),
                    *(f"- {label} 팀 일을 내 일로(후보) {x}" for x in t.team_as_mine),
                    *(f"- {label} 모름에 주체(후보) {x}" for x in t.unknown_actor),
                    *(f"- {label} 없는 사실(후보) {x}" for x in t.opinion_invented),
                    *(f"- {label} 종류 위반 {x}" for x in t.kind_broken),
                    *(f"- {label} 기본 순서 어긋남 {x}" for x in t.order_miss),
                    *(f"- {label} 되풀이(후보) {x}" for x in t.repeated),
                    *(f"- {label} 생각 {x}" for x in t.opinions),
                    *(f"- {label} 문장 {x}" for x in t.sentences),
                    *(f"- {label} omitted {x}" for x in t.omitted),
                ]
            style_rows.append(
                f"| {name} | {len(style.broken)} | {style.refused}/{style.refuse_chances} "
                f"| {style.applied}/{style.apply_chances} | {style.order_hit}/{style.order_total} |"
            )
            lines = [
                *(f"- 지침 결과 위반 {x}" for x in style.broken),
                *(f"- 반영 못 함이어야 {x}" for x in style.refuse_miss),
                *(f"- 틀 순서 어긋남 {x}" for x in style.order_miss),
                *(f"- 지침 {x}" for x in style.notes),
                *lines,
            ]
            details.append(f"\n### {name}\n\n" + "\n".join(lines))
    head = (
        "| 모델 · 회차 · 요청 | 쌍 | 오류 | FACT 문장 | 지어냄(후보) | 없는 숫자 | 팀 일을 내 일로(후보) "
        "| 모름에 주체(후보) | 없는 사실(후보) | 종류 위반 | omitted 씀 | 기본 순서 | 되풀이(후보) |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    style_head = (
        "| 모델 · 회차 | 지침 결과 위반 | 반영 못 함 맞음 | 반영함 | 틀 순서 |\n"
        "| --- | --- | --- | --- | --- |"
    )
    return (
        "\n".join([head, *rows])
        + "\n\n"
        + "\n".join([style_head, *style_rows])
        + "\n"
        + "".join(details)
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="쉼표로 「제공사:모델」 — 기본은 MATCH_LLM_MODELS 첫 모델")
    parser.add_argument("--runs", type=int, default=1, help="같은 요청을 몇 번씩")
    parser.add_argument("--out", type=Path, help="응답 저장 폴더")
    parser.add_argument("--env", type=Path, default=ROOT / ".env", help="키를 읽을 .env")
    parser.add_argument("--rescore", type=Path, help="저장한 응답만 다시 채점(LLM 안 부름)")
    args = parser.parse_args()

    experiences = load_experiences()
    guided = load_guided()
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
        bases = {p["id"]: p for p in load_pairs()}
        pairs = [bases[i] for i in dict.fromkeys(g["base"] for g in guided)] + guided
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = args.out or ROOT / "eval-runs" / f"guided-{stamp}"
        print(f"LLM 호출 {len(pairs) * len(models) * args.runs}회(규칙 위반 시 더) → {out}")
        for i in range(args.runs):
            asyncio.run(run(models, ["tuned"], experiences, pairs, out / f"run{i + 1}", args.env))

    text = report(out, experiences, guided)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
