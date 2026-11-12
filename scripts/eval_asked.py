"""이미 낸 되묻기 · 「딱히 없음」 보완을 채점한다 — 전(before) · 후(after)를 같은 회차로(AI#138 · ETC#354).

    uv run python scripts/eval_asked.py --runs 2
    uv run python scripts/eval_asked.py --env ../WOOWANGTHON_TEAM_12_AI/.env --runs 2   # worktree
    uv run python scripts/eval_asked.py --rescore eval-runs/asked-<시각>                # 다시 채점만(LLM 안 부름)

LLM 호출 수 = 경우 수(7) × 2(전 · 후) × 모델 수 × runs(규칙 위반이면 경우마다 최대 LLM_MAX_ATTEMPTS 번)
— 기본 7 × 2 × 1 × 1 = 14회. 응답은 --out 폴더(기본 eval-runs/asked-<시각>, 커밋하지 않음)의
<모델>/<before|after>/run-<N>/<경우>.json 에 저장한다.

전(before)은 이 PR 전 동작이다 — 요청에서 askedQuestions 를 빼고(예전 서버는 모르는 칸이라 버렸다),
「없다」 규칙을 붙이지 않는다. 후(after)는 지금 코드 그대로다.

자료: samples/asked/cases.json(지어낸 경험 4개 · 경우 7개 — README).

숫자의 뜻(정규식으로 거른 것은 후보 — 전문을 summary.md 에 남기니 사람이 읽고 센다):
- 다시 물음: /match 는 target 경험에 낸 되묻기 중 reask 에 걸린 수, /draft 는 omitted(질문 · 글) 중 걸린 수.
  dup-* 는 이미 낸 되묻기와 같은 사실(내 몫), none-* 는 「없다」고 답한 사실이다 — 0 이어야 한다
- 있다고 씀: /match 는 target 추천의 reason · emphasis, /draft 는 FACT · ELABORATED · SUPPOSED 문장 중 claim 에 걸린 수.
  「없다」고 답한 사실을 있었다고 쓴 것 — 0 이어야 한다
- gaps 에 남김: /match 에서 target 추천의 gaps 수(꼭 필요하면 gaps 에만 — 참고)
- 오류: 다시 시도 뒤에도 규칙을 어겨 끝난 경우 수
"""

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_runs import merge_runs, run_dirs, run_path  # noqa: E402

DATA = ROOT / "samples" / "asked" / "cases.json"
MODES = ("before", "after")


def load_data() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def request_body(case: dict, experiences: dict, mode: str) -> dict:
    exps = []
    for i in case["experienceIds"]:
        exp = dict(experiences[str(i)])
        asked = case.get("asked", {}).get(str(i))
        if asked and mode == "after":
            exp["askedQuestions"] = asked
        exps.append(exp)
    return {"question": case["question"], "experiences": exps}


@dataclass
class Score:
    cases: int = 0
    reasked: list[str] = field(default_factory=list)
    claimed: list[str] = field(default_factory=list)
    gaps: int = 0
    errors: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)


def score_case(case: dict, record: dict, s: Score) -> None:
    cid, target = case["id"], case["target"]
    reask = re.compile(case["reask"])
    claim = re.compile(case["claim"]) if case.get("claim") else None
    res = record["response"]
    if case["endpoint"] == "match":
        for c in res["clarifications"]:
            line = f"{cid} 되묻기 [{c['experienceId']}] {c['question']}"
            s.texts.append(line)
            if c["experienceId"] == target and reask.search(c["question"]):
                s.reasked.append(line)
        for r in res["recommendations"]:
            if r["experienceId"] != target:
                continue
            s.gaps += len(r["gaps"])
            s.texts.append(f"{cid} 이유 {r['reason']} / 강조 {r['emphasis']} / gaps {r['gaps']}")
            said = f"{r['reason']} {r['emphasis']}"
            if claim and claim.search(said):
                s.claimed.append(f"{cid} 추천 「{said}」")
        return
    for i, sent in enumerate(res["sentences"]):
        s.texts.append(f"{cid} [{i} {sent['kind']}] {sent['text']}")
        if claim and sent["kind"] != "OPINION" and claim.search(sent["text"]):
            s.claimed.append(f"{cid} [{i} {sent['kind']}] {sent['text']}")
    for o in res["omitted"]:
        line = f"{cid} omitted 「{o['text']}」 / 질문 {o['question']}"
        s.texts.append(line)
        if reask.search(f"{o['question'] or ''}") and o.get("sentenceIndex") is None:
            s.reasked.append(line)


def score_dir(path: Path, cases: list[dict]) -> Score:
    s = Score()
    for case in cases:
        f = path / f"{case['id']}.json"
        if not f.exists():
            continue
        record = json.loads(f.read_text(encoding="utf-8"))
        s.cases += 1
        if "error" in record:
            s.errors.append(f"{case['id']}: {record['error']}")
            continue
        score_case(case, record, s)
    return s


def _before_patches():
    """전 동작 — 「없다」 규칙을 붙이지 않는다. askedQuestions 는 request_body 가 뺀다."""
    from app import draft, match

    saved = (match.has_none_answer, draft.has_none_answer)
    match.has_none_answer = draft.has_none_answer = lambda _: False
    return lambda: (
        setattr(match, "has_none_answer", saved[0]),
        setattr(draft, "has_none_answer", saved[1]),
    )


async def run(models: list[str], runs: int, data: dict, out: Path, env: Path) -> None:
    from dotenv import load_dotenv

    from app.config import load_settings, parse_models
    from app.draft import draft
    from app.errors import ApiError
    from app.llm import build_llm
    from app.match import match
    from app.schemas import DraftRequest, MatchRequest

    load_dotenv(env)  # 값은 화면에 찍지 않는다
    base = load_settings()
    gate = asyncio.Semaphore(4)
    clients = {}
    for model in models:
        settings = replace(base, match_models=parse_models(model))
        clients[model] = (settings, build_llm(settings, settings.match_models))

    async def one(model: str, mode: str, n: int, case: dict) -> None:
        settings, llm = clients[model]
        body = request_body(case, data["experiences"], mode)
        async with gate:
            started = time.monotonic()
            try:
                if case["endpoint"] == "match":
                    got = await match(MatchRequest.model_validate(body), llm, settings)
                else:
                    got = await draft(DraftRequest.model_validate(body), llm, settings)
                record = {"request": body, "response": json.loads(got.model_dump_json())}
            except ApiError as e:
                record = {"request": body, "error": f"{e.code}: {e.message}"}
            record["seconds"] = round(time.monotonic() - started, 1)
        model_dir = out / model.replace(":", "_").replace("/", "_") / mode
        path = run_path(model_dir, n) / f"{case['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {model} · {mode} · run-{n} · {case['id']} ({record['seconds']}초)")

    # 전 · 후는 차례로 — 전은 모듈의 함수를 잠시 바꿔 돌린다
    for mode in MODES:
        restore = _before_patches() if mode == "before" else None
        try:
            await asyncio.gather(
                *(
                    one(m, mode, n, c)
                    for m in models
                    for n in range(1, runs + 1)
                    for c in data["cases"]
                )
            )
        finally:
            if restore:
                restore()


def report(out: Path, cases: list[dict]) -> str:
    rows, details = [], []
    for model_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        for mode in MODES:
            mode_dir = model_dir / mode
            if not mode_dir.is_dir():
                continue
            for tail, path in run_dirs(mode_dir):
                s = score_dir(path, cases)
                if not s.cases:
                    continue
                name = f"{model_dir.name} · {mode}{tail}"
                rows.append(
                    f"| {name} | {len(s.reasked)} | {len(s.claimed)} | {s.gaps} | {len(s.errors)} |"
                )
                lines = [
                    *(f"- 다시 물음 {x}" for x in s.reasked),
                    *(f"- 있다고 씀 {x}" for x in s.claimed),
                    *(f"- 오류 {x}" for x in s.errors),
                    "",
                    "전문:",
                    *(f"- {x}" for x in s.texts),
                ]
                details.append(f"\n### {name}\n\n" + "\n".join(lines))
    head = "| 모델 · 전후 | 다시 물음 | 있다고 씀 | gaps 에 남김 | 오류 |\n" + "| --- " * 5 + "|"
    merged = "\n".join([head, *merge_runs(rows)])
    return merged + "\n\n회차별:\n\n" + "\n".join([head, *rows]) + "\n" + "".join(details) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="쉼표로 「제공사:모델」 — 기본은 MATCH_LLM_MODELS 첫 모델")
    parser.add_argument("--runs", type=int, default=1, help="같은 경우를 몇 번씩")
    parser.add_argument("--out", type=Path, help="응답 저장 폴더")
    parser.add_argument("--env", type=Path, default=ROOT / ".env", help="키를 읽을 .env")
    parser.add_argument("--rescore", type=Path, help="저장한 응답만 다시 채점(LLM 안 부름)")
    args = parser.parse_args()

    data = load_data()
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
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = args.out or ROOT / "eval-runs" / f"asked-{stamp}"
        calls = len(data["cases"]) * len(MODES) * len(models) * args.runs
        print(f"LLM 호출 {calls}회(규칙 위반 시 더) → {out}")
        asyncio.run(run(models, args.runs, data, out, args.env))

    text = report(out, data["cases"])
    (out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
