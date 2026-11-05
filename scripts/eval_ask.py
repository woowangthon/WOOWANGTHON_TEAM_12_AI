"""시험 질문으로 꺼내기(/ask)를 채점한다 — 프롬프트 · 모델을 바꿀 때 같은 자료 · 같은 숫자로 견준다(AI#34).

    uv run python scripts/eval_ask.py                      # MATCH_LLM_MODELS 첫 모델로, 우리 프롬프트 · 범용 프롬프트
    uv run python scripts/eval_ask.py --prompts tuned --runs 2 --models openai:gpt-6.1-terra,openai:gpt-6-luna
    uv run python scripts/eval_ask.py --rescore eval-runs/ask-<시각>   # 저장한 응답만 다시 채점(LLM 안 부름)

LLM 호출 수 = 질문 수 × 모델 수 × 프롬프트 수 × runs(규칙 위반이면 질문마다 최대 LLM_MAX_ATTEMPTS 번).
응답은 --out 폴더(기본 eval-runs/ask-<시각>, 커밋하지 않음)에 저장한다.
모델 폴백 없이 고른 모델만 부른다. 키는 --env 파일(기본 레포의 .env)에서 읽고 화면에 찍지 않는다.

자료: samples/match/experiences.json(매칭과 같은 지어낸 경험) · samples/ask/queries.json(질문과 손으로 적은 정답 —
칸의 뜻은 samples/ask/README.md).

숫자의 뜻:
- 첫 추천 맞음: match 질문에서 첫 추천이 good 인 수(allowZero 면 0개 · ok 도 맞음)
- 0개여야 할 때 0개: none 질문에서 추천을 하나도 안 낸 수
- 엉뚱한 추천: bad 에 있는 경험을 추천한 수
- 팀 일을 내 일로: actor 가 TEAM · UNKNOWN 인 추천의 reason 에 「내가 · 제가 · 직접 · 주도 …」가 있고, 그 문장에
  「없 · 않 · 아니 · 모르 · 확인」이 없는 수(「내가 이끌었는지는 자료에 없다」는 빼고 센다). 걸린 후보 전부(빼기 전)도 찍는다
- 지어낸 수치: reason 의 숫자 중 그 경험(본문 · 근거 · 보완 · 기간)과 질문에 없는 숫자가 있는 추천 수
- 보완 씀: 보완이 실린 경험을 추천했을 때 evidence 에 보완을 넣은 수 / 그런 추천 수(S-25)
- 오류: 다시 시도 뒤에도 규칙을 어겨 LLM_INVALID_OUTPUT 등으로 끝난 수
- 시간: 질문 하나에 걸린 시간(다시 시도 포함)의 중앙값 · 최대 — 동기 호출이라 사용자가 이만큼 기다린다

낱말로 거른 것은 후보다 — 이유 전문을 summary.md 에 남기니 사람이 읽고 센다.
"""

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ask import SYSTEM_PROMPT  # noqa: E402
from app.schemas import MatchExperience  # noqa: E402
from scripts.eval_match import MINE_WORDS, _experience_text, _ints  # noqa: E402

DATA = ROOT / "samples" / "ask"
EXPERIENCES = ROOT / "samples" / "match" / "experiences.json"

# 범용 프롬프트 — 같은 출력 모양(JSON 스키마) · 같은 경험 목록에 지시만 준다. ID 검사도 같은 코드가 한다
GENERIC_PROMPT = "이 질문에 맞는 내 경험을 찾아 줘."
PROMPTS = {"tuned": SYSTEM_PROMPT, "generic": GENERIC_PROMPT}

# 「내 몫은 자료에 없다」처럼 부정하며 쓴 문장
DENIED = re.compile(r"없|않|아니|모르|모름|확인")


def load_data(only: list[str] | None) -> tuple[list[dict], list[dict]]:
    experiences = json.loads(EXPERIENCES.read_text(encoding="utf-8"))
    queries = json.loads((DATA / "queries.json").read_text(encoding="utf-8"))
    return experiences, [q for q in queries if not only or q["id"] in only]


def claims_mine(reason: str) -> bool:
    """TEAM · UNKNOWN 경험을 내 일로 말했나 — 낱말이 있는 문장에 부정이 없으면."""
    sentences = re.split(r"[.!?。]", reason)
    return any(MINE_WORDS.search(s) and not DENIED.search(s) for s in sentences)


@dataclass
class Score:
    total: int = 0
    match_total: int = 0
    first_right: int = 0
    none_total: int = 0
    none_zero: int = 0
    wrong: list[str] = field(default_factory=list)
    forced: list[str] = field(default_factory=list)
    first_wrong: list[str] = field(default_factory=list)
    team_as_mine: list[str] = field(default_factory=list)
    mine_words: list[str] = field(default_factory=list)
    made_up_numbers: list[str] = field(default_factory=list)
    supplement_recs: int = 0
    supplement_used: int = 0
    picks: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)


def score_dir(path: Path, experiences: list[dict], queries: list[dict]) -> Score:
    by_id = {e["id"]: MatchExperience.model_validate(e) for e in experiences}
    s = Score()
    for q in queries:
        f = path / f"{q['id']}.json"
        if not f.exists():
            continue
        record = json.loads(f.read_text(encoding="utf-8"))
        s.total += 1
        s.seconds.append(record.get("seconds", 0.0))
        if "error" in record:
            s.errors.append(f"{q['id']}: {record['error']}")
            continue
        recs = record["response"]["recommendations"]
        ids = [r["experienceId"] for r in recs]
        if q["kind"] == "none":
            s.none_total += 1
            s.none_zero += not ids
            if ids:
                s.forced.append(f"{q['id']}: {ids}")
        else:
            s.match_total += 1
            right = (ids and ids[0] in q["good"]) or (
                q.get("allowZero") and (not ids or ids[0] in q["ok"])
            )
            s.first_right += bool(right)
            if not right:
                s.first_wrong.append(f"{q['id']}: {ids or '없음'} (good {q['good']})")
        s.wrong += [f"{q['id']}: {i}" for i in ids if i in q["bad"]]
        known_ints = _ints(q["query"]) | set(by_id)
        for r in recs:
            exp = by_id[r["experienceId"]]
            line = f"{q['id']} → {exp.id}({exp.actor}): {r['reason']}"
            s.picks.append(line + f" {[(e['type'], e['id']) for e in r['evidence']]}")
            extra = _ints(r["reason"]) - _ints(_experience_text(exp)) - known_ints
            if extra:
                s.made_up_numbers.append(f"{line} — {sorted(extra)}")
            if exp.actor != "ME" and MINE_WORDS.search(r["reason"]):
                s.mine_words.append(line)
                if claims_mine(r["reason"]):
                    s.team_as_mine.append(line)
            if exp.supplements:
                s.supplement_recs += 1
                s.supplement_used += any(e["type"] == "SUPPLEMENT" for e in r["evidence"])
    return s


async def run(
    models: list[str],
    prompts: list[str],
    runs: int,
    experiences: list[dict],
    queries: list[dict],
    out: Path,
    env: Path,
) -> None:
    from dotenv import load_dotenv

    from app.ask import ask
    from app.config import load_settings, parse_models
    from app.errors import ApiError
    from app.llm import build_llm
    from app.schemas import AskRequest

    load_dotenv(env)  # 값은 화면에 찍지 않는다
    base = load_settings()
    gate = asyncio.Semaphore(4)
    clients = {}  # 모델마다 한 번만 만든다 — eval_route 의 까닭과 같다(AI#37)
    for model in models:
        settings = replace(base, match_models=parse_models(model))
        llm = build_llm(settings, settings.match_models, settings.llm_sync_call_timeout_seconds)
        clients[model] = (settings, llm)

    async def one(model: str, prompt: str, n: int, q: dict) -> None:
        settings, llm = clients[model]
        req = AskRequest.model_validate({"query": q["query"], "experiences": experiences})
        async with gate:
            started = time.monotonic()
            try:
                data = await ask(req, llm, settings, system=PROMPTS[prompt])
                record = {"response": json.loads(data.model_dump_json())}
            except ApiError as e:
                record = {"error": f"{e.code}: {e.message}"}
            record["seconds"] = round(time.monotonic() - started, 1)
        name = prompt + (f"-{n}" if runs > 1 else "")
        path = out / model.replace(":", "_").replace("/", "_") / name / f"{q['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        got = record.get("error") or f"추천 {len(record['response']['recommendations'])}개"
        print(f"  {model} · {name} · {q['id']} — {got} ({record['seconds']}초)")

    await asyncio.gather(
        *(
            one(m, p, n, q)
            for m in models
            for p in prompts
            for n in range(1, runs + 1)
            for q in queries
        )
    )


def report(out: Path, experiences: list[dict], queries: list[dict]) -> str:
    rows, details = [], []
    for model_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        for prompt_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            s = score_dir(prompt_dir, experiences, queries)
            if not s.total:
                continue
            name = f"{model_dir.name} · {prompt_dir.name}"
            secs = s.seconds or [0.0]
            rows.append(
                f"| {name} | {s.first_right}/{s.match_total} | {s.none_zero}/{s.none_total} "
                f"| {len(s.wrong)} | {len(s.team_as_mine)} ({len(s.mine_words)}) "
                f"| {len(s.made_up_numbers)} | {s.supplement_used}/{s.supplement_recs} "
                f"| {len(s.errors)} | {statistics.median(secs):.1f}초 · {max(secs):.1f}초 |"
            )
            lines = [
                *(f"- 첫 추천 틀림 {x}" for x in s.first_wrong),
                *(f"- 억지 추천 {x}" for x in s.forced),
                *(f"- 엉뚱한 추천 {x}" for x in s.wrong),
                *(f"- 팀 일을 내 일로 {x}" for x in s.team_as_mine),
                *(f"- 내 일 낱말(부정 포함) {x}" for x in s.mine_words),
                *(f"- 지어낸 수치 {x}" for x in s.made_up_numbers),
                *(f"- 오류 {x}" for x in s.errors),
                "",
                "추천 전문(사람이 읽고 센다):",
                *(f"- {x}" for x in s.picks),
            ]
            details.append(f"\n### {name}\n\n" + "\n".join(lines))
    head = (
        "| 모델 · 프롬프트 | 첫 추천 맞음 | 0개여야 할 때 0개 | 엉뚱한 추천 | 팀 일을 내 일로 (낱말 후보) "
        "| 지어낸 수치 | 보완 씀 | 오류 | 시간(중앙 · 최대) |\n" + "| --- " * 9 + "|"
    )
    return "\n".join([head, *rows]) + "\n" + "".join(details) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="쉼표로 「제공사:모델」 — 기본은 MATCH_LLM_MODELS 첫 모델")
    parser.add_argument("--prompts", default="tuned,generic", help="tuned · generic")
    parser.add_argument("--runs", type=int, default=1, help="같은 질문을 몇 번씩")
    parser.add_argument("--queries", help="쉼표로 질문 id — 기본은 전부")
    parser.add_argument("--out", type=Path, help="응답 저장 폴더")
    parser.add_argument("--env", type=Path, default=ROOT / ".env", help="키를 읽을 .env")
    parser.add_argument("--rescore", type=Path, help="저장한 응답만 다시 채점(LLM 안 부름)")
    args = parser.parse_args()

    experiences, queries = load_data(args.queries.split(",") if args.queries else None)
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
        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
        for p in prompts:
            if p not in PROMPTS:
                parser.error(f"프롬프트는 {list(PROMPTS)} 중 — {p!r}")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = args.out or ROOT / "eval-runs" / f"ask-{stamp}"
        calls = len(queries) * len(models) * len(prompts) * args.runs
        print(f"LLM 호출 {calls}회(규칙 위반 시 더) → {out}")
        asyncio.run(run(models, prompts, args.runs, experiences, queries, out, args.env))

    text = report(out, experiences, queries)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
