"""시험 공고로 공고 기준 경험 찾기(/posting-match)를 채점한다 — 프롬프트 · 모델을 바꿀 때 같은 자료 · 같은 숫자로(AI#122).

    uv run python scripts/eval_posting_match.py                    # MATCH_LLM_MODELS 첫 모델로, 우리 프롬프트 · 범용 프롬프트
    uv run python scripts/eval_posting_match.py --prompts tuned --runs 2
    uv run python scripts/eval_posting_match.py --rescore eval-runs/posting-match-<시각>   # 다시 채점만(LLM 안 부름)
    uv run python scripts/eval_posting_match.py --env ../다른-클론/.env         # 키를 다른 .env 에서(worktree)

LLM 호출 수 = 공고 수 × 더미 수 × 모델 수 × 프롬프트 수 × runs(규칙 위반이면 경우마다 최대 LLM_MAX_ATTEMPTS 번)
— 기본 2 × 2 × 1 × 2 = 8회. 응답은 --out 폴더(기본 eval-runs/posting-match-<시각>, 커밋하지 않음)의
<모델>/<프롬프트>/run-<N>/<공고>-<더미>.json 에 저장한다. 모델 폴백 없이 고른 모델만 부른다.

자료: samples/posting-match(README — 지어낸 공고 2건 · 경험 더미 2종 · 줄마다 사람 판정).

숫자의 뜻:
- 판정 일치: 줄 판정이 사람 판정(expect)과 같은 줄 수 / 줄 수 · 허용 포함: ok 까지 맞다고 친 수
- 억지로 맞춤: 사람이 NOT_FOUND 만 맞다고 본 줄(ok 없음)을 SUPPORTED · PARTIAL 로 판정한 수 / 그런 줄 수.
  「개발만 × 마케팅」 칸은 그중 개발 경험만 있는 더미의 마케팅 공고 줄만 — 다른 서비스가 억지로 맞춘 그 경우(ETC#323)
- 엉뚱한 근거: 줄의 근거로 bad 경험을 고른 수(다른 분야 · 그 일과 상관없는 경험)
- 지어낸 수치(되묻기): 질문의 숫자 중 그 경험(본문 · 근거 · 보완 · 기간) · 그 줄에 없는 숫자가 있는 질문 수
- 답 유도(되묻기): 「…죠?」「맞나요?」처럼 확인만 받는 어미 수
- 되묻기 수: 질문 수 합(0개인 경우 수)
- 오류: 다시 시도 뒤에도 규칙을 어겨 LLM_INVALID_OUTPUT 등으로 끝난 경우 수
- 시간: 경우 하나(다시 시도 포함)의 중앙값 · 최대

경험 · 근거 ID 를 지어내는 것은 코드(check_output)가 막는다 — 숫자로 남는 「만들어 낸 근거」는 엉뚱한 근거와
지어낸 수치다. 낱말로 거른 것은 후보 — 판정 · 질문 전문을 summary.md 에 남기니 사람이 읽고 센다.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.posting_match import SYSTEM_PROMPT  # noqa: E402
from app.schemas import MatchExperience  # noqa: E402
from scripts.eval_match import LEADING, _experience_text, _ints  # noqa: E402
from scripts.eval_runs import merge_runs, run_dirs, run_path  # noqa: E402

DATA = ROOT / "samples" / "posting-match"
MATCH_EXPERIENCES = ROOT / "samples" / "match" / "experiences.json"
# 개발 경험만 있는 더미 — samples/match 의 개발 활동(바꿔요 · 알고리즘 스터디 · 모여봄 · 가계부 웹앱)
DEV_ACTIVITIES = {1, 2, 3, 6}
DUMMIES = ("dev", "mixed")
DUMMY_NAMES = {"dev": "개발만", "mixed": "섞임"}

# 범용 프롬프트 — 같은 출력 모양 · 같은 입력에 지시만 준다. 줄 · ID 검사도 같은 코드가 한다
GENERIC_PROMPT = "공고 요건 줄마다 내 경험이 받쳐 주는지 판정해 줘."
PROMPTS = {"tuned": SYSTEM_PROMPT, "generic": GENERIC_PROMPT}


def load_data() -> tuple[dict[str, list[dict]], list[dict]]:
    base = json.loads(MATCH_EXPERIENCES.read_text(encoding="utf-8"))
    extra = json.loads((DATA / "extra-experiences.json").read_text(encoding="utf-8"))
    dummies = {
        "dev": [e for e in base if e["activityId"] in DEV_ACTIVITIES],
        "mixed": base + extra,
    }
    postings = json.loads((DATA / "postings.json").read_text(encoding="utf-8"))
    return dummies, postings


def request_body(posting: dict, experiences: list[dict]) -> dict:
    lines = [{k: p[k] for k in ("id", "section", "text", "ksa")} for p in posting["lines"]]
    return {"jobRole": posting["jobRole"], "lines": lines, "experiences": experiences}


def case_name(posting: dict, dummy: str) -> str:
    return f"{posting['id']}-{dummy}"


@dataclass
class Score:
    cases: int = 0
    lines: int = 0
    exact: int = 0
    allowed: int = 0
    strict_none: int = 0
    forced: list[str] = field(default_factory=list)
    dev_marketing_none: int = 0
    dev_marketing_forced: int = 0
    mismatches: list[str] = field(default_factory=list)
    bad_picks: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    zero_questions: int = 0
    made_up_numbers: list[str] = field(default_factory=list)
    leading: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)


def score_dir(path: Path, dummies: dict[str, list[dict]], postings: list[dict]) -> Score:
    s = Score()
    for posting in postings:
        line_by_id = {p["id"]: p for p in posting["lines"]}
        for dummy in DUMMIES:
            name = case_name(posting, dummy)
            f = path / f"{name}.json"
            if not f.exists():
                continue
            exps = {e["id"]: MatchExperience.model_validate(e) for e in dummies[dummy]}
            record = json.loads(f.read_text(encoding="utf-8"))
            s.cases += 1
            s.seconds.append(record.get("seconds", 0.0))
            if "error" in record:
                s.errors.append(f"{name}: {record['error']}")
                continue
            res = record["response"]
            for j in res["lines"]:
                line = line_by_id[j["id"]]
                answer = line[dummy]
                s.lines += 1
                s.exact += j["status"] == answer["expect"]
                s.allowed += j["status"] in (answer["expect"], *answer["ok"])
                tag = f"{name} 줄 {j['id']} 「{line['text']}」 {j['status']} {j['experienceIds']}"
                if j["status"] != answer["expect"]:
                    s.mismatches.append(f"{tag} (사람 {answer['expect']} · 허용 {answer['ok']})")
                if answer["expect"] == "NOT_FOUND" and not answer["ok"]:
                    s.strict_none += 1
                    dev_marketing = dummy == "dev" and posting["id"] == "marketing"
                    s.dev_marketing_none += dev_marketing
                    if j["status"] != "NOT_FOUND":
                        s.forced.append(tag)
                        s.dev_marketing_forced += dev_marketing
                bad = [x for x in j["experienceIds"] if x in line["bad"]]
                if bad:
                    s.bad_picks.append(f"{tag} — 엉뚱한 {bad}")
            s.zero_questions += not res["clarifications"]
            for c in res["clarifications"]:
                exp = exps[c["experienceId"]]
                line = line_by_id[c["lineId"]]
                q = f"{name} 줄 {c['lineId']} → {exp.id}({exp.actor}): {c['question']}"
                s.questions.append(q)
                extra = _ints(c["question"]) - _ints(_experience_text(exp)) - _ints(line["text"])
                if extra:
                    s.made_up_numbers.append(f"{q} — {sorted(extra)}")
                if LEADING.search(c["question"]):
                    s.leading.append(q)
    return s


async def run(
    models: list[str],
    prompts: list[str],
    runs: int,
    dummies: dict[str, list[dict]],
    postings: list[dict],
    out: Path,
    env: Path,
) -> None:
    from dotenv import load_dotenv

    from app.config import load_settings, parse_models
    from app.errors import ApiError
    from app.llm import build_llm
    from app.posting_match import posting_match
    from app.schemas import PostingMatchRequest

    load_dotenv(env)  # 값은 화면에 찍지 않는다
    base = load_settings()
    gate = asyncio.Semaphore(4)
    clients = {}  # 모델마다 한 번만 만든다 — eval_route 의 까닭과 같다(AI#37)
    for model in models:
        settings = replace(base, match_models=parse_models(model))
        clients[model] = (settings, build_llm(settings, settings.match_models))

    async def one(model: str, prompt: str, n: int, posting: dict, dummy: str) -> None:
        settings, llm = clients[model]
        req = PostingMatchRequest.model_validate(request_body(posting, dummies[dummy]))
        async with gate:
            started = time.monotonic()
            try:
                data = await posting_match(req, llm, settings, system=PROMPTS[prompt])
                record = {"response": json.loads(data.model_dump_json())}
            except ApiError as e:
                record = {"error": f"{e.code}: {e.message}"}
            record["seconds"] = round(time.monotonic() - started, 1)
        model_dir = out / model.replace(":", "_").replace("/", "_") / prompt
        path = run_path(model_dir, n) / f"{case_name(posting, dummy)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        got = record.get("error") or " ".join(j["status"][0] for j in record["response"]["lines"])
        print(
            f"  {model} · {prompt} · run-{n} · {case_name(posting, dummy)} — {got} ({record['seconds']}초)"
        )

    await asyncio.gather(
        *(
            one(m, p, n, posting, d)
            for m in models
            for p in prompts
            for n in range(1, runs + 1)
            for posting in postings
            for d in DUMMIES
        )
    )


def report(out: Path, dummies: dict[str, list[dict]], postings: list[dict]) -> str:
    rows, details = [], []
    for model_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        for prompt_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            for tail, path in run_dirs(prompt_dir):
                s = score_dir(path, dummies, postings)
                if not s.cases:
                    continue
                name = f"{model_dir.name} · {prompt_dir.name}{tail}"
                secs = s.seconds or [0.0]
                rows.append(
                    f"| {name} | {s.exact}/{s.lines} | {s.allowed}/{s.lines} "
                    f"| {len(s.forced)}/{s.strict_none} | {s.dev_marketing_forced}/{s.dev_marketing_none} "
                    f"| {len(s.bad_picks)} | {len(s.made_up_numbers)} | {len(s.leading)} "
                    f"| {len(s.questions)} ({s.zero_questions}) | {len(s.errors)} "
                    f"| {statistics.median(secs):.1f}초 · {max(secs):.1f}초 |"
                )
                lines = [
                    *(f"- 억지로 맞춤 {x}" for x in s.forced),
                    *(f"- 엉뚱한 근거 {x}" for x in s.bad_picks),
                    *(f"- 지어낸 수치 {x}" for x in s.made_up_numbers),
                    *(f"- 답 유도 {x}" for x in s.leading),
                    *(f"- 오류 {x}" for x in s.errors),
                    "",
                    "사람 판정과 다른 줄:",
                    *(f"- {x}" for x in s.mismatches),
                    "",
                    "되묻기 전문(사람이 읽고 센다 — 공고 조건을 겪었다고 전제하나 · 이미 있는 사실을 묻나):",
                    *(f"- {x}" for x in s.questions),
                ]
                details.append(f"\n### {name}\n\n" + "\n".join(lines))
    head = (
        "| 모델 · 프롬프트 | 판정 일치 | 허용 포함 | 억지로 맞춤 | 개발만 × 마케팅 억지 | 엉뚱한 근거 "
        "| 지어낸 수치(되묻기) | 답 유도 | 되묻기 수 (0개 경우) | 오류 | 시간(중앙 · 최대) |\n"
        + "| --- " * 11
        + "|"
    )
    merged = "\n".join([head, *merge_runs(rows)])
    return merged + "\n\n회차별:\n\n" + "\n".join([head, *rows]) + "\n" + "".join(details) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="쉼표로 「제공사:모델」 — 기본은 MATCH_LLM_MODELS 첫 모델")
    parser.add_argument("--prompts", default="tuned,generic", help="tuned · generic")
    parser.add_argument("--runs", type=int, default=1, help="같은 경우를 몇 번씩")
    parser.add_argument("--out", type=Path, help="응답 저장 폴더")
    parser.add_argument("--env", type=Path, default=ROOT / ".env", help="키를 읽을 .env")
    parser.add_argument("--rescore", type=Path, help="저장한 응답만 다시 채점(LLM 안 부름)")
    args = parser.parse_args()

    dummies, postings = load_data()
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
        out = args.out or ROOT / "eval-runs" / f"posting-match-{stamp}"
        calls = len(postings) * len(DUMMIES) * len(models) * len(prompts) * args.runs
        print(f"LLM 호출 {calls}회(규칙 위반 시 더) → {out}")
        asyncio.run(run(models, prompts, args.runs, dummies, postings, out, args.env))

    text = report(out, dummies, postings)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
