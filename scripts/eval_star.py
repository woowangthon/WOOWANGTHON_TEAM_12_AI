"""시험 활동으로 STAR 칸 나누기(/star)를 채점한다 — 프롬프트 · 모델을 바꿀 때 같은 자료 · 같은 숫자로(AI#125).

    uv run python scripts/eval_star.py                    # MATCH_LLM_MODELS 첫 모델로, 우리 프롬프트 · 범용 프롬프트
    uv run python scripts/eval_star.py --prompts tuned --runs 2
    uv run python scripts/eval_star.py --rescore eval-runs/star-<시각>   # 다시 채점만(LLM 안 부름)
    uv run python scripts/eval_star.py --env ../다른-클론/.env         # 키를 다른 .env 에서(worktree)

LLM 호출 수 = 활동 수(4) × 모델 수 × 프롬프트 수 × runs(규칙 위반이면 활동마다 최대 LLM_MAX_ATTEMPTS 번)
— 기본 4 × 1 × 2 = 8회. 응답은 --out 폴더(기본 eval-runs/star-<시각>, 커밋하지 않음)의
<모델>/<프롬프트>/run-<N>/<활동>.json 에 저장한다. 모델 폴백 없이 고른 모델만 부른다.

자료: samples/star(README — 지어낸 활동 4개 · 경험 20개 · 칸마다 사람 판정).

숫자의 뜻:
- 원문 그대로: 응답의 찬 칸 중 칸 글이 그 출처 글(근거 발췌문 · 보완 답)에 적혀 있는 칸 / 찬 칸.
  코드(app/star.py)가 어긴 칸을 빈칸으로 내리므로 100% 여야 한다 — 내린 칸 수는 따로 센다
- 내린 칸: 마지막 시도에서도 칸 규칙을 어겨 빈칸으로 내린 칸 수 · 다시 시도한 활동 수
- 칸 판정: 사람 판정(expect)과 찬 / 빈 · 출처가 맞는 칸 / 판정한 칸(expect 가 null 인 칸은 안 셈).
  억지로 채움 = 사람은 빈칸인데 채운 칸 · 놓침 = 사람은 채울 수 있다는데 비운 칸
- 만들어 낸 배운 점: R 을 근거(CITATION)로 채웠는데 칸 글이 근거 속 배운 점 · 느낀 점 · 다짐 문장(learned)과 겹치거나
  배운 점 낱말을 담은 수. 0 이어야 한다(배운 점은 사용자 답으로만). 같은 근거의 결과 사실만 고르면 세지 않는다
- 빈칸 질문: 질문 수(질문 없는 빈칸 수) · 칸에 맞음 후보(칸마다 낱말 — 아래 SLOT_WORDS) · 아무 데나 붙는 질문 ·
  답 유도 · 지어낸 수치(그 경험 글에 없는 숫자)
- 오류: 응답 전체 위반 등으로 실패한 활동 수 · 시간(활동 하나, 다시 시도 포함)의 중앙값 · 최대

낱말로 거른 것은 후보 — 칸 · 질문 전문을 summary.md 에 남기니 사람이 읽고 센다(원문에 없는 전제 · 이미 있는 사실).
"""

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.star import LEARNED, SYSTEM_PROMPT, _squash, written_in  # noqa: E402
from scripts.eval_match import LEADING, VAGUE  # noqa: E402
from scripts.eval_runs import merge_runs, run_dirs, run_path  # noqa: E402

DATA = ROOT / "samples" / "star" / "activities.json"

GENERIC_PROMPT = "경험마다 S · T · A · R 칸으로 나눠 줘."
PROMPTS = {"tuned": SYSTEM_PROMPT, "generic": GENERIC_PROMPT}

# 칸에 맞는 질문인지 후보로 거르는 낱말 — 사람이 summary 를 읽고 확인한다
SLOT_WORDS = {
    "S": re.compile(r"왜|계기|배경|상황|문제|이유|필요|어떤 일"),
    "T": re.compile(r"맡|역할|담당|목표|과제|몫|업무"),
    "A": re.compile(r"어떻게|무엇을 하|무엇을 했|직접|시도|방법|노력|하셨|했나요|하였"),
    "R": re.compile(r"달라|결과|배운|배웠|느낀|느끼|변화|성과|반응|어떻게 되|나아"),
}
REQUEST_KEYS = ("id", "content", "actor", "competencies", "citations", "supplements")


def load_data() -> list[dict]:
    return json.loads(DATA.read_text(encoding="utf-8"))


def request_body(activity: dict) -> dict:
    return {"experiences": [{k: e[k] for k in REQUEST_KEYS} for e in activity["experiences"]]}


def _ints(text: str) -> set[int]:
    return {int(n) for n in re.findall(r"\d+", unicodedata.normalize("NFKC", text))}


def _source(exp: dict, slot: dict) -> str | None:
    if slot["sourceType"] == "CITATION":
        items = [c["quote"] for c in exp["citations"] if c["id"] == slot["sourceId"]]
    else:
        items = [s["answer"] for s in exp["supplements"] if s["id"] == slot["sourceId"]]
    return items[0] if items else None


def _learned(exp: dict, text: str) -> bool:
    """R 칸 글이 근거의 배운 점 · 느낀 점 · 다짐 문장(learned)과 겹치거나 배운 점 낱말을 담나."""
    got = _squash(text)
    return bool(LEARNED.search(text)) or any(
        got in _squash(x) or _squash(x) in got for x in exp["learned"]
    )


def _key(slot: dict) -> str:
    return ("C" if slot["sourceType"] == "CITATION" else "P") + str(slot["sourceId"])


@dataclass
class Score:
    activities: int = 0
    filled: int = 0
    verbatim: int = 0
    not_verbatim: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    retried: int = 0
    judged: int = 0
    right: int = 0
    forced: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    wrong_source: list[str] = field(default_factory=list)
    made_up_learning: list[str] = field(default_factory=list)
    team_t: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    no_question: int = 0
    fits: int = 0
    unfit: list[str] = field(default_factory=list)
    vague: list[str] = field(default_factory=list)
    leading: list[str] = field(default_factory=list)
    made_up_numbers: list[str] = field(default_factory=list)
    fills: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)


def score_dir(path: Path, activities: list[dict]) -> Score:
    s = Score()
    for activity in activities:
        f = path / f"{activity['id']}.json"
        if not f.exists():
            continue
        record = json.loads(f.read_text(encoding="utf-8"))
        s.activities += 1
        s.seconds.append(record.get("seconds", 0.0))
        if "error" in record:
            s.errors.append(f"{activity['id']}: {record['error']}")
            continue
        s.dropped += [f"{activity['id']} {d}" for d in record.get("dropped", [])]
        s.retried += record.get("attempt", 1) > 1
        exps = {e["id"]: e for e in activity["experiences"]}
        for got in record["response"]["experiences"]:
            exp = exps[got["id"]]
            for slot in got["slots"]:
                name = slot["slot"]
                tag = f"{exp['id']}({exp['actor']}) {name}"
                expect = exp["expect"][name]
                filled = slot["sourceType"] is not None
                if filled:
                    s.filled += 1
                    source = _source(exp, slot)
                    if source is not None and written_in(slot["text"], source):
                        s.verbatim += 1
                    else:
                        s.not_verbatim.append(f"{tag} {_key(slot)}: {slot['text']}")
                    s.fills.append(f"{tag} ← {_key(slot)} 「{slot['text']}」")
                    if name == "R" and slot["sourceType"] == "CITATION":
                        if _learned(exp, slot["text"]):
                            s.made_up_learning.append(f"{tag} 「{slot['text']}」")
                    if name == "T" and slot["sourceType"] == "CITATION" and exp["actor"] != "ME":
                        s.team_t.append(f"{tag} 「{slot['text']}」")
                else:
                    q = slot["question"]
                    if q is None:
                        s.no_question += 1
                    else:
                        line = f"{tag}: {q}"
                        s.questions.append(line)
                        if SLOT_WORDS[name].search(q):
                            s.fits += 1
                        else:
                            s.unfit.append(line)
                        if VAGUE.search(q):
                            s.vague.append(line)
                        if LEADING.search(q):
                            s.leading.append(line)
                        known = " ".join(
                            [exp["content"]]
                            + [c["quote"] for c in exp["citations"]]
                            + [f"{x['question']} {x['answer']}" for x in exp["supplements"]]
                        )
                        extra = _ints(q) - _ints(known)
                        if extra:
                            s.made_up_numbers.append(f"{line} — {sorted(extra)}")
                if expect is None:
                    continue
                s.judged += 1
                if not expect:
                    if filled:
                        s.forced.append(f"{tag} ← {_key(slot)} 「{slot['text']}」")
                    else:
                        s.right += 1
                elif not filled:
                    s.missed.append(f"{tag} (사람 {expect})")
                elif _key(slot) in expect:
                    s.right += 1
                else:
                    s.wrong_source.append(f"{tag} ← {_key(slot)} (사람 {expect})")
    return s


async def run(
    models: list[str], prompts: list[str], runs: int, activities: list[dict], out: Path, env: Path
) -> None:
    from dotenv import load_dotenv

    from app.config import load_settings, parse_models
    from app.errors import ApiError
    from app.llm import build_llm
    from app.schemas import StarRequest
    from app.star import star_run

    load_dotenv(env)  # 값은 화면에 찍지 않는다
    base = load_settings()
    gate = asyncio.Semaphore(4)
    clients = {}  # 모델마다 한 번만 만든다 — eval_route 의 까닭과 같다(AI#37)
    for model in models:
        settings = replace(base, match_models=parse_models(model))
        clients[model] = (settings, build_llm(settings, settings.match_models))

    async def one(model: str, prompt: str, n: int, activity: dict) -> None:
        settings, llm = clients[model]
        req = StarRequest.model_validate(request_body(activity))
        async with gate:
            started = time.monotonic()
            try:
                got = await star_run(req, llm, settings, system=PROMPTS[prompt])
                record = {
                    "response": json.loads(got.response.model_dump_json()),
                    "dropped": got.dropped,
                    "attempt": got.attempt,
                }
            except ApiError as e:
                record = {"error": f"{e.code}: {e.message}"}
            record["seconds"] = round(time.monotonic() - started, 1)
        model_dir = out / model.replace(":", "_").replace("/", "_") / prompt
        path = run_path(model_dir, n) / f"{activity['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        if "error" in record:
            got_text = record["error"]
        else:
            got_text = (
                " ".join(
                    "".join(x["slot"] if x["sourceType"] else "·" for x in e["slots"])
                    for e in record["response"]["experiences"]
                )
                + f" 내림 {len(record['dropped'])}"
            )
        print(
            f"  {model} · {prompt} · run-{n} · {activity['id']} — {got_text} ({record['seconds']}초)"
        )

    await asyncio.gather(
        *(
            one(m, p, n, a)
            for m in models
            for p in prompts
            for n in range(1, runs + 1)
            for a in activities
        )
    )


def _pct(a: int, b: int) -> str:
    return f"{a}/{b}"


def report(out: Path, activities: list[dict]) -> str:
    rows, details = [], []
    for model_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        for prompt_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            for tail, path in run_dirs(prompt_dir):
                s = score_dir(path, activities)
                if not s.activities:
                    continue
                name = f"{model_dir.name} · {prompt_dir.name}{tail}"
                secs = s.seconds or [0.0]
                rows.append(
                    f"| {name} | {_pct(s.verbatim, s.filled)} | {len(s.dropped)} ({s.retried}) "
                    f"| {_pct(s.right, s.judged)} | {len(s.forced)} | {len(s.missed)} | {len(s.wrong_source)} "
                    f"| {len(s.made_up_learning)} | {len(s.team_t)} "
                    f"| {len(s.questions)} ({s.no_question}) | {_pct(s.fits, len(s.questions))} "
                    f"| {len(s.vague)} | {len(s.leading)} | {len(s.made_up_numbers)} | {len(s.errors)} "
                    f"| {statistics.median(secs):.1f}초 · {max(secs):.1f}초 |"
                )
                lines = [
                    *(f"- 원문과 다름 {x}" for x in s.not_verbatim),
                    *(f"- 내린 칸 {x}" for x in s.dropped),
                    *(f"- 만들어 낸 배운 점 {x}" for x in s.made_up_learning),
                    *(f"- TEAM · UNKNOWN 의 T 를 근거로 {x}" for x in s.team_t),
                    *(f"- 억지로 채움 {x}" for x in s.forced),
                    *(f"- 놓침 {x}" for x in s.missed),
                    *(f"- 다른 출처 {x}" for x in s.wrong_source),
                    *(f"- 칸에 안 맞는 질문(후보) {x}" for x in s.unfit),
                    *(f"- 아무 데나 붙는 질문 {x}" for x in s.vague),
                    *(f"- 답 유도 {x}" for x in s.leading),
                    *(f"- 지어낸 수치 {x}" for x in s.made_up_numbers),
                    *(f"- 오류 {x}" for x in s.errors),
                    "",
                    "찬 칸 전문:",
                    *(f"- {x}" for x in s.fills),
                    "",
                    "빈칸 질문 전문(사람이 읽고 센다 — 원문에 없는 전제 · 이미 있는 사실 · 칸에 맞나):",
                    *(f"- {x}" for x in s.questions),
                ]
                details.append(f"\n### {name}\n\n" + "\n".join(lines))
    head = (
        "| 모델 · 프롬프트 | 원문 그대로 | 내린 칸 (다시 시도) | 칸 판정 | 억지로 채움 | 놓침 | 다른 출처 "
        "| 만들어 낸 배운 점 | TEAM T 근거 | 빈칸 질문 (질문 없음) | 칸에 맞음(후보) "
        "| 아무 데나 | 답 유도 | 지어낸 수치 | 오류 | 시간(중앙 · 최대) |\n" + "| --- " * 16 + "|"
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

    activities = load_data()
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
        out = args.out or ROOT / "eval-runs" / f"star-{stamp}"
        calls = len(activities) * len(models) * len(prompts) * args.runs
        print(f"LLM 호출 {calls}회(규칙 위반 시 더) → {out}")
        asyncio.run(run(models, prompts, args.runs, activities, out, args.env))

    text = report(out, activities)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
