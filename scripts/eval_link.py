"""시험 자료로 이어짐 제안(/link-suggestions)을 채점한다 — 프롬프트 · 모델을 바꿀 때 같은 자료 · 같은 숫자로 견준다(AI#27).

    uv run python scripts/eval_link.py                      # MATCH_LLM_MODELS 첫 모델로, 우리 프롬프트 · 범용 프롬프트
    uv run python scripts/eval_link.py --prompts tuned --runs 2 --models openai:gpt-6.1-terra,openai:gpt-6-luna
    uv run python scripts/eval_link.py --rescore eval-runs/link-<시각>   # 저장한 응답만 다시 채점(LLM 안 부름)

LLM 호출 수 = 자료 수 × 모델 수 × 프롬프트 수 × runs(규칙 위반이면 자료마다 최대 LLM_MAX_ATTEMPTS 번).
응답은 --out 폴더(기본 eval-runs/link-<시각>, 커밋하지 않음)에 저장한다.
모델 폴백 없이 고른 모델만 부른다. 키는 --env 파일(기본 레포의 .env)에서 읽고 화면에 찍지 않는다.

자료: samples/link/inputs.json(지어낸 활동과 손으로 적은 정답 — 칸의 뜻은 samples/link/README.md).

숫자의 뜻:
- 찾음: 정답 `linked` 후보 중 고른 수
- 억지 제안: `linked` · `maybe` 어디에도 없는 후보를 고른 수 — 날짜 순서 · 같은 낱말로 잇는 실패
- 0개여야 할 자료: `linked` · `maybe` 가 비었는데 1개 이상 낸 자료 수
- 단정: 이유에서 「」 인용을 뺀 뒤 CAUSAL_PATTERN(「때문에」 「이어졌다」 …)에 걸린 수
- 평가: 이유가 PRAISE_PATTERN(「돋보」 「훌륭」 …)에 걸린 수
- 기대지 않음: 이유가 출발 활동의 글(계기 · 이름 · 경험)과 후보의 글 양쪽과 NGRAM 글자 이상 겹치지 않는 수 — 날짜뿐인 이유가 여기 걸린다
- 없는 인용: 이유의 「」 인용이 입력에 없는 수(공백 · 문장 부호 무시)
- 오류: 다시 시도 뒤에도 규칙을 어겨 LLM_INVALID_OUTPUT 등으로 끝난 수
- 시간: 자료 하나에 걸린 시간(다시 시도 포함)의 중앙값 · 최대 — 동기 호출이라 사용자가 이만큼 기다린다

정규식은 후보를 고를 뿐이다 — 이유 전문을 summary.md 에 남기니 사람이 읽고 센다.
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

from app.link_suggest import SYSTEM_PROMPT  # noqa: E402

DATA = ROOT / "samples" / "link"

# 범용 프롬프트 — 같은 출력 모양(JSON 스키마)에 지시만 준다
GENERIC_PROMPT = "이 활동이 어떤 후보 활동으로 이어졌는지 찾아서 이유를 적어 줘. 없으면 빈 배열."
PROMPTS = {"tuned": SYSTEM_PROMPT, "generic": GENERIC_PROMPT}

CAUSAL_PATTERN = re.compile(
    r"때문|덕분|이어졌(?:다|어요|습니다|고|으며|음)|이어진 것(?:이|입)|이어지게 (?:됐|되었)"
    r"|(?:해서|하여|하면서|하고자|하려고) 시작했|계기가 (?:되|됐)|계기로 |(?:으로|로) 인해"
    r"|비롯(?:됐|되었|된)|결실|발전(?:했|시켰)"
)
PRAISE_PATTERN = re.compile(r"돋보|훌륭|대단|멋져|멋집|뛰어나|성장했|성장이")
_QUOTE = re.compile(r"「([^「」]*)」")
NGRAM = 3


def load_inputs(only: list[str] | None) -> list[dict]:
    items = json.loads((DATA / "inputs.json").read_text(encoding="utf-8"))
    return [i for i in items if not only or i["id"] in only]


def _plain(text: str) -> str:
    return re.sub(r"[^\w]", "", text)


def _ngrams(text: str) -> set[str]:
    text = _plain(text)
    return {text[i : i + NGRAM] for i in range(len(text) - NGRAM + 1)}


def activity_text(activity: dict) -> str:
    parts = [activity["name"], activity.get("note") or ""]
    return " ".join([*parts, *(e["content"] for e in activity["experiences"])])


def is_causal(reason: str) -> bool:
    """「」 안은 사용자가 쓴 말을 옮긴 것이다 — 그건 단정이 아니다."""
    return bool(CAUSAL_PATTERN.search(_QUOTE.sub("", reason)))


def leans_on_both(reason: str, origin: dict, candidate: dict) -> bool:
    words = _ngrams(reason)
    return bool(words & _ngrams(activity_text(origin))) and bool(
        words & _ngrams(activity_text(candidate))
    )


def invented_quotes(reason: str, origin: dict, candidates: list[dict]) -> list[str]:
    body = _plain(" ".join(activity_text(a) for a in (origin, *candidates)))
    return [q for q in _QUOTE.findall(reason) if _plain(q) not in body]


@dataclass
class Score:
    total: int = 0
    linked: int = 0
    found: int = 0
    suggested: int = 0
    forced: list[str] = field(default_factory=list)
    zero_failed: list[str] = field(default_factory=list)
    causal: list[str] = field(default_factory=list)
    praise: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    no_quote: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)


def score_dir(path: Path, inputs: list[dict]) -> Score:
    s = Score()
    for item in inputs:
        f = path / f"{item['id']}.json"
        if not f.exists():
            continue
        record = json.loads(f.read_text(encoding="utf-8"))
        s.total += 1
        s.linked += len(item["linked"])
        s.seconds.append(record.get("seconds", 0.0))
        if "error" in record:
            s.errors.append(f"{item['id']}: {record['error']}")
            continue
        origin, candidates = item["request"]["from"], item["request"]["candidates"]
        by_id = {c["id"]: c for c in candidates}
        got = record["response"]["suggestions"]
        s.suggested += len(got)
        ok = {*item["linked"], *item["maybe"]}
        if got and not ok:
            s.zero_failed.append(f"{item['id']}: {[g['toActivityId'] for g in got]}")
        for g in got:
            to, reason = g["toActivityId"], g["reason"]
            line = f"{item['id']} → {to}: {reason}"
            s.reasons.append(line)
            s.found += to in item["linked"]
            if to not in ok:
                s.forced.append(line)
            if is_causal(reason):
                s.causal.append(line)
            if PRAISE_PATTERN.search(reason):
                s.praise.append(line)
            if not leans_on_both(reason, origin, by_id[to]):
                s.unsupported.append(line)
            if invented_quotes(reason, origin, candidates):
                s.no_quote.append(line)
    return s


async def run(
    models: list[str], prompts: list[str], runs: int, inputs: list[dict], out: Path, env: Path
) -> None:
    from dotenv import load_dotenv

    from app.config import load_settings, parse_models
    from app.errors import ApiError
    from app.link_suggest import suggest_links
    from app.llm import build_llm
    from app.schemas import LinkSuggestionsRequest

    load_dotenv(env)  # 값은 화면에 찍지 않는다
    base = load_settings()
    gate = asyncio.Semaphore(4)
    clients = {}  # 모델마다 한 번만 만든다 — eval_route 의 까닭과 같다
    for model in models:
        settings = replace(base, match_models=parse_models(model))
        llm = build_llm(settings, settings.match_models, settings.llm_sync_call_timeout_seconds)
        clients[model] = (settings, llm)

    async def one(model: str, prompt: str, n: int, item: dict) -> None:
        settings, llm = clients[model]
        req = LinkSuggestionsRequest.model_validate(item["request"])
        async with gate:
            started = time.monotonic()
            try:
                data = await suggest_links(req, llm, settings, system=PROMPTS[prompt])
                record = {"response": json.loads(data.model_dump_json())}
            except ApiError as e:
                record = {"error": f"{e.code}: {e.message}"}
            record["seconds"] = round(time.monotonic() - started, 1)
        name = prompt + (f"-{n}" if runs > 1 else "")
        path = out / model.replace(":", "_").replace("/", "_") / name / f"{item['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        got = record.get("error") or f"제안 {len(record['response']['suggestions'])}개"
        print(f"  {model} · {name} · {item['id']} — {got} ({record['seconds']}초)")

    await asyncio.gather(
        *(
            one(m, p, n, item)
            for m in models
            for p in prompts
            for n in range(1, runs + 1)
            for item in inputs
        )
    )


def report(out: Path, inputs: list[dict]) -> str:
    rows, details = [], []
    for model_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        for prompt_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            s = score_dir(prompt_dir, inputs)
            if not s.total:
                continue
            name = f"{model_dir.name} · {prompt_dir.name}"
            secs = s.seconds or [0.0]
            rows.append(
                f"| {name} | {s.found}/{s.linked} | {s.suggested} | {len(s.forced)} "
                f"| {len(s.zero_failed)} | {len(s.causal)} | {len(s.praise)} "
                f"| {len(s.unsupported)} | {len(s.no_quote)} | {len(s.errors)} "
                f"| {statistics.median(secs):.1f}초 · {max(secs):.1f}초 |"
            )
            lines = [
                *(f"- 0개여야 함 {x}" for x in s.zero_failed),
                *(f"- 억지 제안 {x}" for x in s.forced),
                *(f"- 단정 {x}" for x in s.causal),
                *(f"- 평가 {x}" for x in s.praise),
                *(f"- 기대지 않음 {x}" for x in s.unsupported),
                *(f"- 없는 인용 {x}" for x in s.no_quote),
                *(f"- 오류 {x}" for x in s.errors),
                "",
                "이유 전문(사람이 읽고 센다):",
                *(f"- {x}" for x in s.reasons),
            ]
            details.append(f"\n### {name}\n\n" + "\n".join(lines))
    head = (
        "| 모델 · 프롬프트 | 찾음 | 제안 | 억지 제안 | 0개여야 할 자료 | 단정 | 평가 "
        "| 기대지 않음 | 없는 인용 | 오류 | 시간(중앙 · 최대) |\n" + "| --- " * 11 + "|"
    )
    return "\n".join([head, *rows]) + "\n" + "".join(details) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="쉼표로 「제공사:모델」 — 기본은 MATCH_LLM_MODELS 첫 모델")
    parser.add_argument("--prompts", default="tuned,generic", help="tuned · generic")
    parser.add_argument("--runs", type=int, default=1, help="같은 자료를 몇 번씩")
    parser.add_argument("--inputs", help="쉼표로 자료 id — 기본은 전부")
    parser.add_argument("--out", type=Path, help="응답 저장 폴더")
    parser.add_argument("--env", type=Path, default=ROOT / ".env", help="키를 읽을 .env")
    parser.add_argument("--rescore", type=Path, help="저장한 응답만 다시 채점(LLM 안 부름)")
    args = parser.parse_args()

    inputs = load_inputs(args.inputs.split(",") if args.inputs else None)
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
        out = args.out or ROOT / "eval-runs" / f"link-{stamp}"
        calls = len(inputs) * len(models) * len(prompts) * args.runs
        print(f"LLM 호출 {calls}회(규칙 위반 시 더) → {out}")
        asyncio.run(run(models, prompts, args.runs, inputs, out, args.env))

    text = report(out, inputs)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
