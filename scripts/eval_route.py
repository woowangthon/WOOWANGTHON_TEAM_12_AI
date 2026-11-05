"""시험 입력으로 홈 입력 가르기(/route)를 채점한다 — 프롬프트 · 모델을 바꿀 때 같은 자료 · 같은 숫자로 견준다(AI#33).

    uv run python scripts/eval_route.py                      # ROUTE_LLM_MODELS 첫 모델로, 우리 프롬프트 · 범용 프롬프트
    uv run python scripts/eval_route.py --models openai:gpt-5.4-mini,openai:gpt-6.1-terra --prompts tuned
    uv run python scripts/eval_route.py --runs 2             # 같은 입력을 두 번씩(흔들림 보기)
    uv run python scripts/eval_route.py --rescore eval-runs/route-<시각>   # 저장한 응답만 다시 채점(LLM 안 부름)

LLM 호출 수 = 입력 수 × 모델 수 × 프롬프트 수 × runs(규칙 위반이면 입력마다 최대 LLM_MAX_ATTEMPTS 번)
— 기본 40 × 1 × 2 × 1 = 80회. 응답은 --out 폴더(기본 eval-runs/route-<시각>, 커밋하지 않음)에 저장한다.
모델 폴백 없이 고른 모델만 부른다. 키는 --env 파일(기본 레포의 .env)에서 읽고 화면에 찍지 않는다.

자료: samples/route/inputs.json(지어낸 입력과 손으로 적은 정답 — 칸의 뜻은 samples/route/README.md).

숫자의 뜻:
- 맞음: 고른 kind 가 answer 이거나 alsoOk 에 있는 입력 수
- 틀리게 넘김: UNSURE 가 아닌데 틀린 수 — 사용자가 엉뚱한 흐름으로 간다(가장 나쁘다)
- 괜히 물음: 정답이 UNSURE 가 아닌데(alsoOk 에도 없는데) UNSURE 를 고른 수 — 한 번 더 고르게 할 뿐이다
- 오류: 다시 시도 뒤에도 규칙을 어겨 LLM_INVALID_OUTPUT 등으로 끝난 수
- 따로 더한 것: heldOut 입력(첫 채점 뒤 프롬프트 예시 · 규칙에 없는 꼴로 더한 것 — samples/route/README.md)만의 맞음
- 시간: 입력 하나에 걸린 시간(다시 시도 포함)의 중앙값 · 최대 — 동기 호출이라 사용자가 이만큼 기다린다
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

from app.route import MAX_ROUTE_CHARS, SYSTEM_PROMPT  # noqa: E402
from app.schemas import ENTRANCE_KINDS  # noqa: E402

DATA = ROOT / "samples" / "route"

# 범용 프롬프트 — 같은 출력 모양(JSON 스키마)에 종류 이름과 계약의 한 줄 뜻만 준다
GENERIC_PROMPT = (
    "이 글이 어떤 종류인지 하나 골라 줘. SOURCE: 내가 한 일을 적은 글 · APPLICATION: 지원 문항이나 모집 공고 · "
    "ASK: 내 경험을 찾아 달라는 질문 · UNSURE: 애매함."
)
PROMPTS = {"tuned": SYSTEM_PROMPT, "generic": GENERIC_PROMPT}


def load_inputs(only: list[str] | None) -> list[dict]:
    """file 은 내용으로 바꾸고, BE 처럼 앞 MAX_ROUTE_CHARS 자로 자른다. totalLength 가 없으면 글 길이."""
    items = json.loads((DATA / "inputs.json").read_text(encoding="utf-8"))
    out = []
    for item in items:
        if only and item["id"] not in only:
            continue
        text = item.get("text")
        if text is None:
            text = (DATA / item["file"]).read_text(encoding="utf-8")
        total = max(item.get("totalLength", len(text)), len(text))
        out.append({**item, "text": text[:MAX_ROUTE_CHARS], "totalLength": total})
    return out


def judge(item: dict, kind: str) -> str:
    """right · wrong(틀리게 넘김) · asked(괜히 물음)."""
    if kind == item["answer"] or kind in item.get("alsoOk", ()):
        return "right"
    return "asked" if kind == "UNSURE" else "wrong"


@dataclass
class Score:
    total: int = 0
    right: int = 0
    wrong: list[str] = field(default_factory=list)
    asked: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    by_answer: dict[str, list[int]] = field(default_factory=dict)  # 정답 종류별 [맞음, 전체]
    held_out: list[int] = field(default_factory=lambda: [0, 0])  # heldOut 입력 [맞음, 전체]


def score_dir(path: Path, inputs: list[dict]) -> Score:
    s = Score()
    for item in inputs:
        f = path / f"{item['id']}.json"
        if not f.exists():
            continue
        record = json.loads(f.read_text(encoding="utf-8"))
        s.total += 1
        s.seconds.append(record.get("seconds", 0.0))
        tally = s.by_answer.setdefault(item["answer"], [0, 0])
        tally[1] += 1
        held = int(item.get("heldOut", False))
        s.held_out[1] += held
        if "error" in record:
            s.errors.append(f"{item['id']}: {record['error']}")
            continue
        kind, reason = record["response"]["kind"], record["response"]["reason"]
        verdict = judge(item, kind)
        line = f"{item['id']}: 정답 {item['answer']} → {kind} — {reason}"
        if verdict == "right":
            s.right += 1
            tally[0] += 1
            s.held_out[0] += held
        elif verdict == "wrong":
            s.wrong.append(line)
        else:
            s.asked.append(line)
    return s


async def run(
    models: list[str], prompts: list[str], runs: int, inputs: list[dict], out: Path, env: Path
) -> None:
    from dotenv import load_dotenv

    from app.config import load_settings, parse_models
    from app.errors import ApiError
    from app.llm import build_llm
    from app.route import route
    from app.schemas import RouteRequest

    load_dotenv(env)  # 값은 화면에 찍지 않는다
    base = load_settings()
    gate = asyncio.Semaphore(4)
    # 모델마다 클라이언트를 한 번만 만든다(서버도 lru_cache 로 한 번). 호출마다 만들면 생성(하나에 약 0.3초)이
    # 이벤트 루프를 막아, 먼저 시작한 호출의 시간이 수십 초로 부풀고 제한 시간에 걸린다
    clients = {}
    for model in models:
        settings = replace(base, route_models=parse_models(model))
        llm = build_llm(settings, settings.route_models, settings.llm_sync_call_timeout_seconds)
        clients[model] = (settings, llm)

    async def one(model: str, prompt: str, n: int, item: dict) -> None:
        settings, llm = clients[model]
        req = RouteRequest(text=item["text"], totalLength=item["totalLength"])
        async with gate:
            started = time.monotonic()
            try:
                data = await route(req, llm, settings, system=PROMPTS[prompt])
                record = {"response": json.loads(data.model_dump_json())}
            except ApiError as e:
                record = {"error": f"{e.code}: {e.message}"}
            record["seconds"] = round(time.monotonic() - started, 1)
        name = prompt + (f"-{n}" if runs > 1 else "")
        path = out / model.replace(":", "_").replace("/", "_") / name / f"{item['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        got = record.get("error") or record["response"]["kind"]
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
    kinds = [k for k in ENTRANCE_KINDS if any(i["answer"] == k for i in inputs)]
    rows, details = [], []
    for model_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        for prompt_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            s = score_dir(prompt_dir, inputs)
            if not s.total:
                continue
            name = f"{model_dir.name} · {prompt_dir.name}"
            per_kind = " | ".join(
                f"{s.by_answer.get(k, [0, 0])[0]}/{s.by_answer.get(k, [0, 0])[1]}" for k in kinds
            )
            secs = s.seconds or [0.0]
            rows.append(
                f"| {name} | {s.right}/{s.total} | {per_kind} | {s.held_out[0]}/{s.held_out[1]} "
                f"| {len(s.wrong)} | {len(s.asked)} "
                f"| {len(s.errors)} | {statistics.median(secs):.1f}초 · {max(secs):.1f}초 |"
            )
            lines = [
                *(f"- 틀리게 넘김 {x}" for x in s.wrong),
                *(f"- 괜히 물음 {x}" for x in s.asked),
                *(f"- 오류 {x}" for x in s.errors),
            ]
            details.append(f"\n### {name}\n\n" + ("\n".join(lines) or "- 전부 맞음"))
    head = (
        "| 모델 · 프롬프트 | 맞음 | "
        + " | ".join(f"정답 {k}" for k in kinds)
        + " | 따로 더한 것 | 틀리게 넘김 | 괜히 물음 | 오류 | 시간(중앙 · 최대) |\n"
        + "| --- " * (len(kinds) + 7)
        + "|"
    )
    return "\n".join([head, *rows]) + "\n" + "".join(details) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="쉼표로 「제공사:모델」 — 기본은 ROUTE_LLM_MODELS 첫 모델")
    parser.add_argument("--prompts", default="tuned,generic", help="tuned · generic")
    parser.add_argument("--runs", type=int, default=1, help="같은 입력을 몇 번씩")
    parser.add_argument("--inputs", help="쉼표로 입력 id — 기본은 전부")
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
            p, m = load_settings().route_models[0]
            models = [f"{p}:{m}"]
        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
        for p in prompts:
            if p not in PROMPTS:
                parser.error(f"프롬프트는 {list(PROMPTS)} 중 — {p!r}")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = args.out or ROOT / "eval-runs" / f"route-{stamp}"
        calls = len(inputs) * len(models) * len(prompts) * args.runs
        print(f"LLM 호출 {calls}회(규칙 위반 시 더) → {out}")
        asyncio.run(run(models, prompts, args.runs, inputs, out, args.env))

    text = report(out, inputs)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
