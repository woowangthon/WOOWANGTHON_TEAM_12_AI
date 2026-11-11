"""캡처 읽기(/parse-application `images`, S-92)를 채점한다 — 비전 모델 · 프롬프트를 바꿀 때 같은 자료로 견준다(AI#121).

    uv run python scripts/eval_capture.py                                   # VISION_LLM_MODELS 첫 모델, detail high
    uv run python scripts/eval_capture.py --models openai:gpt-6.1-terra,openai:gpt-6-luna --runs 2
    uv run python scripts/eval_capture.py --models openai:gpt-6.1-terra --detail low
    uv run python scripts/eval_capture.py --rescore eval-runs/capture-<시각>   # 저장한 응답만 다시 채점(LLM 안 부름)

LLM 호출 수 = 자료 8벌 × 모델 수 × runs(규칙 위반이면 자료마다 최대 LLM_MAX_ATTEMPTS 번). 한 벌에 이미지 1 ~ 2장.
응답 · 토큰 수는 --out 폴더(기본 eval-runs/capture-<시각>, 커밋하지 않음)에 저장한다. 모델 폴백 없이 고른 모델만 부른다.
키는 --env 파일(기본 레포의 .env)에서 읽고 화면에 찍지 않는다.

자료: samples/capture/ — 지어낸 화면 그림과 정답(scripts/make_captures.py 가 함께 만든다, 칸의 뜻은 README).

숫자의 뜻:
- 문항 그대로 · 줄 그대로: 정답 문항 · 요건 줄 가운데 글자 그대로 옮긴 수(공백 · 앞 번호 · 글머리 · 끝의 필수 표시 「*」만
  같게 본다 — 「(500자)」를 빼거나 낱말 하나를 바꾸면 그대로가 아니다). 오류 · 422 로 끝난 자료도 분모에 남는다
- 글자 다름: 정답 밖 출력 가운데 어느 정답과 거의 같은 것(difflib 0.6 이상) — 옮기다 글자를 틀리거나 다듬은 것
- 만들어 낸 줄: 정답 밖 출력 가운데 어느 정답과도 닮지 않은 것 — 메뉴 · 혜택 · 잘린 문항 · 지시 · 없는 글. 가장 나쁘다
- 함정: 만들어 낸 줄 가운데 정답의 traps(잘린 문항 · 메뉴 · 혜택 · 글 속 지시)를 담은 것
- 지원처 · 종류 · 직무: 값이 정답 목록에 있는 자료 수. 칸: 찾은 줄 가운데 section 이 맞은 수
- 오류: 422 IMAGE_UNREADABLE · LLM_INVALID_OUTPUT · LLM_TIMEOUT 등으로 끝난 수(모두 글자가 있는 자료라 422 도 실패다)
- 시간: 자료 하나에 걸린 시간(다시 시도 포함)의 중앙값 · 최대 — 동기 호출이라 사용자가 이만큼 기다린다(계약 50초)
- 비용: 응답의 토큰 수(usage) × PRICES. 가격이 없는 모델은 토큰만
"""

import argparse
import asyncio
import base64
import contextvars
import difflib
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

from app.parse_application import squeeze  # noqa: E402

DATA = ROOT / "samples" / "capture"
FIELDS = ("organization", "organizationType", "jobRole")
# 1M 토큰당 달러(입력, 출력) — OpenAI 가격표 Standard(2026-11-11 확인, AI#121). 쓰기 전에 가격표를 다시 본다
PRICES = {
    "openai:gpt-6.1-terra": (1.00, 5.00),
    "openai:gpt-6-luna": (0.10, 0.50),
    "openai:gpt-6.1-sol": (2.00, 10.00),
    "openai:gpt-5.4-mini": (0.75, 4.50),
}
SIMILAR = 0.6

_LEADING = re.compile(r"^(?:\d+[.)]|Q\d+[.:]?|[·•-])\s*")
_REQUIRED_MARK = re.compile(r"\s*\*$")

# 자료 하나를 부르는 동안의 토큰 수 — 다시 시도 · 폴백까지 모두 더한다
USAGE: contextvars.ContextVar[list[tuple[int, int]]] = contextvars.ContextVar("usage")


def core(text: str) -> str:
    return _REQUIRED_MARK.sub("", _LEADING.sub("", squeeze(text)))


def load_cases(only: list[str] | None) -> list[dict]:
    cases = json.loads((DATA / "answers.json").read_text(encoding="utf-8"))
    return [c for c in cases if not only or c["id"] in only]


def request_body(case: dict) -> dict:
    return {
        "images": [
            {"mediaType": "image/png", "data": base64.b64encode((DATA / f).read_bytes()).decode()}
            for f in case["files"]
        ]
    }


@dataclass
class Score:
    total: int = 0
    questions: int = 0
    questions_expected: int = 0
    lines: int = 0
    lines_expected: int = 0
    section: int = 0
    misread: list[str] = field(default_factory=list)
    invented: list[str] = field(default_factory=list)
    traps: int = 0
    right: dict[str, int] = field(default_factory=lambda: dict.fromkeys(FIELDS, 0))
    wrong: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0


def _match(s: Score, case: dict, name: str, got: list[str], answer: list[str]) -> list[int]:
    """정답 순서대로 맞은 정답의 번호. 정답 밖 출력은 글자 다름 · 만들어 냄으로 나눠 적는다."""
    remaining = {core(a): i for i, a in enumerate(answer)}
    found = []
    for text in got:
        i = remaining.pop(core(text), None)
        if i is not None:
            found.append(i)
            continue
        best = max(
            (difflib.SequenceMatcher(None, core(text), core(a)).ratio() for a in answer),
            default=0.0,
        )
        line = f"{case['id']}: {name} 「{text}」"
        if best >= SIMILAR:
            s.misread.append(f"{line} (닮음 {best:.2f})")
        else:
            s.invented.append(line)
            s.traps += any(core(t) in core(text) for t in case["traps"])
    for a in remaining:
        s.wrong.append(f"{case['id']}: 못 찾은 {name} 「{answer[remaining[a]]}」")
    return found


def score_dir(path: Path, cases: list[dict]) -> Score:
    s = Score()
    for case in cases:
        f = path / f"{case['id']}.json"
        if not f.exists():
            continue
        record = json.loads(f.read_text(encoding="utf-8"))
        s.total += 1
        s.questions_expected += len(case["questions"])
        s.lines_expected += len(case["postingLines"])
        s.seconds.append(record.get("seconds", 0.0))
        s.tokens_in += record.get("tokens", [0, 0])[0]
        s.tokens_out += record.get("tokens", [0, 0])[1]
        if "error" in record:
            s.errors.append(f"{case['id']}: {record['error']}")
            continue
        got = record["response"]
        s.questions += len(_match(s, case, "문항", got["questions"], case["questions"]))
        answer_lines = case["postingLines"]
        got_lines = got["postingLines"]
        found = _match(
            s, case, "줄", [x["text"] for x in got_lines], [x["text"] for x in answer_lines]
        )
        s.lines += len(found)
        by_core = {core(x["text"]): x for x in got_lines}
        for i in found:
            answer = answer_lines[i]
            mine = by_core[core(answer["text"])]
            if mine["section"] == answer["section"]:
                s.section += 1
            else:
                s.wrong.append(
                    f"{case['id']}: 칸 {mine['section']} → 정답 {answer['section']} 「{answer['text']}」"
                )
        for name in FIELDS:
            value = got[name]
            ok = value in case[name] or (
                name == "organization"
                and value is not None
                and squeeze(value) in {squeeze(v) for v in case[name] if v}
            )
            if ok:
                s.right[name] += 1
            else:
                s.wrong.append(f"{case['id']}: {name} 정답 {case[name]} → {value}")
    return s


def cost(model_dir: str, s: Score) -> str:
    model = model_dir.split("__")[0].replace("_", ":", 1)
    price = PRICES.get(model)
    tokens = f"입력 {s.tokens_in:,} · 출력 {s.tokens_out:,}"
    if price is None or not s.total:
        return tokens
    usd = (s.tokens_in * price[0] + s.tokens_out * price[1]) / 1_000_000
    return f"{tokens} · ${usd:.4f}(자료당 ${usd / s.total:.5f})"


def report(out: Path, cases: list[dict]) -> str:
    rows, details = [], []
    for model_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            s = score_dir(run_dir, cases)
            if not s.total:
                continue
            name = f"{model_dir.name} · {run_dir.name}"
            secs = s.seconds or [0.0]
            q = s.questions / s.questions_expected * 100 if s.questions_expected else 0
            ln = s.lines / s.lines_expected * 100 if s.lines_expected else 0
            fields = " | ".join(f"{s.right[f]}/{s.total}" for f in FIELDS)
            rows.append(
                f"| {name} | {s.questions}/{s.questions_expected} ({q:.0f}%) "
                f"| {s.lines}/{s.lines_expected} ({ln:.0f}%) | {len(s.misread)} | {len(s.invented)} "
                f"| {s.traps} | {fields} | {s.section}/{s.lines} | {len(s.errors)} "
                f"| {statistics.median(secs):.1f}초 · {max(secs):.1f}초 | {cost(model_dir.name, s)} |"
            )
            lines = [
                *(f"- 만들어 냄 {x}" for x in s.invented),
                *(f"- 글자 다름 {x}" for x in s.misread),
                *(f"- 틀림 {x}" for x in s.wrong),
                *(f"- 오류 {x}" for x in s.errors),
            ]
            details.append(f"\n### {name}\n\n" + ("\n".join(lines) or "- 전부 맞음"))
    head = (
        "| 모델 · 회차 | 문항 그대로 | 줄 그대로 | 글자 다름 | 만들어 낸 줄 | 함정 | 지원처 | 종류 | 직무 | 칸 | 오류 "
        "| 시간(중앙 · 최대) | 토큰 · 비용 |\n" + "| --- " * 13 + "|"
    )
    return "\n".join([head, *rows]) + "\n" + "".join(details) + "\n"


def count_usage(llm) -> None:
    """OpenAI 응답의 usage 를 USAGE 에 더한다 — 앱 코드는 토큰을 모른다(채점에서만)."""
    provider = llm.providers["openai"]
    if not provider.available:
        return
    completions = provider._client.chat.completions
    create = completions.create

    async def counted(**kwargs):
        response = await create(**kwargs)
        if response.usage is not None:
            USAGE.get().append((response.usage.prompt_tokens, response.usage.completion_tokens))
        return response

    completions.create = counted


async def run(
    models: list[str], detail: str, runs: int, cases: list[dict], out: Path, env: Path
) -> None:
    from dotenv import load_dotenv

    from app.config import load_settings, parse_models
    from app.errors import ApiError
    from app.llm import build_llm
    from app.parse_application import parse_application
    from app.schemas import ParseApplicationRequest

    load_dotenv(env)  # 값은 화면에 찍지 않는다
    base = load_settings()
    gate = asyncio.Semaphore(4)
    clients = {}
    for model in models:
        settings = replace(base, vision_models=parse_models(model))
        llm = build_llm(
            settings, settings.vision_models, settings.llm_vision_call_timeout_seconds, detail
        )
        count_usage(llm)
        clients[model] = (settings, llm)

    async def one(model: str, n: int, case: dict) -> None:
        settings, llm = clients[model]
        req = ParseApplicationRequest(**request_body(case))
        USAGE.set([])
        async with gate:
            started = time.monotonic()
            try:
                data = await parse_application(req, llm, settings, vision_llm=llm)
                record = {"response": json.loads(data.model_dump_json())}
            except ApiError as e:
                record = {"error": f"{e.code}: {e.message}"}
            record["seconds"] = round(time.monotonic() - started, 1)
        record["tokens"] = [sum(u[0] for u in USAGE.get()), sum(u[1] for u in USAGE.get())]
        folder = model.replace(":", "_").replace("/", "_") + f"__{detail}"
        path = out / folder / f"run-{n}" / f"{case['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        got = record.get("error") or (
            f"문항 {len(record['response']['questions'])} · 줄 {len(record['response']['postingLines'])}"
        )
        print(f"  {model} · {detail} · run-{n} · {case['id']} — {got} ({record['seconds']}초)")

    await asyncio.gather(
        *(one(m, n, case) for m in models for n in range(1, runs + 1) for case in cases)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", help="쉼표로 「제공사:모델」 — 기본은 VISION_LLM_MODELS 첫 모델"
    )
    parser.add_argument("--detail", default="high", choices=["high", "low", "auto"])
    parser.add_argument("--runs", type=int, default=1, help="같은 자료를 몇 번씩")
    parser.add_argument("--cases", help="쉼표로 자료 id — 기본은 전부")
    parser.add_argument("--out", type=Path, help="응답 저장 폴더(있으면 더해 쓴다)")
    parser.add_argument("--env", type=Path, default=ROOT / ".env", help="키를 읽을 .env")
    parser.add_argument("--rescore", type=Path, help="저장한 응답만 다시 채점(LLM 안 부름)")
    args = parser.parse_args()

    cases = load_cases(args.cases.split(",") if args.cases else None)
    if args.rescore:
        out = args.rescore
    else:
        if args.models:
            models = [m.strip() for m in args.models.split(",") if m.strip()]
        else:
            from dotenv import load_dotenv

            from app.config import load_settings

            load_dotenv(args.env)
            p, m = load_settings().vision_models[0]
            models = [f"{p}:{m}"]
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = args.out or ROOT / "eval-runs" / f"capture-{stamp}"
        calls = len(cases) * len(models) * args.runs
        print(
            f"LLM 호출 {calls}회(규칙 위반 시 더) · 이미지 {sum(len(c['files']) for c in cases)}장씩 → {out}"
        )
        asyncio.run(run(models, args.detail, args.runs, cases, out, args.env))

    text = report(out, cases)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
