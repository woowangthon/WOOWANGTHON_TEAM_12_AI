"""시험 글로 지원 나누기(/parse-application)를 채점한다 — 프롬프트 · 모델을 바꿀 때 같은 자료 · 같은 숫자로 견준다(AI#35).

    uv run python scripts/eval_parse.py                      # ROUTE_LLM_MODELS 첫 모델로, 우리 프롬프트 · 범용 프롬프트
    uv run python scripts/eval_parse.py --models openai:gpt-5.4-mini,openai:gpt-6-luna --prompts tuned
    uv run python scripts/eval_parse.py --runs 2             # 같은 글을 두 번씩(흔들림 보기)
    uv run python scripts/eval_parse.py --rescore eval-runs/parse-<시각>   # 저장한 응답만 다시 채점(LLM 안 부름)

LLM 호출 수 = 글 수 × 모델 수 × 프롬프트 수 × runs(규칙 위반이면 글마다 최대 LLM_MAX_ATTEMPTS 번).
응답은 --out 폴더(기본 eval-runs/parse-<시각>, 커밋하지 않음)에 저장한다.
모델 폴백 없이 고른 모델만 부른다. 키는 --env 파일(기본 레포의 .env)에서 읽고 화면에 찍지 않는다.

자료: samples/parse/inputs.json(지어낸 글과 손으로 적은 정답 — 칸의 뜻은 samples/parse/README.md).
공고 요건 줄 자료는 samples/parse/postings.json(지어낸 공고 6건 · 메뉴 잡음 포함).

숫자의 뜻:
- 문항: 꺼낸 문항 목록이 정답과 순서까지 같은 글 수(앞 번호 · 끝의 글자 수 안내는 안 본다)
- 없는 문항 만듦: 정답이 빈 목록인데 문항을 낸 글 수 — 가장 나쁘다. 적혀 있기는 해도 문항이 아닌 문장이다
- 지원처 · 종류 · 직무: 값이 정답 목록에 있는 글 수
- 오류: 다시 시도 뒤에도 규칙을 어겨 LLM_INVALID_OUTPUT 등으로 끝난 수 — 글에 없는 문항 · 이름은 여기로 온다
- 시간: 글 하나에 걸린 시간(다시 시도 포함)의 중앙값 · 최대 — 동기 호출이라 사용자가 이만큼 기다린다
- 줄 재현율 · 잡음 비율: 정답 줄을 찾은 비율 · 정답 밖 출력 줄 비율(줄 정답이 있는 자료만)
- 칸 · KSA: 찾은 줄 중 정답과 같은 수. KSA 정답도 AI 해석의 기준점이며 팀원 검토 전이다
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

from app.parse_application import SYSTEM_PROMPT, squeeze  # noqa: E402

DATA = ROOT / "samples" / "parse"
FIELDS = ("organization", "organizationType", "jobRole")

# 범용 프롬프트 — 같은 출력 모양(JSON 스키마)에 칸 이름의 뜻만 준다
GENERIC_PROMPT = (
    "이 글에서 지원처 이름(organization) · 지원처 종류(organizationType) · 직무(jobRole) · "
    "지원 문항(questions) · 공고 요건 줄(postingLines: section · text · ksa)을 찾아 줘. 없으면 null · 빈 배열."
)
PROMPTS = {"tuned": SYSTEM_PROMPT, "generic": GENERIC_PROMPT}

_LEADING_NUMBER = re.compile(r"^(?:\d+[.)]|Q\d+[.:]?)\s*")
_TRAILING_LIMIT = re.compile(r"\s*\([^()]*\d+\s*자[^()]*\)$")
_LEADING_BULLET = re.compile(r"^(?:[·•-]|\d+[.)])\s+")


def posting_core(text: str) -> str:
    """원문 대조와 같이 공백만 줄이고, 남겨도 되는 앞 글머리는 채점에서만 뺀다."""
    return _LEADING_BULLET.sub("", squeeze(text))


def load_inputs(only: list[str] | None) -> list[dict]:
    items = json.loads((DATA / "inputs.json").read_text(encoding="utf-8"))
    items += json.loads((DATA / "postings.json").read_text(encoding="utf-8"))
    return [i for i in items if not only or i["id"] in only]


def same_question(a: str, b: str) -> bool:
    """앞 번호 · 끝의 글자 수 안내는 지워도 같은 문항이다 — 둘 다 계약이 허용하거나 사용자가 지울 것이다."""

    def core(text: str) -> str:
        return _TRAILING_LIMIT.sub("", _LEADING_NUMBER.sub("", squeeze(text)))

    return core(a) == core(b)


def same_questions(got: list[str], answer: list[str]) -> bool:
    return len(got) == len(answer) and all(map(same_question, got, answer))


@dataclass
class Score:
    total: int = 0
    questions: int = 0
    invented: list[str] = field(default_factory=list)
    right: dict[str, int] = field(default_factory=lambda: dict.fromkeys(FIELDS, 0))
    wrong: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    posting_expected: int = 0
    posting_found: int = 0
    posting_returned: int = 0
    posting_noise: int = 0
    posting_section: int = 0
    posting_ksa: int = 0
    posting_wrong: list[str] = field(default_factory=list)


def score_dir(path: Path, inputs: list[dict]) -> Score:
    s = Score()
    for item in inputs:
        f = path / f"{item['id']}.json"
        if not f.exists():
            continue
        record = json.loads(f.read_text(encoding="utf-8"))
        s.total += 1
        # 오류도 재현율 분모에 남긴다. 옛 응답(postingLines 없음)은 0줄로 전후를 비교한다.
        expected = item.get("postingLines", [])
        s.posting_expected += len(expected)
        s.seconds.append(record.get("seconds", 0.0))
        if "error" in record:
            s.errors.append(f"{item['id']}: {record['error']}")
            continue
        got = record["response"]
        if "postingLines" in item:
            lines = got.get("postingLines", [])
            s.posting_returned += len(lines)
            remaining = {posting_core(line["text"]): line for line in expected}
            for line in lines:
                key = posting_core(line["text"])
                answer = remaining.pop(key, None)
                if answer is not None:
                    s.posting_found += 1
                    s.posting_section += line["section"] == answer["section"]
                    s.posting_ksa += line["ksa"] == answer["ksa"]
                    if line["section"] != answer["section"] or line["ksa"] != answer["ksa"]:
                        s.posting_wrong.append(f"{item['id']}: 분류 {line} → 정답 {answer}")
                else:
                    # 정답 밖의 줄은 모두 잡음으로 센다 — 알려진 메뉴만 세어 과소평가하지 않는다.
                    s.posting_noise += 1
                    s.posting_wrong.append(f"{item['id']}: 불필요한 줄 {line['text']}")
            for line in remaining.values():
                s.posting_wrong.append(f"{item['id']}: 못 찾은 줄 {line['text']}")
        if same_questions(got["questions"], item["questions"]):
            s.questions += 1
        else:
            line = f"{item['id']}: 문항 {len(item['questions'])}개 정답 → {got['questions']}"
            (s.invented if not item["questions"] else s.wrong).append(line)
        for name in FIELDS:
            if got[name] in item[name]:
                s.right[name] += 1
            else:
                s.wrong.append(f"{item['id']}: {name} 정답 {item[name]} → {got[name]}")
    return s


async def run(
    models: list[str], prompts: list[str], runs: int, inputs: list[dict], out: Path, env: Path
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
    clients = {}  # 모델마다 한 번만 만든다 — eval_route 의 까닭과 같다
    for model in models:
        settings = replace(base, route_models=parse_models(model))
        llm = build_llm(settings, settings.route_models, settings.llm_sync_call_timeout_seconds)
        clients[model] = (settings, llm)

    async def one(model: str, prompt: str, n: int, item: dict) -> None:
        settings, llm = clients[model]
        req = ParseApplicationRequest(text=item["text"])
        async with gate:
            started = time.monotonic()
            try:
                data = await parse_application(req, llm, settings, system=PROMPTS[prompt])
                record = {"response": json.loads(data.model_dump_json())}
            except ApiError as e:
                record = {"error": f"{e.code}: {e.message}"}
            record["seconds"] = round(time.monotonic() - started, 1)
        name = prompt + (f"-{n}" if runs > 1 else "")
        path = out / model.replace(":", "_").replace("/", "_") / name / f"{item['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        got = record.get("error") or f"문항 {len(record['response']['questions'])}개"
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
            fields = " | ".join(f"{s.right[f]}/{s.total}" for f in FIELDS)
            recall = s.posting_found / s.posting_expected * 100 if s.posting_expected else 0
            noise = s.posting_noise / s.posting_returned * 100 if s.posting_returned else 0
            rows.append(
                f"| {name} | {s.questions}/{s.total} | {len(s.invented)} | {fields} "
                f"| {len(s.errors)} | {statistics.median(secs):.1f}초 · {max(secs):.1f}초 "
                f"| {s.posting_found}/{s.posting_expected} ({recall:.1f}%) "
                f"| {s.posting_noise}/{s.posting_returned} ({noise:.1f}%) "
                f"| {s.posting_section}/{s.posting_found} | {s.posting_ksa}/{s.posting_found} |"
            )
            lines = [
                *(f"- 없는 문항 만듦 {x}" for x in s.invented),
                *(f"- 틀림 {x}" for x in s.wrong),
                *(f"- 오류 {x}" for x in s.errors),
                *(f"- 공고 {x}" for x in s.posting_wrong),
            ]
            details.append(f"\n### {name}\n\n" + ("\n".join(lines) or "- 전부 맞음"))
    head = (
        "| 모델 · 프롬프트 | 문항 | 없는 문항 만듦 | 지원처 | 종류 | 직무 | 오류 | 시간(중앙 · 최대) | 줄 재현율 | 잡음 비율 | 칸 | KSA |\n"
        + "| --- " * 12
        + "|"
    )
    return "\n".join([head, *rows]) + "\n" + "".join(details) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="쉼표로 「제공사:모델」 — 기본은 ROUTE_LLM_MODELS 첫 모델")
    parser.add_argument("--prompts", default="tuned,generic", help="tuned · generic")
    parser.add_argument("--runs", type=int, default=1, help="같은 글을 몇 번씩")
    parser.add_argument("--inputs", help="쉼표로 글 id — 기본은 전부")
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
        out = args.out or ROOT / "eval-runs" / f"parse-{stamp}"
        calls = len(inputs) * len(models) * len(prompts) * args.runs
        print(f"LLM 호출 {calls}회(규칙 위반 시 더) → {out}")
        asyncio.run(run(models, prompts, args.runs, inputs, out, args.env))

    text = report(out, inputs)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
