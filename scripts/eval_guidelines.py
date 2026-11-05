"""붙여 넣은 작성법 글 → 작성 지침 정리를 채점한다(S-49 · AI#55). 프롬프트 · 모델을 바꿀 때 같은 자료 · 같은 숫자로 견준다.

    uv run python scripts/eval_guidelines.py                    # MATCH_LLM_MODELS 첫 모델로, 우리 프롬프트 · 범용 프롬프트
    uv run python scripts/eval_guidelines.py --models openai:gpt-6-luna --prompts tuned
    uv run python scripts/eval_guidelines.py --rescore eval-runs/guidelines-<시각>   # 저장한 응답만 다시 채점

LLM 호출 수 = 글 수 × 모델 수 × 프롬프트 수 — 기본 6 × 1 × 2 = 12회(규칙 위반이면 글마다 최대 LLM_MAX_ATTEMPTS 번).
응답은 --out 폴더(기본 eval-runs/guidelines-<시각>, 커밋하지 않음)에 저장한다. 모델 폴백 없이 고른 모델만 부른다.
키는 --env 파일(기본 레포의 .env)에서 읽고 화면에 찍지 않는다. 범용 프롬프트는 같은 모델 · 같은 출력 모양에 지시만 한 줄이다(개수 · 길이 검사는 모양이라 같이 켠다).

자료: samples/guidelines/texts.json(지어낸 글 + 손으로 적은 정답 — 칸의 뜻은 그 폴더 README).

숫자의 뜻(정규식으로 거른 것은 후보 — 줄을 나란히 찍으니 사람이 읽는다):
- 줄: 지침 줄 수 합 · 오류: 끝까지 어겨 LLM_INVALID_OUTPUT 등으로 끝난 글 수 · 다시 부름: 검사에 걸려 한 번 더 부른 글 수
- 옮김: 글에 있는 쓰는 법(expect) 중 지침 줄이 옮긴 수 / 기회(many-tips 는 10줄까지라 기회를 10 으로 센다)
- 글에 없는 말(후보): expect 의 어느 것에도 안 걸리는 줄 — 일반론을 더했거나 글의 뜻을 넓힌 것. 사람이 읽는다
- 지어내라 옮김: 사실을 지어내거나 부풀리라는 말 · 합격자 예문 · 글 속 지시(forbidden)가 줄에 나온 수 — 0 이어야 한다
- 빈 글 0줄: 쓰는 법이 없는 글(expect [])에 0줄을 돌려준 수 / 그런 글 수
- 안 합침: 같은 뜻(merge)이 두 줄 이상으로 나온 수
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

from app.guidelines import MAX_LINES, SYSTEM_PROMPT  # noqa: E402
from scripts.eval_match import _slug  # noqa: E402

DATA = ROOT / "samples" / "guidelines" / "texts.json"
GENERIC_PROMPT = "이 글에서 자기소개서 작성 지침을 한 줄씩 뽑아 줘."
PROMPTS = {"tuned": SYSTEM_PROMPT, "generic": GENERIC_PROMPT}


@dataclass
class Score:
    texts: int = 0
    errors: list[str] = field(default_factory=list)
    retried: int = 0
    seconds: float = 0.0
    lines: list[str] = field(default_factory=list)
    kept: int = 0
    chances: int = 0
    dropped: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    empty_hit: int = 0
    empty_chances: int = 0
    unmerged: list[str] = field(default_factory=list)

    def add(self, other: "Score") -> None:
        for name, value in vars(other).items():
            setattr(self, name, getattr(self, name) + value)


def score(item: dict, lines: list[str]) -> Score:
    tid = item["id"]
    s = Score()
    s.lines = [f"{tid}: 「{x}」" for x in lines]
    hits = [e for e in item["expect"] if any(re.search(e["pattern"], x) for x in lines)]
    s.kept = len(hits)
    s.chances = min(len(item["expect"]), MAX_LINES)
    s.dropped = [f"{tid}: {e['line']}" for e in item["expect"] if e not in hits]
    s.added = [
        f"{tid}: 「{x}」"
        for x in lines
        if not any(re.search(e["pattern"], x) for e in item["expect"])
    ]
    s.forbidden = [
        f"{tid}: {f['what']} — 「{x}」"
        for f in item["forbidden"]
        for x in lines
        if re.search(f["pattern"], x)
    ]
    if not item["expect"]:
        s.empty_chances = 1
        s.empty_hit = int(not lines)
    for pattern in item.get("merge", []):
        same = [x for x in lines if re.search(pattern, x)]
        if len(same) > 1:
            s.unmerged.append(f"{tid}: {same}")
    return s


def load_texts(only: list[str] | None = None) -> list[dict]:
    texts = json.loads(DATA.read_text(encoding="utf-8"))
    return [t for t in texts if t["id"] in only] if only else texts


async def run(
    models: list[str], prompts: list[str], texts: list[dict], out: Path, env: Path
) -> None:
    from dotenv import load_dotenv

    from app.config import load_settings, parse_models
    from app.errors import ApiError
    from app.guidelines import guidelines
    from app.llm import build_llm
    from app.schemas import GuidelinesRequest

    load_dotenv(env)  # 값은 화면에 찍지 않는다
    base = load_settings()
    gate = asyncio.Semaphore(4)
    clients = {}
    for model in models:
        settings = replace(base, llm_models=parse_models(model))
        clients[model] = (
            settings,
            build_llm(settings, call_timeout=base.llm_sync_call_timeout_seconds),
        )

    async def one(model: str, prompt: str, item: dict) -> None:
        settings, llm = clients[model]
        async with gate:
            started = time.monotonic()
            try:
                got = await guidelines(
                    GuidelinesRequest(text=item["text"]), llm, settings, PROMPTS[prompt]
                )
                record = {"response": json.loads(got.model_dump_json())}
            except ApiError as e:
                record = {"error": f"{e.code}: {e.message}"}
            record["seconds"] = round(time.monotonic() - started, 1)
        path = out / _slug(model) / prompt / f"{item['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"  {model} · {prompt} · {item['id']} — {record.get('error') or 'ok'} ({record['seconds']}초)"
        )

    await asyncio.gather(*(one(m, p, t) for m in models for p in prompts for t in texts))


def report(out: Path, texts: list[dict]) -> str:
    rows, details = [], []
    for model_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        for prompt_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            t = Score()
            for item in texts:
                path = prompt_dir / f"{item['id']}.json"
                if not path.exists():
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                t.texts += 1
                t.seconds += record.get("seconds", 0.0)
                if "error" in record:
                    t.errors.append(f"{item['id']}: {record['error']}")
                    continue
                lines = [g["text"] for g in record["response"]["guidelines"]]
                t.add(score(item, lines))
            if not t.texts:
                continue
            name = f"{model_dir.name} · {prompt_dir.name}"
            rows.append(
                f"| {name} | {len(t.lines)} | {len(t.errors)}/{t.texts} | {t.kept}/{t.chances} "
                f"| {len(t.added)} | {len(t.forbidden)} | {t.empty_hit}/{t.empty_chances} "
                f"| {len(t.unmerged)} | {t.seconds:.0f}초 |"
            )
            lines = [
                *(f"- 오류 {x}" for x in t.errors),
                *(f"- 지어내라 옮김 {x}" for x in t.forbidden),
                *(f"- 글에 없는 말(후보) {x}" for x in t.added),
                *(f"- 안 합침 {x}" for x in t.unmerged),
                *(f"- 못 옮김 {x}" for x in t.dropped),
                *(f"- 줄 {x}" for x in t.lines),
            ]
            details.append(f"\n### {name}\n\n" + "\n".join(lines))
    head = (
        "| 모델 · 프롬프트 | 줄 | 오류 | 옮김 | 글에 없는 말(후보) | 지어내라 옮김 | 빈 글 0줄 | 안 합침 | 시간 합 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    return "\n".join([head, *rows]) + "\n" + "".join(details) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="쉼표로 「제공사:모델」 — 기본은 MATCH_LLM_MODELS 첫 모델")
    parser.add_argument("--prompts", default="tuned,generic", help="tuned · generic")
    parser.add_argument("--texts", help="쉼표로 글 id — 기본은 전부")
    parser.add_argument("--out", type=Path, help="응답 저장 폴더")
    parser.add_argument("--env", type=Path, default=ROOT / ".env", help="키를 읽을 .env")
    parser.add_argument("--rescore", type=Path, help="저장한 응답만 다시 채점(LLM 안 부름)")
    args = parser.parse_args()

    texts = load_texts(args.texts.split(",") if args.texts else None)
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
        out = args.out or ROOT / "eval-runs" / f"guidelines-{stamp}"
        print(f"LLM 호출 {len(texts) * len(models) * len(prompts)}회(규칙 위반 시 더) → {out}")
        asyncio.run(run(models, prompts, texts, out, args.env))

    text = report(out, texts)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
