"""시험 자료로 추출을 채점한다 — 프롬프트 · 모델을 바꿀 때 같은 자료 · 같은 숫자로 나빠지지 않았는지 본다(AI#5).

    uv run python scripts/eval_extract.py                    # LLM_MODELS 첫 모델로, 우리 프롬프트 · 범용 프롬프트
    uv run python scripts/eval_extract.py --models openai:gpt-6-luna,openai:gpt-5.4-mini
    uv run python scripts/eval_extract.py --prompts tuned    # 우리 프롬프트만
    uv run python scripts/eval_extract.py --rescore eval-runs/<폴더>   # LLM 을 다시 부르지 않고 저장한 응답만 다시 채점

LLM 호출 수 = 자료 수 × 모델 수 × 프롬프트 수(규칙 위반이면 자료마다 최대 LLM_MAX_ATTEMPTS 번) — 기본 10 × 1 × 2 = 20회.
응답은 --out 폴더(기본 eval-runs/<시각>, 커밋하지 않음)에 저장한다. 모델 폴백 없이 고른 모델만 부른다.

정답은 samples/answers/<자료 이름>.json — 손으로 적은 활동 · 경험 · 수행 주체. 경험은 원문 구절(anchor)로 적는다.
LLM 경험 하나를 정답 구절 하나에 짝짓는다: 발췌문이 품은(또는 6글자 이상이고 구절 안에 든) 구절 중 content 와
두 글자 묶음이 가장 많이 겹치는 것(다이스 계수). 한 줄을 통째로 발췌해 여러 구절을 품어도 내용이 가리키는 구절 하나로 본다.
optional 경험은 「찾음」 분모에 넣지 않고, 짝지어졌을 때 수행 주체만 본다.
정답 활동은 keys(원문에 있는 이름 조각)가 LLM 활동 이름에 들어 있거나, 그 활동의 경험이 짝지어지면 찾은 것이다.
이름 · 기간만 있는 활동(이력서 항목 · MOC 링크)은 경험 없이(`experiences: []`) 적는다(SPEC S-03).

숫자의 뜻:
- 근거 일치: 원본에서 찾은 발췌문 / 전체 발췌문(app.citations 기준)
- 지어낸 수치: 경험 content 의 숫자 중 원본에 없는 숫자가 있는 경험 수(「세 번」→「3번」도 걸린다 — 목록을 사람이 본다)
- 주체 틀림: 정답 구절에 짝지어진 LLM 경험의 actor 가 정답 밖인 수. 괄호는 그중 ME 가 아닌 일을 ME 로 단 수
- 경험 찾음: 정답(optional 빼고) 중 찾은 수
- 경험 아닌 것: notExperiences(자기 칭찬 · 기술 목록 등)를 경험으로 뽑은 수
- 섞임 / 쪼개짐: 다른 활동의 경험이 한 활동에 들어간 LLM 활동 수 / 한 활동이 여러 LLM 활동으로 갈린 수
- 활동 찾음: 정답 활동(optional 빼고) 중 찾은 수
- 활동 아닌 것: notActivities(공부 노트 · 스크랩 링크 등)를 활동으로 만든 수
"""

import argparse
import asyncio
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.citations import normalize  # noqa: E402
from app.extract import SYSTEM_PROMPT  # noqa: E402
from app.schemas import ExtractResponse  # noqa: E402

SAMPLES = ROOT / "samples"
ANSWERS = SAMPLES / "answers"

# 범용 프롬프트 — 사용자가 ChatGPT 에 그냥 붙일 만한 한 줄. 숫자를 같이 재려고 출력 모양(JSON 스키마)만 같게 준다
GENERIC_PROMPT = "이 자료에서 내 경험을 정리해 줘."
PROMPTS = {"tuned": SYSTEM_PROMPT, "generic": GENERIC_PROMPT}
MIN_INNER_QUOTE = 6  # 발췌문이 정답 구절 안에 들어갈 때 최소 글자 수(비교용 문자열 기준)


# ---- 채점 ----


@dataclass
class Score:
    files: int = 0
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0
    quotes: int = 0
    matched: int = 0
    experiences: int = 0
    confirmed: int = 0
    made_up_numbers: list[str] = field(default_factory=list)
    actor_wrong: list[str] = field(default_factory=list)
    actor_wrong_me: int = 0
    required: int = 0
    found: int = 0
    not_experiences: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    activity_counts: list[str] = field(default_factory=list)
    activities_expected: int = 0
    activities_got: int = 0
    mixed: int = 0
    split: int = 0
    activities_required: int = 0
    activities_found: int = 0
    missed_activities: list[str] = field(default_factory=list)
    not_activities: list[str] = field(default_factory=list)

    def add(self, other: "Score") -> None:
        for name, value in vars(other).items():
            setattr(self, name, getattr(self, name) + value)


def _ints(text: str) -> set[int]:
    return {int(n) for n in re.findall(r"\d+", unicodedata.normalize("NFKC", text))}


def _covers(quote: str, anchor: str) -> bool:
    return anchor in quote or (len(quote) >= MIN_INNER_QUOTE and quote in anchor)


def _bigrams(text: str) -> set[str]:
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _closeness(content: str, anchor: str) -> float:
    # 다이스 계수 — 짧은 구절이 덮어놓고 이기지 않게 두 쪽 길이를 함께 본다
    c, a = _bigrams(content), _bigrams(anchor)
    return 2 * len(c & a) / (len(c) + len(a)) if c or a else 0.0


def score(stem: str, source: str, answer: dict, data: ExtractResponse) -> Score:
    s = Score(files=1, activities_expected=len(answer["activities"]))
    s.activities_got = len(data.activities)
    if s.activities_got != s.activities_expected:
        names = " · ".join(a.name for a in data.activities)
        s.activity_counts.append(
            f"{stem}: {s.activities_got}개(정답 {s.activities_expected}) {names}"
        )
    source_ints = _ints(source)

    # 짝지을 후보: (정답 활동 번호, 정답 경험) — 정답 활동 번호가 None 이면 notExperiences
    candidates: list[tuple[int | None, dict]] = [
        (ei, want) for ei, act in enumerate(answer["activities"]) for want in act["experiences"]
    ] + [(None, {"anchor": a}) for a in answer.get("notExperiences", [])]
    normalized = [normalize(want["anchor"]) for _, want in candidates]
    paired: dict[int, list[tuple[int, object]]] = {}  # 후보 번호 → [(LLM 활동 번호, 경험)]

    for ai, act in enumerate(data.activities):
        for exp in act.experiences:
            s.experiences += 1
            s.confirmed += exp.sourceStatus == "SOURCE_CONFIRMED"
            s.quotes += len(exp.citations)
            s.matched += sum(bool(c.matched) for c in exp.citations)
            extra = _ints(exp.content) - source_ints
            if extra:
                s.made_up_numbers.append(f"{stem}: {sorted(extra)} 「{exp.content}」")
            quotes = [normalize(c.quote) for c in exp.citations]
            hits = [i for i, a in enumerate(normalized) if any(_covers(q, a) for q in quotes)]
            if hits:
                content = normalize(exp.content)
                best = max(hits, key=lambda i: _closeness(content, normalized[i]))
                paired.setdefault(best, []).append((ai, exp))

    llm_to_expected: dict[int, set[int]] = {}
    expected_to_llm: dict[int, set[int]] = {}
    for i, (ei, want) in enumerate(candidates):
        hits = paired.get(i, [])
        if ei is None:
            s.not_experiences += [f"{stem}: 「{exp.content}」" for _, exp in hits]
            continue
        if not want.get("optional"):
            s.required += 1
            s.found += bool(hits)
            if not hits:
                s.missed.append(f"{stem}: 「{want['anchor']}」")
        for ai, exp in hits:
            llm_to_expected.setdefault(ai, set()).add(ei)
            expected_to_llm.setdefault(ei, set()).add(ai)
            if exp.actor not in want["actor"]:
                s.actor_wrong.append(
                    f"{stem}: 「{want['anchor']}」 정답 {'/'.join(want['actor'])} → {exp.actor}"
                    f" 「{exp.content}」"
                )
                s.actor_wrong_me += exp.actor == "ME"
    s.mixed = sum(len(e) > 1 for e in llm_to_expected.values())
    s.split = sum(len(a) > 1 for a in expected_to_llm.values())

    names = [normalize(a.name) for a in data.activities]
    for ei, act in enumerate(answer["activities"]):
        if act.get("optional"):
            continue
        s.activities_required += 1
        keys = [normalize(k) for k in act.get("keys", [])]
        if ei in expected_to_llm or any(k in n for k in keys for n in names):
            s.activities_found += 1
        else:
            s.missed_activities.append(f"{stem}: {act['name']}")
    for key in answer.get("notActivities", []):
        k = normalize(key)
        hits = [a.name for a, n in zip(data.activities, names, strict=True) if k in n]
        s.not_activities += [f"{stem}: {name}" for name in hits]
    return s


def load_answers(stems: list[str] | None) -> dict[str, tuple[str, dict]]:
    out = {}
    for path in sorted(ANSWERS.glob("*.json")):
        if stems and path.stem not in stems:
            continue
        source = (SAMPLES / f"{path.stem}.md").read_text(encoding="utf-8")
        out[path.stem] = (source, json.loads(path.read_text(encoding="utf-8")))
    return out


# ---- 실행 ----


def _slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


async def run(
    models: list[str], prompts: list[str], samples: dict[str, tuple[str, dict]], out: Path
) -> None:
    from dotenv import load_dotenv

    from app.config import load_settings, parse_models
    from app.errors import ApiError
    from app.extract import extract
    from app.llm import build_llm
    from app.schemas import ExtractRequest

    load_dotenv(ROOT / ".env")  # 값은 화면에 찍지 않는다
    base = load_settings()
    gate = asyncio.Semaphore(4)

    # 모델마다 클라이언트를 한 번만 만든다(eval_route 와 같은 이유 — 호출마다 만들면 생성이 이벤트 루프를 막아
    # 먼저 시작한 호출의 시간이 부푼다, AI#37)
    clients = {}
    for model in models:
        settings = replace(base, llm_models=parse_models(model))
        clients[model] = (settings, build_llm(settings))

    async def one(model: str, prompt: str, stem: str, source: str) -> None:
        settings, llm = clients[model]
        req = ExtractRequest(sourceName=f"{stem}.md", text=source)
        async with gate:
            started = time.monotonic()
            try:
                data = await extract(req, llm, settings, system=PROMPTS[prompt])
                record = {"response": json.loads(data.model_dump_json())}
            except ApiError as e:
                record = {"error": f"{e.code}: {e.message}"}
            record["seconds"] = round(time.monotonic() - started, 1)
        path = out / _slug(model) / prompt / f"{stem}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"  {model} · {prompt} · {stem} — {record.get('error') or 'ok'} ({record['seconds']}초)"
        )

    await asyncio.gather(
        *(
            one(m, p, stem, source)
            for m in models
            for p in prompts
            for stem, (source, _) in samples.items()
        )
    )


def report(out: Path, samples: dict[str, tuple[str, dict]]) -> str:
    rows, details = [], []
    for model_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        for prompt_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            total = Score()
            for stem, (source, answer) in samples.items():
                path = prompt_dir / f"{stem}.json"
                if not path.exists():
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                total.seconds += record.get("seconds", 0.0)
                if "error" in record:
                    total.files += 1
                    total.errors.append(f"{stem}: {record['error']}")
                    continue
                data = ExtractResponse.model_validate(record["response"])
                total.add(score(stem, source, answer, data))
            name = f"{model_dir.name} · {prompt_dir.name}"
            t = total
            rows.append(
                f"| {name} | {t.matched}/{t.quotes} | {t.confirmed}/{t.experiences} "
                f"| {len(t.made_up_numbers)} | {len(t.actor_wrong)} ({t.actor_wrong_me}) "
                f"| {t.found}/{t.required} | {len(t.not_experiences)} "
                f"| {t.activities_found}/{t.activities_required} | {len(t.not_activities)} "
                f"| {t.activities_got} ({t.activities_expected}) | {t.mixed} / {t.split} "
                f"| {len(t.errors)}/{t.files} | {t.seconds:.0f}초 |"
            )
            lines = [
                *(f"- 지어낸 수치 {x}" for x in t.made_up_numbers),
                *(f"- 주체 틀림 {x}" for x in t.actor_wrong),
                *(f"- 경험 아닌 것 {x}" for x in t.not_experiences),
                *(f"- 놓친 경험 {x}" for x in t.missed),
                *(f"- 놓친 활동 {x}" for x in t.missed_activities),
                *(f"- 활동 아닌 것 {x}" for x in t.not_activities),
                *(f"- 활동 수 {x}" for x in t.activity_counts),
                *(f"- 오류 {x}" for x in t.errors),
            ]
            if lines:
                details.append(f"\n### {name}\n\n" + "\n".join(lines))
    head = (
        "| 모델 · 프롬프트 | 근거 일치 | 원문에서 확인된 경험 | 지어낸 수치 | 주체 틀림 (ME 로) "
        "| 경험 찾음 | 경험 아닌 것 | 활동 찾음 | 활동 아닌 것 | 활동 수 (정답) | 섞임 / 쪼개짐 "
        "| 오류 | 시간 합 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    return "\n".join([head, *rows]) + "\n" + "".join(details) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="쉼표로 「제공사:모델」 — 기본은 LLM_MODELS 첫 모델")
    parser.add_argument("--prompts", default="tuned,generic", help="tuned · generic")
    parser.add_argument("--samples", help="쉼표로 자료 이름(확장자 없이) — 기본은 정답이 있는 전부")
    parser.add_argument("--out", type=Path, help="응답 저장 폴더")
    parser.add_argument("--rescore", type=Path, help="저장한 응답만 다시 채점(LLM 안 부름)")
    args = parser.parse_args()

    samples = load_answers(args.samples.split(",") if args.samples else None)
    if args.rescore:
        out = args.rescore
    else:
        if args.models:
            models = [m.strip() for m in args.models.split(",") if m.strip()]
        else:
            from dotenv import load_dotenv

            from app.config import load_settings

            load_dotenv(ROOT / ".env")
            p, m = load_settings().llm_models[0]
            models = [f"{p}:{m}"]
        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
        for p in prompts:
            if p not in PROMPTS:
                parser.error(f"프롬프트는 {list(PROMPTS)} 중 — {p!r}")
        out = args.out or ROOT / "eval-runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
        calls = len(samples) * len(models) * len(prompts)
        print(f"LLM 호출 {calls}회(규칙 위반 시 더) → {out}")
        asyncio.run(run(models, prompts, samples, out))

    text = report(out, samples)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
