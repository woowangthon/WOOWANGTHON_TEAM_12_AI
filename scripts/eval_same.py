"""시험 자료로 같은 활동 제안(/extract 의 sameAsActivityId, S-44 · S-60)을 채점한다(AI#31 · AI#67).

    uv run python scripts/eval_same.py                       # LLM_MODELS 첫 모델로, 우리 프롬프트
    uv run python scripts/eval_same.py --runs 2 --models openai:gpt-6-luna,openai:gpt-6.1-terra
    uv run python scripts/eval_same.py --rescore eval-runs/same-<시각>   # 저장한 응답만 다시 채점(LLM 안 부름)

LLM 호출 수 = 자료 수 × 모델 수 × runs(규칙 위반이면 자료마다 최대 LLM_MAX_ATTEMPTS 번).
응답은 --out 폴더(기본 eval-runs/same-<시각>, 커밋하지 않음)에 저장한다. 모델 폴백 없이 고른 모델만 부른다.
키는 --env 파일(기본 레포의 .env)에서 읽고 화면에 찍지 않는다.

자료: samples/same/inputs.json(지어낸 자료와 손으로 적은 정답 — 칸의 뜻은 samples/same/README.md).

숫자의 뜻:
- 잡음: 정답 `same` 이 id 인 활동 중 그 id 를 단 수(① 이름이 다르게 적힌 같은 활동)
- 틀리게 붙임: `same` · `maybe` 밖의 id 를 단 수 — 정답에 없는 LLM 활동이 id 를 단 것도 센다(② 가장 나쁜 실패)
- 놓침: `same` 이 id 인데 null 을 단 수(`maybe` 에 null 이 있으면 세지 않는다)
- 빈 목록에 id: existingActivities 가 [] 인데 id 를 단 활동 수(③ 0이어야 한다)
- 활동 못 찾음: 정답 활동(keys)이 LLM 활동 이름에 없는 수 — 제안을 채점할 수 없다
- 오류: 다시 시도 뒤에도 규칙을 어겨 LLM_INVALID_OUTPUT 등으로 끝난 수
- 시간: 자료 하나에 걸린 시간(다시 시도 포함)의 중앙값 · 최대 — 추출 전체 시간이다

대상이 있는 자료(`target`, S-60)만 따로 — BE 가 제안 없이 바로 붙이므로 가장 조심할 숫자다. 위 숫자에도 들어간다:
- 대상 찾음: 정답 `same` 이 대상인 활동 중 대상 id 를 단 수
- 대상 놓침: 정답이 대상인데(`maybe` 에 null 없음) 대상 id 를 못 단 수 — 활동을 못 찾은 것도 센다
- 대상에 틀리게 붙임: 정답이 대상이 아닌 활동에 대상 id 를 단 수(대상 쪽으로 쏠림)
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

from app.citations import normalize  # noqa: E402

SAMPLES = ROOT / "samples"
DATA = SAMPLES / "same"


def _json(name: str) -> list[dict]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def build_request(item: dict) -> dict:
    """자료 하나의 /extract 요청 — 원본은 textFrom(samples/ 의 파일), 기존 활동은 existing 이름으로 채운다."""
    req = dict(item["request"])
    if "textFrom" in item:
        req["text"] = (SAMPLES / item["textFrom"]).read_text(encoding="utf-8")
    kind = item.get("existing")
    if kind == "base":
        req["existingActivities"] = _json("existing-base.json")
    elif kind == "long":
        # BE 처럼 최근 것이 앞 — 기본 목록(2023 ~ 2025)이 앞, 100개(2019 ~ 2022)가 뒤
        req["existingActivities"] = _json("existing-base.json") + _json("existing-100.json")
    elif kind == "none":
        req["existingActivities"] = []
    req.setdefault("existingActivities", [])
    if "target" in item:
        req["targetActivityId"] = item["target"]
    return req


def load_inputs(only: list[str] | None) -> list[dict]:
    items = _json("inputs.json")
    return [i for i in items if not only or i["id"] in only]


@dataclass
class Score:
    total: int = 0
    expected_same: int = 0
    caught: int = 0
    wrong: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    id_on_empty: list[str] = field(default_factory=list)
    not_found: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    # 대상이 있는 자료만(S-60)
    target_expected: int = 0
    target_caught: int = 0
    target_missed: list[str] = field(default_factory=list)
    target_wrong: list[str] = field(default_factory=list)


def score_item(s: Score, item: dict, response: dict) -> None:
    req = build_request(item)
    names = {a["id"]: a["name"] for a in req["existingActivities"]}
    acts = response["activities"]
    normalized = [normalize(a["name"]) for a in acts]
    claimed: set[int] = set()  # 정답에 짝지어진 LLM 활동 번호
    target = item.get("target")
    wrong_before = len(s.wrong)

    for want in item["answers"]:
        keys = [normalize(k) for k in want["keys"]]
        # whole: 자료 전체가 한 활동이라 이름과 상관없이 LLM 활동 전부가 이 정답이다(이름 없는 자료)
        hits = [
            i for i, n in enumerate(normalized) if want.get("whole") or any(k in n for k in keys)
        ]
        ok = {want["same"], *want.get("maybe", [])}
        label = want["keys"][0] if want["keys"] else "(자료 전체)"
        if want["same"] is not None:
            s.expected_same += 1
        if target is not None and want["same"] == target:
            s.target_expected += 1
            if target in {acts[i]["sameAsActivityId"] for i in hits}:
                s.target_caught += 1
            elif None not in ok:
                s.target_missed.append(f"{item['id']}: {label}")
        if not hits:
            s.not_found.append(f"{item['id']}: {label}")
            continue
        claimed.update(hits)
        got = {acts[i]["sameAsActivityId"] for i in hits}
        if want["same"] is not None and want["same"] in got:
            s.caught += 1
        elif want["same"] is not None and None not in ok and got == {None}:
            s.missed.append(f"{item['id']}: {label} → null(정답 {want['same']})")
        for i in hits:
            g = acts[i]["sameAsActivityId"]
            if g is not None and g not in ok:  # null 이 틀린 건 놓침이다
                s.wrong.append(
                    f"{item['id']}: 「{acts[i]['name']}」 → {g} 「{names.get(g)}」"
                    f"(정답 {want['same']})"
                )

    for i, act in enumerate(acts):
        g = act["sameAsActivityId"]
        if g is None:
            continue
        line = f"{item['id']}: 「{act['name']}」 → {g} 「{names.get(g)}」"
        s.suggestions.append(line)
        if not req["existingActivities"]:
            s.id_on_empty.append(line)
        if i not in claimed:
            s.wrong.append(line + "(정답에 없는 활동)")
    s.target_wrong.extend(x for x in s.wrong[wrong_before:] if f"→ {target} " in x)


def score_dir(path: Path, inputs: list[dict]) -> Score:
    s = Score()
    for item in inputs:
        f = path / f"{item['id']}.json"
        if not f.exists():
            continue
        record = json.loads(f.read_text(encoding="utf-8"))
        s.total += 1
        s.seconds.append(record.get("seconds", 0.0))
        if "error" in record:
            s.errors.append(f"{item['id']}: {record['error']}")
            s.expected_same += sum(w["same"] is not None for w in item["answers"])
            if "target" in item:
                s.target_expected += sum(w["same"] == item["target"] for w in item["answers"])
            continue
        score_item(s, item, record["response"])
    return s


async def run(models: list[str], runs: int, inputs: list[dict], out: Path, env: Path) -> None:
    from dotenv import load_dotenv

    from app.config import load_settings, parse_models
    from app.errors import ApiError
    from app.extract import extract
    from app.llm import build_llm
    from app.schemas import ExtractRequest

    load_dotenv(env)  # 값은 화면에 찍지 않는다
    base = load_settings()
    gate = asyncio.Semaphore(4)
    clients = {}  # 모델마다 한 번만 만든다 — eval_extract 의 까닭과 같다
    for model in models:
        settings = replace(base, llm_models=parse_models(model))
        clients[model] = (settings, build_llm(settings))

    async def one(model: str, n: int, item: dict) -> None:
        settings, llm = clients[model]
        req = ExtractRequest.model_validate(build_request(item))
        async with gate:
            started = time.monotonic()
            try:
                data = await extract(req, llm, settings)
                record = {"response": json.loads(data.model_dump_json())}
            except ApiError as e:
                record = {"error": f"{e.code}: {e.message}"}
            record["seconds"] = round(time.monotonic() - started, 1)
        name = f"run-{n}"
        path = out / model.replace(":", "_").replace("/", "_") / name / f"{item['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        got = record.get("error") or [
            a["sameAsActivityId"] for a in record["response"]["activities"]
        ]
        print(f"  {model} · {name} · {item['id']} — {got} ({record['seconds']}초)")

    await asyncio.gather(
        *(one(m, n, item) for m in models for n in range(1, runs + 1) for item in inputs)
    )


def report(out: Path, inputs: list[dict]) -> str:
    rows, details = [], []
    for model_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            s = score_dir(run_dir, inputs)
            if not s.total:
                continue
            name = f"{model_dir.name} · {run_dir.name}"
            secs = s.seconds or [0.0]
            rows.append(
                f"| {name} | {s.caught}/{s.expected_same} | {len(s.wrong)} | {len(s.missed)} "
                f"| {len(s.id_on_empty)} | {len(s.not_found)} | {len(s.errors)}/{s.total} "
                f"| {statistics.median(secs):.1f}초 · {max(secs):.1f}초 "
                f"| {s.target_caught}/{s.target_expected} | {len(s.target_missed)} "
                f"| {len(s.target_wrong)} |"
            )
            lines = [
                *(f"- 틀리게 붙임 {x}" for x in s.wrong),
                *(f"- 대상 놓침 {x}" for x in s.target_missed),
                *(f"- 놓침 {x}" for x in s.missed),
                *(f"- 빈 목록에 id {x}" for x in s.id_on_empty),
                *(f"- 활동 못 찾음 {x}" for x in s.not_found),
                *(f"- 오류 {x}" for x in s.errors),
                "",
                "제안 전부:",
                *(f"- {x}" for x in s.suggestions),
            ]
            details.append(f"\n### {name}\n\n" + "\n".join(lines))
    head = (
        "| 모델 · 회차 | 잡음 | 틀리게 붙임 | 놓침 | 빈 목록에 id | 활동 못 찾음 | 오류 "
        "| 시간(중앙 · 최대) | 대상 찾음 | 대상 놓침 | 대상에 틀리게 붙임 |\n" + "| --- " * 11 + "|"
    )
    return "\n".join([head, *rows]) + "\n" + "".join(details) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="쉼표로 「제공사:모델」 — 기본은 LLM_MODELS 첫 모델")
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
            p, m = load_settings().llm_models[0]
            models = [f"{p}:{m}"]
        out = args.out or ROOT / "eval-runs" / f"same-{datetime.now():%Y%m%d-%H%M%S}"
        calls = len(inputs) * len(models) * args.runs
        print(f"LLM 호출 {calls}회(규칙 위반 시 더) → {out}")
        asyncio.run(run(models, args.runs, inputs, out, args.env))

    text = report(out, inputs)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
