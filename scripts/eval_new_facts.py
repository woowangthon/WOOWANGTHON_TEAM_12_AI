"""시험 확정 답으로 새 경험 후보를 채점한다 — 프롬프트 · 모델을 바꿀 때 같은 자료 · 같은 숫자로 견준다(AI#53).

    uv run python scripts/eval_new_facts.py                         # MATCH_LLM_MODELS 첫 모델로 1회
    uv run python scripts/eval_new_facts.py --models openai:gpt-6-luna --runs 2
    uv run python scripts/eval_new_facts.py --rescore eval-runs/new-facts-<시각>   # 저장한 응답만 다시 채점

LLM 호출 수 = 답 수 × 모델 수 × 회 수 — 기본 8회(규칙 위반이면 답마다 최대 LLM_MAX_ATTEMPTS 번).
빼야 할 문장이 다 빠져 남은 문장이 없는 답은 LLM 을 부르지 않는다(제품과 같다).
응답은 --out 폴더(기본 eval-runs/new-facts-<시각>, 커밋하지 않음)에 저장한다. 모델 폴백 없이 고른 모델만 부른다.
키는 --env 파일(기본 레포의 .env)에서 읽고 화면에 찍지 않는다.

자료: samples/match/experiences.json(지어낸 경험 전부) · samples/new-facts/activities.json ·
samples/new-facts/answers.json(문장마다 손으로 적은 정답).

숫자의 뜻:
- 후보: 돌려준 후보 수 합
- 오류: 끝까지 어겨 LLM_INVALID_OUTPUT 등으로 끝난 답 수
- 잘못 고름: 정답 SKIP 문장을 후보로 고른 수. kind 마다 따로 찍는다 — 0 이어야 한다(경험으로 새면 다음 초안의 근거가 된다)
- 찾음: 정답 NEW 문장을 고른 수 / 그런 문장 수(코드가 숫자 대조로 뺀 후보는 빠뜨린 것으로 센다)
- 주체 · 활동 맞음: 찾은 NEW 중 정답 actor · activityId 와 같은 수
- 다시 씀: mixed 문장의 content 에 받치던 사실(notIn)을 다시 쓴 수
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ground_check import split_sentences  # noqa: E402
from app.schemas import NewFactsResponse  # noqa: E402
from scripts.eval_match import _slug  # noqa: E402

DATA = ROOT / "samples" / "new-facts"
EXPERIENCES = ROOT / "samples" / "match" / "experiences.json"

SKIP_KINDS = ("draft", "backed", "opinion", "puff", "mine", "number")
NEW_KINDS = ("new", "mixed")


@dataclass
class Score:
    answers: int = 0
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0
    candidates: int = 0
    lines: list[str] = field(default_factory=list)
    wrong: dict[str, int] = field(default_factory=dict)
    wrong_lines: list[str] = field(default_factory=list)
    found: int = 0
    chances: int = 0
    missed: list[str] = field(default_factory=list)
    actor_ok: int = 0
    activity_ok: int = 0
    off: list[str] = field(default_factory=list)
    restated: list[str] = field(default_factory=list)

    def add(self, other: "Score") -> None:
        for name, value in vars(other).items():
            if isinstance(value, dict):
                mine = getattr(self, name)
                for key, count in value.items():
                    mine[key] = mine.get(key, 0) + count
            else:
                setattr(self, name, getattr(self, name) + value)


def score(answer: dict, data: NewFactsResponse) -> Score:
    aid = answer["id"]
    s = Score(candidates=len(data.candidates))
    picked = {c.sentence: c for c in data.candidates}
    for want in answer["sentences"]:
        got = picked.pop(want["text"], None)
        if want["want"] == "NEW":
            s.chances += 1
            if got is None:
                s.missed.append(f"{aid}: {want['kind']} 「{want['text']}」")
                continue
            s.found += 1
            actors = want["actor"] if isinstance(want["actor"], list) else [want["actor"]]
            s.actor_ok += got.actor in actors
            s.activity_ok += got.activityId == want["activityId"]
            line = (
                f"{aid}: {got.actor} · 활동 {got.activityId} — {got.content} ← 「{want['text']}」"
            )
            s.lines.append(line)
            if got.actor not in actors or got.activityId != want["activityId"]:
                s.off.append(f"정답 {want['actor']} · 활동 {want['activityId']} 아님 — {line}")
            if any(word in got.content for word in want.get("notIn", [])):
                s.restated.append(line)
        elif got is not None:
            kind = want["kind"]
            s.wrong[kind] = s.wrong.get(kind, 0) + 1
            s.wrong_lines.append(f"{kind} {aid}: {got.content} ← 「{want['text']}」")
    for sentence in picked:  # 자료의 문장이 아닌 후보 — 코드가 막으니 나오면 안 된다
        s.wrong["unknown"] = s.wrong.get("unknown", 0) + 1
        s.wrong_lines.append(f"unknown {aid}: 「{sentence}」")
    return s


def load_experiences() -> list[dict]:
    return json.loads(EXPERIENCES.read_text(encoding="utf-8"))


def load_activities() -> list[dict]:
    return json.loads((DATA / "activities.json").read_text(encoding="utf-8"))


def load_answers(only: list[str] | None = None) -> list[dict]:
    answers = json.loads((DATA / "answers.json").read_text(encoding="utf-8"))
    return [a for a in answers if a["id"] in only] if only else answers


def answer_text(answer: dict) -> str:
    return " ".join(s["text"] for s in answer["sentences"])


def request_body(answer: dict, experiences: list[dict], activities: list[dict]) -> dict:
    return {
        "text": answer_text(answer),
        "experiences": experiences,
        "activities": activities,
        "excludedSentences": [s["text"] for s in answer["sentences"] if s.get("excluded")],
    }


# ---- 실행 ----


async def run(
    models: list[str],
    runs: int,
    experiences: list[dict],
    activities: list[dict],
    answers: list[dict],
    out: Path,
    env: Path,
) -> None:
    from dotenv import load_dotenv

    from app.config import load_settings, parse_models
    from app.errors import ApiError
    from app.llm import build_llm
    from app.new_facts import new_facts
    from app.schemas import NewFactsRequest

    load_dotenv(env)  # 값은 화면에 찍지 않는다
    base = load_settings()
    gate = asyncio.Semaphore(4)
    clients = {}  # 모델마다 클라이언트를 한 번만 만든다(AI#37)
    for model in models:
        settings = replace(base, llm_models=parse_models(model))
        clients[model] = (settings, build_llm(settings))

    async def one(model: str, round_no: int, answer: dict) -> None:
        settings, llm = clients[model]
        req = NewFactsRequest.model_validate(request_body(answer, experiences, activities))
        async with gate:
            started = time.monotonic()
            try:
                response = await new_facts(req, llm, settings)
                record = {"response": json.loads(response.model_dump_json())}
            except ApiError as e:
                record = {"error": f"{e.code}: {e.message}"}
            record["seconds"] = round(time.monotonic() - started, 1)
        path = out / _slug(model) / f"r{round_no}" / f"{answer['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        status = record.get("error") or "ok"
        print(f"  {model} · r{round_no} · {answer['id']} — {status} ({record['seconds']}초)")

    await asyncio.gather(
        *(one(m, r, a) for m in models for r in range(1, runs + 1) for a in answers)
    )


def report(out: Path, answers: list[dict]) -> str:
    rows, details = [], []
    for model_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        for round_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            t = Score()
            for answer in answers:
                path = round_dir / f"{answer['id']}.json"
                if not path.exists():
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                t.answers += 1
                t.seconds += record.get("seconds", 0.0)
                if "error" in record:
                    t.errors.append(f"{answer['id']}: {record['error']}")
                    continue
                t.add(score(answer, NewFactsResponse.model_validate(record["response"])))
            if not t.answers:
                continue
            name = f"{model_dir.name} · {round_dir.name}"
            wrong = sum(t.wrong.values())
            kinds = " · ".join(f"{k} {n}" for k, n in sorted(t.wrong.items())) or "없음"
            rows.append(
                f"| {name} | {t.candidates} | {len(t.errors)}/{t.answers} | {wrong} "
                f"| {t.found}/{t.chances} | {t.actor_ok}/{t.found} | {t.activity_ok}/{t.found} "
                f"| {len(t.restated)} | {t.seconds:.0f}초 |"
            )
            lines = [
                f"- 잘못 고름 종류별: {kinds}",
                *(f"- 오류 {x}" for x in t.errors),
                *(f"- 잘못 고름 {x}" for x in t.wrong_lines),
                *(f"- 빠뜨림 {x}" for x in t.missed),
                *(f"- 주체 · 활동 다름 {x}" for x in t.off),
                *(f"- 다시 씀 {x}" for x in t.restated),
                *(f"- 후보 {x}" for x in t.lines),
            ]
            details.append(f"\n### {name}\n\n" + "\n".join(lines))
    head = (
        "| 모델 · 회 | 후보 | 오류 | 잘못 고름 | 찾음 | 주체 맞음 | 활동 맞음 | 다시 씀 | 시간 합 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    return "\n".join([head, *rows]) + "\n" + "".join(details) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="쉼표로 「제공사:모델」 — 기본은 MATCH_LLM_MODELS 첫 모델")
    parser.add_argument("--runs", type=int, default=1, help="모델마다 몇 번 돌리나")
    parser.add_argument("--answers", help="쉼표로 답 id — 기본은 전부")
    parser.add_argument("--out", type=Path, help="응답 저장 폴더")
    parser.add_argument("--env", type=Path, default=ROOT / ".env", help="키를 읽을 .env")
    parser.add_argument("--rescore", type=Path, help="저장한 응답만 다시 채점(LLM 안 부름)")
    args = parser.parse_args()

    answers = load_answers(args.answers.split(",") if args.answers else None)
    for answer in answers:  # 자료가 코드의 문장 나누기와 어긋나면 채점 전에 알린다
        assert split_sentences(answer_text(answer)) == [s["text"] for s in answer["sentences"]]
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
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = args.out or ROOT / "eval-runs" / f"new-facts-{stamp}"
        print(f"LLM 호출 최대 {len(answers) * len(models) * args.runs}회(규칙 위반 시 더) → {out}")
        experiences, activities = load_experiences(), load_activities()
        asyncio.run(run(models, args.runs, experiences, activities, answers, out, args.env))

    text = report(out, answers)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
