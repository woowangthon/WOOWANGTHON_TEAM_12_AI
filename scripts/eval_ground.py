"""시험 답으로 근거 점검을 채점한다 — 프롬프트 · 모델을 바꿀 때 같은 자료 · 같은 숫자로 견준다(AI#32).

    uv run python scripts/eval_ground.py                         # MATCH_LLM_MODELS 첫 모델로 1회
    uv run python scripts/eval_ground.py --models openai:gpt-6-luna --runs 2
    uv run python scripts/eval_ground.py --rescore eval-runs/ground-<시각>   # 저장한 응답만 다시 채점(LLM 안 부름)

LLM 호출 수 = 답 수 × 모델 수 × 회 수 — 기본 15회(규칙 위반이면 답마다 최대 LLM_MAX_ATTEMPTS 번).
응답은 --out 폴더(기본 eval-runs/ground-<시각>, 커밋하지 않음)에 저장한다. 모델 폴백 없이 고른 모델만 부른다.
키는 --env 파일(기본 레포의 .env)에서 읽고 화면에 찍지 않는다.

자료: samples/match/experiences.json(지어낸 경험 전부) · samples/ground/experiences.json(점검에만 쓰는 경험 —
짧게 답한 보완, AI#100)과 samples/ground/answers.json(문장마다 손으로 적은 정답).

숫자의 뜻:
- 문장: 판정을 받은 문장 수 합
- 다시 부름: 코드 검사에 걸려 LLM 을 한 번 더 부른 답 수 · 오류: 끝까지 어겨 LLM_INVALID_OUTPUT 등으로 끝난 답 수
- 잡음: 정답이 UNSUPPORTED 인 문장을 UNSUPPORTED 로 둔 수 / 그런 문장 수. kind 마다 따로 찍는다
- 코드가 잡음: 그중 숫자 대조가 돌린 수(note 로 센다)
- 잘못 칠함: 받쳐진 문장(SOURCE · USER_INPUT)을 UNSUPPORTED 로 둔 수 — 사용자가 멀쩡한 문장을 의심하게 된다
- 생각으로 뺌: 근거 없는 사실(정답 UNSUPPORTED)을 OPINION 으로 둔 수 — 점검을 빠져나간다
- 받침 다름: 받쳐진 문장을 UNSUPPORTED 가 아닌 다른 status 로 둔 수(SOURCE ↔ USER_INPUT · OPINION)
- 생각 칠함: 정답 OPINION 문장을 다른 것으로 둔 수
- 맞는 근거: status 가 맞은 받쳐진 문장 중 정답 evidence 를 붙인 수 / 그런 문장 수
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
from app.schemas import GroundCheckResponse  # noqa: E402
from scripts.eval_match import _slug  # noqa: E402

DATA = ROOT / "samples" / "ground"
EXPERIENCES = ROOT / "samples" / "match" / "experiences.json"

KINDS = ("exaggerate", "mine", "number", "word-number", "none")
NUMBER_NOTE = "문장의 숫자("


@dataclass
class Score:
    answers: int = 0
    errors: list[str] = field(default_factory=list)
    retried: int = 0
    seconds: float = 0.0
    lines: list[str] = field(default_factory=list)
    caught: dict[str, int] = field(default_factory=dict)
    chances: dict[str, int] = field(default_factory=dict)
    by_code: int = 0
    missed: list[str] = field(default_factory=list)
    false_alarm: list[str] = field(default_factory=list)
    to_opinion: list[str] = field(default_factory=list)
    wrong_backing: list[str] = field(default_factory=list)
    opinion_wrong: list[str] = field(default_factory=list)
    evidence_hit: int = 0
    evidence_chances: int = 0
    evidence_miss: list[str] = field(default_factory=list)

    def add(self, other: "Score") -> None:
        for name, value in vars(other).items():
            if isinstance(value, dict):
                mine = getattr(self, name)
                for key, count in value.items():
                    mine[key] = mine.get(key, 0) + count
            else:
                setattr(self, name, getattr(self, name) + value)


def score(answer: dict, data: GroundCheckResponse) -> Score:
    aid = answer["id"]
    s = Score()
    for want, got in zip(answer["sentences"], data.sentences, strict=True):
        pairs = {(e.type, e.id) for e in got.evidence}
        line = f"{aid}: {want['want']} → {got.status} {sorted(pairs)} 「{got.text}」"
        if got.note:
            line += f" — {got.note}"
        s.lines.append(line)
        if want["want"] == "UNSUPPORTED":
            kind = want["kind"]
            s.chances[kind] = s.chances.get(kind, 0) + 1
            if got.status == "UNSUPPORTED":
                s.caught[kind] = s.caught.get(kind, 0) + 1
                s.by_code += got.note is not None and got.note.startswith(NUMBER_NOTE)
            else:
                s.missed.append(f"{kind} {line}")
                if got.status == "OPINION":
                    s.to_opinion.append(line)
        elif want["want"] == "OPINION":
            if got.status != "OPINION":
                s.opinion_wrong.append(line)
        elif got.status == "UNSUPPORTED":
            s.false_alarm.append(line)
        elif got.status != want["want"]:
            s.wrong_backing.append(line)
        else:
            s.evidence_chances += 1
            if {tuple(e) for e in want["evidence"]} & pairs:
                s.evidence_hit += 1
            else:
                s.evidence_miss.append(f"{want['evidence']} 아님 — {line}")
    return s


def load_experiences() -> list[dict]:
    extra = (DATA / "experiences.json").read_text(encoding="utf-8")
    return json.loads(EXPERIENCES.read_text(encoding="utf-8")) + json.loads(extra)


def load_answers(only: list[str] | None = None) -> list[dict]:
    answers = json.loads((DATA / "answers.json").read_text(encoding="utf-8"))
    return [a for a in answers if a["id"] in only] if only else answers


def answer_text(answer: dict) -> str:
    """번호 목록 답(문장에 mark 가 있다)은 줄마다 번호를 붙이고, 그 밖엔 공백으로 잇는다."""
    sentences = answer["sentences"]
    if "mark" in sentences[0]:
        return "\n".join(f"{s['mark']}{s['text']}" for s in sentences)
    return " ".join(s["text"] for s in sentences)


# ---- 실행 ----


async def run(
    models: list[str],
    runs: int,
    experiences: list[dict],
    answers: list[dict],
    out: Path,
    env: Path,
) -> None:
    from dotenv import load_dotenv

    from app.config import load_settings, parse_models
    from app.errors import ApiError
    from app.ground_check import (
        LLM_JSON_SCHEMA,
        SYSTEM_PROMPT,
        build_user_message,
        check_request,
        parse_output,
    )
    from app.llm import build_llm, generate_valid
    from app.schemas import GroundCheckRequest

    load_dotenv(env)  # 값은 화면에 찍지 않는다
    base = load_settings()
    gate = asyncio.Semaphore(4)
    clients = {}  # 모델마다 클라이언트를 한 번만 만든다(AI#37)
    for model in models:
        settings = replace(base, llm_models=parse_models(model))
        clients[model] = (settings, build_llm(settings))

    async def one(model: str, round_no: int, answer: dict) -> None:
        settings, llm = clients[model]
        req = GroundCheckRequest.model_validate(
            {"text": answer_text(answer), "experiences": experiences}
        )
        sentences, exps = check_request(req, settings)
        async with gate:
            started = time.monotonic()
            try:
                got = await generate_valid(
                    llm,
                    SYSTEM_PROMPT,
                    build_user_message(sentences, exps),
                    LLM_JSON_SCHEMA,
                    lambda text: parse_output(text, sentences, exps),
                    settings,
                )
                response = GroundCheckResponse(sentences=got.value, model=got.model)
                record = {"response": json.loads(response.model_dump_json())}
                record["attempt"] = got.attempt
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
                t.retried += record.get("attempt", 1) > 1
                t.add(score(answer, GroundCheckResponse.model_validate(record["response"])))
            if not t.answers:
                continue
            name = f"{model_dir.name} · {round_dir.name}"
            caught, chances = sum(t.caught.values()), sum(t.chances.values())
            kinds = " · ".join(
                f"{k} {t.caught.get(k, 0)}/{t.chances[k]}" for k in KINDS if k in t.chances
            )
            rows.append(
                f"| {name} | {len(t.lines)} | {t.retried} | {len(t.errors)}/{t.answers} "
                f"| {caught}/{chances} | {t.by_code} | {len(t.false_alarm)} "
                f"| {len(t.to_opinion)} | {len(t.wrong_backing)} | {len(t.opinion_wrong)} "
                f"| {t.evidence_hit}/{t.evidence_chances} | {t.seconds:.0f}초 |"
            )
            lines = [
                f"- 종류별 잡음: {kinds}",
                *(f"- 오류 {x}" for x in t.errors),
                *(f"- 놓침 {x}" for x in t.missed),
                *(f"- 잘못 칠함 {x}" for x in t.false_alarm),
                *(f"- 받침 다름 {x}" for x in t.wrong_backing),
                *(f"- 생각 칠함 {x}" for x in t.opinion_wrong),
                *(f"- 근거 다름 {x}" for x in t.evidence_miss),
                *(f"- 문장 {x}" for x in t.lines),
            ]
            details.append(f"\n### {name}\n\n" + "\n".join(lines))
    head = (
        "| 모델 · 회 | 문장 | 다시 부름 | 오류 | 잡음 | 코드가 잡음 | 잘못 칠함 | 생각으로 뺌 "
        "| 받침 다름 | 생각 칠함 | 맞는 근거 | 시간 합 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
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
        out = args.out or ROOT / "eval-runs" / f"ground-{stamp}"
        print(f"LLM 호출 {len(answers) * len(models) * args.runs}회(규칙 위반 시 더) → {out}")
        asyncio.run(run(models, args.runs, load_experiences(), answers, out, args.env))

    text = report(out, answers)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
