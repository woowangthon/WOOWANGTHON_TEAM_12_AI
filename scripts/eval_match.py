"""시험 문항으로 매칭을 채점한다 — 프롬프트 · 모델을 바꿀 때 같은 자료 · 같은 숫자로 견준다(AI#20).

    uv run python scripts/eval_match.py                    # MATCH_LLM_MODELS 첫 모델로, 우리 프롬프트 · 범용 프롬프트
    uv run python scripts/eval_match.py --models openai:gpt-6-luna,openai:gpt-5.4-mini
    uv run python scripts/eval_match.py --prompts tuned    # 우리 프롬프트만
    uv run python scripts/eval_match.py --rescore eval-runs/match-<시각>   # 저장한 응답만 다시 채점(LLM 안 부름)

LLM 호출 수 = 문항 수 × 모델 수 × 프롬프트 수(규칙 위반이면 문항마다 최대 LLM_MAX_ATTEMPTS 번) — 기본 10 × 1 × 2 = 20회.
응답은 --out 폴더(기본 eval-runs/match-<시각>, 커밋하지 않음)에 저장한다. 모델 폴백 없이 고른 모델만 부른다.

자료: samples/match/experiences.json(지어낸 사용자 한 명의 경험 · 근거 · 보완 — /match 요청 모양 그대로)과
samples/match/questions.json(문항마다 손으로 적은 정답). 정답 칸:
- kind: match(맞는 경험이 있다) · partial(일부만 받친다) · none(맞는 경험이 없다 — 0개여야 한다)
- good: 첫 추천으로 맞는 경험 · ok: 추천해도 되는 경험 · bad: 추천하면 틀린 경험 · allowZero: 0개도 맞음

숫자의 뜻:
- 첫 추천 맞음: match · partial 문항에서 첫 추천이 good 인 수(allowZero 면 0개도 맞음)
- 엉뚱한 추천: bad 에 있는 경험을 추천한 수
- 억지 추천: none 문항에 1개 이상 추천한 문항 수
- 요구사항 판정: match 는 SUPPORTED 가 하나 이상 · partial 은 SUPPORTED 아닌 것이 하나 이상 · none 은 전부 NOT_FOUND 인 문항 수
- 지어낸 수치: 추천의 reason · emphasis 숫자 중 그 경험(본문 · 근거 · 보완 · 기간)과 문항에 없는 숫자가 있는 추천 수
  (요청의 경험 ID 는 뺀다 — 「115번 경험」은 수치가 아니다)
- 팀 일을 내 일로: actor 가 TEAM · UNKNOWN 인 추천의 reason · emphasis 에 「내가 · 제가 · 본인 · 직접 · 주도 …」가 있는 수
  — 낱말로 거른 후보다(「누가 고쳤는지 자료에 없다」도 걸릴 수 있다). 목록을 사람이 읽고 판단한다
- 보완 씀: 보완이 실린 경험을 추천했을 때 evidence 에 보완을 넣은 수 / 그런 추천 수(S-25)
- 오류: 다시 시도 뒤에도 규칙(ID 가 요청에 있나 등)을 어겨 LLM_INVALID_OUTPUT 등으로 끝난 문항 수
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

from app.match import SYSTEM_PROMPT  # noqa: E402
from app.schemas import MatchExperience, MatchResponse  # noqa: E402

DATA = ROOT / "samples" / "match"

# 범용 프롬프트 — 사용자가 ChatGPT 에 그냥 붙일 만한 한 줄. 숫자를 같이 재려고 출력 모양(JSON 스키마)과
# 경험 목록 모양만 같게 준다. ID 검사도 같은 코드가 한다
GENERIC_PROMPT = "이 문항에 맞는 내 경험을 골라 줘."
PROMPTS = {"tuned": SYSTEM_PROMPT, "generic": GENERIC_PROMPT}

# 팀 · 모르는 일을 내 일로 말했는지 거르는 낱말(후보 — 사람이 본다)
# 「문제가」의 「제가」처럼 낱말 가운데는 거르지 않는다. 「맡아」는 「팀원이 맡아」도 있어 넣지 않는다
MINE_WORDS = re.compile(r"(?<![가-힣])(?:내가|제가|나는|저는|본인)|직접|주도|이끌")


@dataclass
class Score:
    questions: int = 0
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0
    picked_total: int = 0
    picked_right: int = 0
    wrong_picks: list[str] = field(default_factory=list)
    forced: list[str] = field(default_factory=list)
    req_total: int = 0
    req_right: int = 0
    req_wrong: list[str] = field(default_factory=list)
    made_up_numbers: list[str] = field(default_factory=list)
    team_as_mine: list[str] = field(default_factory=list)
    supplement_recs: int = 0
    supplement_used: int = 0
    picks: list[str] = field(default_factory=list)

    def add(self, other: "Score") -> None:
        for name, value in vars(other).items():
            setattr(self, name, getattr(self, name) + value)


def _ints(text: str) -> set[int]:
    return {int(n) for n in re.findall(r"\d+", unicodedata.normalize("NFKC", text))}


def _experience_text(exp: MatchExperience) -> str:
    parts = [exp.activityName or "", exp.content]
    if exp.period:
        parts += [exp.period.start or "", exp.period.end or ""]
    parts += [c.quote for c in exp.citations]
    parts += [f"{s.question} {s.answer}" for s in exp.supplements]
    return " ".join(parts)


def score(question: dict, experiences: list[MatchExperience], data: MatchResponse) -> Score:
    qid = question["id"]
    kind = question["kind"]
    by_id = {e.id: e for e in experiences}
    s = Score(questions=1)
    recs = data.recommendations
    ids = [r.experienceId for r in recs]
    statuses = [r.status for r in data.requirements]
    s.picks.append(f"{qid}: 추천 {ids or '없음'} · 요구사항 {statuses}")

    if kind in ("match", "partial"):
        s.picked_total += 1
        if (ids and ids[0] in question["good"]) or (not ids and question.get("allowZero")):
            s.picked_right += 1
    elif ids:
        s.forced.append(f"{qid}: {ids}")
    s.wrong_picks += [f"{qid}: {i}" for i in ids if i in question["bad"]]

    s.req_total += 1
    right = {
        "match": "SUPPORTED" in statuses,
        "partial": any(x != "SUPPORTED" for x in statuses),
        "none": all(x == "NOT_FOUND" for x in statuses),
    }[kind]
    s.req_right += right
    if not right:
        s.req_wrong.append(
            f"{qid}({kind}): " + " · ".join(f"{r.text}={r.status}" for r in data.requirements)
        )

    known_ints = _ints(question["text"]) | set(by_id)
    for rec in recs:
        exp = by_id[rec.experienceId]
        said = f"{rec.reason} / {rec.emphasis}"
        extra = _ints(said) - _ints(_experience_text(exp)) - known_ints
        if extra:
            s.made_up_numbers.append(f"{qid}: 경험 {exp.id} {sorted(extra)} 「{said}」")
        words = MINE_WORDS.findall(said)
        if exp.actor != "ME" and words:
            s.team_as_mine.append(f"{qid}: 경험 {exp.id}({exp.actor}) {words} 「{said}」")
        if exp.supplements:
            s.supplement_recs += 1
            s.supplement_used += any(e.type == "SUPPLEMENT" for e in rec.evidence)
    return s


def load_data(only: list[str] | None) -> tuple[list[dict], list[dict]]:
    experiences = json.loads((DATA / "experiences.json").read_text(encoding="utf-8"))
    questions = json.loads((DATA / "questions.json").read_text(encoding="utf-8"))
    if only:
        questions = [q for q in questions if q["id"] in only]
    return experiences, questions


# ---- 실행 ----


def _slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


async def run(
    models: list[str], prompts: list[str], experiences: list[dict], questions: list[dict], out: Path
) -> None:
    from dotenv import load_dotenv

    from app.config import load_settings, parse_models
    from app.errors import ApiError
    from app.llm import build_llm
    from app.match import match
    from app.schemas import MatchRequest

    load_dotenv(ROOT / ".env")  # 값은 화면에 찍지 않는다
    base = load_settings()
    gate = asyncio.Semaphore(4)

    async def one(model: str, prompt: str, q: dict) -> None:
        settings = replace(base, llm_models=parse_models(model))
        llm = build_llm(settings)
        req = MatchRequest.model_validate(
            {
                "question": {"organization": q["organization"], "text": q["text"]},
                "experiences": experiences,
            }
        )
        async with gate:
            started = time.monotonic()
            try:
                data = await match(req, llm, settings, system=PROMPTS[prompt])
                record = {"response": json.loads(data.model_dump_json())}
            except ApiError as e:
                record = {"error": f"{e.code}: {e.message}"}
            record["seconds"] = round(time.monotonic() - started, 1)
        path = out / _slug(model) / prompt / f"{q['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"  {model} · {prompt} · {q['id']} — {record.get('error') or 'ok'} ({record['seconds']}초)"
        )

    await asyncio.gather(*(one(m, p, q) for m in models for p in prompts for q in questions))


def report(out: Path, experiences: list[dict], questions: list[dict]) -> str:
    exps = [MatchExperience.model_validate(e) for e in experiences]
    rows, details = [], []
    for model_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        for prompt_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            t = Score()
            for q in questions:
                path = prompt_dir / f"{q['id']}.json"
                if not path.exists():
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                t.seconds += record.get("seconds", 0.0)
                if "error" in record:
                    t.questions += 1
                    t.errors.append(f"{q['id']}: {record['error']}")
                    continue
                t.add(score(q, exps, MatchResponse.model_validate(record["response"])))
            name = f"{model_dir.name} · {prompt_dir.name}"
            rows.append(
                f"| {name} | {t.picked_right}/{t.picked_total} | {len(t.wrong_picks)} "
                f"| {len(t.forced)} | {t.req_right}/{t.req_total} | {len(t.made_up_numbers)} "
                f"| {len(t.team_as_mine)} | {t.supplement_used}/{t.supplement_recs} "
                f"| {len(t.errors)}/{t.questions} | {t.seconds:.0f}초 |"
            )
            lines = [
                *(f"- 엉뚱한 추천 {x}" for x in t.wrong_picks),
                *(f"- 억지 추천 {x}" for x in t.forced),
                *(f"- 요구사항 판정 틀림 {x}" for x in t.req_wrong),
                *(f"- 지어낸 수치 {x}" for x in t.made_up_numbers),
                *(f"- 팀 일을 내 일로(후보) {x}" for x in t.team_as_mine),
                *(f"- 오류 {x}" for x in t.errors),
                *(f"- 고른 것 {x}" for x in t.picks),
            ]
            details.append(f"\n### {name}\n\n" + "\n".join(lines))
    head = (
        "| 모델 · 프롬프트 | 첫 추천 맞음 | 엉뚱한 추천 | 억지 추천 | 요구사항 판정 | 지어낸 수치 "
        "| 팀 일을 내 일로(후보) | 보완 씀 | 오류 | 시간 합 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    return "\n".join([head, *rows]) + "\n" + "".join(details) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="쉼표로 「제공사:모델」 — 기본은 MATCH_LLM_MODELS 첫 모델")
    parser.add_argument("--prompts", default="tuned,generic", help="tuned · generic")
    parser.add_argument("--questions", help="쉼표로 문항 id — 기본은 전부")
    parser.add_argument("--out", type=Path, help="응답 저장 폴더")
    parser.add_argument("--rescore", type=Path, help="저장한 응답만 다시 채점(LLM 안 부름)")
    args = parser.parse_args()

    experiences, questions = load_data(args.questions.split(",") if args.questions else None)
    if args.rescore:
        out = args.rescore
    else:
        if args.models:
            models = [m.strip() for m in args.models.split(",") if m.strip()]
        else:
            from dotenv import load_dotenv

            from app.config import load_settings

            load_dotenv(ROOT / ".env")
            p, m = load_settings().match_models[0]
            models = [f"{p}:{m}"]
        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
        for p in prompts:
            if p not in PROMPTS:
                parser.error(f"프롬프트는 {list(PROMPTS)} 중 — {p!r}")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = args.out or ROOT / "eval-runs" / f"match-{stamp}"
        calls = len(questions) * len(models) * len(prompts)
        print(f"LLM 호출 {calls}회(규칙 위반 시 더) → {out}")
        asyncio.run(run(models, prompts, experiences, questions, out))

    text = report(out, experiences, questions)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
