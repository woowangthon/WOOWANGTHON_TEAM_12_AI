"""시험 문항으로 매칭 · 되묻기를 채점한다 — 프롬프트 · 모델을 바꿀 때 같은 자료 · 같은 숫자로 견준다(AI#20 · AI#21).

    uv run python scripts/eval_match.py                    # MATCH_LLM_MODELS 첫 모델로, 우리 프롬프트 · 범용 프롬프트
    uv run python scripts/eval_match.py --models openai:gpt-6-luna,openai:gpt-5.4-mini
    uv run python scripts/eval_match.py --prompts tuned    # 우리 프롬프트만
    uv run python scripts/eval_match.py --variants with    # 보완을 뺀 자료 · 유형만 보는 문항은 돌리지 않음
    uv run python scripts/eval_match.py --variants types   # 문항 유형(S-41)만 — 자주 나오는 꼴 문항
    uv run python scripts/eval_match.py --rescore eval-runs/match-<시각>   # 저장한 응답만 다시 채점(LLM 안 부름)
    uv run python scripts/eval_match.py --env ../다른-클론/.env         # 키를 다른 .env 에서(worktree)

LLM 호출 수 = (문항 수 + 보완 없음 문항 수 + 유형 문항 수) × 모델 수 × 프롬프트 수
(규칙 위반이면 문항마다 최대 LLM_MAX_ATTEMPTS 번) — 기본 (10 + 3 + 15) × 1 × 2 = 56회. 「보완 없음」은 보완이 실린 경험이 정답(good · ok)인 문항만, 경험에서 보완을 모두 빼고
한 번 더 돌린다 — 문항 A 에서 답하기 전(데모 4)과 답한 뒤(데모 6 · S-25)를 견준다. 폴더 이름 끝 -nosupp.
「유형 문항」은 samples/match/question-types.json — 「자주 나오는 문항」(FE src/lib/common-questions.ts) 꼴을 본떠
지어낸 문항이다. 같은 경험으로 /match 를 그대로 부르되 문항 유형만 채점한다. 폴더 이름 끝 -types.
응답은 --out 폴더(기본 eval-runs/match-<시각>, 커밋하지 않음)에 저장한다. 모델 폴백 없이 고른 모델만 부른다.

자료: samples/match/experiences.json(지어낸 사용자 한 명의 경험 · 근거 · 보완 — /match 요청 모양 그대로)과
samples/match/questions.json(문항마다 손으로 적은 정답). 정답 칸:
- kind: match(맞는 경험이 있다) · partial(일부만 받친다) · none(맞는 경험이 없다 — 0개여야 한다)
- good: 첫 추천으로 맞는 경험 · ok: 추천해도 되는 경험 · bad: 추천하면 틀린 경험 · allowZero: 0개도 맞음
- questionType: 문항이 주로 묻는 유형(S-41) · questionTypeOk: 겹치는 문항이라 그것도 틀렸다고 하기 어려운 유형

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
- 문항 유형 정답: questionType 과 같은 수 · 허용 포함: questionTypeOk 까지 맞다고 친 수(S-41)
- 추천 수 · 활동 수(S-84 — 0 ~ 8개 · 활동 5개까지): 답한 문항의 추천 경험 수 합 · 서로 다른 활동 수 합
- 정답 밖(후보): good · ok 에도 bad 에도 없는 경험을 추천한 수 — 상한이 늘어 억지로 채웠나 사람이 읽는다
- 활동 통째(후보): 경험이 3개 이상인 활동의 경험을 전부 추천한 수 — 「문항에 쓸 만한 줄만」을 어겼나 본다

되묻기(S-22 · S-25) — samples/match/clarify.json 의 「경험에 이미 있는 사실」과 그걸 묻는 정규식으로 센다.
정규식 · 낱말로 거른 것은 전부 후보다 — 목록을 사람이 읽고 판단한다(질문 전체도 찍는다):
- 되묻기 수: 질문 수 합(0개 문항 수 / 답한 문항 수)
- 이미 있는 사실 다시 물음: 질문이 그 경험의 근거 · 보완에 이미 있는 사실을 묻는 수 / 그런 사실이 있는 경험을 추천한 기회
- 보완 없을 때 그 빈칸 물음: -nosupp 에서, 뺀 보완이 답하던 사실을 물은 수 / 그 사실이 필요한 문항(askIfMissing)에서
  그 경험을 추천한 기회(데모 4)
- 수치 · 답 유도: 질문에 그 경험 · 문항에 없는 숫자가 있거나 「…죠?」「맞나요?」처럼 확인만 받는 어미인 수
- 막연한 질문: 「더 자세히 알려 주세요」「구체적으로 설명해 주세요」 꼴인 수
- 한 질문에 여러 개: 물음표가 둘 이상인 질문 수 — 「최대 3개」를 한 질문에 몰아 넘는다
- 내 몫 물음: asksMyRole 문항에서 TEAM · UNKNOWN 경험을 추천했을 때 그 경험에 내 몫(맡은 · 직접 · 역할 …)을 물은 수(S-20)
- 질문 갈래(후보, AI#99): 질문마다 낱말로 한 갈래 — 내 몫(맡은 · 직접 · 어떤 부분 …) → 판단 · 원인 · 입장(왜 · 이유 · 기준 · 원인 · 중요하게 · 걱정 …) →
  확인 · 결과(확인 · 검증 · 달라 · 결과 …) → 그 밖(배경 · 상황 · 무엇을 했나) 순으로 먼저 걸린 것. 평가자가 보는 순서
  (내 몫 → 판단 이유 → 확인 · 전후 변화)를 따르면 「그 밖」이 줄고 판단 · 확인이 는다
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
GENERIC_PROMPT = "이 문항에 맞는 내 경험을 골라 주고, 부족한 걸 물어봐 줘."
PROMPTS = {"tuned": SYSTEM_PROMPT, "generic": GENERIC_PROMPT}

# 팀 · 모르는 일을 내 일로 말했는지 거르는 낱말(후보 — 사람이 본다)
# 「문제가」의 「제가」처럼 낱말 가운데는 거르지 않는다. 「맡아」는 「팀원이 맡아」도 있어 넣지 않는다
MINE_WORDS = re.compile(r"(?<![가-힣])(?:내가|제가|나는|저는|본인)|직접|주도|이끌")

# 되묻기 — 확인만 받는 어미 · 막연한 질문 · 내 몫을 묻는 낱말(후보 — 사람이 본다)
# 되묻기 갈래(후보 — 사람이 읽는다, AI#99). 앞의 것부터 먼저 걸린 갈래 하나
ASK_KINDS = (
    ("내 몫", re.compile(r"맡|직접|본인|내 몫|담당|역할|(어느|어떤) 부분")),
    (
        "판단 · 원인 · 입장",
        re.compile(
            r"왜|이유|골랐|고른|선택|대신|견주|견줬|비교|결정|기준|원인|파악|중요하게|원했|걱정|우려"
        ),
    ),
    ("확인 · 결과", re.compile(r"확인|검증|테스트|달라|결과|줄었|늘었|효과|반응|변화|어떻게 됐")),
)
LEADING = re.compile(r"(죠|지요|맞나요|맞으신가요|맞습니까|아니었나요|않았나요|않으셨나요)\s*\?")
VAGUE = re.compile(
    r"(자세히|자세하게|구체적으로|좀 더|더|추가로)\s*(알려|설명|말씀|이야기|적어|들려|공유)"
)
SHARE = re.compile(r"맡|직접|본인|몫|역할|기여|담당|하신 일|한 일|참여")
NOSUPP = "-nosupp"
TYPES = "-types"


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
    clarifications: list[str] = field(default_factory=list)
    clar_empty: int = 0
    repeats: list[str] = field(default_factory=list)
    repeat_chances: int = 0
    gap_asked: int = 0
    gap_chances: int = 0
    leading: list[str] = field(default_factory=list)
    vague: list[str] = field(default_factory=list)
    compound: list[str] = field(default_factory=list)
    share_asked: int = 0
    share_chances: int = 0
    type_total: int = 0
    type_right: int = 0
    type_ok: int = 0
    type_wrong: list[str] = field(default_factory=list)
    rec_count: int = 0
    activity_count: int = 0
    unlabeled: list[str] = field(default_factory=list)
    whole_activity: list[str] = field(default_factory=list)
    ask_mine: int = 0
    ask_why: int = 0
    ask_check: int = 0
    ask_other: list[str] = field(default_factory=list)

    def add(self, other: "Score") -> None:
        for name, value in vars(other).items():
            setattr(self, name, getattr(self, name) + value)


def ask_kind(question: str) -> str:
    """되묻기 갈래 — ASK_KINDS 중 먼저 걸린 것, 없으면 「그 밖」."""
    return next((name for name, pattern in ASK_KINDS if pattern.search(question)), "그 밖")


def _ints(text: str) -> set[int]:
    return {int(n) for n in re.findall(r"\d+", unicodedata.normalize("NFKC", text))}


def _experience_text(exp: MatchExperience) -> str:
    parts = [exp.activityName or "", exp.content]
    if exp.period:
        parts += [exp.period.start or "", exp.period.end or ""]
    parts += [c.quote for c in exp.citations]
    parts += [f"{s.question} {s.answer}" for s in exp.supplements]
    return " ".join(parts)


def _has(exp: MatchExperience, fact: dict) -> bool:
    items = exp.citations if fact["source"] == "CITATION" else exp.supplements
    return any(x.id == fact["id"] for x in items)


def score_type(question: dict, data: MatchResponse) -> Score:
    """문항 유형(S-41)만 — 유형 문항(-types)은 이것만, 매칭 문항은 score 안에서도 부른다."""
    s = Score(questions=1)
    if "questionType" not in question:
        return s
    s.type_total = 1
    got, want = data.questionType, question["questionType"]
    s.type_right = got == want
    s.type_ok = got == want or got in question.get("questionTypeOk", ())
    if got != want:
        mark = "허용" if s.type_ok else "틀림"
        s.type_wrong.append(f"{question['id']}: {got}({mark}, 정답 {want}) 「{question['text']}」")
    return s


def score(
    question: dict,
    experiences: list[MatchExperience],
    data: MatchResponse,
    known: list[dict] = (),
    stripped: bool = False,
) -> Score:
    """known: clarify.json 의 사실들. stripped: 보완을 뺀 자료(-nosupp)로 돌린 응답인가."""
    qid = question["id"]
    kind = question["kind"]
    by_id = {e.id: e for e in experiences}
    s = score_type(question, data)
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

    # ---- 개수(S-84) ----
    s.rec_count += len(ids)
    s.activity_count += len({by_id[i].activityId for i in ids})
    labeled = {*question["good"], *question.get("ok", ()), *question["bad"]}
    s.unlabeled += [f"{qid}: {i} 「{by_id[i].content}」" for i in ids if i not in labeled]
    members: dict[int | None, set[int]] = {}
    for e in experiences:
        members.setdefault(e.activityId, set()).add(e.id)
    for aid, group in members.items():
        if aid is not None and len(group) >= 3 and group <= set(ids):
            s.whole_activity.append(f"{qid}: 활동 {aid} 경험 {sorted(group)}")

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

    # ---- 되묻기 ----
    clars = data.clarifications
    s.clar_empty += not clars
    for c in clars:
        exp = by_id[c.experienceId]
        line = f"{qid}: 경험 {exp.id}({exp.actor}) 「{c.question}」"
        s.clarifications.append(line)
        extra = _ints(c.question) - _ints(_experience_text(exp)) - known_ints
        if extra or LEADING.search(c.question):
            s.leading.append(line)
        if VAGUE.search(c.question):
            s.vague.append(line)
        if c.question.count("?") > 1:
            s.compound.append(line)
        kind = ask_kind(c.question)
        if kind == "내 몫":
            s.ask_mine += 1
        elif kind == "판단 · 원인 · 입장":
            s.ask_why += 1
        elif kind == "확인 · 결과":
            s.ask_check += 1
        else:
            s.ask_other.append(line)
    for fact in known:
        if fact["experienceId"] not in ids:
            continue
        exp = by_id[fact["experienceId"]]
        asked = [
            c.question
            for c in clars
            if c.experienceId == exp.id and re.search(fact["pattern"], c.question)
        ]
        if _has(exp, fact):
            s.repeat_chances += 1
            s.repeats += [
                f"{qid}: 경험 {exp.id} 「{q}」 — 이미 있음: {fact['fact']}" for q in asked
            ]
        elif stripped and qid in fact.get("askIfMissing", ()):
            s.gap_chances += 1
            s.gap_asked += bool(asked)
    if question.get("asksMyRole"):
        for rec in recs:
            exp = by_id[rec.experienceId]
            if exp.actor == "ME":
                continue
            s.share_chances += 1
            s.share_asked += any(
                c.experienceId == exp.id and SHARE.search(c.question) for c in clars
            )
    return s


def strip_supplements(experiences: list[dict]) -> list[dict]:
    return [{**e, "supplements": []} for e in experiences]


def nosupp_questions(experiences: list[dict], questions: list[dict]) -> list[dict]:
    """보완이 실린 경험이 정답(good · ok)인 문항 — 보완을 빼고 한 번 더 돌린다."""
    with_supp = {e["id"] for e in experiences if e["supplements"]}
    return [q for q in questions if with_supp & set(q["good"] + q["ok"])]


def load_data(only: list[str] | None) -> tuple[list[dict], list[dict]]:
    experiences = json.loads((DATA / "experiences.json").read_text(encoding="utf-8"))
    questions = json.loads((DATA / "questions.json").read_text(encoding="utf-8"))
    if only:
        questions = [q for q in questions if q["id"] in only]
    return experiences, questions


def load_types(only: list[str] | None) -> list[dict]:
    questions = json.loads((DATA / "question-types.json").read_text(encoding="utf-8"))
    return [q for q in questions if q["id"] in only] if only else questions


def load_known() -> list[dict]:
    return json.loads((DATA / "clarify.json").read_text(encoding="utf-8"))["known"]


# ---- 실행 ----


def _slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


async def run(
    models: list[str],
    prompts: list[str],
    variants: list[str],
    experiences: list[dict],
    questions: list[dict],
    type_questions: list[dict],
    out: Path,
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

    # 모델마다 클라이언트를 한 번만 만든다(eval_route 와 같은 이유 — 호출마다 만들면 생성이 이벤트 루프를 막아
    # 먼저 시작한 호출의 시간이 부푼다, AI#37)
    clients = {}
    for model in models:
        settings = replace(base, llm_models=parse_models(model))
        clients[model] = (settings, build_llm(settings))

    async def one(model: str, prompt: str, variant: str, q: dict) -> None:
        settings, llm = clients[model]
        exps = strip_supplements(experiences) if variant == "without" else experiences
        req = MatchRequest.model_validate(
            {
                "question": {"organization": q["organization"], "text": q["text"]},
                "experiences": exps,
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
        name = prompt + {"with": "", "without": NOSUPP, "types": TYPES}[variant]
        path = out / _slug(model) / name / f"{q['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"  {model} · {name} · {q['id']} — {record.get('error') or 'ok'} ({record['seconds']}초)"
        )

    jobs = [("with", q) for q in questions] if "with" in variants else []
    if "without" in variants:
        jobs += [("without", q) for q in nosupp_questions(experiences, questions)]
    if "types" in variants:
        jobs += [("types", q) for q in type_questions]
    await asyncio.gather(*(one(m, p, v, q) for m in models for p in prompts for v, q in jobs))


def report(
    out: Path, experiences: list[dict], questions: list[dict], type_questions: list[dict]
) -> str:
    known = load_known()
    rows, clar_rows, type_rows, count_rows, details = [], [], [], [], []
    for model_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        for prompt_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            stripped = prompt_dir.name.endswith(NOSUPP)
            exps = [
                MatchExperience.model_validate(e)
                for e in (strip_supplements(experiences) if stripped else experiences)
            ]
            only_type = prompt_dir.name.endswith(TYPES)
            t = Score()
            for q in type_questions if only_type else questions:
                path = prompt_dir / f"{q['id']}.json"
                if not path.exists():
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                t.seconds += record.get("seconds", 0.0)
                if "error" in record:
                    t.questions += 1
                    t.errors.append(f"{q['id']}: {record['error']}")
                    continue
                data = MatchResponse.model_validate(record["response"])
                if only_type:
                    t.add(score_type(q, data))
                else:
                    t.add(score(q, exps, data, known, stripped))
            if not t.questions:
                continue
            name = f"{model_dir.name} · {prompt_dir.name}"
            if t.type_total or only_type:
                type_rows.append(
                    f"| {name} | {t.type_right}/{t.type_total} | {t.type_ok}/{t.type_total} "
                    f"| {len(t.errors)}/{t.questions} |"
                )
            if only_type:
                lines = [
                    *(f"- 유형 다름 {x}" for x in t.type_wrong),
                    *(f"- 오류 {x}" for x in t.errors),
                ]
                details.append(f"\n### {name}\n\n" + "\n".join(lines))
                continue
            answered = t.questions - len(t.errors)
            gap = f"{t.gap_asked}/{t.gap_chances}" if stripped else "-"
            clar_rows.append(
                f"| {name} | {len(t.clarifications)}({t.clar_empty}/{answered}) "
                f"| {len(t.repeats)}/{t.repeat_chances} | {gap} "
                f"| {len(t.leading)} | {len(t.vague)} | {len(t.compound)} "
                f"| {t.share_asked}/{t.share_chances} "
                f"| {t.ask_mine} · {t.ask_why} · {t.ask_check} · {len(t.ask_other)} "
                f"| {len(t.errors)}/{t.questions} |"
            )
            rows.append(
                f"| {name} | {t.picked_right}/{t.picked_total} | {len(t.wrong_picks)} "
                f"| {len(t.forced)} | {t.req_right}/{t.req_total} | {len(t.made_up_numbers)} "
                f"| {len(t.team_as_mine)} | {t.supplement_used}/{t.supplement_recs} "
                f"| {len(t.errors)}/{t.questions} | {t.seconds:.0f}초 |"
            )
            count_rows.append(
                f"| {name} | {t.rec_count} | {t.activity_count} | {len(t.unlabeled)} "
                f"| {len(t.whole_activity)} |"
            )
            lines = [
                *(f"- 정답 밖(후보) {x}" for x in t.unlabeled),
                *(f"- 활동 통째(후보) {x}" for x in t.whole_activity),
                *(f"- 엉뚱한 추천 {x}" for x in t.wrong_picks),
                *(f"- 억지 추천 {x}" for x in t.forced),
                *(f"- 요구사항 판정 틀림 {x}" for x in t.req_wrong),
                *(f"- 지어낸 수치 {x}" for x in t.made_up_numbers),
                *(f"- 팀 일을 내 일로(후보) {x}" for x in t.team_as_mine),
                *(f"- 오류 {x}" for x in t.errors),
                *(f"- 유형 다름 {x}" for x in t.type_wrong),
                *(f"- 이미 있는 사실 다시 물음(후보) {x}" for x in t.repeats),
                *(f"- 수치 · 답 유도(후보) {x}" for x in t.leading),
                *(f"- 막연한 질문(후보) {x}" for x in t.vague),
                *(f"- 갈래 그 밖(후보) {x}" for x in t.ask_other),
                *(f"- 고른 것 {x}" for x in t.picks),
                *(f"- 되묻기 {x}" for x in t.clarifications),
            ]
            details.append(f"\n### {name}\n\n" + "\n".join(lines))
    head = (
        "| 모델 · 프롬프트 | 첫 추천 맞음 | 엉뚱한 추천 | 억지 추천 | 요구사항 판정 | 지어낸 수치 "
        "| 팀 일을 내 일로(후보) | 보완 씀 | 오류 | 시간 합 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    clar_head = (
        "| 모델 · 프롬프트 | 되묻기 수(0개 문항/답한 문항) | 이미 있는 사실 다시 물음(후보) "
        "| 보완 없을 때 그 빈칸 물음 | 수치 · 답 유도(후보) | 막연한 질문(후보) "
        "| 한 질문에 여러 개 | 팀 · 모름 경험에 내 몫 물음 "
        "| 갈래 내 몫 · 판단/원인/입장 · 확인/결과 · 그 밖(후보) | 오류 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    type_head = "| 모델 · 프롬프트 | 문항 유형 정답 | 허용 포함 | 오류 |\n| --- | --- | --- | --- |"
    count_head = (
        "| 모델 · 프롬프트 | 추천 수 | 활동 수 | 정답 밖(후보) | 활동 통째(후보) |\n"
        "| --- | --- | --- | --- | --- |"
    )
    tables = "\n\n".join(
        "\n".join(t)
        for t in (
            [head, *rows],
            [count_head, *count_rows],
            [clar_head, *clar_rows],
            [type_head, *type_rows],
        )
        if len(t) > 1
    )
    return tables + "\n" + "".join(details) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="쉼표로 「제공사:모델」 — 기본은 MATCH_LLM_MODELS 첫 모델")
    parser.add_argument("--prompts", default="tuned,generic", help="tuned · generic")
    parser.add_argument(
        "--variants",
        default="with,without,types",
        help="with(자료 그대로) · without(보완을 뺀 자료) · types(유형만 보는 문항)",
    )
    parser.add_argument("--questions", help="쉼표로 문항 id(유형 문항 포함) — 기본은 전부")
    parser.add_argument("--out", type=Path, help="응답 저장 폴더")
    parser.add_argument("--rescore", type=Path, help="저장한 응답만 다시 채점(LLM 안 부름)")
    parser.add_argument("--env", type=Path, default=ROOT / ".env", help="키를 읽을 .env")
    args = parser.parse_args()

    only = args.questions.split(",") if args.questions else None
    experiences, questions = load_data(only)
    type_questions = load_types(only)
    if args.rescore:
        out = args.rescore
    else:
        from dotenv import load_dotenv

        load_dotenv(args.env)  # 이미 읽은 값이 이긴다 — 아래 ROOT/.env 는 덮지 않는다
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
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
        for v in variants:
            if v not in ("with", "without", "types"):
                parser.error(f"--variants 는 with · without · types 중 — {v!r}")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = args.out or ROOT / "eval-runs" / f"match-{stamp}"
        n = len(questions) * ("with" in variants)
        n += len(nosupp_questions(experiences, questions)) * ("without" in variants)
        n += len(type_questions) * ("types" in variants)
        calls = n * len(models) * len(prompts)
        print(f"LLM 호출 {calls}회(규칙 위반 시 더) → {out}")
        asyncio.run(run(models, prompts, variants, experiences, questions, type_questions, out))

    text = report(out, experiences, questions, type_questions)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
