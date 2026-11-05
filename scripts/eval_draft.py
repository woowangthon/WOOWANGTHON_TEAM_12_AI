"""시험 문항 · 고른 경험으로 답변 초안을 채점한다 — 프롬프트 · 모델을 바꿀 때 같은 자료 · 같은 숫자로 견준다(AI#26).

    uv run python scripts/eval_draft.py                    # MATCH_LLM_MODELS 첫 모델로, 우리 프롬프트 · 범용 프롬프트
    uv run python scripts/eval_draft.py --models openai:gpt-6-luna --prompts tuned
    uv run python scripts/eval_draft.py --rescore eval-runs/draft-<시각>   # 저장한 응답만 다시 채점(LLM 안 부름)

LLM 호출 수 = 쌍 수 × 모델 수 × 프롬프트 수 — 기본 14 × 1 × 2 = 28회(규칙 위반이면 쌍마다 최대 LLM_MAX_ATTEMPTS 번).
응답은 --out 폴더(기본 eval-runs/draft-<시각>, 커밋하지 않음)에 저장한다. 모델 폴백 없이 고른 모델만 부른다.

자료: samples/match/experiences.json · samples/draft/experiences.json(지어낸 경험)과
samples/draft/pairs.json(문항 + 고른 경험 + 손으로 적은 정답).

범용 프롬프트는 같은 모델 · 같은 경험 목록 · 같은 출력 모양에 지시만 한 줄이다. **코드 검사를 끄고** 돌린다 — 검사가
다시 부르면 지어낸 것이 숫자에서 사라지기 때문이다. 우리 쪽은 제품과 같이 검사를 켠다.

숫자의 뜻(정규식 · 낱말로 거른 것은 전부 후보 — 문장 전부를 사람이 읽고 판단한다. 문장 · 근거를 나란히 찍는다):
- 문장: 문장 수 합
- 다시 부름: 코드 검사에 걸려 LLM 을 한 번 더 부른 쌍 수 · 오류: 끝까지 어겨 LLM_INVALID_OUTPUT 등으로 끝난 쌍 수
- 없는 ID: 요청의 근거 · 보완 · 입력한 본문이 아닌 것을 붙인 문장 수
- 없는 숫자: 문장 속 숫자가 붙인 글에 없는 문장 수(코드 검사와 같은 기준 — 우리 쪽은 0 이어야 한다)
- 지어냄(후보): 그 문항에 필요한데 자료에 없는 사실(missing)을 문장이 쓴 것 같은 수
- 팀 일을 내 일로(후보): 붙인 근거가 전부 TEAM · UNKNOWN 경험인 문장에 「제가 · 직접 · 주도 …」가 있거나, TEAM 근거인데
  「팀 · 회장 · 팀원」 같은 주어가 없는 수 — 주어 없는 문장은 내가 한 일로 읽힌다
- 모름에 주체(후보): 근거가 전부 UNKNOWN 경험인 문장이 「팀은」처럼 누가 했는지 말하거나 「고쳤습니다」처럼 사람의
  행동을 주어 없이 쓴 수(AI#44)
- omitted 씀: 문항이 대놓고 묻는데 자료에 없는 사실(mustOmit)을 omitted 에 적은 수 / 기회
- 맞는 근거: 문장이 그 사실을 쓸 때 맞는 근거 · 보완을 붙인 수 / 쓴 수(expect)
- 고른 경험 안 씀: 사용자가 골랐는데 어느 문장도 그 경험을 붙이지 않은 수 / 고른 경험 수
- 글자(쌍 평균): 문장을 이어 붙인 글자 수 평균. 길수록 좋다는 뜻이 아니다 — 재료를 얼마나 썼는지 본다
- 구성 순서: beats 의 사실이 전부 나오고 적은 순서대로인 쌍 수 / beats 가 있는 쌍 수
- 되풀이(후보): 문장 둘이 공백을 뺀 8글자 이상을 같이 쓰는 수 + once 의 사실이 두 문장 이상에 나온 수 — 사람이 읽는다
"""

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.draft import SYSTEM_PROMPT, backings, missing_numbers  # noqa: E402
from app.schemas import LLMDraftOutput, MatchExperience  # noqa: E402
from scripts.eval_match import MINE_WORDS, _slug  # noqa: E402

DATA = ROOT / "samples" / "draft"
EXPERIENCES = ROOT / "samples" / "match" / "experiences.json"

GENERIC_PROMPT = "이 경험으로 이 문항 답을 써 줘."
PROMPTS = {"tuned": SYSTEM_PROMPT, "generic": GENERIC_PROMPT}

# TEAM · UNKNOWN 문장에 있어야 할 주어(후보 — 사람이 본다)
TEAM_SUBJECT = re.compile(r"팀|회장|우리|다 ?같이|함께")
# UNKNOWN 문장이 누가 했는지 말하는지(후보 — 사람이 본다)
WHO = re.compile(r"팀|회장|우리")
ACTED = re.compile(r"고쳤|수정했|해결했|바꿨|찾았|만들었")


@dataclass
class Score:
    pairs: int = 0
    errors: list[str] = field(default_factory=list)
    retried: int = 0
    seconds: float = 0.0
    sentences: list[str] = field(default_factory=list)
    fake_ids: list[str] = field(default_factory=list)
    bad_numbers: list[str] = field(default_factory=list)
    invented: list[str] = field(default_factory=list)
    team_as_mine: list[str] = field(default_factory=list)
    unknown_actor: list[str] = field(default_factory=list)
    omit_hit: int = 0
    omit_chances: int = 0
    omit_miss: list[str] = field(default_factory=list)
    expect_hit: int = 0
    expect_chances: int = 0
    expect_miss: list[str] = field(default_factory=list)
    omitted: list[str] = field(default_factory=list)
    picked_total: int = 0
    unused: list[str] = field(default_factory=list)
    chars: int = 0
    order_hit: int = 0
    order_total: int = 0
    order_miss: list[str] = field(default_factory=list)
    repeated: list[str] = field(default_factory=list)

    def add(self, other: "Score") -> None:
        for name, value in vars(other).items():
            setattr(self, name, getattr(self, name) + value)


def shared_run(a: str, b: str, size: int = 8) -> str | None:
    """두 문장이 공백을 빼고 같이 쓰는 size 글자 토막. 없으면 None."""
    a, b = a.replace(" ", ""), b.replace(" ", "")
    return next((a[i : i + size] for i in range(len(a) - size + 1) if a[i : i + size] in b), None)


def score(pair: dict, experiences: list[MatchExperience], data: LLMDraftOutput) -> Score:
    pid = pair["id"]
    s = Score()
    known = backings(experiences)
    s.omitted = [f"{pid}: 「{x}」" for x in data.omitted]
    used = {
        known[e.type, e.id].experience.id
        for sent in data.sentences
        for e in sent.evidence
        if (e.type, e.id) in known
    }
    s.picked_total = len(experiences)
    s.unused = [f"{pid}: 경험 {e.id}" for e in experiences if e.id not in used]
    for sent in data.sentences:
        text = sent.text
        pairs = [(e.type, e.id) for e in sent.evidence]
        cited = [known[p] for p in pairs if p in known]
        shown = " / ".join(f"{p[0][0]}{p[1]} {known[p].text}" for p in pairs if p in known)
        line = f"{pid}: 「{text}」 ← {shown or '(없음)'}"
        s.sentences.append(line)
        fake = [p for p in pairs if p not in known]
        if fake:
            s.fake_ids.append(f"{pid}: {fake} 「{text}」")
        numbers = missing_numbers(text, cited)
        if numbers:
            s.bad_numbers.append(f"{pid}: {numbers} {line}")
        s.invented += [
            f"{pid}: {m['fact']} — {line}" for m in pair["missing"] if re.search(m["pattern"], text)
        ]
        actors = {b.experience.actor for b in cited}
        if actors and "ME" not in actors:
            words = MINE_WORDS.findall(text)
            if words or ("TEAM" in actors and not TEAM_SUBJECT.search(text)):
                s.team_as_mine.append(f"{pid}: {words or '주어 없음'} {line}")
        if actors == {"UNKNOWN"} and (WHO.search(text) or ACTED.search(text)):
            s.unknown_actor.append(line)
        for want in pair["expect"]:
            if re.search(want["pattern"], text):
                s.expect_chances += 1
                if {tuple(e) for e in want["evidence"]} & set(pairs):
                    s.expect_hit += 1
                else:
                    s.expect_miss.append(f"{pid}: {want['evidence']} 아님 — {line}")
    s.chars = sum(len(x.text) for x in data.sentences)
    # 문장 쌍마다 한 번만 센다 — 토막과 once 에 같이 걸려도 되풀이 하나다
    texts = [x.text for x in data.sentences]
    twice: dict[tuple[int, int], str] = {}
    for fact in pair.get("once", []):
        hits = [i for i, text in enumerate(texts) if re.search(fact, text)]
        twice.update(dict.fromkeys(combinations(hits, 2), fact))
    for i, j in combinations(range(len(texts)), 2):
        run = shared_run(texts[i], texts[j])
        if run:
            twice.setdefault((i, j), run)
    s.repeated = [
        f"{pid}: 「{v}」 — 「{texts[i]}」 / 「{texts[j]}」" for (i, j), v in twice.items()
    ]
    if pair.get("beats"):
        s.order_total = 1
        spots = [
            next((i for i, x in enumerate(data.sentences) if re.search(b, x.text)), None)
            for b in pair["beats"]
        ]
        if None not in spots and spots == sorted(spots):
            s.order_hit = 1
        else:
            s.order_miss.append(f"{pid}: 문장 위치 {spots}")
    for m in pair["missing"]:
        if m.get("mustOmit"):
            s.omit_chances += 1
            if any(re.search(m["omitPattern"], x) for x in data.omitted):
                s.omit_hit += 1
            else:
                s.omit_miss.append(f"{pid}: {m['fact']}")
    return s


def load_experiences() -> list[dict]:
    extra = (DATA / "experiences.json").read_text(encoding="utf-8")
    return json.loads(EXPERIENCES.read_text(encoding="utf-8")) + json.loads(extra)


def load_pairs(only: list[str] | None = None) -> list[dict]:
    pairs = json.loads((DATA / "pairs.json").read_text(encoding="utf-8"))
    return [p for p in pairs if p["id"] in only] if only else pairs


def picked(pair: dict, experiences: list[dict]) -> list[dict]:
    """고른 순서대로 — /draft 요청의 experiences."""
    by_id = {e["id"]: e for e in experiences}
    return [by_id[i] for i in pair["experienceIds"]]


# ---- 실행 ----


async def run(
    models: list[str], prompts: list[str], experiences: list[dict], pairs: list[dict], out: Path
) -> None:
    from dotenv import load_dotenv

    from app.config import load_settings, parse_models
    from app.draft import LLM_JSON_SCHEMA, check_output, check_request
    from app.errors import ApiError
    from app.llm import build_llm, generate_valid
    from app.match import build_user_message
    from app.schemas import DraftRequest

    load_dotenv(ROOT / ".env")  # 값은 화면에 찍지 않는다
    base = load_settings()
    gate = asyncio.Semaphore(4)
    clients = {}  # 모델마다 클라이언트를 한 번만 만든다(AI#37)
    for model in models:
        settings = replace(base, llm_models=parse_models(model))
        clients[model] = (settings, build_llm(settings))

    async def one(model: str, prompt: str, pair: dict) -> None:
        settings, llm = clients[model]
        req = DraftRequest.model_validate(
            {
                "question": {"organization": pair["organization"], "text": pair["text"]},
                "experiences": picked(pair, experiences),
            }
        )
        question, exps = check_request(req, settings)

        def parse(text: str) -> LLMDraftOutput:
            output = LLMDraftOutput.model_validate_json(text)
            return check_output(output, exps) if prompt == "tuned" else output

        async with gate:
            started = time.monotonic()
            try:
                got = await generate_valid(
                    llm,
                    PROMPTS[prompt],
                    build_user_message(question, exps),
                    LLM_JSON_SCHEMA,
                    parse,
                    settings,
                )
                record = {"response": json.loads(got.value.model_dump_json())}
                record["attempt"] = got.attempt
            except ApiError as e:
                record = {"error": f"{e.code}: {e.message}"}
            record["seconds"] = round(time.monotonic() - started, 1)
        path = out / _slug(model) / prompt / f"{pair['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        status = record.get("error") or "ok"
        print(f"  {model} · {prompt} · {pair['id']} — {status} ({record['seconds']}초)")

    await asyncio.gather(*(one(m, p, q) for m in models for p in prompts for q in pairs))


def report(out: Path, experiences: list[dict], pairs: list[dict]) -> str:
    rows, details = [], []
    for model_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        for prompt_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            t = Score()
            for pair in pairs:
                path = prompt_dir / f"{pair['id']}.json"
                if not path.exists():
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                t.pairs += 1
                t.seconds += record.get("seconds", 0.0)
                if "error" in record:
                    t.errors.append(f"{pair['id']}: {record['error']}")
                    continue
                t.retried += record.get("attempt", 1) > 1
                exps = [MatchExperience.model_validate(e) for e in picked(pair, experiences)]
                t.add(score(pair, exps, LLMDraftOutput.model_validate(record["response"])))
            if not t.pairs:
                continue
            name = f"{model_dir.name} · {prompt_dir.name}"
            rows.append(
                f"| {name} | {len(t.sentences)} | {t.retried} | {len(t.errors)}/{t.pairs} "
                f"| {len(t.fake_ids)} | {len(t.bad_numbers)} | {len(t.invented)} "
                f"| {len(t.team_as_mine)} | {len(t.unknown_actor)} | {t.omit_hit}/{t.omit_chances} "
                f"| {t.expect_hit}/{t.expect_chances} "
                f"| {len(t.unused)}/{t.picked_total} "
                f"| {t.chars // max(t.pairs - len(t.errors), 1)} "
                f"| {t.order_hit}/{t.order_total} | {len(t.repeated)} | {t.seconds:.0f}초 |"
            )
            lines = [
                *(f"- 오류 {x}" for x in t.errors),
                *(f"- 없는 ID {x}" for x in t.fake_ids),
                *(f"- 없는 숫자 {x}" for x in t.bad_numbers),
                *(f"- 지어냄(후보) {x}" for x in t.invented),
                *(f"- 팀 일을 내 일로(후보) {x}" for x in t.team_as_mine),
                *(f"- 모름에 주체(후보) {x}" for x in t.unknown_actor),
                *(f"- omitted 안 씀 {x}" for x in t.omit_miss),
                *(f"- 근거 다름 {x}" for x in t.expect_miss),
                *(f"- 고른 경험 안 씀 {x}" for x in t.unused),
                *(f"- 구성 순서 어긋남 {x}" for x in t.order_miss),
                *(f"- 되풀이(후보) {x}" for x in t.repeated),
                *(f"- 문장 {x}" for x in t.sentences),
                *(f"- omitted {x}" for x in t.omitted),
            ]
            details.append(f"\n### {name}\n\n" + "\n".join(lines))
    head = (
        "| 모델 · 프롬프트 | 문장 | 다시 부름 | 오류 | 없는 ID | 없는 숫자 | 지어냄(후보) "
        "| 팀 일을 내 일로(후보) | 모름에 주체(후보) | omitted 씀 | 맞는 근거 | 고른 경험 안 씀 | 글자(쌍 평균) | 구성 순서 | 시간 합 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    return "\n".join([head, *rows]) + "\n" + "".join(details) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="쉼표로 「제공사:모델」 — 기본은 MATCH_LLM_MODELS 첫 모델")
    parser.add_argument("--prompts", default="tuned,generic", help="tuned · generic")
    parser.add_argument("--pairs", help="쉼표로 쌍 id — 기본은 전부")
    parser.add_argument("--out", type=Path, help="응답 저장 폴더")
    parser.add_argument("--rescore", type=Path, help="저장한 응답만 다시 채점(LLM 안 부름)")
    args = parser.parse_args()

    experiences = load_experiences()
    pairs = load_pairs(args.pairs.split(",") if args.pairs else None)
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
        out = args.out or ROOT / "eval-runs" / f"draft-{stamp}"
        print(f"LLM 호출 {len(pairs) * len(models) * len(prompts)}회(규칙 위반 시 더) → {out}")
        asyncio.run(run(models, prompts, experiences, pairs, out))

    text = report(out, experiences, pairs)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
