"""시험 문항 · 고른 경험으로 답변 초안을 채점한다 — 프롬프트 · 모델을 바꿀 때 같은 자료 · 같은 숫자로 견준다(AI#26).

    uv run python scripts/eval_draft.py                    # MATCH_LLM_MODELS 첫 모델로, 우리 프롬프트 · 범용 프롬프트
    uv run python scripts/eval_draft.py --models openai:gpt-6-luna --prompts tuned
    uv run python scripts/eval_draft.py --prompts tuned --runs 2   # 같은 쌍을 두 번씩 — 칸에 회차별 `a · b`
    uv run python scripts/eval_draft.py --rescore eval-runs/draft-<시각>   # 저장한 응답만 다시 채점(LLM 안 부름)

LLM 호출 수 = 쌍 수 × 모델 수 × 프롬프트 수 × runs — 기본 33 × 1 × 2 × 1 = 66회(규칙 위반이면 쌍마다 최대 LLM_MAX_ATTEMPTS 번).
응답은 --out 폴더(기본 eval-runs/draft-<시각>, 커밋하지 않음)의 <모델>/<프롬프트>/run-<N>/ 에 저장한다. 모델 폴백 없이 고른 모델만 부른다.

자료: samples/match/experiences.json · samples/draft/experiences.json(지어낸 경험)과
samples/draft/pairs.json(문항 + 고른 경험 + 손으로 적은 정답). 「이미 있는 사실」은 samples/match/clarify.json 과
samples/draft/clarify.json. addSupplements 가 있는 쌍은 빠진 사실에 답한 뒤 다시 만든 요청이다 — 그 보완을 경험에 더해 부른다.
jobRole · requirements · hints 가 있는 쌍은 매칭 결과 · 직무를 선택 칸으로 같이 싣는다(ETC#141 — 근거가 아니다).

범용 프롬프트는 같은 모델 · 같은 경험 목록 · 같은 출력 모양에 지시만 한 줄이다. **코드 검사를 끄고** 돌린다 — 검사가
다시 부르면 지어낸 것이 숫자에서 사라지기 때문이다. 우리 쪽은 제품과 같이 검사를 켠다.

숫자의 뜻(정규식 · 낱말로 거른 것은 전부 후보 — 문장 전부를 사람이 읽고 판단한다. 문장 · 근거를 나란히 찍는다):
- 문장: FACT 문장 수 합
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
- 질문 붙음: 문항이 대놓고 묻는 빈칸 중 경험 · 질문이 붙은 수 / 기회(noQuestion — 경험에 붙지 않는 빈칸은 뺀다)
- 억지 질문: 경험에 붙지 않는 빈칸(noQuestion)에 경험 · 질문을 붙인 수
- 질문: omitted 에 붙은 질문 수(omitted 전체 수)
- 짝 위반: 질문 · 경험의 null 짝이 틀렸거나 고르지 않은 경험을 붙인 수(코드 검사와 같은 기준 — 우리 쪽은 0 이어야 한다)
- 수치 유도(후보): 질문 · 빈칸 구절에 그 경험 · 문항에 없는 숫자가 있거나 「…죠?」처럼 확인만 받는 어미인 수
- 막연함(후보): 「더 자세히 알려 주세요」처럼 아무 경험에나 붙는 질문 수
- 두 가지(후보): 한 질문에 물음표가 둘 이상인 수
- 있는 사실 물음(후보): 그 경험의 근거 · 보완에 이미 있는 사실을 묻는 질문 수 / 그런 사실이 있는 경험을 고른 기회
- 보완 씀: 다시 만들 때 더한 보완을 붙이고 그 사실을 쓴 문장이 있는 수 / 더한 보완 수
- 보완 또 물음(후보): 더한 보완이 답한 사실을 omitted 에 다시 적은 수
- 구성 순서: beats 의 사실이 전부 나오고 적은 순서대로인 쌍 수 / beats 가 있는 쌍 수
- 되풀이(후보): 문장 둘이 공백을 뺀 8글자 이상을 같이 쓰는 수 + once 의 사실이 두 문장 이상에 나온 수 — 사람이 읽는다
- 생각 문장: OPINION 수(문장 수 합과 따로 센다)
- 종류 위반: FACT 근거 0개 · OPINION 근거 · OPINION 숫자 · 초안당 4개 이상 · 생각만 있는 초안(코드 검사와 같은 기준 — 우리 쪽은 0 이어야 한다)
- 사실로 뺌(후보): OPINION 이 「~했습니다」처럼 일어난 일로 끝나는 수 — 받칠 수 있으면 FACT 로, 없으면 omitted 로 갔어야 한다
- 없는 사실(후보): 그 문항의 자료에 없는 사실(missing)을 OPINION 이 쓴 것 같은 수 — 가장 큰 위험, 근거 규칙이 새는 곳
- 감정 · 칭찬 · 단정(후보): OPINION 에 지어낸 감정 · 칭찬 · 의미 단정 낱말이 있는 수 — 낱말로 거른 것이라 문장을 읽는다
- 소제목(S-86): 소제목이 있는 쌍 수. 쌍마다 찍는다 — 그 답이 보여 주는 것인지 사람이 읽는다
- 소제목 위반: 문장이 있는데 없음 · 문장 0개인데 있음 · 40자 넘음 · 숫자(코드 검사와 같은 기준 — 우리 쪽은 0 이어야 한다)
- 소제목 지어냄(후보): 그 문항의 자료에 없는 사실(missing)을 소제목이 쓴 것 같은 수
- 소제목 칭찬(후보): 소제목에 감정 · 칭찬 · 의미 단정 낱말(생각 문장과 같은 목록)이 있는 수
- 다짐 첫 문장 · 다짐 맺음(후보, AI#97): 첫 문장 · 끝 문장이 생각 문장이고 「기여하 · 이어가겠 · 살피겠 …」 다짐 낱말이 든 쌍 수.
  평가자는 무슨 역량인지 안 보이는 다짐으로 열고 닫는 글을 형식적으로 본다. 지원 동기 · 포부 문항이면 자연스러울 수 있다
- 분류명 소제목: 「~ 경험」으로 끝나는 소제목 수(「기능 구현과 장애 대응 경험」 — 이 답의 문제 · 변화가 안 보인다, AI#97)
- 강조 앞: 강조점(hints)의 받칠 수 있는 사실(lead)이 FACT 앞 leadWithin(기본 2)문장 안에 나온 쌍 수 / lead 가 있는 쌍 수.
  같은 경험 151 을 다른 문항 · 강조점으로 부른 쌍(rich-problem · emph-*)의 문장을 나란히 읽어 앞세운 사실이 달라졌는지 본다
- 강조 쌍 지어냄(후보): hints 가 있는 쌍의 지어냄 + 없는 사실 — 강조점 · 직무 때문에 없는 사실이 늘었나(강조점에만 있는 판단 ·
  숫자 · 직무 경력을 missing 에 적어 둔다)
- 지난 빈칸 또 물음(후보): 매칭 때 gaps 에 있었지만 지금 보완이 채운 사실(staleGaps)을 omitted 에 다시 적은 수

글 짜임 · AI 보탬 · 빈칸 · 목표 글자 수(S-75 · S-77 · S-78, AI#81) — contextExperienceIds · targetLength 가 있는 쌍은 같이 싣는다:
- 나열: 문장(FACT · ELABORATED)을 받은 경험이 둘 이상인데 경험마다 한 문장씩뿐인 쌍 수 / 경험(고른 것 + 보조 재료)이 둘 이상인 쌍 수
- 대표 깊이: 첫째로 고른 경험을 받친 문장이 둘 이상인 쌍 수 / 받칠 것이 있는 쌍 수
- 첫 문장: 쌍마다 첫 문장을 찍는다 — 주제문인지 사람이 읽는다
- AI 보탬: ELABORATED 수. 보탬 > 사실: ELABORATED 가 FACT 보다 많은 쌍 수
- 보탬 지어냄(후보): 그 문항의 자료에 없는 사실(missing)을 ELABORATED 가 쓴 것 같은 수(지어냄(후보)에도 들어 있다 — 따로 본다)
- 보탬 판단 · 감정(후보): ELABORATED 에 「보았 · 생각했 · 판단했 · 마음먹 …」 같은 사용자의 판단 · 생각이나 감정 · 칭찬 낱말이 있는 수
- 모름에 「제가」: 근거가 전부 UNKNOWN 경험인데 「제가」로 쓴 문장 — ELABORATED 면 허용(규칙 5), FACT 면 팀 일을 내 일로(후보)에 센다
- 빈칸: 문장 속 자리 표시 수
- 목표 맞음: 글자 수가 목표의 80 ~ 100% 인 쌍 수 / targetLength 가 있는 쌍 수. 넘음 · 모자람은 비율을 찍는다
- 보조 재료 씀: 보조 재료 경험을 받친 문장이 있는 쌍 수 / 보조 재료가 있는 쌍 수. 보조가 뼈대: 보조 재료만 받친 문장이 고른 경험을 받친 문장보다 많은 쌍 수

AI 가정(SUPPOSED · ADR-014, AI#84) — 가정 문장은 근거 규칙(지어냄 · 없는 숫자 · 맞는 근거)에서 빼고 따로 센다.
나열 · 대표 깊이에서 가정 문장은 근거 · 짝 omitted 의 경험에 속한다:
- 수동형(후보): FACT · ELABORATED · SUPPOSED 중 「…됐습니다 · 되었습니다 · 졌습니다」로 사람을 감춘 문장 수(근거가 전부 UNKNOWN 이면 뺀다)
- 가정: SUPPOSED 수(specific 수). 가정 > 사실: SUPPOSED 가 FACT · ELABORATED 보다 많은 쌍 수(위반 아님 — 얼마나 지어 썼나)
- specific 누락(후보): specific false 인 가정에 숫자 · 수량(코드와 같은 기준) · 수상 · 직책 · 기관 낱말이 있는 수
- 가정 짝 위반: 가정 문장마다 짝 omitted 가 정확히 하나가 아닌 수(코드 검사와 같은 기준 — 우리 쪽은 0)
- 가정이 사실 덮음(후보): 자료가 받치는 사실(expect)을 SUPPOSED 로 쓴 수 — FACT 로 썼어야 한다
- 가정 팀 「제가」(후보): 근거 · 짝 경험이 전부 TEAM 인 가정에 「제가 · 직접 · 주도 …」가 있는 수
- 지원처 가정(후보): 가정에 지원처 이름이나 「귀사 · 지원처 · 회사의」가 있는 수
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

from app.citations import numbers  # noqa: E402
from app.draft import MAX_OPINIONS, SYSTEM_PROMPT, backings, missing_numbers  # noqa: E402
from app.schemas import TITLE_MAX, LLMDraftOutput, MatchExperience  # noqa: E402
from scripts.eval_match import (  # noqa: E402
    LEADING,
    MINE_WORDS,
    VAGUE,
    _experience_text,
    _has,
    _ints,
    _slug,
    load_known,
)
from scripts.eval_runs import merge_runs, run_dirs, run_path  # noqa: E402

DATA = ROOT / "samples" / "draft"
EXPERIENCES = ROOT / "samples" / "match" / "experiences.json"

GENERIC_PROMPT = "이 경험으로 이 문항 답을 써 줘."
PROMPTS = {"tuned": SYSTEM_PROMPT, "generic": GENERIC_PROMPT}

# TEAM · UNKNOWN 문장에 있어야 할 주어(후보 — 사람이 본다)
TEAM_SUBJECT = re.compile(r"팀|회장|우리|다 ?같이|함께")
# UNKNOWN 문장이 누가 했는지 말하는지(후보 — 사람이 본다)
WHO = re.compile(r"팀|회장|우리")
ACTED = re.compile(r"고쳤|수정했|해결했|바꿨|찾았|만들었")
# 사용자가 보완의 답에 자기가 했다고 말한 것 — 이 보완을 붙인 문장은 「제가」로 써도 된다
SAID_MINE = re.compile(r"(?<![가-힣])(?:내가|제가|나는|저는)")
# 생각 문장 후보(사람이 본다) — 일어난 일로 끝남 · 지어낸 감정 · 칭찬 · 의미 단정
ENDING = re.compile(r"([가-힣])습니다")
FLATTERY = re.compile(
    r"뿌듯|기뻤|보람|즐거|즐겁|감동|설레|행복|벅차|느꼈"
    r"|훌륭|뛰어난|뛰어납|탁월|적임|최고|값진|소중한|밑거름|증명|큰 의미|성장할 수 있었"
)
# 아무 지원자에게나 붙는 다짐(후보 — AI#97). 지원 동기 · 포부 문항에선 자연스러울 수 있으니 사람이 본다
VOW = re.compile(
    r"기여하|이바지|보탬이 되|활용하고 싶|발휘하고 싶|이어가겠|이어 가겠|살피겠|노력하겠|성장하겠|되겠습니다"
)
# AI 보탬이 사용자의 판단 · 생각 · 동기를 만들어 넣었나(후보 — 사람이 본다)
JUDGED = re.compile(r"보았|봤습|생각했|판단했|마음먹|결심|다짐|깨달|느꼈|여겼|싶었")
# 문장 속 빈칸 자리 표시
MARK = re.compile(r"\[[^\[\]]*\]")
# 사람을 감춘 수동형(후보 — 사람이 본다). 「…하게 됐습니다」도 걸리니 문장을 읽는다
PASSIVE = re.compile(r"(?:됐|되었|졌)습니다")
# specific 이어야 할 낱말(후보) — 숫자는 app.citations.numbers 가 본다
SPECIFIC_WORDS = re.compile(
    r"수상|대상|최우수|우수상|장려상|입상|팀장|회장|부회장|리더|대표|총무|기관|공사|재단|주식회사|㈜"
)
# 지원처 정보를 가정했나(후보) — 지원처 이름은 쌍마다 더한다
EMPLOYER = re.compile(r"귀사|지원처|회사의|기업의")


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
    omitted_total: int = 0
    asked: list[str] = field(default_factory=list)
    bad_pairs: list[str] = field(default_factory=list)
    leading: list[str] = field(default_factory=list)
    vague: list[str] = field(default_factory=list)
    compound: list[str] = field(default_factory=list)
    known_asked: list[str] = field(default_factory=list)
    known_chances: int = 0
    gap_asked: int = 0
    gap_chances: int = 0
    gap_miss: list[str] = field(default_factory=list)
    forced: list[str] = field(default_factory=list)
    fill_hit: int = 0
    fill_chances: int = 0
    fill_miss: list[str] = field(default_factory=list)
    fill_reasked: list[str] = field(default_factory=list)
    opinions: list[str] = field(default_factory=list)
    kind_broken: list[str] = field(default_factory=list)
    opinion_happened: list[str] = field(default_factory=list)
    opinion_invented: list[str] = field(default_factory=list)
    opinion_flattery: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    title_broken: list[str] = field(default_factory=list)
    title_invented: list[str] = field(default_factory=list)
    title_flattery: list[str] = field(default_factory=list)
    title_category: list[str] = field(default_factory=list)
    vow_open: list[str] = field(default_factory=list)
    vow_end: list[str] = field(default_factory=list)
    lead_hit: int = 0
    lead_chances: int = 0
    lead_miss: list[str] = field(default_factory=list)
    hinted_invented: list[str] = field(default_factory=list)
    stale_reasked: list[str] = field(default_factory=list)
    listed: list[str] = field(default_factory=list)
    list_chances: int = 0
    deep_hit: int = 0
    deep_chances: int = 0
    shallow: list[str] = field(default_factory=list)
    firsts: list[str] = field(default_factory=list)
    elaborated: list[str] = field(default_factory=list)
    elab_heavy: list[str] = field(default_factory=list)
    elab_invented: list[str] = field(default_factory=list)
    elab_judged: list[str] = field(default_factory=list)
    unknown_mine: list[str] = field(default_factory=list)
    marks: list[str] = field(default_factory=list)
    target_hit: int = 0
    target_chances: int = 0
    target_miss: list[str] = field(default_factory=list)
    context_used: int = 0
    context_chances: int = 0
    context_heavy: list[str] = field(default_factory=list)
    passive: list[str] = field(default_factory=list)
    supposed: list[str] = field(default_factory=list)
    specific: int = 0
    guess_heavy: list[str] = field(default_factory=list)
    specific_miss: list[str] = field(default_factory=list)
    guess_unpaired: list[str] = field(default_factory=list)
    guess_over_fact: list[str] = field(default_factory=list)
    guess_mine: list[str] = field(default_factory=list)
    guess_employer: list[str] = field(default_factory=list)

    def add(self, other: "Score") -> None:
        for name, value in vars(other).items():
            setattr(self, name, getattr(self, name) + value)


def shared_run(a: str, b: str, size: int = 8) -> str | None:
    """두 문장이 공백을 빼고 같이 쓰는 size 글자 토막. 없으면 None."""
    a, b = a.replace(" ", ""), b.replace(" ", "")
    return next((a[i : i + size] for i in range(len(a) - size + 1) if a[i : i + size] in b), None)


def score_omitted(
    pair: dict,
    experiences: list[MatchExperience],
    data: LLMDraftOutput,
    facts: list[dict],
) -> Score:
    """omitted 의 경험 · 질문 — S-22 규칙과 다시 만들기(S-27)."""
    pid = pair["id"]
    s = Score()
    by_id = {e.id: e for e in experiences}
    s.omitted = [
        f"{pid}: 「{x.text}」 ← 경험 {x.experienceId} 「{x.question}」" for x in data.omitted
    ]
    s.omitted_total = len(data.omitted)
    asked = [x for x in data.omitted if x.question is not None and x.experienceId in by_id]
    for x in data.omitted:
        if (x.question is None) != (x.experienceId is None) or (
            x.experienceId is not None and x.experienceId not in by_id
        ):
            s.bad_pairs.append(f"{pid}: 「{x.text}」 ← 경험 {x.experienceId} 「{x.question}」")
    for x in asked:
        exp = by_id[x.experienceId]
        line = f"{pid}: 경험 {exp.id}({exp.actor}) 「{x.question}」"
        s.asked.append(line)
        allowed = _ints(_experience_text(exp)) | _ints(pair["text"])
        # 가정의 짝은 그 문장의 값을 확인하는 질문이라 숫자를 묻는 건 유도가 아니다 — 「…죠?」만 센다
        guess_pair = x.placeholder is None and x.sentenceIndex is not None
        invented = (_ints(x.question) | _ints(x.text)) - allowed
        if (invented and not guess_pair) or LEADING.search(x.question):
            s.leading.append(line)
        if VAGUE.search(x.question):
            s.vague.append(line)
        if x.question.count("?") > 1:
            s.compound.append(line)
    for fact in facts:
        exp = by_id.get(fact["experienceId"])
        if exp is None or not _has(exp, fact):
            continue
        s.known_chances += 1
        s.known_asked += [
            f"{pid}: 경험 {exp.id} 「{x.question}」 — 이미 있음: {fact['fact']}"
            for x in asked
            if x.experienceId == exp.id and re.search(fact["pattern"], x.question)
        ]
    for m in pair["missing"]:
        hits = [x for x in data.omitted if re.search(m["omitPattern"], x.text)]
        if m.get("noQuestion"):
            s.forced += [
                f"{pid}: {m['fact']} ← 경험 {x.experienceId} 「{x.question}」"
                for x in hits
                if x.experienceId is not None
            ]
        elif m.get("mustOmit"):
            s.gap_chances += 1
            if any(x.question for x in hits):
                s.gap_asked += 1
            else:
                s.gap_miss.append(f"{pid}: {m['fact']}")
    for add in pair.get("addSupplements", []):
        s.fill_chances += 1
        if any(
            re.search(add["pattern"], x.text)
            and any((e.type, e.id) == ("SUPPLEMENT", add["id"]) for e in x.evidence)
            for x in data.sentences
        ):
            s.fill_hit += 1
        else:
            s.fill_miss.append(f"{pid}: 보완 {add['id']} 「{add['answer']}」")
        s.fill_reasked += [
            f"{pid}: 보완 {add['id']} — 「{x.text}」 「{x.question}」"
            for x in data.omitted
            if re.search(add["omitPattern"], f"{x.text} {x.question or ''}")
        ]
    return s


def score_hints(pair: dict, data: LLMDraftOutput) -> Score:
    """매칭 결과 · 직무를 실은 쌍(ETC#141) — 강조점을 앞세웠나, 보완이 채운 지난 빈칸을 또 묻나."""
    pid = pair["id"]
    s = Score()
    if pair.get("lead"):
        s.lead_chances = 1
        facts = [x.text for x in data.sentences if x.kind == "FACT"]
        spot = next((i for i, t in enumerate(facts) if re.search(pair["lead"], t)), None)
        if spot is not None and spot < pair.get("leadWithin", 2):
            s.lead_hit = 1
        else:
            s.lead_miss.append(f"{pid}: FACT 위치 {spot} — 「{pair['lead']}」")
    for gap in pair.get("staleGaps", []):
        s.stale_reasked += [
            f"{pid}: {gap['fact']} — 「{x.text}」 「{x.question}」"
            for x in data.omitted
            if re.search(gap["pattern"], f"{x.text} {x.question or ''}")
        ]
    return s


def happened(text: str) -> bool:
    """「~했습니다 · ~느꼈습니다」처럼 지난 일로 끝나는가 — 「습니다」 앞 글자의 받침이 ㅆ(있 · 겠 빼고)."""
    return any((ord(c) - 0xAC00) % 28 == 20 and c not in "있겠" for c in ENDING.findall(text))


def score_opinions(pair: dict, data: LLMDraftOutput) -> Score:
    """생각 문장(S-27) — 코드가 막는 규칙과, 근거 규칙이 OPINION 으로 새는지."""
    pid = pair["id"]
    s = Score()
    opinions = [x for x in data.sentences if x.kind == "OPINION"]
    s.kind_broken = [
        f"{pid}: {x.kind} 근거 0개 「{x.text}」"
        for x in data.sentences
        if x.kind in ("FACT", "ELABORATED") and not x.evidence
    ]
    if len(opinions) > MAX_OPINIONS:
        s.kind_broken.append(f"{pid}: OPINION {len(opinions)}개")
    if opinions and len(opinions) == len(data.sentences):
        s.kind_broken.append(f"{pid}: 생각 문장만")
    for x in opinions:
        line = f"{pid}: 「{x.text}」"
        s.opinions.append(line)
        if x.evidence:
            s.kind_broken.append(f"{pid}: OPINION 근거 「{x.text}」")
        if numbers(x.text):
            s.kind_broken.append(f"{pid}: OPINION 숫자 「{x.text}」")
        if happened(x.text):
            s.opinion_happened.append(line)
        if FLATTERY.search(x.text):
            s.opinion_flattery.append(line)
        s.opinion_invented += [
            f"{pid}: {m['fact']} — 「{x.text}」"
            for m in pair["missing"]
            if re.search(m["pattern"], x.text)
        ]
    if data.sentences:
        first, last = data.sentences[0], data.sentences[-1]
        if first.kind == "OPINION" and VOW.search(first.text):
            s.vow_open.append(f"{pid}: 「{first.text}」")
        if len(data.sentences) > 1 and last.kind == "OPINION" and VOW.search(last.text):
            s.vow_end.append(f"{pid}: 「{last.text}」")
    return s


def score_title(pair: dict, data: LLMDraftOutput) -> Score:
    """첫 줄 소제목(S-86) — 코드가 막는 규칙과, 생각 문장처럼 지어냄 · 칭찬이 새는지."""
    pid = pair["id"]
    s = Score()
    title = data.title
    if data.sentences and title is None:
        s.title_broken.append(f"{pid}: 문장이 있는데 소제목 없음")
    if not data.sentences and title is not None:
        s.title_broken.append(f"{pid}: 문장 0개인데 소제목 「{title}」")
    if title is None:
        return s
    line = f"{pid}: 「{title}」"
    s.titles.append(line)
    if len(title) > TITLE_MAX or numbers(title):
        s.title_broken.append(f"{pid}: {len(title)}자 · 숫자 {sorted(numbers(title))} {line}")
    if FLATTERY.search(title):
        s.title_flattery.append(line)
    if title.rstrip().endswith("경험"):
        s.title_category.append(line)
    s.title_invented += [
        f"{pid}: {m['fact']} — {line}" for m in pair["missing"] if re.search(m["pattern"], title)
    ]
    return s


def score_essay(
    pair: dict,
    experiences: list[MatchExperience],
    context: list[MatchExperience],
    data: LLMDraftOutput,
) -> Score:
    """글 짜임 · AI 보탬 · 빈칸 · 목표 글자 수 · 보조 재료(S-75 · S-77 · S-78)."""
    pid = pair["id"]
    s = Score()
    known = backings([*experiences, *context])
    chosen = {e.id for e in experiences}
    owners_by = sentence_owners(data, known)
    written = [x for x in data.sentences if x.kind != "OPINION"]
    per: dict[int, int] = {}
    from_context = from_chosen = 0
    for i, x in enumerate(data.sentences):
        if x.kind == "OPINION":
            continue
        owners = owners_by[i]
        for owner in owners:
            per[owner] = per.get(owner, 0) + 1
        if owners & chosen:
            from_chosen += 1
        elif owners:
            from_context += 1
    if len(experiences) + len(context) > 1:
        s.list_chances = 1
        if len(per) > 1 and max(per.values()) == 1:
            s.listed.append(f"{pid}: 경험마다 한 문장 {sorted(per)}")
    if backings(experiences[:1]):
        s.deep_chances = 1
        if per.get(experiences[0].id, 0) >= 2:
            s.deep_hit = 1
        else:
            s.shallow.append(
                f"{pid}: 경험 {experiences[0].id} 문장 {per.get(experiences[0].id, 0)}개"
            )
    if data.sentences:
        s.firsts.append(f"{pid}: [{data.sentences[0].kind}] 「{data.sentences[0].text}」")
    elaborated = [x for x in data.sentences if x.kind == "ELABORATED"]
    facts = [x for x in data.sentences if x.kind == "FACT"]
    if len(elaborated) > len(facts):
        s.elab_heavy.append(f"{pid}: ELABORATED {len(elaborated)} > FACT {len(facts)}")
    for x in elaborated:
        line = f"{pid}: 「{x.text}」 — note 「{x.note}」"
        s.elaborated.append(line)
        s.elab_invented += [
            f"{pid}: {m['fact']} — {line}"
            for m in pair["missing"]
            if re.search(m["pattern"], x.text)
        ]
        if JUDGED.search(x.text) or FLATTERY.search(x.text):
            s.elab_judged.append(line)
    for x in written:
        actors = {
            known[e.type, e.id].experience.actor for e in x.evidence if (e.type, e.id) in known
        }
        if actors == {"UNKNOWN"} and MINE_WORDS.search(x.text):
            s.unknown_mine.append(f"{pid}: [{x.kind}] 「{x.text}」 — note 「{x.note}」")
    for i, x in enumerate(data.sentences):
        s.marks += [f"{pid}: [{i}] {mark} 「{x.text}」" for mark in MARK.findall(x.text)]
    target = pair.get("targetLength")
    if target:
        s.target_chances = 1
        ratio = sum(len(x.text) for x in data.sentences) / target
        if 0.8 <= ratio <= 1.0:
            s.target_hit = 1
        else:
            s.target_miss.append(f"{pid}: 목표 {target}자의 {ratio:.0%}")
    if context:
        s.context_chances = 1
        s.context_used = int(from_context > 0 or any(per.get(e.id) for e in context))
        if from_context > from_chosen:
            s.context_heavy.append(f"{pid}: 보조만 {from_context} > 고른 경험 {from_chosen}")
    s.add(score_supposed(pair, [*experiences, *context], data, owners_by))
    return s


def sentence_owners(data: LLMDraftOutput, known: dict) -> list[set[int]]:
    """문장마다 속한 경험 ID — 근거의 경험, 가정(SUPPOSED)이면 짝 omitted 의 경험도."""
    owners = [
        {known[e.type, e.id].experience.id for e in x.evidence if (e.type, e.id) in known}
        for x in data.sentences
    ]
    for o in data.omitted:
        if (
            o.placeholder is None
            and o.sentenceIndex is not None
            and 0 <= o.sentenceIndex < len(owners)
            and data.sentences[o.sentenceIndex].kind == "SUPPOSED"
            and o.experienceId is not None
        ):
            owners[o.sentenceIndex].add(o.experienceId)
    return owners


def score_supposed(
    pair: dict,
    experiences: list[MatchExperience],
    data: LLMDraftOutput,
    owners_by: list[set[int]],
) -> Score:
    """수동형 · AI 가정(ADR-014) — 지어 쓴 문장이 근거 문장과 섞이지 않나."""
    pid = pair["id"]
    s = Score()
    by_id = {e.id: e for e in experiences}
    actors_by = [{by_id[o].actor for o in owners if o in by_id} for owners in owners_by]
    for i, x in enumerate(data.sentences):
        if x.kind != "OPINION" and PASSIVE.search(x.text) and actors_by[i] != {"UNKNOWN"}:
            s.passive.append(f"{pid}: [{x.kind}] 「{x.text}」")
    guesses = [(i, x) for i, x in enumerate(data.sentences) if x.kind == "SUPPOSED"]
    written = sum(x.kind in ("FACT", "ELABORATED") for x in data.sentences)
    if len(guesses) > written:
        s.guess_heavy.append(f"{pid}: SUPPOSED {len(guesses)} > FACT · ELABORATED {written}")
    org = (pair.get("organization") or "").split()
    employer = re.compile(
        "|".join([EMPLOYER.pattern, *(re.escape(w) for w in org[:1] if len(w) > 1)])
    )
    for i, x in guesses:
        line = f"{pid}: [{i}] 「{x.text}」 specific={x.specific}"
        s.supposed.append(line)
        s.specific += bool(x.specific)
        if not x.specific and (numbers(x.text) or SPECIFIC_WORDS.search(x.text)):
            s.specific_miss.append(line)
        mates = [o for o in data.omitted if o.placeholder is None and o.sentenceIndex == i]
        if len(mates) != 1:
            s.guess_unpaired.append(f"{line} — 짝 {len(mates)}개")
        s.guess_over_fact += [
            f"{line} — 받치는 사실 {want['evidence']}"
            for want in pair["expect"]
            if re.search(want["pattern"], x.text)
        ]
        if actors_by[i] == {"TEAM"} and MINE_WORDS.search(x.text):
            s.guess_mine.append(line)
        if employer.search(x.text):
            s.guess_employer.append(line)
    return s


def score(
    pair: dict,
    experiences: list[MatchExperience],
    data: LLMDraftOutput,
    facts: list[dict] = (),
    context: list[MatchExperience] = (),
) -> Score:
    pid = pair["id"]
    context = list(context)
    s = score_omitted(pair, [*experiences, *context], data, list(facts))
    s.add(score_opinions(pair, data))
    s.add(score_title(pair, data))
    s.add(score_essay(pair, experiences, context, data))
    known = backings([*experiences, *context])
    used = {
        known[e.type, e.id].experience.id
        for sent in data.sentences
        for e in sent.evidence
        if (e.type, e.id) in known
    }
    s.picked_total = len(experiences)
    s.unused = [f"{pid}: 경험 {e.id}" for e in experiences if e.id not in used]
    for sent in data.sentences:
        # 근거 규칙 밖 — OPINION 은 score_opinions, SUPPOSED 는 score_supposed 가 따로 본다
        if sent.kind in ("OPINION", "SUPPOSED"):
            continue
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
        # 사용자가 보완에 자기가 했다고 말한 문장은 내 일이 맞다(프롬프트 규칙 5 의 예외)
        actors = {
            "ME"
            if p[0] == "SUPPLEMENT" and SAID_MINE.search(known[p].text)
            else known[p].experience.actor
            for p in pairs
            if p in known
        }
        allowed = sent.kind == "ELABORATED" and actors == {
            "UNKNOWN"
        }  # 규칙 5 — 확인 필요로 「제가」
        if actors and "ME" not in actors and not allowed:
            words = MINE_WORDS.findall(text)
            if words or ("TEAM" in actors and not TEAM_SUBJECT.search(text)):
                s.team_as_mine.append(f"{pid}: {words or '주어 없음'} {line}")
        if actors == {"UNKNOWN"} and not allowed and (WHO.search(text) or ACTED.search(text)):
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
    s.add(score_hints(pair, data))
    if pair.get("hints"):
        s.hinted_invented = [*s.invented, *s.opinion_invented]
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
            if any(re.search(m["omitPattern"], x.text) for x in data.omitted):
                s.omit_hit += 1
            else:
                s.omit_miss.append(f"{pid}: {m['fact']}")
    return s


def load_experiences() -> list[dict]:
    extra = (DATA / "experiences.json").read_text(encoding="utf-8")
    return json.loads(EXPERIENCES.read_text(encoding="utf-8")) + json.loads(extra)


def load_facts() -> list[dict]:
    """경험에 이미 있는 사실 — 되묻기 채점의 것 + 초안에만 쓰는 경험의 것."""
    extra = json.loads((DATA / "clarify.json").read_text(encoding="utf-8"))["known"]
    return load_known() + extra


def question(pair: dict) -> dict:
    """/draft 요청의 question — jobRole · requirements · frame 은 있는 쌍만."""
    keys = ("organization", "text", "jobRole", "requirements", "frame", "targetLength")
    return {k: pair[k] for k in keys if k in pair}


def guidelines(pair: dict) -> list[dict]:
    """/draft 요청의 guidelines — 정답 칸(expect 등)은 빼고 id · text 만. 없으면 []."""
    return [{"id": g["id"], "text": g["text"]} for g in pair.get("guidelines", [])]


def load_pairs(only: list[str] | None = None) -> list[dict]:
    pairs = json.loads((DATA / "pairs.json").read_text(encoding="utf-8"))
    return [p for p in pairs if p["id"] in only] if only else pairs


def picked(pair: dict, experiences: list[dict]) -> list[dict]:
    """고른 순서대로 — /draft 요청의 experiences. addSupplements 는 그 경험의 보완에 더하고,
    hints(경험 ID → reason · emphasis · gaps)는 그 경험의 선택 칸으로 싣는다."""
    by_id = {e["id"]: e for e in experiences}
    added = pair.get("addSupplements", [])

    def with_added(exp: dict) -> dict:
        more = [
            {"id": a["id"], "question": a["question"], "answer": a["answer"]}
            for a in added
            if a["experienceId"] == exp["id"]
        ]
        hint = pair.get("hints", {}).get(str(exp["id"]), {})
        return {**exp, **hint, "supplements": [*exp.get("supplements", []), *more]}

    return [with_added(by_id[i]) for i in pair["experienceIds"]]


def context_of(pair: dict, experiences: list[dict]) -> list[dict]:
    """/draft 요청의 contextExperiences — 쌍에 적은 순서대로. 없으면 [](S-75)."""
    by_id = {e["id"]: e for e in experiences}
    return [by_id[i] for i in pair.get("contextExperienceIds", [])]


# ---- 실행 ----


async def run(
    models: list[str],
    prompts: list[str],
    experiences: list[dict],
    pairs: list[dict],
    out: Path,
    env: Path = ROOT / ".env",
    runs: int | None = None,
) -> None:
    """runs 를 주면 회차 폴더(run-N)에, 안 주면(eval_draft_guided · eval_frame_cards) 프롬프트 폴더에 바로 쓴다."""
    from dotenv import load_dotenv

    from app.config import load_settings, parse_models
    from app.draft import (
        build_draft_message,
        check_output,
        check_request,
        schema_for,
        system_prompt,
    )
    from app.errors import ApiError
    from app.llm import build_llm, generate_valid
    from app.schemas import DraftRequest

    load_dotenv(env)  # 값은 화면에 찍지 않는다
    base = load_settings()
    gate = asyncio.Semaphore(4)
    clients = {}  # 모델마다 클라이언트를 한 번만 만든다(AI#37)
    for model in models:
        settings = replace(base, llm_models=parse_models(model))
        clients[model] = (settings, build_llm(settings))

    async def one(model: str, prompt: str, n: int | None, pair: dict) -> None:
        settings, llm = clients[model]
        req = DraftRequest.model_validate(
            {
                "question": question(pair),
                "experiences": picked(pair, experiences),
                "contextExperiences": context_of(pair, experiences),
                "guidelines": guidelines(pair),
            }
        )
        q, exps, gs = check_request(req, settings)
        ctx = req.contextExperiences or []

        def parse(text: str) -> LLMDraftOutput:
            output = LLMDraftOutput.model_validate_json(text)
            return check_output(output, exps, gs, ctx) if prompt == "tuned" else output

        async with gate:
            started = time.monotonic()
            try:
                got = await generate_valid(
                    llm,
                    system_prompt(q, exps, gs, ctx) if prompt == "tuned" else PROMPTS[prompt],
                    build_draft_message(q, exps, gs, ctx),
                    schema_for(gs),
                    parse,
                    settings,
                )
                record = {"response": json.loads(got.value.model_dump_json())}
                record["attempt"] = got.attempt
            except ApiError as e:
                record = {"error": f"{e.code}: {e.message}"}
            record["seconds"] = round(time.monotonic() - started, 1)
        folder = out / _slug(model) / prompt
        path = (folder if n is None else run_path(folder, n)) / f"{pair['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        status = record.get("error") or "ok"
        name = prompt if n is None else f"{prompt} · run-{n}"
        print(f"  {model} · {name} · {pair['id']} — {status} ({record['seconds']}초)")

    numbers = [None] if runs is None else range(1, runs + 1)
    await asyncio.gather(
        *(one(m, p, n, q) for m in models for p in prompts for n in numbers for q in pairs)
    )


def report(out: Path, experiences: list[dict], pairs: list[dict]) -> str:
    facts = load_facts()
    rows, details, question_rows, opinion_rows, hint_rows, essay_rows = [], [], [], [], [], []
    guess_rows = []
    for model_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        for prompt_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            for suffix, run_dir in run_dirs(prompt_dir):
                t = Score()
                for pair in pairs:
                    path = run_dir / f"{pair['id']}.json"
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
                    ctx = [MatchExperience.model_validate(e) for e in context_of(pair, experiences)]
                    data = LLMDraftOutput.model_validate(record["response"])
                    t.add(score(pair, exps, data, facts, ctx))
                if not t.pairs:
                    continue
                name = f"{model_dir.name} · {prompt_dir.name}{suffix}"
                rows.append(
                    f"| {name} | {len(t.sentences)} | {t.retried} | {len(t.errors)}/{t.pairs} "
                    f"| {len(t.fake_ids)} | {len(t.bad_numbers)} | {len(t.invented)} "
                    f"| {len(t.team_as_mine)} | {len(t.unknown_actor)} | {t.omit_hit}/{t.omit_chances} "
                    f"| {t.expect_hit}/{t.expect_chances} "
                    f"| {len(t.unused)}/{t.picked_total} "
                    f"| {t.chars // max(t.pairs - len(t.errors), 1)} "
                    f"| {t.order_hit}/{t.order_total} | {len(t.repeated)} | {t.seconds:.0f}초 |"
                )
                question_rows.append(
                    f"| {name} | {len(t.asked)}({t.omitted_total}) | {len(t.bad_pairs)} "
                    f"| {t.gap_asked}/{t.gap_chances} | {len(t.forced)} | {len(t.leading)} "
                    f"| {len(t.vague)} | {len(t.compound)} "
                    f"| {len(t.known_asked)}/{t.known_chances} "
                    f"| {t.fill_hit}/{t.fill_chances} | {len(t.fill_reasked)} |"
                )
                opinion_rows.append(
                    f"| {name} | {len(t.opinions)} | {len(t.kind_broken)} "
                    f"| {len(t.opinion_happened)} | {len(t.opinion_invented)} "
                    f"| {len(t.opinion_flattery)} | {len(t.titles)} | {len(t.title_broken)} "
                    f"| {len(t.title_invented)} | {len(t.title_flattery)} | {len(t.title_category)} "
                    f"| {len(t.vow_open)} | {len(t.vow_end)} |"
                )
                hint_rows.append(
                    f"| {name} | {t.lead_hit}/{t.lead_chances} | {len(t.hinted_invented)} "
                    f"| {len(t.stale_reasked)} |"
                )
                essay_rows.append(
                    f"| {name} | {len(t.listed)}/{t.list_chances} | {t.deep_hit}/{t.deep_chances} "
                    f"| {len(t.elaborated)} | {len(t.elab_heavy)} | {len(t.elab_invented)} "
                    f"| {len(t.elab_judged)} | {len(t.unknown_mine)} | {len(t.marks)} "
                    f"| {t.target_hit}/{t.target_chances} | {t.context_used}/{t.context_chances} "
                    f"| {len(t.context_heavy)} |"
                )
                guess_rows.append(
                    f"| {name} | {len(t.passive)} | {len(t.supposed)}({t.specific}) "
                    f"| {len(t.guess_heavy)} | {len(t.specific_miss)} | {len(t.guess_unpaired)} "
                    f"| {len(t.guess_over_fact)} | {len(t.guess_mine)} | {len(t.guess_employer)} |"
                )
                lines = [
                    *(f"- 오류 {x}" for x in t.errors),
                    *(f"- 수동형(후보) {x}" for x in t.passive),
                    *(f"- specific 누락(후보) {x}" for x in t.specific_miss),
                    *(f"- 가정 짝 위반 {x}" for x in t.guess_unpaired),
                    *(f"- 가정이 사실 덮음(후보) {x}" for x in t.guess_over_fact),
                    *(f"- 가정 팀 「제가」(후보) {x}" for x in t.guess_mine),
                    *(f"- 지원처 가정(후보) {x}" for x in t.guess_employer),
                    *(f"- 가정 > 사실 {x}" for x in t.guess_heavy),
                    *(f"- AI 가정 {x}" for x in t.supposed),
                    *(f"- 나열 {x}" for x in t.listed),
                    *(f"- 대표 얕음 {x}" for x in t.shallow),
                    *(f"- 보탬 > 사실 {x}" for x in t.elab_heavy),
                    *(f"- 보탬 지어냄(후보) {x}" for x in t.elab_invented),
                    *(f"- 보탬 판단 · 감정(후보) {x}" for x in t.elab_judged),
                    *(f"- 모름에 「제가」 {x}" for x in t.unknown_mine),
                    *(f"- 목표 어긋남 {x}" for x in t.target_miss),
                    *(f"- 보조가 뼈대 {x}" for x in t.context_heavy),
                    *(f"- 첫 문장 {x}" for x in t.firsts),
                    *(f"- AI 보탬 {x}" for x in t.elaborated),
                    *(f"- 빈칸 {x}" for x in t.marks),
                    *(f"- 강조 뒤로 {x}" for x in t.lead_miss),
                    *(f"- 강조 쌍 지어냄(후보) {x}" for x in t.hinted_invented),
                    *(f"- 지난 빈칸 또 물음(후보) {x}" for x in t.stale_reasked),
                    *(f"- 종류 위반 {x}" for x in t.kind_broken),
                    *(f"- 사실로 뺌(후보) {x}" for x in t.opinion_happened),
                    *(f"- 없는 사실(후보) {x}" for x in t.opinion_invented),
                    *(f"- 감정 · 칭찬 · 단정(후보) {x}" for x in t.opinion_flattery),
                    *(f"- 소제목 위반 {x}" for x in t.title_broken),
                    *(f"- 소제목 지어냄(후보) {x}" for x in t.title_invented),
                    *(f"- 소제목 칭찬(후보) {x}" for x in t.title_flattery),
                    *(f"- 분류명 소제목 {x}" for x in t.title_category),
                    *(f"- 다짐 첫 문장(후보) {x}" for x in t.vow_open),
                    *(f"- 다짐 맺음(후보) {x}" for x in t.vow_end),
                    *(f"- 소제목 {x}" for x in t.titles),
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
                    *(f"- 짝 위반 {x}" for x in t.bad_pairs),
                    *(f"- 질문 없는 빈칸 {x}" for x in t.gap_miss),
                    *(f"- 억지 질문 {x}" for x in t.forced),
                    *(f"- 수치 유도(후보) {x}" for x in t.leading),
                    *(f"- 막연함(후보) {x}" for x in t.vague),
                    *(f"- 두 가지(후보) {x}" for x in t.compound),
                    *(f"- 있는 사실 물음(후보) {x}" for x in t.known_asked),
                    *(f"- 보완 안 씀 {x}" for x in t.fill_miss),
                    *(f"- 보완 또 물음(후보) {x}" for x in t.fill_reasked),
                    *(f"- 생각 {x}" for x in t.opinions),
                    *(f"- 문장 {x}" for x in t.sentences),
                    *(f"- omitted {x}" for x in t.omitted),
                ]
                details.append(f"\n### {name}\n\n" + "\n".join(lines))
    head = (
        "| 모델 · 프롬프트 | 문장 | 다시 부름 | 오류 | 없는 ID | 없는 숫자 | 지어냄(후보) "
        "| 팀 일을 내 일로(후보) | 모름에 주체(후보) | omitted 씀 | 맞는 근거 | 고른 경험 안 씀 | 글자(쌍 평균) | 구성 순서 | 되풀이(후보) | 시간 합 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    question_head = (
        "| 모델 · 프롬프트 | 질문(omitted) | 짝 위반 | 질문 붙음 | 억지 질문 | 수치 유도(후보) "
        "| 막연함(후보) | 두 가지(후보) | 있는 사실 물음(후보) | 보완 씀 | 보완 또 물음(후보) |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    opinion_head = (
        "| 모델 · 프롬프트 | 생각 문장 | 종류 위반 | 사실로 뺌(후보) | 없는 사실(후보) "
        "| 감정 · 칭찬 · 단정(후보) | 소제목 | 소제목 위반 | 소제목 지어냄(후보) | 소제목 칭찬(후보) "
        "| 분류명 소제목 | 다짐 첫 문장(후보) | 다짐 맺음(후보) |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    hint_head = (
        "| 모델 · 프롬프트 | 강조 앞 | 강조 쌍 지어냄(후보) | 지난 빈칸 또 물음(후보) |\n"
        "| --- | --- | --- | --- |"
    )
    essay_head = (
        "| 모델 · 프롬프트 | 나열 | 대표 깊이 | AI 보탬 | 보탬 > 사실 | 보탬 지어냄(후보) "
        "| 보탬 판단 · 감정(후보) | 모름에 「제가」 | 빈칸 | 목표 맞음 | 보조 재료 씀 | 보조가 뼈대 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    guess_head = (
        "| 모델 · 프롬프트 | 수동형(후보) | 가정(specific) | 가정 > 사실 | specific 누락(후보) "
        "| 가정 짝 위반 | 가정이 사실 덮음(후보) | 가정 팀 「제가」(후보) | 지원처 가정(후보) |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    tables = "\n\n".join(
        "\n".join(part)
        for part in (
            [head, *merge_runs(rows)],
            [question_head, *merge_runs(question_rows)],
            [opinion_head, *merge_runs(opinion_rows)],
            [hint_head, *merge_runs(hint_rows)],
            [essay_head, *merge_runs(essay_rows)],
            [guess_head, *merge_runs(guess_rows)],
        )
    )
    return tables + "\n" + "".join(details) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="쉼표로 「제공사:모델」 — 기본은 MATCH_LLM_MODELS 첫 모델")
    parser.add_argument("--prompts", default="tuned,generic", help="tuned · generic")
    parser.add_argument("--runs", type=int, default=1, help="같은 쌍을 몇 번씩")
    parser.add_argument("--pairs", help="쉼표로 쌍 id — 기본은 전부")
    parser.add_argument("--out", type=Path, help="응답 저장 폴더")
    parser.add_argument("--rescore", type=Path, help="저장한 응답만 다시 채점(LLM 안 부름)")
    parser.add_argument("--env", type=Path, default=ROOT / ".env", help="키를 읽을 .env")
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

            load_dotenv(args.env)
            p, m = load_settings().match_models[0]
            models = [f"{p}:{m}"]
        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
        for p in prompts:
            if p not in PROMPTS:
                parser.error(f"프롬프트는 {list(PROMPTS)} 중 — {p!r}")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = args.out or ROOT / "eval-runs" / f"draft-{stamp}"
        print(
            f"LLM 호출 {len(pairs) * len(models) * len(prompts) * args.runs}회(규칙 위반 시 더) → {out}"
        )
        asyncio.run(run(models, prompts, experiences, pairs, out, args.env, args.runs))

    text = report(out, experiences, pairs)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
