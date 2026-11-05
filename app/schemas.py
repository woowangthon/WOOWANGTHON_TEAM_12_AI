"""요청 · 응답 · LLM 출력 모양 — ETC docs/contracts/ai-api.md 「POST /extract」."""

import re
from typing import Annotated, Literal, get_args

from pydantic import AfterValidator, BaseModel, Field, model_validator


def _not_blank(value: str) -> str:
    # 값을 고치지 않고 검사만 한다 — 발췌문은 원문 그대로여야 한다
    if not value.strip():
        raise ValueError("빈 문자열")
    return value


_PERIOD_RE = re.compile(r"^\d{4}(-(0[1-9]|1[0-2])(-(0[1-9]|[12]\d|3[01]))?)?$")


def _period_format(value: str) -> str:
    if not _PERIOD_RE.match(value):
        raise ValueError("YYYY · YYYY-MM · YYYY-MM-DD 중 하나여야 한다")
    return value


NonBlank = Annotated[str, AfterValidator(_not_blank)]
PeriodValue = Annotated[str, AfterValidator(_period_format)]
Actor = Literal["ME", "TEAM", "UNKNOWN"]

# 역량 — 닫힌 목록. 원본은 ETC docs/contracts/README.md 「값 목록」(바꾸려면 SPEC 미결정 2 부터)
Competency = Literal[
    "IMPLEMENTATION",
    "PROBLEM_SOLVING",
    "PLANNING",
    "ANALYSIS",
    "OPERATION",
    "COLLABORATION",
    "COMMUNICATION",
    "LEADERSHIP",
    "INITIATIVE",
    "RESPONSIBILITY",
    "PERSEVERANCE",
    "LEARNING",
    "PRESENTATION",
]
COMPETENCIES: tuple[str, ...] = get_args(Competency)


def _no_duplicates(values: list[str]) -> list[str]:
    # 중복을 지우지 않고 규칙 위반으로 본다 — 조용히 고쳐 넣지 않는다
    if len(set(values)) != len(values):
        raise ValueError("역량이 중복됐다")
    return values


# LLM 은 1 ~ 3개(계약). 사용자가 지워 0개가 되는 건 BE 쪽 일이다
Competencies = Annotated[
    list[Competency], Field(min_length=1, max_length=3), AfterValidator(_no_duplicates)
]


# ---- 요청 ----


class ExtractRequest(BaseModel):
    sourceName: str | None = None
    text: str | None = None


# ---- 기간 (요청 · 응답 공통) ----


class Period(BaseModel):
    start: PeriodValue | None
    end: PeriodValue | None
    ongoing: bool

    @model_validator(mode="after")
    def _ongoing_has_no_end(self) -> "Period":
        if self.ongoing and self.end is not None:
            raise ValueError("ongoing 이 true 면 end 는 null")
        return self


# ---- LLM 이 내야 하는 모양 ----
# sourceStatus · matched · location 은 LLM 이 정하지 않는다 — 코드가 붙인다(계약).


class LLMExperience(BaseModel):
    content: NonBlank
    competencies: Competencies
    actor: Actor
    quotes: list[NonBlank]


class LLMActivity(BaseModel):
    name: NonBlank
    period: Period
    experiences: list[LLMExperience]


class LLMOutput(BaseModel):
    activities: list[LLMActivity]


# ---- 응답 ----


class Location(BaseModel):
    startLine: int
    endLine: int


class Citation(BaseModel):
    quote: str
    matched: bool | None
    location: Location | None


class Experience(BaseModel):
    content: str
    competencies: list[Competency]
    actor: Actor
    sourceStatus: Literal["SOURCE_CONFIRMED", "NEEDS_REVIEW"]
    citations: list[Citation]


class Activity(BaseModel):
    name: str
    period: Period
    experiences: list[Experience]


class ExtractResponse(BaseModel):
    activities: list[Activity]
    model: str | None


# ==== POST /match — ETC docs/contracts/ai-api.md 「POST /match」 ====
# 요청은 BE 가 저장한 값 그대로라 너그럽게 받는다(역량 · 기간 모양까지 다시 따지지 않는다 — 매칭에 쓰지 않거나
# 문맥 단서로만 쓴다). LLM 출력은 엄격하게 본다.

RequirementStatus = Literal["SUPPORTED", "PARTIAL", "NOT_FOUND"]
EvidenceType = Literal["CITATION", "SUPPLEMENT"]
# 문항 유형(S-41) — 닫힌 목록. 원본은 ETC docs/contracts/README.md 「값 목록」 questionType
QuestionType = Literal[
    "MOTIVATION",
    "SELF_INTRO",
    "STRENGTH_WEAKNESS",
    "ACHIEVEMENT",
    "PROBLEM_SOLVING",
    "COLLABORATION",
    "CONFLICT",
    "FAILURE",
    "GROWTH",
    "IDEA",
    "OTHER",
]
QUESTION_TYPES: tuple[str, ...] = get_args(QuestionType)


class MatchQuestion(BaseModel):
    organization: str | None = None
    text: str | None = None


class MatchPeriod(BaseModel):
    start: str | None = None
    end: str | None = None
    ongoing: bool = False


class MatchCitation(BaseModel):
    id: int
    quote: str


class MatchSupplement(BaseModel):
    id: int
    question: str
    answer: str


class MatchExperience(BaseModel):
    id: int
    activityName: str | None = None
    period: MatchPeriod | None = None
    content: str
    competencies: list[str] = []
    actor: Actor
    sourceStatus: str | None = None
    citations: list[MatchCitation] = []
    supplements: list[MatchSupplement] = []


class MatchRequest(BaseModel):
    question: MatchQuestion | None = None
    experiences: list[MatchExperience] | None = None


class Requirement(BaseModel):
    text: NonBlank
    status: RequirementStatus
    experienceIds: list[int]


class Evidence(BaseModel):
    type: EvidenceType
    id: int


class Recommendation(BaseModel):
    experienceId: int
    reason: NonBlank
    emphasis: NonBlank
    evidence: list[Evidence]
    gaps: list[NonBlank]


class Clarification(BaseModel):
    experienceId: int
    question: NonBlank


# LLM 이 내는 모양 — 개수가 넘치면 잘라 내지 않고 규칙 위반(다시 부른다)
class LLMMatchOutput(BaseModel):
    questionType: QuestionType
    requirements: list[Requirement] = Field(min_length=1, max_length=6)
    recommendations: list[Recommendation] = Field(max_length=3)
    clarifications: list[Clarification] = Field(max_length=3)


class MatchResponse(BaseModel):
    questionType: QuestionType
    requirements: list[Requirement]
    recommendations: list[Recommendation]
    clarifications: list[Clarification]
    model: str | None


# ==== POST /draft — ETC docs/contracts/ai-api.md 「POST /draft」 ====

DraftEvidenceType = Literal["CITATION", "SUPPLEMENT", "EXPERIENCE"]


def _one_line(value: str) -> str:
    if "\n" in value or "\r" in value:
        raise ValueError("문장에 줄바꿈이 있다")
    return value


class DraftRequest(MatchRequest):
    # 모양은 /match 와 같고 고른 경험만 싣는다. 개수는 app/draft.py check_request 가 본다(400)
    pass


class DraftEvidence(BaseModel):
    type: DraftEvidenceType
    id: int


class DraftSentence(BaseModel):
    text: Annotated[NonBlank, AfterValidator(_one_line)]
    evidence: list[DraftEvidence] = Field(min_length=1)


class LLMDraftOutput(BaseModel):
    sentences: list[DraftSentence] = Field(max_length=15)
    omitted: list[NonBlank] = Field(max_length=5)


class DraftResponse(LLMDraftOutput):
    model: str | None


# ==== POST /route — ETC docs/contracts/ai-api.md 「POST /route」 ====

# 홈 입력 가르기 — 닫힌 목록. 원본은 ETC docs/contracts/README.md 「값 목록」 entranceKind
EntranceKind = Literal["SOURCE", "APPLICATION", "ASK", "UNSURE"]
ENTRANCE_KINDS: tuple[str, ...] = get_args(EntranceKind)


class RouteRequest(BaseModel):
    # 비었는지 · 4000자 · totalLength 와 맞는지는 app/route.py check_request 가 본다(400 INVALID_REQUEST)
    text: str
    totalLength: int


# LLM 이 내는 모양 — reason 이 200자를 넘으면 잘라 내지 않고 규칙 위반(다시 부른다)
class LLMRouteOutput(BaseModel):
    reason: Annotated[NonBlank, Field(max_length=200)]
    kind: EntranceKind


class RouteResponse(BaseModel):
    kind: EntranceKind
    reason: str
    model: str | None
