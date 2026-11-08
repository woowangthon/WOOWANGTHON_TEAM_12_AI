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

# 활동 분류(S-57) — 닫힌 목록. 원본은 ETC docs/contracts/README.md 「값 목록」 activityCategory
ActivityCategory = Literal["PROJECT", "STUDY", "CLUB"]
ACTIVITY_CATEGORIES: tuple[str, ...] = get_args(ActivityCategory)


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


class ExistingActivity(BaseModel):
    """이미 가진 활동(S-44) — BE 가 저장한 값 그대로라 너그럽게 받는다. 이름 · 기간만 단서로 쓴다."""

    id: int
    name: str
    period: "MatchPeriod | None" = None


class ExtractRequest(BaseModel):
    sourceName: str | None = None
    text: str | None = None
    # 빼면 [] 와 같다(S-44 전 BE 와 맞물리게)
    existingActivities: list[ExistingActivity] = []
    # 「자료 더하기」로 사용자가 고른 활동(S-60) — existingActivities 의 id 또는 null
    targetActivityId: int | None = None


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
    # 목록 안의 id 인가는 app/extract.py check_output 이 본다 — 칸이 빠져도 규칙 위반(기본값 없음)
    sameAsActivityId: int | None
    # 목록 밖 값 · 칸 빠짐은 규칙 위반 — null 로 바꾸지 않고 다시 부른다
    category: ActivityCategory | None
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
    # 늘 싣는다(계약). 기본값은 S-44 전에 저장한 채점 응답을 다시 읽으려고
    sameAsActivityId: int | None = None
    # 늘 싣는다(계약). 기본값은 S-57 전에 저장한 채점 응답을 다시 읽으려고
    category: ActivityCategory | None = None
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


# ==== POST /ask — ETC docs/contracts/ai-api.md 「POST /ask」 ====


class AskRequest(BaseModel):
    query: str | None = None
    experiences: list[MatchExperience] | None = None


class AskRecommendation(BaseModel):
    experienceId: int
    reason: NonBlank
    evidence: list[Evidence]


class LLMAskOutput(BaseModel):
    recommendations: list[AskRecommendation] = Field(max_length=3)


class AskResponse(LLMAskOutput):
    model: str | None


# ==== POST /draft — ETC docs/contracts/ai-api.md 「POST /draft」 ====

DraftEvidenceType = Literal["CITATION", "SUPPLEMENT", "EXPERIENCE"]
# 값 목록은 ETC docs/contracts/README.md 「값 목록」 draftSentenceKind
DraftSentenceKind = Literal["FACT", "OPINION", "ELABORATED"]
# 구성 틀 — README 「값 목록」 draftFrame(S-50). NONE 이나 없으면 기본 순서(AI#46)
DraftFrame = Literal["STAR", "CARL", "PREP", "NONE"]


def _one_line(value: str) -> str:
    if "\n" in value or "\r" in value:
        raise ValueError("문장에 줄바꿈이 있다")
    return value


class DraftRequirement(BaseModel):
    # question-api Requirement 에서 experienceIds 를 뺀 것(ETC#141)
    text: NonBlank
    status: RequirementStatus


class DraftQuestion(MatchQuestion):
    # 선택 칸 — 없으면(null 도) 지금과 같다. 모양이 틀리면 400(RequestValidationError).
    # jobRole 의 값 목록 JobRole(S-42)은 아래에 있어 이름으로 가리키고 파일 끝에서 푼다
    jobRole: "JobRole | None" = None
    requirements: list[DraftRequirement] | None = None
    frame: DraftFrame | None = None
    # 목표 글자 수(S-77). 목표의 80 ~ 100% 를 노리되 못 채워도 실패가 아니다
    targetLength: Annotated[int, Field(ge=100, le=5000)] | None = None


class DraftExperience(MatchExperience):
    # 그 문항 추천의 이 경험 몫 — question-api Recommendation 그대로. 근거가 아니다(ETC#141)
    reason: NonBlank | None = None
    emphasis: NonBlank | None = None
    gaps: list[NonBlank] | None = None


class DraftGuideline(BaseModel):
    # 사용자가 켜 둔 작성 지침(S-49). 개수 · id 중복은 app/draft.py check_request 가 본다(400)
    id: int
    text: NonBlank


class DraftRequest(MatchRequest):
    # 모양은 /match 와 같고 고른 경험만 싣는다. 개수는 app/draft.py check_request 가 본다(400)
    question: DraftQuestion | None = None
    experiences: list[DraftExperience] | None = None
    # 같은 활동의 나머지 경험 — 뒷받침 재료(S-75). 개수 · id 겹침은 app/draft.py check_request 가 본다(400)
    contextExperiences: list[MatchExperience] | None = None
    guidelines: list[DraftGuideline] | None = None


class DraftEvidence(BaseModel):
    type: DraftEvidenceType
    id: int


class DraftSentence(BaseModel):
    # kind 마다 근거 · 숫자 · note 규칙이 달라 app/draft.py check_output 이 본다
    kind: DraftSentenceKind
    text: Annotated[NonBlank, AfterValidator(_one_line)]
    evidence: list[DraftEvidence]
    # ELABORATED 만 — 왜 확인이 필요한지 한 줄(S-78)
    note: Annotated[NonBlank, AfterValidator(_one_line), Field(max_length=60)] | None = None


class DraftOmitted(BaseModel):
    # 셋 다 필수 — 예전 문자열 응답을 추측해 고치지 않는다.
    # 질문 · 경험의 null 짝과 요청 안의 경험인지는 app/draft.py check_output 이 본다
    text: NonBlank
    question: NonBlank | None
    experienceId: int | None
    # 문장 속 빈칸(S-78) — 둘 다 있거나 둘 다 null. 모양 · 문장 안 짝은 app/draft.py check_output 이 본다
    placeholder: str | None = None
    sentenceIndex: int | None = None


class DraftGuidelineResult(BaseModel):
    # 요청 지침마다 하나, 같은 순서 · applied 와 note 의 짝은 app/draft.py check_output 이 본다
    id: int
    applied: bool
    note: Annotated[NonBlank, AfterValidator(_one_line), Field(max_length=80)] | None


class LLMDraftOutput(BaseModel):
    sentences: list[DraftSentence] = Field(max_length=15)
    omitted: list[DraftOmitted] = Field(max_length=5)
    # 요청에 지침이 없으면 LLM 에 묻지 않는다 — 빠지면 [](계약)
    guidelines: list[DraftGuidelineResult] = []


class DraftResponse(LLMDraftOutput):
    model: str | None


# ==== POST /ground-check — ETC docs/contracts/ai-api.md 「POST /ground-check」 ====

GroundingStatus = Literal["SOURCE", "USER_INPUT", "UNSUPPORTED", "OPINION"]


class GroundCheckRequest(BaseModel):
    text: str | None = None
    experiences: list[MatchExperience] | None = None


class GroundVerdict(BaseModel):
    index: int
    status: GroundingStatus
    evidence: list[DraftEvidence]
    note: Annotated[NonBlank, AfterValidator(_one_line)] | None


class LLMGroundOutput(BaseModel):
    verdicts: list[GroundVerdict]


class GroundSentence(BaseModel):
    text: str
    status: GroundingStatus
    evidence: list[DraftEvidence]
    note: str | None


class GroundCheckResponse(BaseModel):
    sentences: list[GroundSentence]
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


# 지원처 종류 · 직무 — 닫힌 목록. 원본은 ETC docs/contracts/README.md 「값 목록」(S-42 · 미결정 17)
OrganizationType = Literal[
    "CLUB", "EXTRACURRICULAR", "EXTERNAL", "BOOTCAMP", "INTERN", "JOB", "OTHER"
]
ORGANIZATION_TYPES: tuple[str, ...] = get_args(OrganizationType)
JobRole = Literal[
    "BACKEND",
    "FRONTEND",
    "MOBILE",
    "AI_DATA",
    "INFRA",
    "DESIGN",
    "PLANNING",
    "MARKETING",
    "BUSINESS",
    "OTHER",
]
JOB_ROLES: tuple[str, ...] = get_args(JobRole)


class ParseApplicationRequest(BaseModel):
    # 둘 중 하나만 · 비었는지 · 20000자는 app/parse_application.py check_request 가 본다
    text: str | None = None
    url: str | None = None


# LLM 이 내는 모양. 글에 적혀 있나 · 순서 · 중복은 app/parse_application.py check_written 이 본다
class LLMParseApplicationOutput(BaseModel):
    organization: Annotated[NonBlank, Field(max_length=100)] | None
    organizationType: OrganizationType | None
    jobRole: JobRole | None
    questions: Annotated[list[Annotated[NonBlank, Field(max_length=5000)]], Field(max_length=10)]


class ParseApplicationResponse(LLMParseApplicationOutput):
    model: str | None


# POST /link-suggestions (S-35). 비었나 · 개수 · id 중복 · 글자 수는 app/link_suggest.py check_request 가 본다
class LinkExperience(BaseModel):
    id: int
    content: str


class LinkActivity(BaseModel):
    id: int
    name: str
    period: MatchPeriod | None = None
    note: str | None = None
    experiences: list[LinkExperience] = []


class LinkSuggestionsRequest(BaseModel):
    # 「from」은 파이썬 예약어라 별칭으로 받는다
    origin: LinkActivity = Field(alias="from")
    candidates: list[LinkActivity]


# LLM 이 내는 모양. 후보 안인가 · 중복 · 줄바꿈은 app/link_suggest.py check_output 이 본다
class LinkSuggestion(BaseModel):
    toActivityId: int
    reason: Annotated[NonBlank, Field(max_length=200)]


class LLMLinkSuggestionsOutput(BaseModel):
    suggestions: Annotated[list[LinkSuggestion], Field(max_length=3)]


class LinkSuggestionsResponse(LLMLinkSuggestionsOutput):
    model: str | None


# ExistingActivity 의 기간은 아래에 정의한 MatchPeriod 를 쓴다(앞에서 이름으로 가리킨 것을 풀어 둔다)
ExistingActivity.model_rebuild()
ExtractRequest.model_rebuild()
DraftQuestion.model_rebuild()
DraftRequest.model_rebuild()


class AgentQuestion(BaseModel):
    id: int
    text: str
    organization: str | None = None


class AgentContext(BaseModel):
    question: AgentQuestion | None = None


class AgentRequest(BaseModel):
    # 전부 선택으로 두고 코드로 검사한다 — pydantic 의 「없음」 오류는 본문 전체를 실어 mcpUrl 토큰이 응답으로 샌다
    text: str | None = None
    context: AgentContext | None = None
    mcpUrl: str | None = None
    maxToolCalls: int | None = None


# ==== POST /new-facts — ETC docs/contracts/ai-api.md 「POST /new-facts」 ====


class NewFactsRequest(BaseModel):
    text: str | None = None
    experiences: list[MatchExperience] | None = None
    activities: list[ExistingActivity] | None = None
    excludedSentences: list[str] | None = None


class LLMNewFact(BaseModel):
    index: int
    content: Annotated[NonBlank, AfterValidator(_one_line)]
    actor: Actor
    activityId: int | None


class LLMNewFactsOutput(BaseModel):
    candidates: list[LLMNewFact]


class NewFactCandidate(BaseModel):
    sentence: str
    content: str
    actor: Actor
    activityId: int | None


class NewFactsResponse(BaseModel):
    candidates: list[NewFactCandidate]
    model: str | None


# ==== POST /guidelines — ETC docs/contracts/ai-api.md 「POST /guidelines」 ====


class GuidelinesRequest(BaseModel):
    text: str | None = None


class GuidelineLine(BaseModel):
    text: Annotated[NonBlank, AfterValidator(_one_line), Field(max_length=100)]


class LLMGuidelinesOutput(BaseModel):
    guidelines: list[GuidelineLine] = Field(max_length=10)


class GuidelinesResponse(LLMGuidelinesOutput):
    model: str | None
