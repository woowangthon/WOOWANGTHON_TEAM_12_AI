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
