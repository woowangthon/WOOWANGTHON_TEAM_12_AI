"""/extract · /health — LLM 은 가짜로 바꿔 계약 모양 · 오류 처리만 본다(네트워크 없음)."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import load_settings
from app.errors import LLMError, LLMTimeout
from app.extract import (
    SYSTEM_PROMPT,
    TARGET_RULE,
    WORK_LOG_RULE,
    build_user_message,
    is_work_log,
)
from app.main import app
from app.schemas import ExistingActivity
from tests.conftest import FakeLLM

GOOD = {
    "activities": [
        {
            "name": "바꿔요",
            "period": {"start": "2025-03", "end": "2025-06", "ongoing": False},
            "sameAsActivityId": None,
            "category": "PROJECT",
            "experiences": [
                {
                    "content": "폴링을 WebSocket 으로 바꿨다",
                    "competencies": ["PROBLEM_SOLVING", "IMPLEMENTATION"],
                    "actor": "ME",
                    "quotes": ["내가 폴링을 WebSocket 으로 바꿔 해결했다."],
                },
                {
                    "content": "AWS 에 배포했다",
                    "competencies": ["IMPLEMENTATION"],
                    "actor": "TEAM",
                    "quotes": [],
                },
            ],
        },
        {
            "name": "경험 없는 활동",
            "period": {"start": None, "end": None, "ongoing": False},
            "sameAsActivityId": None,
            "category": None,
            "experiences": [],
        },
    ]
}


BODY = {"sourceName": "회고.md", "text": "내가 폴링을 WebSocket 으로 바꿔 해결했다."}


def test_health(client_with):
    res = client_with(FakeLLM()).get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_extract_contract_shape(client_with):
    llm = FakeLLM(GOOD)
    res = client_with(llm).post("/extract", json=BODY)
    assert res.status_code == 200
    data = res.json()
    assert data["model"] == "fake-model"
    # 이름 · 기간만 있는 활동도 경험 0개로 남는다(S-03, 2026-11-03)
    assert [a["name"] for a in data["activities"]] == ["바꿔요", "경험 없는 활동"]
    assert data["activities"][1]["experiences"] == []
    exps = data["activities"][0]["experiences"]
    # 근거를 원본에서 찾았다 → SOURCE_CONFIRMED · 줄 번호
    assert exps[0]["sourceStatus"] == "SOURCE_CONFIRMED"
    assert exps[0]["citations"] == [
        {
            "quote": "내가 폴링을 WebSocket 으로 바꿔 해결했다.",
            "matched": True,
            "location": {"startLine": 1, "endLine": 1},
        }
    ]
    # 근거가 없으면 확인 필요
    assert exps[1]["sourceStatus"] == "NEEDS_REVIEW"
    assert exps[1]["citations"] == []
    assert exps[1]["actor"] == "TEAM"
    # 역량은 LLM 이 준 순서 그대로(가장 잘 맞는 것부터) · 관점 문자열은 없다
    assert exps[0]["competencies"] == ["PROBLEM_SOLVING", "IMPLEMENTATION"]
    assert "perspective" not in exps[0]


def test_quote_not_in_source_needs_review(client_with):
    # 근거 하나라도 원본에 없으면 그 경험은 확인 필요(S-04) — 찾은 근거는 그대로 matched: true
    out = _bad(
        lambda d: d["activities"][0]["experiences"][0]["quotes"].append(
            "나는 백엔드 API 설계를 전부 맡았다."
        )
    )
    res = client_with(FakeLLM(out)).post("/extract", json=BODY)
    assert res.status_code == 200
    exp = res.json()["activities"][0]["experiences"][0]
    assert exp["sourceStatus"] == "NEEDS_REVIEW"
    assert [c["matched"] for c in exp["citations"]] == [True, False]
    assert exp["citations"][1]["location"] is None


def test_competency_meanings_match_contract_list():
    # 프롬프트의 역량 설명이 계약 목록(app.schemas.COMPETENCIES)과 같은 값 · 같은 순서인지
    from app.extract import COMPETENCY_MEANINGS, LLM_JSON_SCHEMA, SYSTEM_PROMPT
    from app.schemas import COMPETENCIES

    assert len(COMPETENCIES) == 13
    assert tuple(COMPETENCY_MEANINGS) == COMPETENCIES
    assert all(f"- {value}:" in SYSTEM_PROMPT for value in COMPETENCIES)
    exp = LLM_JSON_SCHEMA["properties"]["activities"]["items"]["properties"]["experiences"]
    comp = exp["items"]["properties"]["competencies"]
    assert comp["items"]["enum"] == list(COMPETENCIES)
    assert (comp["minItems"], comp["maxItems"]) == (1, 3)


def test_no_activities_is_ok(client_with):
    res = client_with(FakeLLM({"activities": []})).post("/extract", json=BODY)
    assert res.status_code == 200
    assert res.json()["activities"] == []


@pytest.mark.parametrize(
    "body", [{}, {"text": ""}, {"text": "  \n "}, {"sourceName": "a", "text": None}, {"text": 3}]
)
def test_invalid_request(client_with, body):
    llm = FakeLLM()
    res = client_with(llm).post("/extract", json=body)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_REQUEST"
    assert llm.calls == 0


def test_text_too_long(client_with):
    llm = FakeLLM()
    res = client_with(llm, max_text_chars=10).post("/extract", json={"text": "가" * 11})
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "TEXT_TOO_LONG"
    assert llm.calls == 0


def _bad(mutate):
    data = json.loads(json.dumps(GOOD))
    mutate(data)
    return data


BAD_OUTPUTS = [
    "not json",
    "",
    {"activity": []},
    _bad(lambda d: d["activities"][0]["experiences"][0].update(actor="나")),
    _bad(lambda d: d["activities"][0]["experiences"][0].update(content="  ")),
    _bad(lambda d: d["activities"][0]["experiences"][0]["quotes"].append("")),
    _bad(lambda d: d["activities"][0].update(name="")),
    _bad(lambda d: d["activities"][0]["period"].update(start="2025년 3월")),
    _bad(lambda d: d["activities"][0]["period"].update(start="2025-13")),
    _bad(lambda d: d["activities"][0]["period"].update(ongoing=True)),  # ongoing 인데 end 가 있음
    _bad(lambda d: d["activities"][0]["experiences"][0].pop("competencies")),
    # 역량: 목록 밖 · 한국어 · 0개 · 4개 이상 · 중복 · 문자열 하나 — 고쳐 넣지 않고 오류
    _bad(lambda d: d["activities"][0]["experiences"][0].update(competencies=["TEAMWORK"])),
    _bad(lambda d: d["activities"][0]["experiences"][0].update(competencies=["협업"])),
    _bad(lambda d: d["activities"][0]["experiences"][0].update(competencies=[])),
    _bad(
        lambda d: d["activities"][0]["experiences"][0].update(
            competencies=["PLANNING", "ANALYSIS", "LEARNING", "OPERATION"]
        )
    ),
    _bad(
        lambda d: d["activities"][0]["experiences"][0].update(competencies=["PLANNING", "PLANNING"])
    ),
    _bad(lambda d: d["activities"][0]["experiences"][0].update(competencies="PLANNING")),
]


@pytest.mark.parametrize("bad", BAD_OUTPUTS)
def test_invalid_output_retries_then_errors(client_with, bad):
    llm = FakeLLM(bad, bad)
    res = client_with(llm, max_attempts=2).post("/extract", json=BODY)
    assert res.status_code == 502
    assert res.json()["error"]["code"] == "LLM_INVALID_OUTPUT"
    assert llm.calls == 2


def test_invalid_then_valid_recovers(client_with):
    llm = FakeLLM("not json", GOOD)
    res = client_with(llm, max_attempts=2).post("/extract", json=BODY)
    assert res.status_code == 200
    assert llm.calls == 2


def test_llm_error(client_with):
    res = client_with(FakeLLM(LLMError("키 없음"))).post("/extract", json=BODY)
    assert res.status_code == 502
    assert res.json()["error"]["code"] == "LLM_ERROR"


def test_llm_timeout(client_with):
    res = client_with(FakeLLM(LLMTimeout("느림"))).post("/extract", json=BODY)
    assert res.status_code == 504
    assert res.json()["error"]["code"] == "LLM_TIMEOUT"


def test_unexpected_error_is_internal(client_with):
    client_with(FakeLLM(RuntimeError("boom")))  # 의존성만 바꾸고
    client = TestClient(app, raise_server_exceptions=False)  # 서버 예외를 응답으로 받는다
    res = client.post("/extract", json=BODY)
    assert res.status_code == 500
    assert res.json()["error"]["code"] == "INTERNAL_ERROR"


def test_missing_api_keys_is_llm_error(client_with):
    from app.llm import build_llm

    settings = replace(load_settings(), openai_api_key=None, gemini_api_key=None)
    res = client_with(build_llm(settings)).post("/extract", json=BODY)
    assert res.status_code == 502
    assert res.json()["error"]["code"] == "LLM_ERROR"


# ---- 같은 활동 제안 sameAsActivityId (S-44) ----

EXISTING = [
    {
        "id": 7,
        "name": "바꿔요 — 캠퍼스 중고거래 앱",
        "period": {"start": "2025-03", "end": "2025-06", "ongoing": False},
    },
    {
        "id": 8,
        "name": "알고리즘 스터디",
        "period": {"start": "2024-09", "end": None, "ongoing": True},
    },
]


class SpyLLM(FakeLLM):
    """받은 사용자 메시지 · 스키마를 남긴다."""

    async def generate_json(self, system, user, schema, deadline):
        self.system, self.user, self.schema = system, user, schema
        return await super().generate_json(system, user, schema, deadline)


def _same_schema(schema):
    return schema["properties"]["activities"]["items"]["properties"]["sameAsActivityId"]


def test_no_existing_list_means_null_only(client_with):
    # existingActivities 를 빼면 [] 와 같다 — 스키마가 null 만 허락한다
    llm = SpyLLM(GOOD)
    res = client_with(llm).post("/extract", json=BODY)
    assert res.status_code == 200
    assert [a["sameAsActivityId"] for a in res.json()["activities"]] == [None, None]
    assert _same_schema(llm.schema) == {"type": "null"}
    assert "<기존 활동>\n(없음)\n</기존 활동>" in llm.user


def test_same_as_existing_activity(client_with):
    out = _bad(lambda d: d["activities"][0].update(sameAsActivityId=7))
    llm = SpyLLM(out)
    res = client_with(llm).post("/extract", json={**BODY, "existingActivities": EXISTING})
    assert res.status_code == 200
    assert [a["sameAsActivityId"] for a in res.json()["activities"]] == [7, None]
    # 목록 id 와 null 만 — Structured Outputs enum
    assert _same_schema(llm.schema) == {
        "anyOf": [{"type": "integer", "enum": [7, 8]}, {"type": "null"}]
    }
    # 이름 · 기간만 싣는다
    assert "[ID 7] 바꿔요 — 캠퍼스 중고거래 앱 (2025-03 ~ 2025-06)" in llm.user
    assert "[ID 8] 알고리즘 스터디 (2024-09 ~ 진행 중)" in llm.user


def test_two_new_activities_may_point_to_same_id(client_with):
    out = _bad(lambda d: [a.update(sameAsActivityId=8) for a in d["activities"]])
    res = client_with(SpyLLM(out)).post("/extract", json={**BODY, "existingActivities": EXISTING})
    assert res.status_code == 200
    assert [a["sameAsActivityId"] for a in res.json()["activities"]] == [8, 8]


def test_period_less_existing_activity_is_listed():
    from app.extract import build_user_message
    from app.schemas import ExistingActivity

    msg = build_user_message("a.md", "본문", [ExistingActivity(id=3, name="멋사")])
    assert "[ID 3] 멋사 (기간 모름)" in msg


@pytest.mark.parametrize(
    ("existing", "same"),
    [
        (EXISTING, 99),  # 목록 밖 id
        ([], 7),  # 목록이 비었는데 id
    ],
)
def test_unknown_id_retries_then_errors(client_with, existing, same):
    # 목록 밖 id 를 null 로 바꿔 「같은 활동 없음」처럼 돌려주지 않는다
    out = _bad(lambda d: d["activities"][0].update(sameAsActivityId=same))
    llm = FakeLLM(out, out)
    res = client_with(llm, max_attempts=2).post(
        "/extract", json={**BODY, "existingActivities": existing}
    )
    assert res.status_code == 502
    assert res.json()["error"]["code"] == "LLM_INVALID_OUTPUT"
    assert llm.calls == 2


def test_missing_same_as_field_is_invalid(client_with):
    out = _bad(lambda d: d["activities"][0].pop("sameAsActivityId"))
    llm = FakeLLM(out, out)
    res = client_with(llm, max_attempts=2).post("/extract", json=BODY)
    assert res.status_code == 502
    assert res.json()["error"]["code"] == "LLM_INVALID_OUTPUT"


def test_activity_category_in_response(client_with):
    # 경험 0개 활동도 분류가 붙고, 없으면 null 이 늘 실린다(S-57)
    out = _bad(lambda d: d["activities"][1].update(category="CLUB"))
    res = client_with(FakeLLM(out)).post("/extract", json=BODY)
    assert res.status_code == 200
    assert [a["category"] for a in res.json()["activities"]] == ["PROJECT", "CLUB"]

    res = client_with(FakeLLM(GOOD)).post("/extract", json=BODY)
    assert [a["category"] for a in res.json()["activities"]] == ["PROJECT", None]


@pytest.mark.parametrize(
    "change",
    [
        lambda a: a.update(category="COMPETITION"),  # 목록 밖 값
        lambda a: a.update(category=""),
        lambda a: a.pop("category"),  # 칸이 빠짐
    ],
)
def test_bad_category_retries_then_errors(client_with, change):
    # 목록 밖 값을 null 로 바꿔 「미분류」처럼 돌려주지 않는다
    out = _bad(lambda d: change(d["activities"][0]))
    llm = FakeLLM(out, out)
    res = client_with(llm, max_attempts=2).post("/extract", json=BODY)
    assert res.status_code == 502
    assert res.json()["error"]["code"] == "LLM_INVALID_OUTPUT"
    assert llm.calls == 2


def test_bad_category_then_good_retry(client_with):
    llm = FakeLLM(_bad(lambda d: d["activities"][0].update(category="ETC")), GOOD)
    res = client_with(llm, max_attempts=2).post("/extract", json=BODY)
    assert res.status_code == 200
    assert res.json()["activities"][0]["category"] == "PROJECT"
    assert llm.calls == 2


def test_category_meanings_match_contract_list():
    from app.extract import ACTIVITY_CATEGORY_MEANINGS, LLM_JSON_SCHEMA, SYSTEM_PROMPT
    from app.schemas import ACTIVITY_CATEGORIES

    assert ACTIVITY_CATEGORIES == ("PROJECT", "STUDY", "CLUB")
    assert tuple(ACTIVITY_CATEGORY_MEANINGS) == ACTIVITY_CATEGORIES
    assert all(f"- {value}:" in SYSTEM_PROMPT for value in ACTIVITY_CATEGORIES)
    assert "{category_lines}" not in SYSTEM_PROMPT
    item = LLM_JSON_SCHEMA["properties"]["activities"]["items"]
    assert item["properties"]["category"] == {
        "anyOf": [{"type": "string", "enum": ["PROJECT", "STUDY", "CLUB"]}, {"type": "null"}]
    }
    assert "category" in item["required"]
    # 범위 — 공모전 · 대회 · 아르바이트 / 자격증 · 어학 / 봉사 · 멘토링
    meanings = ACTIVITY_CATEGORY_MEANINGS
    assert all(w in meanings["PROJECT"] for w in ("공모전", "대회", "아르바이트"))
    assert all(w in meanings["STUDY"] for w in ("자격증", "어학"))
    assert all(w in meanings["CLUB"] for w in ("봉사", "멘토링"))


# ---- 활동에 자료 더하기 targetActivityId (S-60) ----


@pytest.mark.parametrize(
    ("existing", "target"),
    [
        (EXISTING, 99),  # 목록 밖
        ([], 7),  # 목록이 비었는데 대상
    ],
)
def test_target_outside_existing_is_400_without_calling_llm(client_with, existing, target):
    llm = FakeLLM()
    res = client_with(llm).post(
        "/extract", json={**BODY, "existingActivities": existing, "targetActivityId": target}
    )
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_REQUEST"
    assert llm.calls == 0


def test_target_reaches_prompt_and_id_comes_back(client_with):
    out = _bad(lambda d: d["activities"][0].update(sameAsActivityId=7))
    llm = SpyLLM(out)
    res = client_with(llm).post(
        "/extract", json={**BODY, "existingActivities": EXISTING, "targetActivityId": 7}
    )
    assert res.status_code == 200
    assert [a["sameAsActivityId"] for a in res.json()["activities"]] == [7, None]
    line = "[ID 7] 바꿔요 — 캠퍼스 중고거래 앱 (2025-03 ~ 2025-06)"
    assert f"<대상 활동>\n{line}\n</대상 활동>" in llm.user
    assert TARGET_RULE in llm.system


@pytest.mark.parametrize("body", [{}, {"targetActivityId": None}], ids=["빼기", "null"])
def test_no_target_keeps_prompt_exactly(client_with, body):
    # 대상이 없으면 S-44 채점이 그대로 유효하다 — 시스템 · 사용자 메시지가 지금과 한 글자도 같다
    llm = SpyLLM(GOOD)
    res = client_with(llm).post("/extract", json={**BODY, "existingActivities": EXISTING, **body})
    assert res.status_code == 200
    assert llm.system == SYSTEM_PROMPT
    existing = [ExistingActivity.model_validate(a) for a in EXISTING]
    assert llm.user == build_user_message(BODY["sourceName"], BODY["text"], existing)
    assert "대상 활동" not in llm.user


# ---- 작업 로그 묶기는 GitHub 자료일 때만(AI#113) ----

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def test_base_prompt_has_no_work_log_rule():
    # 늘 붙는 본문은 AI#109 전과 같다 — 이력 목록 · 회고로 번지지 않게
    assert "작업 로그" not in SYSTEM_PROMPT
    assert "커밋" not in SYSTEM_PROMPT


@pytest.mark.parametrize(
    "text,want",
    [
        (
            "## 내가 쓴 커밋 제목 (GitHub 아이디 kim-dev · 최근 44개)\n- 2026-01-20 readme 수정",
            True,
        ),
        ("# r\n\n## 내가 연 PR 제목 (3개)\n- 2026-01-10 #12 로그인 (merged)", True),
        ("커밋 메모: 내가 쓴 커밋 제목을 정리했다", False),  # 절 제목이 아니다
        (
            "## 커밋 메모\n- 2025-03-02 로그인 수정",
            False,
        ),  # 붙여 넣은 커밋 메모(commit-memo) — AI#109 전과 같다
        ("- 2024.12 SQLD 자격증 취득\n- 2023.08 정보처리기사", False),
    ],
)
def test_is_work_log(text, want):
    assert is_work_log(text) is want


@pytest.mark.parametrize(
    "sample,want",
    [("github-commits", True), ("commit-memo", False), ("category-list", False)],
)
def test_samples_detect_as_expected(sample, want):
    assert is_work_log((SAMPLES / f"{sample}.md").read_text(encoding="utf-8")) is want


def test_github_source_gets_work_log_rule_at_end(client_with):
    llm = SpyLLM(GOOD)
    text = (SAMPLES / "github-commits.md").read_text(encoding="utf-8")
    res = client_with(llm).post("/extract", json={"sourceName": "recipe-box", "text": text})
    assert res.status_code == 200
    # 앞부분이 늘 같다 — 프롬프트 캐시가 산다
    assert llm.system == SYSTEM_PROMPT + WORK_LOG_RULE


def test_github_source_with_target_keeps_target_rule_first(client_with):
    llm = SpyLLM(GOOD)
    text = (SAMPLES / "github-commits.md").read_text(encoding="utf-8")
    body = {"sourceName": "recipe-box", "text": text, "existingActivities": EXISTING}
    res = client_with(llm).post("/extract", json={**body, "targetActivityId": 7})
    assert res.status_code == 200
    assert llm.system == SYSTEM_PROMPT + TARGET_RULE + WORK_LOG_RULE


def test_non_github_source_keeps_prompt_exactly(client_with):
    llm = SpyLLM(GOOD)
    text = (SAMPLES / "commit-memo.md").read_text(encoding="utf-8")
    res = client_with(llm).post("/extract", json={"sourceName": "메모.md", "text": text})
    assert res.status_code == 200
    assert llm.system == SYSTEM_PROMPT
