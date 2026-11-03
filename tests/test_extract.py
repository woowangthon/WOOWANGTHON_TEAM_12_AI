"""/extract · /health — LLM 은 가짜로 바꿔 계약 모양 · 오류 처리만 본다(네트워크 없음)."""

import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.config import load_settings
from app.errors import LLMError, LLMTimeout
from app.main import app
from tests.conftest import FakeLLM

GOOD = {
    "activities": [
        {
            "name": "바꿔요",
            "period": {"start": "2025-03", "end": "2025-06", "ongoing": False},
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
