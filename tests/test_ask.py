"""/ask — LLM 은 가짜로 바꿔 계약 모양 · 응답 규칙 · 오류 처리만 본다(네트워크 없음).

요청 · 응답 예시는 ETC docs/contracts/ai-api.md 「POST /ask」 그대로다.
"""

import pytest

from app.errors import LLMError, LLMTimeout
from tests.conftest import FakeLLM
from tests.test_match import EXPERIENCES

QUERY = "동아리에서 의견 모아 본 적 있었나?"
BODY = {"query": QUERY, "experiences": EXPERIENCES}
REASON = "팀이 매주 회고로 의견을 모은 자리라 질문과 맞는다. 회고를 내가 이끌었는지는 자료에 없다"
GOOD = {
    "recommendations": [
        {"experienceId": 102, "reason": REASON, "evidence": [{"type": "CITATION", "id": 202}]}
    ]
}


def post(client_with, *outputs, body=None, **settings):
    llm = FakeLLM(*outputs)
    return llm, client_with(llm, **settings).post("/ask", json=body or BODY)


def code(res):
    return res.json()["error"]["code"]


def _rec(exp_id, evidence=(), reason="질문과 맞는다"):
    return {
        "experienceId": exp_id,
        "reason": reason,
        "evidence": [{"type": t, "id": i} for t, i in evidence],
    }


def test_contract_shape(client_with):
    _, res = post(client_with, GOOD)
    assert res.status_code == 200
    assert res.json() == {**GOOD, "model": "fake-model"}


def test_zero_recommendations_is_normal(client_with):
    _, res = post(client_with, {"recommendations": []})
    assert res.status_code == 200
    assert res.json() == {"recommendations": [], "model": "fake-model"}


def test_three_with_supplement_ok(client_with):
    out = {
        "recommendations": [
            _rec(101, [("CITATION", 201), ("SUPPLEMENT", 301)]),
            _rec(102, [("CITATION", 202)]),
            _rec(103),
        ]
    }
    _, res = post(client_with, out)
    assert res.status_code == 200
    assert [r["experienceId"] for r in res.json()["recommendations"]] == [101, 102, 103]


def test_prompt_gets_query_and_every_experience(client_with):
    seen = {}

    class Spy(FakeLLM):
        async def generate_json(self, system, user, schema, deadline):
            seen["user"] = user
            return await super().generate_json(system, user, schema, deadline)

    client_with(Spy(GOOD)).post("/ask", json=BODY)
    for text in (QUERY, "[경험 101]", "[경험 103]", "보완 301", "근거 202", "수행 주체: TEAM"):
        assert text in seen["user"]
    assert (
        "COLLABORATION" not in seen["user"]
    )  # 판정은 본문 · 근거 · 보완으로 — 태그는 넘기지 않는다


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param({"recommendations": [_rec(999)]}, id="요청에 없는 경험"),
        pytest.param({"recommendations": [_rec(101), _rec(101)]}, id="같은 경험 두 번"),
        pytest.param({"recommendations": [_rec(102, [("CITATION", 201)])]}, id="다른 경험의 근거"),
        pytest.param({"recommendations": [_rec(102, [("SUPPLEMENT", 999)])]}, id="없는 보완"),
        pytest.param(
            {"recommendations": [_rec(102, [("CITATION", 202), ("CITATION", 202)])]},
            id="근거 중복",
        ),
        pytest.param({"recommendations": [_rec(101), _rec(102), _rec(103), _rec(101)]}, id="4개"),
        pytest.param({"recommendations": [_rec(101, reason="  ")]}, id="빈 이유"),
        pytest.param("not json", id="JSON 아님"),
    ],
)
def test_breaking_the_rules_retries_then_fails(client_with, bad):
    llm, res = post(client_with, bad, bad)
    assert res.status_code == 502
    assert code(res) == "LLM_INVALID_OUTPUT"
    assert llm.calls == 2


def test_retry_can_recover(client_with):
    llm, res = post(client_with, {"recommendations": [_rec(999)]}, GOOD)
    assert res.status_code == 200
    assert llm.calls == 2


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"experiences": EXPERIENCES}, id="query 없음"),
        pytest.param({"query": "  ", "experiences": EXPERIENCES}, id="query 빔"),
        pytest.param({"query": QUERY}, id="experiences 없음"),
        pytest.param({"query": QUERY, "experiences": []}, id="experiences 0개"),
        pytest.param({"query": QUERY, "experiences": EXPERIENCES * 2}, id="id 중복"),
    ],
)
def test_invalid_request_does_not_call_llm(client_with, body):
    llm, res = post(client_with, GOOD, body=body)
    assert res.status_code == 400
    assert code(res) == "INVALID_REQUEST"
    assert llm.calls == 0


def test_too_long_is_413(client_with):
    llm, res = post(client_with, GOOD, max_match_chars=10)
    assert res.status_code == 413
    assert code(res) == "TEXT_TOO_LONG"
    assert llm.calls == 0


@pytest.mark.parametrize(
    ("error", "status", "expected"),
    [
        pytest.param(LLMError("실패"), 502, "LLM_ERROR", id="LLM_ERROR"),
        pytest.param(LLMTimeout("시간 초과"), 504, "LLM_TIMEOUT", id="LLM_TIMEOUT"),
    ],
)
def test_llm_failure_is_not_covered_up(client_with, error, status, expected):
    _, res = post(client_with, error)
    assert res.status_code == status
    assert code(res) == expected


def test_uses_sync_timeout(client_with, monkeypatch):
    seen = {}

    class Spy(FakeLLM):
        async def generate_json(self, system, user, schema, deadline):
            seen["deadline"] = deadline
            return await super().generate_json(system, user, schema, deadline)

    monkeypatch.setattr("app.llm.time.monotonic", lambda: 1000.0)
    client_with(Spy(GOOD), llm_sync_timeout_seconds=50.0, llm_timeout_seconds=100.0).post(
        "/ask", json=BODY
    )
    assert seen["deadline"] == 1050.0
