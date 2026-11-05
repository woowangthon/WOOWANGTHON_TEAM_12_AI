"""/link-suggestions — LLM 은 가짜로 바꿔 계약 모양 · 응답 규칙 · 오류 처리만 본다(네트워크 없음).

요청 · 응답 예시는 ETC docs/contracts/ai-api.md 「POST /link-suggestions」 그대로다.
"""

import pytest

from app.errors import LLMError, LLMTimeout
from tests.conftest import FakeLLM

STUDY = {
    "id": 8,
    "name": "알고리즘 스터디",
    "period": {"start": "2024-09", "end": None, "ongoing": True},
    "note": "문제만 풀다 보니 직접 서비스를 만들어 보고 싶어졌다",
    "experiences": [{"id": 105, "content": "매주 풀 문제를 고른다"}],
}
BAKKWOYO = {
    "id": 7,
    "name": "바꿔요 — 캠퍼스 중고거래 앱",
    "period": {"start": "2025-03", "end": "2025-06", "ongoing": False},
    "note": None,
    "experiences": [
        {
            "id": 101,
            "content": "채팅 메시지가 늦게 도착하는 문제를 폴링에서 WebSocket 으로 바꿔 해결했다",
        },
        {"id": 102, "content": "매주 금요일 스프린트 회고를 했다"},
    ],
}
OTHER = {**BAKKWOYO, "id": 9, "name": "교내 축제 스태프", "experiences": []}
BODY = {"from": STUDY, "candidates": [BAKKWOYO, OTHER]}
REASON = "계기에 적은 「직접 서비스를 만들어 보고 싶어졌다」와 뒤에 시작한 「바꿔요」 앱 만들기가 닿아 보여요"
GOOD = {"suggestions": [{"toActivityId": 7, "reason": REASON}]}


def post(client_with, *outputs, body=None, **settings):
    llm = FakeLLM(*outputs)
    return llm, client_with(llm, **settings).post("/link-suggestions", json=body or BODY)


def code(res):
    return res.json()["error"]["code"]


def test_contract_shape(client_with):
    _, res = post(client_with, GOOD)
    assert res.status_code == 200
    assert res.json() == {**GOOD, "model": "fake-model"}


def test_no_suggestion_is_normal(client_with):
    _, res = post(client_with, {"suggestions": []})
    assert res.status_code == 200
    assert res.json() == {"suggestions": [], "model": "fake-model"}


def test_candidate_may_have_no_note_or_experiences(client_with):
    bare = {"id": 7, "name": "바꿔요"}
    _, res = post(client_with, GOOD, body={"from": STUDY, "candidates": [bare]})
    assert res.status_code == 200


def test_prompt_gets_note_and_experiences_of_every_activity(client_with):
    seen = {}

    class Spy(FakeLLM):
        async def generate_json(self, system, user, schema, deadline):
            seen["user"] = user
            return await super().generate_json(system, user, schema, deadline)

    client_with(Spy(GOOD)).post("/link-suggestions", json=BODY)
    for text in (
        STUDY["note"],
        "매주 풀 문제를 고른다",
        "후보 ID 7",
        "후보 ID 9",
        "2024-09 ~ 진행 중",
    ):
        assert text in seen["user"]


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param({"suggestions": [{"toActivityId": 99, "reason": REASON}]}, id="후보 밖"),
        pytest.param({"suggestions": [{"toActivityId": 8, "reason": REASON}]}, id="출발 활동"),
        pytest.param(
            {"suggestions": [{"toActivityId": 7, "reason": REASON}] * 2},
            id="중복",
        ),
        pytest.param(
            {"suggestions": [{"toActivityId": i, "reason": REASON} for i in (7, 9, 7, 9)]},
            id="4개",
        ),
        pytest.param({"suggestions": [{"toActivityId": 7, "reason": "  "}]}, id="빈 이유"),
        pytest.param({"suggestions": [{"toActivityId": 7, "reason": "가" * 201}]}, id="201자"),
        pytest.param({"suggestions": [{"toActivityId": 7, "reason": "한 줄\n두 줄"}]}, id="줄바꿈"),
        pytest.param("JSON 이 아님", id="JSON 아님"),
    ],
)
def test_breaking_the_rules_retries_then_fails(client_with, bad):
    llm, res = post(client_with, bad, bad)
    assert res.status_code == 502
    assert code(res) == "LLM_INVALID_OUTPUT"
    assert llm.calls == 2


def test_retry_can_recover(client_with):
    bad = {"suggestions": [{"toActivityId": 99, "reason": REASON}]}
    llm, res = post(client_with, bad, GOOD)
    assert res.status_code == 200
    assert res.json()["suggestions"] == GOOD["suggestions"]
    assert llm.calls == 2


def test_reason_of_200_chars_is_allowed(client_with):
    _, res = post(client_with, {"suggestions": [{"toActivityId": 7, "reason": "가" * 200}]})
    assert res.status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"candidates": [BAKKWOYO]}, id="from 없음"),
        pytest.param({"from": STUDY}, id="candidates 없음"),
        pytest.param({"from": {**STUDY, "note": None}, "candidates": [BAKKWOYO]}, id="계기 없음"),
        pytest.param({"from": {**STUDY, "note": "  "}, "candidates": [BAKKWOYO]}, id="계기 비었음"),
        pytest.param({"from": STUDY, "candidates": []}, id="후보 0개"),
        pytest.param(
            {"from": STUDY, "candidates": [{**BAKKWOYO, "id": i} for i in range(100, 121)]},
            id="후보 21개",
        ),
        pytest.param({"from": STUDY, "candidates": [BAKKWOYO, BAKKWOYO]}, id="id 중복"),
        pytest.param({"from": STUDY, "candidates": [{**BAKKWOYO, "id": 8}]}, id="출발이 후보에"),
    ],
)
def test_invalid_request_does_not_call_llm(client_with, body):
    llm, res = post(client_with, GOOD, body=body)
    assert res.status_code == 400
    assert code(res) == "INVALID_REQUEST"
    assert llm.calls == 0


def test_twenty_candidates_are_allowed(client_with):
    body = {"from": STUDY, "candidates": [{**BAKKWOYO, "id": i} for i in range(100, 120)]}
    _, res = post(client_with, {"suggestions": []}, body=body)
    assert res.status_code == 200


def test_too_long_is_413(client_with):
    big = {**BAKKWOYO, "experiences": [{"id": 1, "content": "가" * 101}]}
    llm, res = post(
        client_with, GOOD, body={"from": STUDY, "candidates": [big]}, max_match_chars=100
    )
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
        "/link-suggestions", json=BODY
    )
    assert seen["deadline"] == 1050.0
