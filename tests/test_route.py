"""/route — LLM 은 가짜로 바꿔 계약 모양 · 응답 규칙 · 오류 처리만 본다(네트워크 없음).

요청 · 응답 예시는 ETC docs/contracts/ai-api.md 「POST /route」 그대로다.
"""

import time

import pytest

from app.errors import LLMError, LLMTimeout
from tests.conftest import FakeLLM

TEXT = "팀 프로젝트에서 맡은 역할과, 팀과 함께 일한 방식을 적어 주세요."
BODY = {"text": TEXT, "totalLength": 35}
GOOD = {"reason": "지원서가 묻는 문항 문장이다", "kind": "APPLICATION"}


def test_route_contract_shape(client_with):
    llm = FakeLLM(GOOD)
    res = client_with(llm).post("/route", json=BODY)
    assert res.status_code == 200
    assert res.json() == {
        "kind": "APPLICATION",
        "reason": "지원서가 묻는 문항 문장이다",
        "model": "fake-model",
    }
    assert llm.calls == 1


@pytest.mark.parametrize("kind", ["SOURCE", "APPLICATION", "ASK", "UNSURE"])
def test_every_entrance_kind(client_with, kind):
    res = client_with(FakeLLM({"reason": "단서", "kind": kind})).post("/route", json=BODY)
    assert res.status_code == 200
    assert res.json()["kind"] == kind


def test_prompt_shows_truncation(client_with):
    # 긴 글은 앞 4000자만 온다 — 전체 길이를 단서로 싣는다
    seen = {}

    class Spy(FakeLLM):
        async def generate_json(self, system, user, schema, deadline):
            seen["user"] = user
            return await super().generate_json(system, user, schema, deadline)

    text = "가" * 4000
    res = client_with(Spy(GOOD)).post("/route", json={"text": text, "totalLength": 12000})
    assert res.status_code == 200
    assert "전체 12000자 중 앞 4000자" in seen["user"]
    assert text in seen["user"]


@pytest.mark.parametrize(
    "body",
    [
        {"text": "", "totalLength": 0},
        {"text": "  \n ", "totalLength": 4},
        {
            "text": "가" * 4001,
            "totalLength": 4001,
        },  # BE 가 4000자로 자른다 — 413 이 아니라 400(계약)
        {"text": TEXT},
        {"totalLength": 35},
        {"text": None, "totalLength": 35},
    ],
)
def test_invalid_request(client_with, body):
    llm = FakeLLM(GOOD)
    res = client_with(llm).post("/route", json=body)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_REQUEST"
    assert llm.calls == 0


def test_total_length_is_only_a_hint(client_with):
    # 계약 예시는 37자 글에 totalLength 35 — 세는 방식이 조금 달라도 거절하지 않는다
    assert len(TEXT) != BODY["totalLength"]
    res = client_with(FakeLLM(GOOD)).post("/route", json={"text": TEXT, "totalLength": 3})
    assert res.status_code == 200


def test_4000_chars_is_ok(client_with):
    res = client_with(FakeLLM(GOOD)).post("/route", json={"text": "가" * 4000, "totalLength": 4000})
    assert res.status_code == 200


@pytest.mark.parametrize(
    "bad",
    [
        {"reason": "단서", "kind": "QUESTION"},  # 목록 밖
        {"reason": "단서", "kind": "source"},
        {"reason": "", "kind": "SOURCE"},
        {"reason": "   ", "kind": "SOURCE"},
        {"reason": "가" * 201, "kind": "SOURCE"},  # 200자 넘음 — 잘라 내지 않는다
        {"kind": "SOURCE"},
        {"reason": "단서"},
        "아마 SOURCE 같습니다",
        "",
    ],
)
def test_invalid_output_retries_then_fails(client_with, bad):
    llm = FakeLLM(bad, bad)
    res = client_with(llm, max_attempts=2).post("/route", json=BODY)
    assert res.status_code == 502
    assert res.json()["error"]["code"] == "LLM_INVALID_OUTPUT"
    assert llm.calls == 2


def test_invalid_then_valid(client_with):
    llm = FakeLLM({"reason": "단서", "kind": "MAYBE"}, GOOD)
    res = client_with(llm, max_attempts=2).post("/route", json=BODY)
    assert res.status_code == 200
    assert res.json()["kind"] == "APPLICATION"
    assert llm.calls == 2


def test_reason_200_chars_is_ok(client_with):
    res = client_with(FakeLLM({"reason": "가" * 200, "kind": "SOURCE"})).post("/route", json=BODY)
    assert res.status_code == 200


def test_sync_deadline(client_with):
    # 입구 셋은 동기 호출 — 작업 호출(100초)이 아니라 LLM_SYNC_TIMEOUT_SECONDS 안에 끝내야 한다(계약)
    seen = {}

    class Spy(FakeLLM):
        async def generate_json(self, system, user, schema, deadline):
            seen["left"] = deadline - time.monotonic()
            return await super().generate_json(system, user, schema, deadline)

    res = client_with(Spy(GOOD), llm_timeout_seconds=100, llm_sync_timeout_seconds=50).post(
        "/route", json=BODY
    )
    assert res.status_code == 200
    assert 45 < seen["left"] <= 50


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (LLMTimeout("시간 초과"), 504, "LLM_TIMEOUT"),
        (LLMError("호출 실패"), 502, "LLM_ERROR"),
    ],
)
def test_llm_failures(client_with, error, status, code):
    res = client_with(FakeLLM(error)).post("/route", json=BODY)
    assert res.status_code == status
    assert res.json()["error"]["code"] == code


def test_route_llm_uses_sync_call_timeout():
    # 동기 호출 — 모델 하나가 멈춰도 50초 안에 다음 모델로 넘어가게 한 번 제한을 짧게(작업 호출 45초가 아니라)
    from app.main import get_route_llm, get_settings

    settings = get_settings()
    llm = get_route_llm.__wrapped__()
    assert llm.models == settings.route_models
    assert llm.call_timeout == settings.llm_sync_call_timeout_seconds
    assert llm.call_timeout < settings.llm_sync_timeout_seconds / 2
