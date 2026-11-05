"""/guidelines — LLM 은 가짜로 바꿔 계약 모양 · 개수 · 길이 · 오류 처리만 본다(네트워크 없음).

요청 · 응답 예시는 ETC docs/contracts/ai-api.md 「POST /guidelines」 그대로다.
"""

import time

import pytest

from app.errors import LLMError
from tests.conftest import FakeLLM

BODY = {
    "text": "자소서는 결론부터. 첫 문장에 뭘 했고 어떻게 됐는지 박아라. 형용사로 꾸미지 말고 행동을 써라. "
    "팀 프로젝트면 내 몫을 분명히. 문단 하나에 이야기 하나만. 솔직히 없는 경험도 좀 부풀려야 붙는다…"
}
GOOD = {
    "guidelines": [
        {"text": "첫 문장에 무엇을 했고 어떻게 됐는지 먼저 쓴다"},
        {"text": "형용사로 꾸미지 않고 한 행동을 쓴다"},
        {"text": "팀 프로젝트는 내가 맡은 몫을 분명히 쓴다"},
        {"text": "문단 하나에 이야기 하나만 쓴다"},
    ]
}


def test_contract_example(client_with):
    llm = FakeLLM(GOOD)
    r = client_with(llm).post("/guidelines", json=BODY)
    assert r.status_code == 200
    assert r.json() == {**GOOD, "model": "fake-model"}
    assert llm.calls == 1


def test_zero_lines_is_valid(client_with):
    r = client_with(FakeLLM({"guidelines": []})).post("/guidelines", json={"text": "합격했어요!"})
    assert r.status_code == 200
    assert r.json()["guidelines"] == []


@pytest.mark.parametrize(
    "lines",
    [
        pytest.param([{"text": f"지침 {i}"} for i in range(11)], id="11줄"),
        pytest.param([{"text": " "}], id="빈 줄"),
        pytest.param([{"text": "가" * 101}], id="100자 넘음"),
        pytest.param([{"text": "첫 줄\n둘째 줄"}], id="줄바꿈"),
    ],
)
def test_bad_output_retries_then_fails(client_with, lines):
    bad = {"guidelines": lines}
    llm = FakeLLM(bad, bad)
    r = client_with(llm, max_attempts=2).post("/guidelines", json=BODY)
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "LLM_INVALID_OUTPUT"
    assert llm.calls == 2


def test_bad_output_then_good(client_with):
    llm = FakeLLM({"guidelines": [{"text": " "}]}, GOOD)
    r = client_with(llm, max_attempts=2).post("/guidelines", json=BODY)
    assert r.status_code == 200
    assert llm.calls == 2


@pytest.mark.parametrize(
    "body, status, code",
    [
        pytest.param({}, 400, "INVALID_REQUEST", id="text 없음"),
        pytest.param({"text": "   \n"}, 400, "INVALID_REQUEST", id="공백뿐"),
        pytest.param({"text": "가" * 20001}, 413, "TEXT_TOO_LONG", id="20000자 넘음"),
    ],
)
def test_bad_request(client_with, body, status, code):
    llm = FakeLLM()
    r = client_with(llm).post("/guidelines", json=body)
    assert r.status_code == status
    assert r.json()["error"]["code"] == code
    assert llm.calls == 0


def test_20000_chars_is_ok(client_with):
    r = client_with(FakeLLM(GOOD)).post("/guidelines", json={"text": "가" * 20000})
    assert r.status_code == 200


def test_llm_error(client_with):
    r = client_with(FakeLLM(LLMError("끊김"))).post("/guidelines", json=BODY)
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "LLM_ERROR"


def test_sync_deadline(client_with):
    # 작업 호출(100초)이 아니라 동기 호출 50초 안에 끝낸다(계약 「오류」 504)
    seen = {}

    class Spy(FakeLLM):
        async def generate_json(self, system, user, schema, deadline):
            seen["left"] = deadline - time.monotonic()
            return await super().generate_json(system, user, schema, deadline)

    r = client_with(Spy(GOOD), llm_timeout_seconds=100, llm_sync_timeout_seconds=50).post(
        "/guidelines", json=BODY
    )
    assert r.status_code == 200
    assert 45 < seen["left"] <= 50
