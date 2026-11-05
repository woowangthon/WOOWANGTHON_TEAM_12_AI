"""/parse-application — LLM 은 가짜로 바꿔 계약 모양 · 응답 규칙 · 오류 처리만 본다(네트워크 없음).

요청 · 응답 예시는 ETC docs/contracts/ai-api.md 「POST /parse-application」 그대로다.
"""

import time

import pytest

from app.errors import LLMError, LLMTimeout
from app.parse_application import MAX_APPLICATION_CHARS
from tests.conftest import FakeLLM

TEXT = (
    "[OO 백엔드 부트캠프 4기 모집]\n모집 분야: 백엔드(Spring)\n\n자기소개서 문항\n"
    "1. 기술적인 문제를 해결한 경험을 구체적으로 적어 주세요. (500자)\n"
    "2. 지원 동기와 수료 후 목표를 적어 주세요. (500자)"
)
BODY = {"text": TEXT}
Q1 = "기술적인 문제를 해결한 경험을 구체적으로 적어 주세요. (500자)"
Q2 = "지원 동기와 수료 후 목표를 적어 주세요. (500자)"
GOOD = {
    "organization": "OO 백엔드 부트캠프 4기",
    "organizationType": "BOOTCAMP",
    "jobRole": "BACKEND",
    "questions": [Q1, Q2],
}


def post(client_with, *outputs, body=None, **settings):
    llm = FakeLLM(*outputs)
    return llm, client_with(llm, **settings).post("/parse-application", json=body or BODY)


def test_contract_shape(client_with):
    _, res = post(client_with, GOOD)
    assert res.status_code == 200
    assert res.json() == {**GOOD, "model": "fake-model"}


def test_nothing_found_is_nulls_and_empty(client_with):
    empty = {"organization": None, "organizationType": None, "jobRole": None, "questions": []}
    _, res = post(client_with, empty)
    assert res.status_code == 200
    assert res.json() == {**empty, "model": "fake-model"}


def test_whitespace_differences_are_still_written(client_with):
    # 계약 「적혀 있나」 — 줄바꿈 · 연속 공백 · 줄 끝 공백은 같은 글로 본다. 돌려줄 땐 LLM 이 쓴 그대로다
    text = "문항\n지원 동기를\n  적어   주세요. (500자)\n"
    written = "지원 동기를 적어 주세요. (500자)"
    out = {**GOOD, "organization": None, "questions": [written]}
    _, res = post(client_with, out, body={"text": text})
    assert res.status_code == 200
    assert res.json()["questions"] == [written]


def test_number_may_be_left_out(client_with):
    out = {**GOOD, "questions": [f"1. {Q1}", Q2]}
    _, res = post(client_with, out)
    assert res.status_code == 200


def test_repeated_question_in_text_is_ok_once(client_with):
    # 글에 두 번 나오는 문항도 순서를 맞출 수 있다 — 안내에 먼저 나온 뒤 본문에 다시 나오는 공고
    text = "협업 경험을 적어 주세요.\n지원 동기를 적어 주세요.\n협업 경험을 적어 주세요."
    out = {**GOOD, "organization": None, "questions": ["지원 동기를 적어 주세요."]}
    _, res = post(client_with, out, body={"text": text})
    assert res.status_code == 200
    out = {**out, "questions": ["지원 동기를 적어 주세요.", "협업 경험을 적어 주세요."]}
    _, res = post(client_with, out, body={"text": text})
    assert res.status_code == 200


@pytest.mark.parametrize(
    "bad",
    [
        # 글에 없는 것 — 지어내거나 다듬으면 위반
        {**GOOD, "organization": "OO 부트캠프"},
        {**GOOD, "questions": [Q1, "수료 후 목표를 적어 주십시오."]},
        {**GOOD, "questions": ["기술적 문제 해결 경험을 적어 주세요."]},
        {**GOOD, "questions": [Q1, "리더십을 발휘한 경험을 적어 주세요."]},
        # 순서 · 중복
        {**GOOD, "questions": [Q2, Q1]},
        {**GOOD, "questions": [Q1, Q1]},
        {**GOOD, "questions": [Q1, Q1.replace(" ", "  ")]},
        # 목록 밖 · 길이 · 빈 값
        {**GOOD, "organizationType": "COMPANY"},
        {**GOOD, "jobRole": "서버"},
        {**GOOD, "organization": ""},
        {**GOOD, "organization": "   "},
        {**GOOD, "questions": [""]},
        {**GOOD, "questions": "기술적인 문제"},
        # 모양
        {k: v for k, v in GOOD.items() if k != "questions"},
        {k: v for k, v in GOOD.items() if k != "organization"},
        "지원처는 OO 부트캠프입니다",
        "",
    ],
)
def test_invalid_output_retries_then_fails(client_with, bad):
    llm, res = post(client_with, bad, bad, max_attempts=2)
    assert res.status_code == 502
    assert res.json()["error"]["code"] == "LLM_INVALID_OUTPUT"
    assert llm.calls == 2


def test_invalid_error_does_not_carry_user_text(client_with):
    bad = {**GOOD, "questions": ["리더십을 발휘한 경험을 적어 주세요."]}
    _, res = post(client_with, bad, bad, max_attempts=2)
    assert "리더십" not in res.json()["error"]["message"]


def test_invalid_then_valid(client_with):
    llm, res = post(client_with, {**GOOD, "organization": "지어낸 이름"}, GOOD, max_attempts=2)
    assert res.status_code == 200
    assert res.json()["organization"] == GOOD["organization"]
    assert llm.calls == 2


def test_limits(client_with):
    questions = [f"문항 {i} 을 적어 주세요." for i in range(11)]
    text = "\n".join(questions)
    empty = {"organization": None, "organizationType": None, "jobRole": None}
    _, res = post(client_with, {**empty, "questions": questions[:10]}, body={"text": text})
    assert res.status_code == 200
    bad = {**empty, "questions": questions}
    _, res = post(client_with, bad, bad, body={"text": text}, max_attempts=2)
    assert res.json()["error"]["code"] == "LLM_INVALID_OUTPUT"

    name = "가" * 100
    _, res = post(
        client_with, {**empty, "organization": name, "questions": []}, body={"text": name}
    )
    assert res.status_code == 200
    long_name = "가" * 101
    bad = {**empty, "organization": long_name, "questions": []}
    _, res = post(client_with, bad, bad, body={"text": long_name}, max_attempts=2)
    assert res.json()["error"]["code"] == "LLM_INVALID_OUTPUT"

    q = "가" * 5000
    _, res = post(client_with, {**empty, "questions": [q]}, body={"text": q})
    assert res.status_code == 200
    q = "가" * 5001
    bad = {**empty, "questions": [q]}
    _, res = post(client_with, bad, bad, body={"text": q}, max_attempts=2)
    assert res.json()["error"]["code"] == "LLM_INVALID_OUTPUT"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"text": ""},
        {"text": " \n\t "},
        {"text": TEXT, "url": "https://example.com/1"},
        {"text": None, "url": None},
        {"text": 3},
    ],
)
def test_bad_request_does_not_call_llm(client_with, body):
    llm = FakeLLM()
    res = client_with(llm).post("/parse-application", json=body)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_REQUEST"
    assert llm.calls == 0


def test_url_is_refused_until_ai39(client_with):
    # 정해진 결과로 채우지 않는다 — 링크 받기가 생길 때까지 오류다(AI#39)
    llm = FakeLLM()
    res = client_with(llm).post("/parse-application", json={"url": "https://example.com/1"})
    assert res.status_code == 400
    assert llm.calls == 0


def test_text_length_limit(client_with):
    empty = {"organization": None, "organizationType": None, "jobRole": None, "questions": []}
    _, res = post(client_with, empty, body={"text": "가" * MAX_APPLICATION_CHARS})
    assert res.status_code == 200
    llm = FakeLLM()
    res = client_with(llm).post(
        "/parse-application", json={"text": "가" * (MAX_APPLICATION_CHARS + 1)}
    )
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "TEXT_TOO_LONG"
    assert llm.calls == 0


def test_sync_deadline(client_with):
    seen = {}

    class Spy(FakeLLM):
        async def generate_json(self, system, user, schema, deadline):
            seen["left"] = deadline - time.monotonic()
            return await super().generate_json(system, user, schema, deadline)

    res = client_with(Spy(GOOD), llm_timeout_seconds=100, llm_sync_timeout_seconds=50).post(
        "/parse-application", json=BODY
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
    _, res = post(client_with, error)
    assert res.status_code == status
    assert res.json()["error"]["code"] == code
