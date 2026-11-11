"""S-91 공고 요건 줄 — 원문 보존 · 순서 · 한도 · 재시도 · 서버가 정하는 출처."""

import pytest

from tests.conftest import FakeLLM

LINES = [
    {"section": "DUTY", "text": "Spring 기반 API 개발", "ksa": "SKILL"},
    {"section": "REQUIRED", "text": "트랜잭션에 대한 이해", "ksa": "KNOWLEDGE"},
    {"section": "PREFERRED", "text": "피드백을 주고받는 태도", "ksa": "ATTITUDE"},
    {"section": "STACK", "text": "Java, PostgreSQL", "ksa": None},
]
TEXT = "돌담페이 채용\n" + "\n".join(f"- {line['text']}" for line in LINES)
GOOD = {
    "organization": "돌담페이",
    "organizationType": "JOB",
    "jobRole": "BACKEND",
    "questions": [],
    "postingLines": LINES,
}


def test_posting_without_questions_and_read_from(client_with):
    # LLM이 출처를 써도 서버가 요청 종류로 붙인다.
    llm = FakeLLM({**GOOD, "readFrom": "IMAGE"})
    res = client_with(llm).post("/parse-application", json={"text": TEXT})
    assert res.status_code == 200
    assert res.json() == {**GOOD, "readFrom": "TEXT", "model": "fake-model"}


def test_url_lines_are_checked_against_fetched_text(client_with, monkeypatch):
    async def read_url(url):
        assert url == "https://example.com/posting"
        return TEXT

    monkeypatch.setattr("app.parse_application.read_url", read_url)
    llm = FakeLLM({**GOOD, "postingLines": [{**LINES[0], "text": "없는 업무"}]}, GOOD)
    res = client_with(llm).post("/parse-application", json={"url": "https://example.com/posting"})
    assert res.status_code == 200
    assert res.json()["readFrom"] == "URL"
    assert res.json()["postingLines"] == LINES
    assert llm.calls == 2


@pytest.mark.parametrize("bullet", ["", "· ", "- ", "• ", "1. "])
def test_verbatim_whitespace_and_bullet(client_with, bullet):
    line = {**LINES[0], "text": bullet + "Spring 기반 API 개발"}
    llm = FakeLLM({**GOOD, "postingLines": [line]})
    res = client_with(llm).post(
        "/parse-application", json={"text": "돌담페이\n" + bullet + "Spring\n  기반 API\t개발"}
    )
    assert res.status_code == 200
    assert res.json()["postingLines"] == [line]


@pytest.mark.parametrize(
    "lines",
    [
        [{**LINES[0], "text": "Spring 기반 서버 개발"}],  # 비슷한 뜻으로 바꿔도 위반
        [LINES[1], LINES[0]],
        [LINES[0], LINES[0]],
        [LINES[0], {**LINES[0], "text": "Spring  기반 API 개발"}],
        [{**LINES[0], "text": ""}],
        [{**LINES[0], "text": " \n\t "}],
        [{**LINES[0], "text": "가" * 301}],
        [{**LINES[0], "section": "BENEFITS"}],
        [{**LINES[0], "ksa": "TECHNOLOGY"}],
        [{k: v for k, v in LINES[0].items() if k != "ksa"}],
        None,
    ],
)
def test_invalid_lines_retry_then_fail_without_echoing_text(client_with, lines):
    bad = {**GOOD, "postingLines": lines}
    llm = FakeLLM(bad, bad)
    res = client_with(llm).post("/parse-application", json={"text": TEXT})
    assert res.status_code == 502
    assert res.json()["error"]["code"] == "LLM_INVALID_OUTPUT"
    assert "Spring" not in res.json()["error"]["message"]
    assert llm.calls == 2


def test_missing_lines_is_invalid_not_silently_empty(client_with):
    bad = {k: v for k, v in GOOD.items() if k != "postingLines"}
    llm = FakeLLM(bad, bad)
    res = client_with(llm).post("/parse-application", json={"text": TEXT})
    assert res.json()["error"]["code"] == "LLM_INVALID_OUTPUT"
    assert llm.calls == 2


@pytest.mark.parametrize(("count", "width", "valid"), [(40, 300, True), (41, 300, False)])
def test_line_count_and_text_length_boundaries(client_with, count, width, valid):
    lines = [{**LINES[0], "text": str(n).ljust(width, "가")} for n in range(count)]
    out = {**GOOD, "postingLines": lines}
    llm = FakeLLM(out, out)
    res = client_with(llm).post(
        "/parse-application",
        json={"text": "돌담페이\n" + "\n".join(line["text"] for line in lines)},
    )
    assert res.status_code == (200 if valid else 502)
    if valid:
        assert res.json()["postingLines"] == lines
    else:
        assert res.json()["error"]["code"] == "LLM_INVALID_OUTPUT"
