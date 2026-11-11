"""/posting-match — LLM 은 가짜로 바꿔 계약 모양 · 줄 · ID 검사 · 오류 처리만 본다(네트워크 없음).

요청 · 응답 예시는 ETC docs/contracts/ai-api.md 「POST /posting-match」 그대로다.
"""

import json

import pytest

from app.posting_match import SYSTEM_PROMPT
from tests.conftest import FakeLLM
from tests.test_match import EXPERIENCES

# 101 ME(WebSocket) · 102 TEAM(회고) · 103 TEAM(AWS 배포) — test_match 와 같은 경험
LINES = [
    {
        "id": 901,
        "section": "REQUIRED",
        "text": "대규모 실시간 트래픽 처리 시스템 개발 경험",
        "ksa": "SKILL",
    },
    {"id": 902, "section": "REQUIRED", "text": "장애 경험 및 문제 해결 경험", "ksa": "SKILL"},
    {
        "id": 903,
        "section": "PREFERRED",
        "text": "Spring Framework 기반의 B2C 서비스 개발 경험",
        "ksa": None,
    },
]
BODY = {"jobRole": "BACKEND", "lines": LINES, "experiences": EXPERIENCES}

GOOD = {
    "lines": [
        {"id": 902, "status": "PARTIAL", "experienceIds": [101]},
        {"id": 901, "status": "NOT_FOUND", "experienceIds": []},
        {"id": 903, "status": "SUPPORTED", "experienceIds": [101, 103]},
    ],
    "clarifications": [
        {
            "lineId": 902,
            "experienceId": 101,
            "question": "채팅이 늦게 도착하던 문제를 처음 어떻게 알아차렸나요?",
        }
    ],
}


def _bad(mutate):
    data = json.loads(json.dumps(GOOD))
    mutate(data)
    return data


def test_contract_shape_in_request_order(client_with):
    llm = FakeLLM(GOOD)
    res = client_with(llm).post("/posting-match", json=BODY)
    assert res.status_code == 200
    data = res.json()
    assert set(data) == {"lines", "clarifications", "model"}
    assert [j["id"] for j in data["lines"]] == [901, 902, 903]  # LLM 순서가 달라도 요청 순서
    assert data["clarifications"] == GOOD["clarifications"]
    assert data["model"] == "fake-model"
    assert llm.calls == 1


def test_prompt_shows_lines_context_and_experiences(client_with):
    llm = FakeLLM(GOOD)
    client_with(llm).post("/posting-match", json=BODY)
    user = llm.seen[0]["user"]
    assert "지원 직무: BACKEND" in user
    assert "[줄 901] REQUIRED(자격요건) · 기술: 대규모 실시간 트래픽 처리 시스템 개발 경험" in user
    assert "[줄 903] PREFERRED(우대사항) · 미분류:" in user
    assert "[경험 101]" in user and "보완 301" in user
    assert "COLLABORATION" not in user  # 역량 태그는 넘기지 않는다(/match 와 같다)
    assert llm.seen[0]["system"] == SYSTEM_PROMPT


def test_without_job_role(client_with):
    llm = FakeLLM(GOOD)
    body = {k: v for k, v in BODY.items() if k != "jobRole"}
    assert client_with(llm).post("/posting-match", json=body).status_code == 200
    assert "지원 직무: (없음)" in llm.seen[0]["user"]


def test_no_experiences_all_not_found_without_llm(client_with):
    llm = FakeLLM()
    res = client_with(llm).post("/posting-match", json={**BODY, "experiences": []})
    assert res.status_code == 200
    data = res.json()
    assert data["lines"] == [
        {"id": i, "status": "NOT_FOUND", "experienceIds": []} for i in (901, 902, 903)
    ]
    assert data["clarifications"] == []
    assert llm.calls == 0


def _line(i, **kw):
    return {"id": i, "section": "DUTY", "text": f"줄 {i}", "ksa": "SKILL"} | kw


@pytest.mark.parametrize(
    "body",
    [
        {"experiences": []},
        {"lines": [], "experiences": []},
        {"lines": [_line(i) for i in range(41)], "experiences": []},
        {"lines": [_line(1), _line(1)], "experiences": []},
        {"lines": [_line(1, text=" ")], "experiences": []},
        {"lines": [_line(1, text="가" * 301)], "experiences": []},
        {"lines": [_line(1, section="ETC")], "experiences": []},
        {"lines": [_line(1, ksa="SKILLS")], "experiences": []},
        {"lines": [_line(1)], "experiences": [], "jobRole": "DEVELOPER"},
        {"lines": [_line(1)]},
        {"lines": [_line(1)], "experiences": None},
        {"lines": [_line(1)], "experiences": [EXPERIENCES[0], EXPERIENCES[0]]},
    ],
)
def test_invalid_request(client_with, body):
    llm = FakeLLM()
    res = client_with(llm).post("/posting-match", json=body)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_REQUEST"
    assert llm.calls == 0


def test_contract_limits_ok(client_with):
    lines = [_line(i, text="가" * 300, ksa=None) for i in range(40)]
    out = {"lines": [{"id": i, "status": "NOT_FOUND", "experienceIds": []} for i in range(40)]}
    out["clarifications"] = []
    res = client_with(FakeLLM(out)).post(
        "/posting-match", json={"lines": lines, "experiences": EXPERIENCES}
    )
    assert res.status_code == 200


def test_too_long_counts_line_text(client_with):
    llm = FakeLLM()
    chars = sum(len(e["content"]) for e in EXPERIENCES[:1])  # 경험 하나 본문만으로는 한도 안
    exp = {**EXPERIENCES[0], "citations": [], "supplements": []}
    body = {"lines": [_line(1, text="가" * 10)], "experiences": [exp]}
    res = client_with(llm, max_match_chars=chars + 5).post("/posting-match", json=body)
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "TEXT_TOO_LONG"
    assert llm.calls == 0


BAD_OUTPUTS = [
    "not json",
    {"lines": []},
    # 줄: 빠짐 · 중복 · 요청에 없는 줄 · 칸 없음
    _bad(lambda d: d["lines"].pop()),
    _bad(lambda d: d["lines"].append(d["lines"][0])),
    _bad(lambda d: d["lines"][1].update(id=999)),
    _bad(lambda d: d.pop("lines")),
    # 판정: 값 밖 · NOT_FOUND 인데 경험 · PARTIAL 인데 경험 없음 · 요청에 없는 경험 · 중복 · 4개
    _bad(lambda d: d["lines"][0].update(status="FOUND")),
    _bad(lambda d: d["lines"][1].update(experienceIds=[101])),
    _bad(lambda d: d["lines"][0].update(experienceIds=[])),
    _bad(lambda d: d["lines"][0].update(experienceIds=[999])),
    _bad(lambda d: d["lines"][2].update(experienceIds=[101, 101])),
    _bad(lambda d: d["lines"][2].update(experienceIds=[101, 102, 103, 104])),
    # TEAM 경험만으로 SUPPORTED
    _bad(lambda d: d["lines"][2].update(experienceIds=[102, 103])),
    # 되묻기: 4개 · SUPPORTED 줄 · NOT_FOUND 줄 · 그 줄의 경험이 아님 · 없는 줄 · 빈 질문 · 81자 · 칸 없음
    _bad(lambda d: d.update(clarifications=d["clarifications"] * 4)),
    _bad(lambda d: d["clarifications"][0].update(lineId=903)),
    _bad(lambda d: d["clarifications"][0].update(lineId=901)),
    _bad(lambda d: d["clarifications"][0].update(experienceId=102)),
    _bad(lambda d: d["clarifications"][0].update(lineId=999)),
    _bad(lambda d: d["clarifications"][0].update(question=" ")),
    _bad(lambda d: d["clarifications"][0].update(question="가" * 81)),
    _bad(lambda d: d.pop("clarifications")),
]


@pytest.mark.parametrize("bad", BAD_OUTPUTS)
def test_invalid_output_retries_then_errors(client_with, bad):
    llm = FakeLLM(bad, bad)
    res = client_with(llm, max_attempts=2).post("/posting-match", json=BODY)
    assert res.status_code == 502
    assert res.json()["error"]["code"] == "LLM_INVALID_OUTPUT"
    assert llm.calls == 2


def test_invalid_then_valid_recovers(client_with):
    bad = _bad(lambda d: d["lines"].pop())
    llm = FakeLLM(bad, GOOD)
    res = client_with(llm, max_attempts=2).post("/posting-match", json=BODY)
    assert res.status_code == 200
    assert llm.calls == 2


def test_edge_outputs_ok(client_with):
    # 80자 질문 · 되묻기 3개(같은 줄 여러 개도 된다) · TEAM 섞인 SUPPORTED · TEAM 만의 PARTIAL
    out = _bad(
        lambda d: (
            d["lines"][1].update(status="PARTIAL", experienceIds=[103]),
            d.update(
                clarifications=[
                    {"lineId": 902, "experienceId": 101, "question": "가" * 80},
                    {
                        "lineId": 901,
                        "experienceId": 103,
                        "question": "배포에서 맡은 일이 있었나요?",
                    },
                    {"lineId": 902, "experienceId": 101, "question": "원인을 어떻게 찾았나요?"},
                ]
            ),
        )
    )
    res = client_with(FakeLLM(out)).post("/posting-match", json=BODY)
    assert res.status_code == 200
    assert len(res.json()["clarifications"]) == 3


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [("LLM_ERROR", 502, "LLM_ERROR"), ("LLM_TIMEOUT", 504, "LLM_TIMEOUT")],
)
def test_llm_errors(client_with, error, status, code):
    from app.errors import LLMError, LLMTimeout

    exc = LLMError("x") if error == "LLM_ERROR" else LLMTimeout("x")
    res = client_with(FakeLLM(exc)).post("/posting-match", json=BODY)
    assert res.status_code == status
    assert res.json()["error"]["code"] == code
