"""/star — LLM 은 가짜로 바꿔 계약 모양 · 칸 규칙 · 다시 시도 · 칸만 빈칸으로 내리기를 본다(네트워크 없음).

요청 · 응답 예시는 ETC docs/contracts/ai-api.md 「POST /star」 그대로다.
"""

import json

from app.errors import LLMError
from tests.conftest import FakeLLM

EXP_101 = {
    "id": 101,
    "content": "채팅 메시지가 늦게 도착하는 문제를 폴링에서 WebSocket 으로 바꿔 해결했다",
    "actor": "ME",
    "competencies": ["PROBLEM_SOLVING", "IMPLEMENTATION"],
    "citations": [
        {
            "id": 201,
            "quote": "채팅 메시지가 늦게 도착하는 문제가 있었다. 내가 폴링을 WebSocket 으로 바꿔 해결했다.",
        }
    ],
    "supplements": [
        {
            "id": 301,
            "question": "폴링을 WebSocket 으로 바꿀 때 직접 고친 부분은 어디였나요?",
            "answer": "서버에 STOMP 엔드포인트를 만들고, 앱의 3초 폴링 코드를 구독 방식으로 바꿨다",
        }
    ],
}
EXP_102 = {
    "id": 102,
    "content": "서비스를 AWS 에 배포했다",
    "actor": "TEAM",
    "competencies": ["OPERATION"],
    "citations": [
        {
            "id": 202,
            "quote": "배포는 팀원이 맡아 AWS 에 올렸다. 회고 때 문서화가 중요하다는 걸 배웠다.",
        }
    ],
    "supplements": [],
}
BODY = {"experiences": [EXP_101, EXP_102]}


def _slot(slot, source_type=None, source_id=None, text=None, question=None):
    return {
        "slot": slot,
        "sourceType": source_type,
        "sourceId": source_id,
        "text": text,
        "question": question,
    }


SLOTS_101 = [
    _slot("S", "CITATION", 201, "채팅 메시지가 늦게 도착하는 문제가 있었다"),
    _slot("T", question="이 문제를 고치는 일은 누가 맡았나요?"),
    _slot(
        "A",
        "SUPPLEMENT",
        301,
        "서버에 STOMP 엔드포인트를 만들고, 앱의 3초 폴링 코드를 구독 방식으로 바꿨다",
    ),
    _slot("R", question="바꾼 뒤 무엇이 달라졌나요?"),
]
SLOTS_102 = [
    _slot("S", question="배포가 왜 필요했나요?"),
    _slot("T", question="이 배포에서 직접 맡은 업무는 무엇이었나요?"),
    _slot("A", question=None),
    _slot("R", question="배포 뒤 무엇이 달라졌나요?"),
]
GOOD = {
    "experiences": [
        {"id": 102, "slots": SLOTS_102},
        {"id": 101, "slots": SLOTS_101},
    ]
}


def _with(exp_id, slot, **change):
    data = json.loads(json.dumps(GOOD))
    for e in data["experiences"]:
        if e["id"] == exp_id:
            for s in e["slots"]:
                if s["slot"] == slot:
                    s.update(change)
    return data


def _slot_of(res, exp_id, slot):
    exp = next(e for e in res.json()["experiences"] if e["id"] == exp_id)
    return next(s for s in exp["slots"] if s["slot"] == slot)


def test_contract_shape_in_request_order(client_with):
    llm = FakeLLM(GOOD)
    res = client_with(llm).post("/star", json=BODY)
    assert res.status_code == 200
    data = res.json()
    assert set(data) == {"experiences", "model"}
    assert [e["id"] for e in data["experiences"]] == [101, 102]  # LLM 순서가 달라도 요청 순서
    assert data["experiences"][0]["slots"] == SLOTS_101
    assert data["model"] == "fake-model"
    assert llm.calls == 1


def test_prompt_shows_experiences(client_with):
    llm = FakeLLM(GOOD)
    client_with(llm).post("/star", json=BODY)
    user = llm.seen[0]["user"]
    assert "[경험 101]" in user and "수행 주체: TEAM" in user
    assert "- 근거 201: 「채팅 메시지가" in user
    assert "- 보완 301: 질문 「폴링을" in user and "답 「서버에 STOMP" in user
    assert "역량: OPERATION" in user


def test_text_may_differ_in_spaces_only(client_with):
    out = _with(101, "S", text="  채팅 메시지가   늦게 도착하는\n문제가 있었다 ")
    res = client_with(FakeLLM(out)).post("/star", json=BODY)
    assert res.status_code == 200
    assert _slot_of(res, 101, "S")["sourceType"] == "CITATION"


def test_rewritten_text_retries_then_only_that_slot_empty(client_with):
    out = _with(101, "S", text="채팅 메시지가 늦게 도착했다")  # 다듬은 글 — 원문에 없다
    llm = FakeLLM(out, out)
    res = client_with(llm, max_attempts=2).post("/star", json=BODY)
    assert res.status_code == 200
    assert llm.calls == 2
    assert _slot_of(res, 101, "S") == _slot("S")  # 질문도 null
    assert _slot_of(res, 101, "A")["sourceId"] == 301  # 다른 칸은 그대로
    assert _slot_of(res, 102, "T")["question"] == "이 배포에서 직접 맡은 업무는 무엇이었나요?"


def test_retry_fixes_slot(client_with):
    bad = _with(101, "A", sourceId=999)
    llm = FakeLLM(bad, GOOD)
    res = client_with(llm, max_attempts=2).post("/star", json=BODY)
    assert llm.calls == 2
    assert _slot_of(res, 101, "A")["sourceId"] == 301


def test_slot_violations(client_with):
    cases = [
        _with(101, "A", sourceType="CITATION", sourceId=301),  # 종류와 ID 가 안 맞음
        _with(101, "S", sourceId=202),  # 다른 경험의 근거
        _with(101, "S", sourceType=None),  # 종류 없이 ID · 글
        _with(101, "S", question="왜 늦었나요?"),  # 찬 칸에 질문
        _with(101, "S", text=""),  # 빈 글
        _with(101, "T", question="가" * 81),  # 질문 80자 넘음
        _with(101, "T", question="첫 줄\n둘째 줄"),
        _with(101, "S", sourceType="EXPERIENCE"),
        # 수행 주체 TEAM 인데 T 를 근거로
        _with(
            102, "T", sourceType="CITATION", sourceId=202, text="배포는 팀원이 맡아", question=None
        ),
        # 근거 원문의 배운 점을 R 에
        _with(
            102,
            "R",
            sourceType="CITATION",
            sourceId=202,
            text="회고 때 문서화가 중요하다는 걸 배웠다",
            question=None,
        ),
    ]
    for out in cases:
        llm = FakeLLM(out)
        res = client_with(llm, max_attempts=1).post("/star", json=BODY)
        assert res.status_code == 200, out
        slots = {e["id"]: e["slots"] for e in res.json()["experiences"]}
        changed = [
            (i, s["slot"])
            for i in (101, 102)
            for s in slots[i]
            if s
            != next(x for x in (SLOTS_101 if i == 101 else SLOTS_102) if x["slot"] == s["slot"])
        ]
        assert len(changed) == 1, (out, changed)
        i, name = changed[0]
        assert next(s for s in slots[i] if s["slot"] == name) == _slot(name)


def test_learned_point_from_supplement_is_fine(client_with):
    body = json.loads(json.dumps(BODY))
    body["experiences"][1]["supplements"] = [
        {
            "id": 302,
            "question": "배운 점은?",
            "answer": "배포 문서를 미리 써 두는 게 중요하다는 걸 배웠다",
        }
    ]
    out = _with(
        102,
        "R",
        sourceType="SUPPLEMENT",
        sourceId=302,
        text="배포 문서를 미리 써 두는 게 중요하다는 걸 배웠다",
        question=None,
    )
    res = client_with(FakeLLM(out)).post("/star", json=body)
    assert _slot_of(res, 102, "R")["sourceId"] == 302


def test_missing_and_duplicate_slots_become_empty(client_with):
    data = json.loads(json.dumps(GOOD))
    exp = next(e for e in data["experiences"] if e["id"] == 101)
    exp["slots"] = [exp["slots"][0], exp["slots"][0], exp["slots"][2]]  # S 두 번 · T R 빠짐
    res = client_with(FakeLLM(data), max_attempts=1).post("/star", json=BODY)
    assert res.status_code == 200
    got = [s["slot"] for s in res.json()["experiences"][0]["slots"]]
    assert got == ["S", "T", "A", "R"]
    assert _slot_of(res, 101, "S")["sourceId"] == 201
    assert _slot_of(res, 101, "T") == _slot("T")


def test_whole_violation_is_invalid_output(client_with):
    missing = {"experiences": [{"id": 101, "slots": SLOTS_101}]}
    extra = {"experiences": [*GOOD["experiences"], {"id": 999, "slots": SLOTS_102}]}
    twice = {"experiences": [*GOOD["experiences"], {"id": 101, "slots": SLOTS_101}]}
    for out in (missing, extra, twice):
        llm = FakeLLM(out, out)
        res = client_with(llm, max_attempts=2).post("/star", json=BODY)
        assert res.status_code == 502
        assert res.json()["error"]["code"] == "LLM_INVALID_OUTPUT"
        assert llm.calls == 2


def test_not_json_is_invalid_output(client_with):
    res = client_with(FakeLLM("nope", "nope"), max_attempts=2).post("/star", json=BODY)
    assert res.json()["error"]["code"] == "LLM_INVALID_OUTPUT"


def test_bad_requests(client_with):
    many = [{**EXP_102, "id": i} for i in range(31)]
    for body in (
        {},
        {"experiences": []},
        {"experiences": many},
        {"experiences": [EXP_101, EXP_101]},
        {"experiences": [{"id": 1, "content": "x", "actor": "SOMEONE"}]},
    ):
        llm = FakeLLM()
        res = client_with(llm).post("/star", json=body)
        assert res.status_code == 400, body
        assert res.json()["error"]["code"] == "INVALID_REQUEST"
        assert llm.calls == 0


def test_too_long(client_with):
    llm = FakeLLM()
    res = client_with(llm, max_match_chars=50).post("/star", json=BODY)
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "TEXT_TOO_LONG"


def test_llm_error(client_with):
    res = client_with(FakeLLM(LLMError("down"))).post("/star", json=BODY)
    assert res.status_code == 502
    assert res.json()["error"]["code"] == "LLM_ERROR"
