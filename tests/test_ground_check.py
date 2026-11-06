"""/ground-check — LLM 은 가짜로 바꿔 문장 나누기 · 계약 모양 · 코드 확인 · 숫자 대조 · 오류 처리만 본다(네트워크 없음).

요청 · 응답 예시는 ETC docs/contracts/ai-api.md 「POST /ground-check」 그대로다.
"""

import pytest

from app.errors import LLMError
from app.ground_check import split_sentences
from tests.conftest import FakeLLM

ACTIVITY = "바꿔요 — 캠퍼스 중고거래 앱"
PERIOD = {"start": "2025-03", "end": "2025-06", "ongoing": False}
EXP_101 = {
    "id": 101,
    "activityName": ACTIVITY,
    "period": PERIOD,
    "content": "채팅 메시지가 늦게 도착하는 문제를 폴링에서 WebSocket 으로 바꿔 해결했다",
    "competencies": ["PROBLEM_SOLVING", "IMPLEMENTATION"],
    "actor": "ME",
    "sourceStatus": "SOURCE_CONFIRMED",
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
EXP_103 = {
    "id": 103,
    "activityName": ACTIVITY,
    "period": PERIOD,
    "content": "서비스를 AWS 에 배포했다",
    "competencies": ["IMPLEMENTATION"],
    "actor": "TEAM",
    "sourceStatus": "SOURCE_CONFIRMED",
    "citations": [{"id": 203, "quote": "배포는 팀원이 맡아 AWS 에 올렸다."}],
    "supplements": [],
}
EXP_106 = {
    "id": 106,
    "activityName": "알고리즘 스터디",
    "period": {"start": "2024-09", "end": None, "ongoing": True},
    "content": "스터디원 6명의 풀이를 매주 리뷰했다",
    "competencies": ["LEARNING"],
    "actor": "ME",
    "sourceStatus": "USER_ENTERED",
    "citations": [],
    "supplements": [],
}
SENTENCES = [
    "저는 「바꿔요」 프로젝트에서 채팅 메시지가 늦게 도착하는 문제를 WebSocket 으로 바꿔 해결했습니다.",
    "서버에 STOMP 엔드포인트를 만들고 앱의 폴링 코드를 구독 방식으로 바꿨습니다.",
    "그 결과 응답 속도가 80% 빨라졌고 사용자 만족도가 크게 올랐습니다.",
    "배포까지 제가 맡아 AWS 에 올렸습니다.",
    "이 경험으로 문제의 원인을 끝까지 찾는 개발자가 되고 싶습니다.",
]
BODY = {"text": " ".join(SENTENCES), "experiences": [EXP_101, EXP_103, EXP_106]}
TEAM_NOTE = "자료엔 배포를 팀원이 맡았다고 되어 있어요"


def _v(index, status, *evidence, note=None):
    return {
        "index": index,
        "status": status,
        "evidence": [{"type": t, "id": i} for t, i in evidence],
        "note": note,
    }


GOOD = [
    _v(1, "SOURCE", ("CITATION", 201)),
    _v(2, "USER_INPUT", ("SUPPLEMENT", 301)),
    _v(3, "SOURCE", ("CITATION", 201)),  # 「80」은 201 에 없다 — 코드가 돌린다
    _v(4, "UNSUPPORTED", note=TEAM_NOTE),
    _v(5, "OPINION"),
]


def _out(*replaced):
    """GOOD 에서 그 번호의 판정만 바꾼 출력."""
    verdicts = list(GOOD)
    for v in replaced:
        verdicts[v["index"] - 1] = v
    return {"verdicts": verdicts}


def _post(client_with, out, body=BODY):
    return client_with(FakeLLM(out)).post("/ground-check", json=body)


# ---- 문장 나누기 ----


@pytest.mark.parametrize(
    ("text", "want"),
    [
        (
            "한다. 했다? 그렇다! 아마… 끝。 마지막",
            ["한다.", "했다?", "그렇다!", "아마…", "끝。", "마지막"],
        ),
        ("학점 3.5 를 받았다. v2.1 로 올렸다.", ["학점 3.5 를 받았다.", "v2.1 로 올렸다."]),
        ("1,200명이 왔다.\n둘째 줄\r\n셋째 줄", ["1,200명이 왔다.", "둘째 줄", "셋째 줄"]),
        ("  앞뒤 공백.   \n\n  다음  ", ["앞뒤 공백.", "다음"]),
        ("정말?! 네.", ["정말?!", "네."]),
        ("「끝났다.」 그리고 갔다", ["「끝났다.」 그리고 갔다"]),
        (" \n ", []),
    ],
)
def test_split_sentences(text, want):
    assert split_sentences(text) == want


@pytest.mark.parametrize(
    "text, want",
    [
        ("1. 첫째입니다.\n2. 둘째입니다.", ["첫째입니다.", "둘째입니다."]),
        ("1) 첫째\r\n(2) 둘째\n① 셋째\n⑳ 넷째", ["첫째", "둘째", "셋째", "넷째"]),
        ("1.붙여 썼다\n  12. 들여 썼다", ["붙여 썼다", "들여 썼다"]),
        ("1.\n번호만 있는 줄", ["번호만 있는 줄"]),
    ],
)
def test_split_drops_list_markers(text, want):
    assert split_sentences(text) == want


@pytest.mark.parametrize(
    "text, want",
    [
        ("3.5배 늘었다", ["3.5배 늘었다"]),
        ("2025. 3월에 시작했다", ["2025.", "3월에 시작했다"]),
        ("순위는 2. 번째였다", ["순위는 2.", "번째였다"]),
        ("3명이 왔다", ["3명이 왔다"]),
        ("1. 80% 올렸다", ["80% 올렸다"]),
    ],
)
def test_split_keeps_numbers_that_are_not_markers(text, want):
    assert split_sentences(text) == want


def test_list_marker_is_not_checked_as_number(client_with):
    body = {"text": "1. 스터디원 6명의 풀이를 리뷰했습니다.", "experiences": [EXP_106]}
    out = {"verdicts": [_v(1, "USER_INPUT", ("EXPERIENCE", 106))]}
    assert _post(client_with, out, body).json()["sentences"][0]["status"] == "USER_INPUT"


def test_split_keeps_text_as_continuous_parts():
    text = "  첫 문장입니다.  둘째는 3.5 예요!\n\n 1. 셋째 "
    at = 0
    for sentence in split_sentences(text):
        at = text.index(sentence, at) + len(sentence)
    assert at == len(text.rstrip())


# ---- 계약 ----


def test_contract_example(client_with):
    llm = FakeLLM(_out())
    r = client_with(llm).post("/ground-check", json=BODY)
    assert r.status_code == 200
    assert r.json() == {
        "sentences": [
            {
                "text": SENTENCES[0],
                "status": "SOURCE",
                "evidence": [{"type": "CITATION", "id": 201}],
                "note": None,
            },
            {
                "text": SENTENCES[1],
                "status": "USER_INPUT",
                "evidence": [{"type": "SUPPLEMENT", "id": 301}],
                "note": None,
            },
            {
                "text": SENTENCES[2],
                "status": "UNSUPPORTED",
                "evidence": [],
                "note": "문장의 숫자(80)가 받치는 근거 · 보완에 없어요",
            },
            {"text": SENTENCES[3], "status": "UNSUPPORTED", "evidence": [], "note": TEAM_NOTE},
            {"text": SENTENCES[4], "status": "OPINION", "evidence": [], "note": None},
        ],
        "model": "fake-model",
    }
    assert llm.calls == 1


def test_llm_gets_numbered_sentences(client_with):
    seen = {}

    class Spy(FakeLLM):
        async def generate_json(self, system, user, schema, deadline):
            seen["user"] = user
            return await super().generate_json(system, user, schema, deadline)

    client_with(Spy(_out())).post("/ground-check", json=BODY)
    assert f"3. {SENTENCES[2]}" in seen["user"]
    assert "[경험 106]" in seen["user"]


def test_user_entered_body_is_evidence(client_with):
    body = {"text": "스터디원 6명의 풀이를 리뷰했습니다.", "experiences": [EXP_106]}
    out = {"verdicts": [_v(1, "USER_INPUT", ("EXPERIENCE", 106))]}
    r = _post(client_with, out, body)
    assert r.status_code == 200
    assert r.json()["sentences"][0]["evidence"] == [{"type": "EXPERIENCE", "id": 106}]


def test_source_may_also_carry_supplement(client_with):
    r = _post(client_with, _out(_v(1, "SOURCE", ("CITATION", 201), ("SUPPLEMENT", 301))))
    assert r.status_code == 200
    assert r.json()["sentences"][0]["status"] == "SOURCE"


def test_no_experiences_is_not_an_error(client_with):
    out = {"verdicts": [_v(1, "UNSUPPORTED", note="자료에 없는 내용이에요"), _v(2, "OPINION")]}
    r = _post(client_with, out, {"text": "상을 받았습니다. 더 배우고 싶습니다.", "experiences": []})
    assert r.status_code == 200
    assert [s["status"] for s in r.json()["sentences"]] == ["UNSUPPORTED", "OPINION"]


# ---- 숫자 대조 ----


def _one(client_with, text, verdict):
    r = _post(client_with, {"verdicts": [verdict]}, {"text": text, "experiences": [EXP_101]})
    assert r.status_code == 200
    return r.json()["sentences"][0]


def test_number_in_evidence_stays(client_with):
    s = _one(client_with, "3초 폴링 코드를 바꿨습니다.", _v(1, "USER_INPUT", ("SUPPLEMENT", 301)))
    assert s["status"] == "USER_INPUT"


def test_number_is_not_matched_inside_other_number(client_with):
    s = _one(client_with, "30초 폴링을 바꿨습니다.", _v(1, "USER_INPUT", ("SUPPLEMENT", 301)))
    assert s["status"] == "UNSUPPORTED"
    assert s["evidence"] == []


def test_number_may_come_from_period_and_activity(client_with):
    s = _one(client_with, "2025년 3월에 시작했습니다.", _v(1, "SOURCE", ("CITATION", 201)))
    assert s["status"] == "SOURCE"


def test_number_only_in_uncited_evidence_is_turned(client_with):
    """3 은 보완 301 에만 있는데 이 문장은 근거 201 만 붙였다 — 붙인 글만 본다."""
    body = {"text": "3초 폴링을 바꿨습니다.", "experiences": [{**EXP_101, "period": None}]}
    out = {"verdicts": [_v(1, "SOURCE", ("CITATION", 201))]}
    s = _post(client_with, out, body).json()["sentences"][0]
    assert s["status"] == "UNSUPPORTED"
    assert s["note"] == "문장의 숫자(3)가 받치는 근거 · 보완에 없어요"


def test_opinion_with_number_is_turned(client_with):
    s = _one(client_with, "7년 뒤에는 시니어가 되고 싶습니다.", _v(1, "OPINION"))
    assert s == {
        "text": "7년 뒤에는 시니어가 되고 싶습니다.",
        "status": "UNSUPPORTED",
        "evidence": [],
        "note": "문장의 숫자(7)가 받치는 근거 · 보완에 없어요",
    }


def test_several_numbers_in_note(client_with):
    s = _one(client_with, "80명 중 90명이 왔습니다.", _v(1, "SOURCE", ("CITATION", 201)))
    assert s["note"] == "문장의 숫자(80, 90)가 받치는 근거 · 보완에 없어요"


def test_llm_unsupported_keeps_its_note(client_with):
    s = _one(client_with, "제가 7번 배포했습니다.", _v(1, "UNSUPPORTED", note=TEAM_NOTE))
    assert s["note"] == TEAM_NOTE


# ---- 위반 → 다시 시도 → LLM_INVALID_OUTPUT ----

BAD = {
    "지어낸 근거": _out(_v(1, "SOURCE", ("CITATION", 999))),
    "지어낸 보완": _out(_v(2, "USER_INPUT", ("SUPPLEMENT", 999))),
    "USER_ENTERED 아닌 경험": _out(_v(2, "USER_INPUT", ("EXPERIENCE", 101))),
    "SOURCE 에 근거 없음": _out(_v(1, "SOURCE", ("SUPPLEMENT", 301))),
    "SOURCE 에 빈 근거": _out(_v(1, "SOURCE")),
    "USER_INPUT 에 근거": _out(_v(2, "USER_INPUT", ("CITATION", 201))),
    "USER_INPUT 에 빈 근거": _out(_v(2, "USER_INPUT")),
    "UNSUPPORTED 에 근거": _out(_v(4, "UNSUPPORTED", ("CITATION", 203), note="x")),
    "OPINION 에 근거": _out(_v(5, "OPINION", ("CITATION", 201))),
    "근거 중복": _out(_v(1, "SOURCE", ("CITATION", 201), ("CITATION", 201))),
    "UNSUPPORTED note 없음": _out(_v(4, "UNSUPPORTED")),
    "UNSUPPORTED note 빈 글": _out(_v(4, "UNSUPPORTED", note=" ")),
    "note 줄바꿈": _out(_v(4, "UNSUPPORTED", note="한 줄\n두 줄")),
    "SOURCE 에 note": _out(_v(1, "SOURCE", ("CITATION", 201), note="x")),
    "문장 수 모자람": {"verdicts": GOOD[:4]},
    "문장 수 넘침": {"verdicts": [*GOOD, _v(6, "OPINION")]},
    "번호 어긋남": {"verdicts": [GOOD[4], *GOOD[1:4], GOOD[0]]},
    "없는 status": _out({**GOOD[0], "status": "PARTIAL"}),
    "JSON 아님": "그냥 글",
}


@pytest.mark.parametrize("out", BAD.values(), ids=BAD.keys())
def test_invalid_output_retries_then_fails(client_with, out):
    llm = FakeLLM(out, out)
    r = client_with(llm, max_attempts=2).post("/ground-check", json=BODY)
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "LLM_INVALID_OUTPUT"
    assert llm.calls == 2


def test_retry_recovers(client_with):
    llm = FakeLLM(BAD["지어낸 근거"], _out())
    r = client_with(llm, max_attempts=2).post("/ground-check", json=BODY)
    assert r.status_code == 200
    assert llm.calls == 2


# ---- 요청 오류 ----


@pytest.mark.parametrize(
    "body",
    [
        {"experiences": []},
        {"text": "   \n ", "experiences": []},
        {"text": "글."},
        {"text": "글.", "experiences": None},
        {"text": "글.", "experiences": [EXP_106, EXP_106]},
        {"text": 3, "experiences": []},
    ],
)
def test_invalid_request(client_with, body):
    llm = FakeLLM()
    r = client_with(llm).post("/ground-check", json=body)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "INVALID_REQUEST"
    assert llm.calls == 0


def test_too_long(client_with):
    llm = FakeLLM()
    r = client_with(llm, max_match_chars=5).post(
        "/ground-check", json={"text": "여섯 글자 글.", "experiences": []}
    )
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "TEXT_TOO_LONG"
    assert llm.calls == 0


def test_llm_error_passes_through(client_with):
    r = _post(client_with, LLMError("막힘"))
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "LLM_ERROR"
