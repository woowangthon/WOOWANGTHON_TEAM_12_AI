"""/new-facts — LLM 은 가짜로 바꿔 빼는 문장 · 계약 모양 · 코드 확인 · 숫자 대조 · 오류 처리만 본다(네트워크 없음).

요청 · 응답 예시는 ETC docs/contracts/ai-api.md 「POST /new-facts」 그대로다.
"""

import pytest

from app.errors import LLMError
from app.new_facts import remaining_sentences
from tests.conftest import FakeLLM
from tests.test_ground_check import EXP_101, EXP_103

SENTENCES = [
    "「바꿔요」 프로젝트에서 채팅 메시지가 늦게 도착하는 문제가 있었고, 제가 폴링을 WebSocket 으로 바꿔 해결했습니다.",
    "서버에 STOMP 엔드포인트를 만들고, 앱의 3초 폴링 코드를 구독 방식으로 바꿨습니다.",
    "바꾼 뒤에는 두 기기로 메시지를 보내 양쪽 화면에 바로 도착하는지 확인했습니다.",
    "이어서 알림 기능도 같은 방식으로 바꿔 서버 요청을 하루 1,200회 줄였습니다.",
    "배포까지 제가 맡아 AWS 에 올렸습니다.",
    "누구보다 끈질기게 문제를 파고들었습니다.",
    "앞으로도 문제의 원인을 살피는 개발자가 되고 싶습니다.",
]
PERIOD = {"start": "2025-03", "end": "2025-06", "ongoing": False}
ACTIVITIES = [
    {"id": 7, "name": "바꿔요 — 캠퍼스 중고거래 앱", "period": PERIOD},
    {
        "id": 8,
        "name": "알고리즘 스터디",
        "period": {"start": "2024-09", "end": None, "ongoing": True},
    },
]
BODY = {
    "text": " ".join(SENTENCES),
    "experiences": [EXP_101, EXP_103],
    "activities": ACTIVITIES,
    "excludedSentences": [SENTENCES[0], SENTENCES[1], SENTENCES[6]],
}
# LLM 이 받는 번호 — 1·2·7 번째를 코드가 빼서 남은 4문장(원래 3 ~ 6번째)
REMAINING = SENTENCES[2:6]
CONTENT_1 = "WebSocket 으로 바꾼 뒤 두 기기로 메시지를 보내 양쪽 화면에 바로 도착하는지 확인했다"
CONTENT_2 = "알림 기능도 구독 방식으로 바꿔 서버 요청을 하루 1,200회 줄였다"


def _c(index, content, actor="ME", activity_id=7):
    return {"index": index, "content": content, "actor": actor, "activityId": activity_id}


GOOD = {"candidates": [_c(1, CONTENT_1), _c(2, CONTENT_2)]}


def _post(client_with, out, body=BODY, **settings):
    llm = out if isinstance(out, FakeLLM) else FakeLLM(out)
    return client_with(llm, **settings).post("/new-facts", json=body)


# ---- 빼는 문장 ----


def test_remaining_drops_only_same_sentences():
    sentences = ["제가 배포했습니다.", "팀이 회고했습니다.", "제가 배포했습니다!"]
    excluded = [" 제가  배포했습니다. ", "팀이 회고 했어요."]
    assert remaining_sentences(sentences, excluded) == ["팀이 회고했습니다.", "제가 배포했습니다!"]


def test_remaining_uses_nfkc():
    assert remaining_sentences(["ＡＷＳ 에 올렸다."], ["AWS에 올렸다."]) == []


def test_all_excluded_skips_llm(client_with):
    llm = FakeLLM()
    body = {**BODY, "excludedSentences": SENTENCES}
    r = _post(client_with, llm, body)
    assert r.status_code == 200
    assert r.json() == {"candidates": [], "model": None}
    assert llm.calls == 0


# ---- 계약 ----


def test_contract_example(client_with):
    llm = FakeLLM(GOOD)
    r = _post(client_with, llm)
    assert r.status_code == 200
    assert r.json() == {
        "candidates": [
            {"sentence": SENTENCES[2], "content": CONTENT_1, "actor": "ME", "activityId": 7},
            {"sentence": SENTENCES[3], "content": CONTENT_2, "actor": "ME", "activityId": 7},
        ],
        "model": "fake-model",
    }
    assert llm.calls == 1


def test_llm_gets_only_remaining_sentences(client_with):
    seen = {}

    class Spy(FakeLLM):
        async def generate_json(self, system, user, schema, deadline):
            seen["user"] = user
            return await super().generate_json(system, user, schema, deadline)

    _post(client_with, Spy(GOOD))
    listed = seen["user"].split("<후보 문장 목록>")[1].split("</후보 문장 목록>")[0]
    assert listed.strip().splitlines() == [f"{i}. {s}" for i, s in enumerate(REMAINING, 1)]
    assert "[경험 103]" in seen["user"]
    assert "[활동 8] 알고리즘 스터디 (2024-09 ~ 진행 중)" in seen["user"]


def test_empty_candidates(client_with):
    r = _post(client_with, {"candidates": []})
    assert r.status_code == 200
    assert r.json()["candidates"] == []


def test_no_experiences_and_activities(client_with):
    body = {**BODY, "experiences": [], "activities": [], "excludedSentences": []}
    r = _post(
        client_with, {"candidates": [_c(3, "배포를 맡아 AWS 에 올렸다", activity_id=7)]}, body
    )
    assert r.status_code == 200
    assert r.json()["candidates"] == [
        {
            "sentence": SENTENCES[2],
            "content": "배포를 맡아 AWS 에 올렸다",
            "actor": "ME",
            "activityId": None,
        }
    ]


def test_candidates_follow_answer_order(client_with):
    r = _post(client_with, {"candidates": [GOOD["candidates"][1], GOOD["candidates"][0]]})
    assert [c["sentence"] for c in r.json()["candidates"]] == SENTENCES[2:4]


# ---- 엄한 쪽 고치기 ----


def test_made_up_activity_becomes_null(client_with):
    r = _post(client_with, {"candidates": [_c(1, CONTENT_1, activity_id=99)]})
    assert r.status_code == 200
    assert r.json()["candidates"][0]["activityId"] is None


def test_number_not_in_sentence_drops_candidate(client_with):
    out = {"candidates": [_c(1, "두 기기로 3초 안에 도착하는지 확인했다"), _c(2, CONTENT_2)]}
    r = _post(client_with, out)
    assert r.status_code == 200
    assert [c["content"] for c in r.json()["candidates"]] == [CONTENT_2]


def test_number_is_compared_as_number(client_with):
    """1,200 ↔ 1200 은 같다. 120 은 1,200 안의 글자여도 다르다."""
    out = {"candidates": [_c(2, "서버 요청을 하루 1200회 줄였다")]}
    assert len(_post(client_with, out).json()["candidates"]) == 1
    out = {"candidates": [_c(2, "서버 요청을 하루 120회 줄였다")]}
    assert _post(client_with, out).json()["candidates"] == []


def test_more_than_five_keeps_first_five(client_with):
    sentences = [f"제가 일 {chr(0xAC00 + i)}를 했습니다." for i in range(7)]
    body = {**BODY, "text": " ".join(sentences), "excludedSentences": []}
    out = {"candidates": [_c(i, "일을 했다") for i in range(7, 0, -1)]}
    r = _post(client_with, out, body)
    assert [c["sentence"] for c in r.json()["candidates"]] == sentences[:5]


# ---- 위반 → 다시 시도 → LLM_INVALID_OUTPUT ----

BAD = {
    "번호 0": {"candidates": [_c(0, CONTENT_1)]},
    "번호 범위 밖": {"candidates": [_c(5, "일을 했다")]},
    "한 문장에 둘": {"candidates": [_c(1, CONTENT_1), _c(1, "확인했다")]},
    "빈 content": {"candidates": [_c(1, " ")]},
    "content 줄바꿈": {"candidates": [_c(1, "확인했다\n바꿨다")]},
    "~다 꼴 아님": {"candidates": [_c(1, "두 기기로 확인함")]},
    "없는 actor": {"candidates": [_c(1, CONTENT_1, actor="ME_AND_TEAM")]},
    "칸 빠짐": {"candidates": [{"index": 1, "content": CONTENT_1, "actor": "ME"}]},
    "JSON 아님": "그냥 글",
}


@pytest.mark.parametrize("out", BAD.values(), ids=BAD.keys())
def test_invalid_output_retries_then_fails(client_with, out):
    llm = FakeLLM(out, out)
    r = _post(client_with, llm, max_attempts=2)
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "LLM_INVALID_OUTPUT"
    assert llm.calls == 2


def test_retry_recovers(client_with):
    llm = FakeLLM(BAD["번호 범위 밖"], GOOD)
    r = _post(client_with, llm, max_attempts=2)
    assert r.status_code == 200
    assert llm.calls == 2


# ---- 요청 오류 ----


def _without(key):
    return {k: v for k, v in BODY.items() if k != key}


@pytest.mark.parametrize(
    "body",
    [
        _without("text"),
        {**BODY, "text": "  \n "},
        _without("experiences"),
        {**BODY, "experiences": None},
        _without("activities"),
        {**BODY, "activities": None},
        _without("excludedSentences"),
        {**BODY, "excludedSentences": None},
        {**BODY, "experiences": [EXP_101, EXP_101]},
        {**BODY, "activities": [ACTIVITIES[0], ACTIVITIES[0]]},
        {**BODY, "text": 3},
    ],
)
def test_invalid_request(client_with, body):
    llm = FakeLLM()
    r = _post(client_with, llm, body)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "INVALID_REQUEST"
    assert llm.calls == 0


def test_too_long(client_with):
    llm = FakeLLM()
    r = _post(client_with, llm, max_match_chars=5)
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "TEXT_TOO_LONG"
    assert llm.calls == 0


def test_llm_error_passes_through(client_with):
    r = _post(client_with, LLMError("막힘"))
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "LLM_ERROR"
