"""/draft — LLM 은 가짜로 바꿔 계약 모양 · 근거 · 숫자 검사 · 오류 처리만 본다(네트워크 없음).

요청 · 응답 예시는 ETC docs/contracts/ai-api.md 「POST /draft」 그대로다.
"""

import pytest

from app.citations import numbers, unsupported_numbers
from app.config import load_settings
from app.draft import (
    CARD_EXAMPLES,
    CARD_MODE,
    CARD_STEPS,
    FRAME_STEPS,
    LLM_JSON_SCHEMA,
    backings,
    build_draft_message,
    check_output,
    check_request,
    schema_for,
)
from app.match import build_user_message, experience_chars, experience_lines
from app.schemas import DraftRequest, LLMDraftOutput, MatchExperience
from tests.conftest import FakeLLM


def _exp(id, content, actor, citations=(), supplements=(), status="SOURCE_CONFIRMED"):
    return {
        "id": id,
        "activityName": "바꿔요 — 캠퍼스 중고거래 앱",
        "period": {"start": "2025-03", "end": "2025-06", "ongoing": False},
        "content": content,
        "competencies": ["PROBLEM_SOLVING"],
        "actor": actor,
        "sourceStatus": status,
        "citations": [{"id": cid, "quote": q} for cid, q in citations],
        "supplements": [{"id": sid, "question": q, "answer": a} for sid, q, a in supplements],
    }


EXP_101 = _exp(
    101,
    "채팅 메시지가 늦게 도착하는 문제를 폴링에서 WebSocket 으로 바꿔 해결했다",
    "ME",
    [(201, "채팅 메시지가 늦게 도착하는 문제가 있었다. 내가 폴링을 WebSocket 으로 바꿔 해결했다.")],
    [
        (
            301,
            "폴링을 WebSocket 으로 바꿀 때 직접 고친 부분은 어디였나요?",
            "서버에 STOMP 엔드포인트를 만들고, 앱의 3초 폴링 코드를 구독 방식으로 바꿨다",
        )
    ],
)
EXP_102 = _exp(
    102,
    "매주 금요일 스프린트 회고를 했다",
    "TEAM",
    [(202, "팀은 매주 금요일 스프린트 회고를 했다.")],
)
EXP_141 = _exp(
    141, "혼자 만든 가계부 웹앱을 친구 5명이 한 달 동안 썼다", "ME", status="USER_ENTERED"
)
QUESTION = {
    "organization": "OO 백엔드 부트캠프",
    "text": "기술적인 문제를 해결한 경험을 구체적으로 적어 주세요.",
}
BODY = {"question": QUESTION, "experiences": [EXP_101]}

FIRST = {
    "kind": "FACT",
    "text": "「바꿔요」 프로젝트에서 채팅 메시지가 늦게 도착하는 문제가 있었고, 제가 폴링을 WebSocket 으로 바꿔 해결했습니다.",
    "evidence": [{"type": "CITATION", "id": 201}],
}
SECOND = {
    "kind": "FACT",
    "text": "서버에 STOMP 엔드포인트를 만들고, 앱의 3초 폴링 코드를 구독 방식으로 바꿨습니다.",
    "evidence": [{"type": "SUPPLEMENT", "id": 301}],
}
# 첫 줄 소제목(S-86) — 문장이 있으면 필수
TITLE = "폴링을 구독으로 바꿔 푼 채팅 지연"
GOOD = {
    "title": TITLE,
    "sentences": [FIRST, SECOND],
    "omitted": [
        {
            "text": "바꾼 뒤 메시지가 제때 오는지 확인한 방법",
            "question": "WebSocket 으로 바꾼 뒤 메시지가 제때 오는지 어떻게 확인했나요?",
            "experienceId": 101,
        },
        {"text": "Java · Spring 을 쓴 경험", "question": None, "experienceId": None},
        {
            "text": "이 경험에서 배운 점 · 느낀 점",
            "question": "채팅 방식을 바꾸며 배운 점이나 느낀 점이 있나요?",
            "experienceId": 101,
        },
    ],
}
OPINION = {
    "kind": "OPINION",
    "text": "앞으로도 문제의 원인을 살피는 개발자가 되고 싶습니다.",
    "evidence": [],
}
LOOSE = {"text": "지원 동기", "question": None, "experienceId": None}
ASKED = {"text": "확인한 방법", "question": "어떻게 확인했나요?", "experienceId": 101}


def _with(*sentences, omitted=()):
    return {
        "title": TITLE if sentences else None,
        "sentences": list(sentences),
        "omitted": list(omitted),
    }


def test_contract_example(client_with):
    llm = FakeLLM(GOOD)
    r = client_with(llm).post("/draft", json=BODY)
    assert r.status_code == 200
    # note · specific · placeholder · sentenceIndex 는 쓰지 않았어도 null 로 온다(S-78 · ADR-014)
    assert r.json() == {
        "title": TITLE,
        "sentences": [{**s, "note": None, "specific": None} for s in GOOD["sentences"]],
        "omitted": [{**o, "placeholder": None, "sentenceIndex": None} for o in GOOD["omitted"]],
        "guidelines": [],
        "model": "fake-model",
    }
    assert llm.calls == 1


def test_opinion_returned_with_kind(client_with):
    r = client_with(FakeLLM(_with(FIRST, SECOND, *[OPINION] * 3))).post("/draft", json=BODY)
    assert r.status_code == 200
    assert [s["kind"] for s in r.json()["sentences"]] == ["FACT", "FACT", *["OPINION"] * 3]
    assert r.json()["sentences"][-1]["evidence"] == []


def test_team_experience_written_as_team(client_with):
    """TEAM 경험 102 를 같이 골랐을 때 — 계약 「예시 읽는 법」의 문장 그대로 통과한다."""
    team = {
        "kind": "FACT",
        "text": "팀은 매주 금요일 스프린트 회고를 했습니다.",
        "evidence": [{"type": "CITATION", "id": 202}],
    }
    body = {"question": QUESTION, "experiences": [EXP_101, EXP_102]}
    r = client_with(FakeLLM(_with(FIRST, team))).post("/draft", json=body)
    assert r.status_code == 200
    assert r.json()["sentences"][1]["evidence"] == [{"type": "CITATION", "id": 202}]


def test_user_entered_body_is_evidence(client_with):
    out = _with(
        {
            "kind": "FACT",
            "text": "혼자 만든 가계부 웹앱을 친구 5명이 한 달 동안 썼습니다.",
            "evidence": [{"type": "EXPERIENCE", "id": 141}],
        }
    )
    body = {"question": QUESTION, "experiences": [EXP_141]}
    assert client_with(FakeLLM(out)).post("/draft", json=body).status_code == 200


def test_sentence_may_cite_two_experiences(client_with):
    out = _with(
        {
            "kind": "FACT",
            "text": "폴링을 WebSocket 으로 바꿨고, 팀은 매주 금요일 스프린트 회고를 했습니다.",
            "evidence": [{"type": "CITATION", "id": 201}, {"type": "CITATION", "id": 202}],
        }
    )
    body = {"question": QUESTION, "experiences": [EXP_101, EXP_102]}
    assert client_with(FakeLLM(out)).post("/draft", json=body).status_code == 200


def test_no_backing_allows_zero_sentences(client_with):
    """고른 경험에 받칠 것이 하나도 없으면 0개가 맞다 — omitted 만 돌려준다."""
    bare = _exp(105, "매주 풀 문제를 고른다", "ME", status="NEEDS_REVIEW")
    body = {"question": QUESTION, "experiences": [bare]}
    r = client_with(FakeLLM(_with(omitted=[LOOSE]))).post("/draft", json=body)
    assert r.status_code == 200
    assert r.json()["sentences"] == []


def test_no_backing_rejects_opinion_only(client_with):
    """받칠 것이 없으면 생각만으로 채우지 않는다 — 계약 「sentences」."""
    bare = _exp(105, "매주 풀 문제를 고른다", "ME", status="NEEDS_REVIEW")
    body = {"question": QUESTION, "experiences": [bare]}
    llm = FakeLLM(_with(OPINION), _with(OPINION))
    r = client_with(llm, max_attempts=2).post("/draft", json=body)
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "LLM_INVALID_OUTPUT"


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(_with(), id="받칠 것이 있는데 0개"),
        pytest.param(_with({**FIRST, "evidence": []}), id="FACT 근거 0개"),
        pytest.param(_with({k: v for k, v in FIRST.items() if k != "kind"}), id="kind 빠짐"),
        pytest.param(_with({**FIRST, "kind": "GUESS"}), id="kind 값 목록 밖"),
        pytest.param(_with(FIRST, {**OPINION, "evidence": FIRST["evidence"]}), id="OPINION 근거"),
        pytest.param(
            _with(FIRST, {**OPINION, "text": "3년 안에 해 보고 싶습니다."}), id="OPINION 숫자"
        ),
        pytest.param(
            _with(FIRST, SECOND, {**OPINION, "text": "앱의 3초 폴링처럼 살피겠습니다."}),
            id="근거에 있는 숫자도 OPINION 엔 금지",
        ),
        pytest.param(_with(OPINION), id="OPINION 만"),
        pytest.param(_with(FIRST, *[OPINION] * 4), id="OPINION 4개"),
        pytest.param(_with({**SECOND, "text": "  "}), id="빈 문장"),
        pytest.param(_with({**SECOND, "text": "한 줄\n두 줄"}), id="줄바꿈"),
        pytest.param(
            _with({**FIRST, "evidence": [{"type": "CITATION", "id": 999}]}), id="없는 근거"
        ),
        pytest.param(
            _with({**FIRST, "evidence": [{"type": "SUPPLEMENT", "id": 201}]}),
            id="근거 ID 를 보완으로",
        ),
        pytest.param(
            _with({**FIRST, "evidence": [{"type": "EXPERIENCE", "id": 101}]}),
            id="USER_ENTERED 아닌 본문",
        ),
        pytest.param(_with({**FIRST, "evidence": [{"type": "OTHER", "id": 201}]}), id="값 목록 밖"),
        pytest.param(_with({**FIRST, "evidence": FIRST["evidence"] * 2}), id="근거 중복"),
        pytest.param(
            _with({**SECOND, "text": "응답 시간을 30% 줄였습니다."}), id="근거에 없는 숫자"
        ),
        pytest.param(
            _with({**SECOND, "text": "앱의 30초 폴링 코드를 바꿨습니다."}),
            id="숫자가 다른 숫자의 일부로만 있음",
        ),
        pytest.param(_with(SECOND, omitted=[LOOSE] * 11), id="omitted 11개"),
        pytest.param(_with(SECOND, omitted=["확인한 방법"]), id="옛 문자열 omitted"),
        pytest.param(_with(SECOND, omitted=[{"text": "확인한 방법"}]), id="omitted 칸 빠짐"),
        pytest.param(_with(SECOND, omitted=[{**LOOSE, "text": " "}]), id="빈 omitted"),
        pytest.param(_with(SECOND, omitted=[{**ASKED, "question": " "}]), id="빈 질문"),
        pytest.param(_with(SECOND, omitted=[{**ASKED, "experienceId": 999}]), id="요청 밖 경험"),
        pytest.param(_with(SECOND, omitted=[{**ASKED, "experienceId": None}]), id="질문만 있음"),
        pytest.param(_with(SECOND, omitted=[{**ASKED, "question": None}]), id="경험만 있음"),
        pytest.param(_with(*[SECOND] * 16), id="문장 16개"),
    ],
)
def test_rule_violation_retries_then_fails(client_with, bad):
    llm = FakeLLM(bad, bad)
    r = client_with(llm, max_attempts=2).post("/draft", json=BODY)
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "LLM_INVALID_OUTPUT"
    assert llm.calls == 2


def test_retry_recovers(client_with):
    bad = _with({**SECOND, "text": "응답 시간을 30% 줄였습니다."})
    llm = FakeLLM(bad, GOOD)
    r = client_with(llm, max_attempts=2).post("/draft", json=BODY)
    assert r.status_code == 200
    assert llm.calls == 2


def test_number_may_come_from_activity_and_period():
    """활동 이름 · 기간의 숫자는 그 경험이면 받친다 — 「2025년 3월」(기간 2025-03)."""
    out = _with(
        {
            "kind": "FACT",
            "text": "2025년 3월부터 6월까지 채팅 문제를 해결했습니다.",
            "evidence": [{"type": "CITATION", "id": 201}],
        }
    )
    check_output(LLMDraftOutput.model_validate(out), [MatchExperience.model_validate(EXP_101)])


def test_number_in_other_evidence_does_not_back():
    """보완 301 의 「3초」는 근거 201 만 붙인 문장을 받치지 못한다 — 기간 2025-03 의 3 도 못 받친다."""
    out = _with(
        {"kind": "FACT", "text": "앱의 3초 폴링 코드를 바꿨습니다.", "evidence": FIRST["evidence"]}
    )
    with pytest.raises(ValueError, match="숫자"):
        check_output(LLMDraftOutput.model_validate(out), [MatchExperience.model_validate(EXP_101)])


def test_supplement_question_does_not_back_numbers():
    """보완의 질문은 사실이 아니다 — 답에 있는 숫자만 받친다."""
    exp = _exp(101, "본문", "ME", supplements=[(301, "3초 폴링을 어떻게 바꿨나요?", "구독으로")])
    out = _with(
        {
            "kind": "FACT",
            "text": "3초 폴링을 바꿨습니다.",
            "evidence": [{"type": "SUPPLEMENT", "id": 301}],
        }
    )
    with pytest.raises(ValueError, match="숫자"):
        check_output(LLMDraftOutput.model_validate(out), [MatchExperience.model_validate(exp)])


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("3초", {"3"}),
        ("1,200명 · 3.5배 · 80%", {"1200", "3.5", "80"}),
        ("２０２５-03", {"2025", "3"}),
        ("3,4주차", {"3", "4"}),
        ("숫자 없음.", set()),
        ("세 번 · 두 달간 · 열두 명이서 · 스무 살", {"3", "2", "12", "20"}),
        ("세번 고쳤고 다섯 곳에 두 배로", {"3", "5", "2"}),
    ],
)
def test_numbers(text, want):
    assert numbers(text) == want


@pytest.mark.parametrize(
    "text",
    [
        "한 번 더 도전하고 싶습니다.",
        "한 달 동안 한 명씩 만났습니다.",
        "두 번째로 배운 점은 소통입니다.",
        "세 가지를 배웠습니다.",
        "두세 번 고쳤습니다.",
        "네다섯 명이 모였습니다.",
        "한두 달 걸렸습니다.",
        "두 개발자가 세 번역본을 맡았습니다.",
        "네, 맞습니다. 둘 다 좋습니다.",
        "세상을 바꾸고 두려움을 이겼습니다.",
    ],
)
def test_words_that_are_not_numbers(text):
    assert numbers(text) == set()


def test_unsupported_numbers():
    assert unsupported_numbers("1200명 중 3명", "1,200명이 왔다") == ["3"]
    assert unsupported_numbers("세 번", "") == ["3"]
    assert unsupported_numbers("세 번 고쳤다", "3번 고쳤다") == []
    assert unsupported_numbers("3번 고쳤다", "세 번 고쳤다") == []


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("2025년 3월부터 6월까지", []),
        ("2025.03 ~ 2025-06", []),
        ("3초로 줄였습니다", ["3"]),
        ("3명이 3개월 동안", ["3"]),
        ("3년 동안", ["3"]),
        ("3월에 3초로 줄였습니다", ["3"]),
        ("4월에 시작했습니다", ["4"]),
        ("2024년에 시작했습니다", ["2024"]),
        ("세 번 고쳤습니다", ["3"]),
    ],
)
def test_period_backs_only_dates(text, want):
    assert unsupported_numbers(text, "", ["2025-03", "2025-06"]) == want


def test_year_only_period_backs_year():
    assert unsupported_numbers("2025년 3월", "", ["2025"]) == ["3"]


@pytest.mark.parametrize(
    "experiences",
    [
        pytest.param([], id="0개"),
        pytest.param([_exp(i, "본문", "ME", [(200 + i, "발췌")]) for i in range(1, 10)], id="9개"),
        pytest.param([EXP_101, EXP_101], id="중복"),
        pytest.param(None, id="없음"),
    ],
)
def test_bad_experiences(client_with, experiences):
    llm = FakeLLM()
    r = client_with(llm).post("/draft", json={"question": QUESTION, "experiences": experiences})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "INVALID_REQUEST"
    assert llm.calls == 0


def test_blank_question(client_with):
    body = {"question": {"text": " "}, "experiences": [EXP_101]}
    assert client_with(FakeLLM()).post("/draft", json=body).status_code == 400


def test_text_too_long(client_with):
    r = client_with(FakeLLM(), max_match_chars=10).post("/draft", json=BODY)
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "TEXT_TOO_LONG"


def test_prompt_gets_only_picked_experiences(client_with):
    seen = {}

    class Spy(FakeLLM):
        async def generate_json(self, system, user, schema, deadline):
            seen["user"] = user
            return await super().generate_json(system, user, schema, deadline)

    client_with(Spy(GOOD)).post("/draft", json=BODY)
    assert "근거 201" in seen["user"] and "보완 301" in seen["user"]
    assert seen["user"].count("[경험 ") == 1


# ---- 매칭 결과 · 직무(ETC#141) — 선택 칸이고 근거가 아니다 ----

HINTED_101 = {
    **EXP_101,
    "reason": "채팅 지연이라는 기술 문제를 직접 해결한 사례라 문항과 맞는다",
    "emphasis": "폴링의 한계를 보고 WebSocket 으로 바꿔 지연을 40% 줄인 판단",
    "gaps": ["직접 고친 범위가 자료에 없다", "고친 뒤 확인한 방법이 자료에 없다"],
}
HINTED = {
    "question": {
        **QUESTION,
        "jobRole": "BACKEND",
        "requirements": [
            {"text": "기술 문제의 원인을 찾아 해결한 경험", "status": "SUPPORTED"},
            {"text": "Java · Spring 사용 경험", "status": "NOT_FOUND"},
        ],
    },
    "experiences": [HINTED_101],
}


def _spy():
    seen = {}

    class Spy(FakeLLM):
        async def generate_json(self, system, user, schema, deadline):
            seen["user"] = user
            return await super().generate_json(system, user, schema, deadline)

    return Spy, seen


def test_hints_go_to_prompt_as_clues(client_with):
    Spy, seen = _spy()
    r = client_with(Spy(GOOD)).post("/draft", json=HINTED)
    assert r.status_code == 200
    user = seen["user"]
    assert "<매칭 단서>" in user and "지원 직무: BACKEND" in user
    assert "Java · Spring 사용 경험 — 자료에 없음" in user
    assert "강조점: 폴링의 한계" in user and "부족한 점: 직접 고친 범위" in user
    # 단서는 경험 목록 뒤에 따로 — 근거 · 보완 줄에 섞이지 않는다
    assert user.index("</경험 목록>") < user.index("<매칭 단서>")


def test_without_hints_prompt_is_unchanged(client_with):
    Spy, seen = _spy()
    client_with(Spy(GOOD)).post("/draft", json=BODY)
    question, exps, _ = check_request(DraftRequest.model_validate(BODY), load_settings())
    assert seen["user"] == build_user_message(question, exps)
    assert "<매칭 단서>" not in seen["user"]


def test_null_job_role_and_empty_hints_are_unchanged():
    body = {
        "question": {**QUESTION, "jobRole": None, "requirements": []},
        "experiences": [{**EXP_101, "gaps": []}],
    }
    question, exps, _ = check_request(DraftRequest.model_validate(body), load_settings())
    assert build_draft_message(question, exps) == build_user_message(question, exps)


def test_hints_do_not_back_numbers():
    # 강조점에만 있는 40 은 FACT 에 못 쓴다 — 숫자 대조의 받치는 글에 넣지 않는다
    exps = DraftRequest.model_validate(HINTED).experiences
    bad = LLMDraftOutput.model_validate(
        _with({**FIRST, "text": "제가 폴링을 WebSocket 으로 바꿔 지연을 40% 줄였습니다."})
    )
    with pytest.raises(ValueError, match="없는 숫자"):
        check_output(bad, exps)
    assert all("40" not in b.text for b in backings(exps).values())


def test_hint_number_in_fact_retries_then_fails(client_with):
    bad = _with({**FIRST, "text": "제가 폴링을 WebSocket 으로 바꿔 지연을 40% 줄였습니다."})
    llm = FakeLLM(bad, bad)
    r = client_with(llm, max_attempts=2).post("/draft", json=HINTED)
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "LLM_INVALID_OUTPUT"


@pytest.mark.parametrize(
    "question, experience",
    [
        pytest.param({"jobRole": "CHEF"}, {}, id="jobRole 목록 밖"),
        pytest.param({"requirements": [{"text": "x", "status": "DONE"}]}, {}, id="status 목록 밖"),
        pytest.param(
            {"requirements": [{"text": " ", "status": "SUPPORTED"}]}, {}, id="빈 요구사항"
        ),
        pytest.param({}, {"gaps": "직접 고친 범위"}, id="gaps 문자열"),
        pytest.param({}, {"emphasis": " "}, id="빈 강조점"),
    ],
)
def test_bad_hints(client_with, question, experience):
    llm = FakeLLM()
    body = {"question": {**QUESTION, **question}, "experiences": [{**EXP_101, **experience}]}
    r = client_with(llm).post("/draft", json=body)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "INVALID_REQUEST"
    assert llm.calls == 0


def test_hints_count_toward_text_limit(client_with):
    limit = len(QUESTION["text"]) + experience_chars(DraftRequest.model_validate(BODY).experiences)
    assert (
        client_with(FakeLLM(GOOD), max_match_chars=limit).post("/draft", json=BODY).status_code
        == 200
    )
    r = client_with(FakeLLM(GOOD), max_match_chars=limit).post("/draft", json=HINTED)
    assert r.status_code == 413


# ---- 구성 틀 · 작성 지침 (S-49 · S-50) ----

GUIDES = [
    {"id": 901, "text": "첫 문장에 무엇을 했고 어떻게 됐는지 먼저 쓴다"},
    {"id": 905, "text": "결과는 꼭 수치로 쓴다"},
]
GUIDED = {"question": {**QUESTION, "frame": "STAR"}, "experiences": [EXP_101], "guidelines": GUIDES}
RESULTS = [
    {"id": 901, "applied": True, "note": None},
    {"id": 905, "applied": False, "note": "결과를 보여 줄 수치가 자료에 없어 넣지 않았어요"},
]


def test_guided_contract_example(client_with):
    Spy, seen = _spy()
    llm = Spy({**GOOD, "guidelines": RESULTS})
    r = client_with(llm).post("/draft", json=GUIDED)
    assert r.status_code == 200
    assert r.json()["guidelines"] == RESULTS
    assert "<구성 틀>\nSTAR: 상황" in seen["user"]
    assert "- [905] 결과는 꼭 수치로 쓴다" in seen["user"]
    # 틀 · 지침은 단서 뒤 맨 끝 — 경험 목록에 섞이지 않는다
    assert seen["user"].index("</경험 목록>") < seen["user"].index("<작성 지침>")


@pytest.mark.parametrize(
    "frame,first_step",
    [
        ("MOTIVE_LINK", "지원처에서 본 것"),
        ("CAUSE_FIX", "한 줄 요약"),
        ("ALIGN", "함께 하던 일"),
        ("CHANGE", "기존 방식"),
        ("SKILL_STEPS", "역량 한 줄"),
        ("ORIGIN", "지금의 나 한 줄"),
    ],
)
def test_structure_card_steps_are_in_prompt(client_with, frame, first_step):
    # 구조 카드(S-82) — 계약 README 「구성 틀」 단계 문구가 <구성 틀> 에 실린다
    Spy, seen = _spy()
    body = {**GUIDED, "question": {**QUESTION, "frame": frame}, "guidelines": []}
    r = client_with(Spy(GOOD)).post("/draft", json=body)
    assert r.status_code == 200
    assert f"<구성 틀>\n{frame}: {first_step}" in seen["user"]


def test_none_frame_and_no_guidelines_are_unchanged():
    body = {"question": {**QUESTION, "frame": "NONE"}, "experiences": [EXP_101], "guidelines": []}
    question, exps, gs = check_request(DraftRequest.model_validate(body), load_settings())
    assert build_draft_message(question, exps, gs) == build_user_message(question, exps)
    assert schema_for(gs) is LLM_JSON_SCHEMA


def test_schema_asks_guidelines_only_when_given():
    gs = DraftRequest.model_validate(GUIDED).guidelines
    schema = schema_for(gs)
    assert "guidelines" in schema["required"]
    assert schema["properties"]["guidelines"]["items"]["properties"]["id"]["enum"] == [901, 905]


@pytest.mark.parametrize(
    "results",
    [
        pytest.param(RESULTS[::-1], id="순서 다름"),
        pytest.param(RESULTS[:1], id="하나 빠짐"),
        pytest.param([*RESULTS, {"id": 999, "applied": True, "note": None}], id="없는 id"),
        pytest.param([{**RESULTS[0], "note": "했어요"}, RESULTS[1]], id="true 인데 note"),
        pytest.param([RESULTS[0], {**RESULTS[1], "note": None}], id="false 인데 note 없음"),
        pytest.param([RESULTS[0], {**RESULTS[1], "note": "가" * 81}], id="note 80자 넘음"),
        pytest.param([RESULTS[0], {**RESULTS[1], "note": "가\n나"}], id="note 줄바꿈"),
    ],
)
def test_bad_guideline_results_retry_then_fail(client_with, results):
    bad = {**GOOD, "guidelines": results}
    llm = FakeLLM(bad, bad)
    r = client_with(llm, max_attempts=2).post("/draft", json=GUIDED)
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "LLM_INVALID_OUTPUT"
    assert llm.calls == 2


def test_guideline_results_without_request_guidelines_fail():
    exps = DraftRequest.model_validate(BODY).experiences
    with pytest.raises(ValueError, match="guidelines id"):
        check_output(LLMDraftOutput.model_validate({**GOOD, "guidelines": RESULTS}), exps)


def test_guideline_does_not_back_numbers(client_with):
    # 「수치로」 지침이 있어도 근거에 없는 숫자는 코드가 막는다 — 지침은 받치는 글이 아니다
    bad = {
        **_with({**FIRST, "text": "제가 폴링을 WebSocket 으로 바꿔 지연을 40% 줄였습니다."}),
        "guidelines": RESULTS,
    }
    llm = FakeLLM(bad, bad)
    r = client_with(llm, max_attempts=2).post("/draft", json=GUIDED)
    assert r.status_code == 502


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"question": {**QUESTION, "frame": "SWOT"}}, id="frame 목록 밖"),
        pytest.param(
            {"guidelines": [{"id": i, "text": f"지침 {i}"} for i in range(11)]}, id="지침 11개"
        ),
        pytest.param({"guidelines": [GUIDES[0], {**GUIDES[1], "id": 901}]}, id="지침 id 중복"),
        pytest.param({"guidelines": [{"id": 1, "text": " "}]}, id="빈 지침"),
    ],
)
def test_bad_style_request(client_with, change):
    llm = FakeLLM()
    r = client_with(llm).post("/draft", json={**GUIDED, **change})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "INVALID_REQUEST"
    assert llm.calls == 0


def test_guidelines_count_toward_text_limit(client_with):
    limit = len(QUESTION["text"]) + experience_chars(DraftRequest.model_validate(BODY).experiences)
    r = client_with(FakeLLM(GOOD), max_match_chars=limit).post("/draft", json=GUIDED)
    assert r.status_code == 413


def test_korean_number_in_opinion_fails():
    """한글 수사도 숫자다 — 생각 문장에 쓰면 아라비아 숫자와 같이 막는다."""
    out = _with(FIRST, {"kind": "OPINION", "text": "세 번 더 도전하겠습니다.", "evidence": []})
    with pytest.raises(ValueError, match="숫자"):
        check_output(LLMDraftOutput.model_validate(out), [MatchExperience.model_validate(EXP_101)])


# ---- AI 보탬 · 보조 재료 · 빈칸 · 목표 글자 수 (S-75 · S-77 · S-78) ----

CTX_103 = _exp(
    103,
    "관리자 페이지를 만들어 운영 시간을 줄였다",
    "ME",
    [(203, "관리자 페이지를 직접 만들어 운영 부담을 덜었다.")],
    [(303, "관리자 페이지에서 무엇을 만들었나요?", "신고 목록과 차단 버튼을 만들었다")],
)
CTX_143 = _exp(143, "혼자 배포 스크립트를 짰다", "ME", status="USER_ENTERED")
WITH_CTX = {**BODY, "contextExperiences": [CTX_103]}
ELAB = {
    "kind": "ELABORATED",
    "text": "당시 채팅이 늦어지면 사용자가 불편했을 것이라 이 문제를 먼저 풀어야 했습니다.",
    "evidence": [{"type": "CITATION", "id": 201}],
    "note": "문제의 영향은 자료에 없어 풀어 쓴 문장입니다",
}
CTX_FACT = {
    "kind": "FACT",
    "text": "신고 목록과 차단 버튼을 만들었습니다.",
    "evidence": [{"type": "SUPPLEMENT", "id": 303}],
}
MARKED = {
    "kind": "FACT",
    "text": "폴링을 WebSocket 으로 바꿔 지연을 [줄인 정도]만큼 줄였습니다.",
    "evidence": [{"type": "CITATION", "id": 201}],
}
BLANK = {
    "text": "바꾼 뒤 줄어든 지연",
    "question": None,
    "experienceId": None,
    "placeholder": "[줄인 정도]",
    "sentenceIndex": 0,
}


def _models(*exps):
    return [MatchExperience.model_validate(e) for e in exps]


def _check(out, exps=(EXP_101,), context=()):
    return check_output(LLMDraftOutput.model_validate(out), _models(*exps), (), _models(*context))


def test_elaborated_context_placeholder_end_to_end(client_with):
    out = _with(
        FIRST,
        ELAB,
        CTX_FACT,
        MARKED,
        omitted=[{**BLANK, "sentenceIndex": 3}, {**ASKED, "experienceId": 103}],
    )
    body = {**WITH_CTX, "question": {**QUESTION, "targetLength": 600}}
    llm = FakeLLM(out)
    r = client_with(llm).post("/draft", json=body)
    assert r.status_code == 200
    got = r.json()
    assert [s["kind"] for s in got["sentences"]] == ["FACT", "ELABORATED", "FACT", "FACT"]
    assert got["sentences"][1]["note"] == ELAB["note"]
    assert got["sentences"][0]["note"] is None
    assert got["sentences"][2]["evidence"] == [{"type": "SUPPLEMENT", "id": 303}]
    assert got["omitted"][0]["placeholder"] == "[줄인 정도]"
    assert got["omitted"][0]["sentenceIndex"] == 3
    assert got["omitted"][1]["placeholder"] is None
    assert got["omitted"][1]["experienceId"] == 103
    assert llm.calls == 1


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(_with(FIRST, {**ELAB, "evidence": []}), id="ELABORATED 근거 0개"),
        pytest.param(_with(FIRST, {**ELAB, "note": None}), id="ELABORATED note 없음"),
        pytest.param(_with(FIRST, {**ELAB, "note": " "}), id="ELABORATED note 공백"),
        pytest.param(_with(FIRST, {**ELAB, "note": "가" * 61}), id="note 60자 넘음"),
        pytest.param(_with(FIRST, {**ELAB, "note": "가\n나"}), id="note 줄바꿈"),
        pytest.param(_with({**FIRST, "note": "왜 확인이 필요한지"}), id="FACT 에 note"),
        pytest.param(_with(FIRST, {**OPINION, "note": "생각"}), id="OPINION 에 note"),
        pytest.param(
            _with(FIRST, {**ELAB, "text": "지연을 30% 줄여야 했습니다."}), id="ELABORATED 없는 숫자"
        ),
        pytest.param(
            _with(FIRST, {**ELAB, "evidence": [{"type": "CITATION", "id": 999}]}),
            id="ELABORATED 없는 근거",
        ),
    ],
)
def test_elaborated_rules_retry_then_fail(client_with, bad):
    llm = FakeLLM(bad, bad)
    r = client_with(llm, max_attempts=2).post("/draft", json=BODY)
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "LLM_INVALID_OUTPUT"
    assert llm.calls == 2


def test_elaborated_note_60_chars_ok():
    _check(_with(FIRST, {**ELAB, "note": "가" * 60}))


def test_elaborated_counts_as_written_sentence():
    """FACT 없이 ELABORATED + OPINION 은 통과, OPINION 만은 막힌다."""
    _check(_with(ELAB, OPINION))
    with pytest.raises(ValueError, match="생각 · 가정 문장만"):
        _check(_with(OPINION))


def test_elaborated_number_backed_like_fact():
    text = "앱의 3초 폴링을 바꿔야 했습니다."
    _check(_with(FIRST, {**ELAB, "text": text, "evidence": [{"type": "SUPPLEMENT", "id": 301}]}))
    # 보완 301 의 3초 는 근거 201 만 붙인 ELABORATED 를 받치지 못한다
    with pytest.raises(ValueError, match="숫자"):
        _check(_with(FIRST, {**ELAB, "text": text}))


def test_context_evidence_allowed():
    _check(_with(FIRST, CTX_FACT), context=[CTX_103])
    cite = {
        **CTX_FACT,
        "text": "관리자 페이지를 직접 만들었습니다.",
        "evidence": [{"type": "CITATION", "id": 203}],
    }
    _check(_with(FIRST, cite), context=[CTX_103])


def test_context_user_entered_body_is_evidence():
    out = _with(
        FIRST,
        {
            "kind": "FACT",
            "text": "혼자 배포 스크립트를 짰습니다.",
            "evidence": [{"type": "EXPERIENCE", "id": 143}],
        },
    )
    _check(out, context=[CTX_143])


def test_context_evidence_without_context_is_foreign():
    with pytest.raises(ValueError, match="근거"):
        _check(_with(FIRST, CTX_FACT))


def test_context_number_needs_backing():
    bad = {**CTX_FACT, "text": "신고 목록과 차단 버튼 2개를 만들었습니다."}
    with pytest.raises(ValueError, match="숫자"):
        _check(_with(FIRST, bad), context=[CTX_103])


def test_omitted_may_point_to_context_experience():
    out = _with(FIRST, omitted=[{**ASKED, "experienceId": 103}])
    _check(out, context=[CTX_103])
    with pytest.raises(ValueError, match="경험 ID"):
        _check(out)
    with pytest.raises(ValueError, match="경험 ID"):
        _check(_with(FIRST, omitted=[{**ASKED, "experienceId": 999}]), context=[CTX_103])


def test_zero_sentences_judged_by_chosen_backing_only():
    bare = _exp(105, "매주 풀 문제를 고른다", "ME", status="NEEDS_REVIEW")
    # 보조 재료에만 받칠 것이 있어도 고른 경험에 없으면 0개가 맞다
    _check(_with(omitted=[LOOSE]), exps=[bare], context=[CTX_103])
    with pytest.raises(ValueError, match="0개"):
        _check(_with(omitted=[LOOSE]), exps=[EXP_101], context=[CTX_103])


@pytest.mark.parametrize(
    "target",
    [
        pytest.param(99, id="99"),
        pytest.param(5001, id="5001"),
        pytest.param(0, id="0"),
        pytest.param("많이", id="문자열"),
    ],
)
def test_bad_target_length(client_with, target):
    llm = FakeLLM()
    body = {"question": {**QUESTION, "targetLength": target}, "experiences": [EXP_101]}
    r = client_with(llm).post("/draft", json=body)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "INVALID_REQUEST"
    assert llm.calls == 0


@pytest.mark.parametrize("target", [100, 5000])
def test_target_length_bounds_ok(client_with, target):
    body = {"question": {**QUESTION, "targetLength": target}, "experiences": [EXP_101]}
    assert client_with(FakeLLM(GOOD)).post("/draft", json=body).status_code == 200


@pytest.mark.parametrize(
    "context",
    [
        pytest.param([_exp(300 + i, "본문", "ME") for i in range(13)], id="13개"),
        pytest.param([CTX_103, CTX_103], id="중복"),
        pytest.param([EXP_101], id="고른 경험과 겹침"),
        pytest.param("없음", id="배열 아님"),
    ],
)
def test_bad_context(client_with, context):
    llm = FakeLLM()
    r = client_with(llm).post("/draft", json={**BODY, "contextExperiences": context})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "INVALID_REQUEST"
    assert llm.calls == 0


def test_context_12_ok_and_empty_ok(client_with):
    twelve = [_exp(300 + i, "본문", "ME") for i in range(12)]
    ok = client_with(FakeLLM(GOOD)).post("/draft", json={**BODY, "contextExperiences": twelve})
    assert ok.status_code == 200
    empty = client_with(FakeLLM(GOOD)).post("/draft", json={**BODY, "contextExperiences": []})
    assert empty.status_code == 200


def test_context_counts_toward_text_limit(client_with):
    base = len(QUESTION["text"]) + experience_chars(DraftRequest.model_validate(BODY).experiences)
    r = client_with(FakeLLM(GOOD), max_match_chars=base).post("/draft", json=WITH_CTX)
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "TEXT_TOO_LONG"
    full = base + experience_chars(DraftRequest.model_validate(WITH_CTX).contextExperiences)
    r = client_with(FakeLLM(GOOD), max_match_chars=full).post("/draft", json=WITH_CTX)
    assert r.status_code == 200


def test_context_and_target_go_to_prompt(client_with):
    Spy, seen = _spy()
    body = {**WITH_CTX, "question": {**QUESTION, "targetLength": 600}}
    assert client_with(Spy(GOOD)).post("/draft", json=body).status_code == 200
    user = seen["user"]
    assert "<보조 재료>" in user and "근거 203" in user and "보완 303" in user
    assert "<목표 글자 수>\n600자" in user
    assert user.index("</경험 목록>") < user.index("<보조 재료>")
    assert user.count("[경험 ") == 2


def test_context_alone_and_target_alone_blocks(client_with):
    Spy, seen = _spy()
    client_with(Spy(GOOD)).post("/draft", json=WITH_CTX)
    assert "<보조 재료>" in seen["user"] and "<목표 글자 수>" not in seen["user"]
    Spy, seen = _spy()
    body = {"question": {**QUESTION, "targetLength": 300}, "experiences": [EXP_101]}
    client_with(Spy(GOOD)).post("/draft", json=body)
    assert "<목표 글자 수>" in seen["user"] and "<보조 재료>" not in seen["user"]


def test_neither_context_nor_target_message_unchanged():
    body = {**BODY, "contextExperiences": [], "question": {**QUESTION, "targetLength": None}}
    question, exps, gs = check_request(DraftRequest.model_validate(body), load_settings())
    assert build_draft_message(question, exps, gs, []) == build_user_message(question, exps)


def test_context_lines_same_as_experience_lines():
    req = DraftRequest.model_validate(WITH_CTX)
    question, exps, gs = check_request(req, load_settings())
    msg = build_draft_message(question, exps, gs, req.contextExperiences)
    for line in experience_lines(req.contextExperiences)[1:-1]:
        assert line in msg


@pytest.mark.parametrize(
    "omitted",
    [
        pytest.param([{**BLANK, "sentenceIndex": None}], id="sentenceIndex 만 없음"),
        pytest.param([{**BLANK, "placeholder": None}], id="placeholder 만 없음"),
        pytest.param([{**BLANK, "placeholder": "줄인 정도"}], id="대괄호 없음"),
        pytest.param([{**BLANK, "placeholder": "[ ]"}], id="속이 공백"),
        pytest.param([{**BLANK, "placeholder": "[]"}], id="속이 없음"),
        pytest.param([{**BLANK, "placeholder": "[가나다라마바사아자차카]"}], id="13자"),
        pytest.param([{**BLANK, "placeholder": "[[줄인 정도]]"}], id="중첩"),
        pytest.param([{**BLANK, "sentenceIndex": 1}], id="범위 밖"),
        pytest.param([{**BLANK, "sentenceIndex": -1}], id="음수 인덱스"),
        pytest.param([{**BLANK, "placeholder": "[다른 값]"}], id="문장에 없음"),
        pytest.param([], id="문장 속 빈칸에 짝 없음"),
        pytest.param([BLANK, BLANK], id="같은 짝 중복"),
    ],
)
def test_broken_placeholder_retries_then_fails(client_with, omitted):
    bad = _with(MARKED, omitted=omitted)
    llm = FakeLLM(bad, bad)
    r = client_with(llm, max_attempts=2).post("/draft", json=BODY)
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "LLM_INVALID_OUTPUT"
    assert llm.calls == 2


def test_placeholder_12_chars_ok_and_twice_in_sentence_fails():
    mark = "[가나다라마바사아자차]"
    assert len(mark) == 12
    blank = {**BLANK, "placeholder": mark}
    _check(_with({**FIRST, "text": f"지연을 {mark}만큼 줄였습니다."}, omitted=[blank]))
    with pytest.raises(ValueError, match="한 번"):
        _check(_with({**FIRST, "text": f"{mark}와 {mark}"}, omitted=[blank]))


def test_placeholder_pairs_by_sentence_index():
    second = {
        **SECOND,
        "text": "앱의 3초 폴링을 [바꾼 범위]까지 바꿨습니다.",
    }
    other = {**BLANK, "placeholder": "[바꾼 범위]", "sentenceIndex": 1}
    _check(_with(MARKED, second, omitted=[BLANK, other]))
    # 같은 글자라도 다른 문장이면 다른 짝이다
    with pytest.raises(ValueError):
        _check(_with(MARKED, second, omitted=[BLANK, {**other, "sentenceIndex": 0}]))


def test_placeholder_recovers_on_retry(client_with):
    llm = FakeLLM(_with(MARKED), _with(MARKED, omitted=[BLANK]))
    r = client_with(llm, max_attempts=2).post("/draft", json=BODY)
    assert r.status_code == 200
    assert r.json()["omitted"][0]["placeholder"] == "[줄인 정도]"
    assert llm.calls == 2


# ---- AI 가정(SUPPOSED · ADR-014) ----

GUESS = {
    "kind": "SUPPOSED",
    "text": "바꾼 뒤 직접 메시지를 주고받으며 지연이 사라졌는지 확인했습니다.",
    "evidence": [{"type": "CITATION", "id": 201}],
    "specific": False,
}
GUESS_NUM = {
    "kind": "SUPPOSED",
    "text": "바꾼 뒤 메시지가 오기까지 걸리는 시간이 1초 안으로 줄었습니다.",
    "evidence": [],
    "specific": True,
}
PAIR = {
    "text": "바꾼 뒤 지연이 사라졌는지 확인한 방법",
    "question": "바꾼 뒤 메시지를 주고받으며 지연이 사라졌는지 확인했나요?",
    "experienceId": 101,
    "placeholder": None,
    "sentenceIndex": 1,
}


def test_supposed_end_to_end(client_with):
    out = _with(FIRST, GUESS, GUESS_NUM, omitted=[PAIR, {**PAIR, "sentenceIndex": 2}, LOOSE])
    r = client_with(FakeLLM(out)).post("/draft", json=BODY)
    assert r.status_code == 200
    got = r.json()
    assert [s["kind"] for s in got["sentences"]] == ["FACT", "SUPPOSED", "SUPPOSED"]
    assert [s["specific"] for s in got["sentences"]] == [None, False, True]
    assert [o["sentenceIndex"] for o in got["omitted"]] == [1, 2, None]
    assert all(o["placeholder"] is None for o in got["omitted"])


def test_supposed_may_outnumber_facts_and_skips_number_check():
    """가정은 FACT 보다 많아도 되고, 근거에 없는 숫자도 specific 이면 통과한다."""
    guesses = [{**GUESS_NUM, "text": f"{n}번째 시도에서 원인을 찾았습니다."} for n in (2, 3, 4)]
    pairs = [{**PAIR, "sentenceIndex": i} for i in (1, 2, 3)]
    _check(_with(FIRST, *guesses, omitted=pairs))


def test_supposed_pair_without_experience_ok():
    _check(_with(FIRST, GUESS, omitted=[{**PAIR, "question": None, "experienceId": None}]))


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(_with(FIRST, GUESS), id="짝 없음"),
        pytest.param(_with(FIRST, GUESS, omitted=[PAIR, PAIR]), id="짝 둘"),
        pytest.param(
            _with(FIRST, GUESS, omitted=[PAIR, {**PAIR, "sentenceIndex": 0}]),
            id="FACT 에 가정 짝",
        ),
        pytest.param(_with(FIRST, GUESS, omitted=[{**PAIR, "sentenceIndex": 5}]), id="범위 밖 짝"),
        pytest.param(
            _with(FIRST, {k: v for k, v in GUESS.items() if k != "specific"}, omitted=[PAIR]),
            id="specific 없음",
        ),
        pytest.param(
            _with(FIRST, {**GUESS_NUM, "specific": False}, omitted=[PAIR]),
            id="숫자 있는데 specific false",
        ),
        pytest.param(_with({**FIRST, "specific": False}), id="FACT 에 specific"),
        pytest.param(_with(FIRST, {**OPINION, "specific": False}), id="OPINION 에 specific"),
        pytest.param(
            _with(FIRST, {**GUESS, "note": "확인해 주세요"}, omitted=[PAIR]), id="가정에 note"
        ),
        pytest.param(
            _with(FIRST, {**GUESS, "evidence": [{"type": "CITATION", "id": 999}]}, omitted=[PAIR]),
            id="가정의 없는 근거",
        ),
        pytest.param(
            _with(FIRST, {**GUESS, "evidence": GUESS["evidence"] * 2}, omitted=[PAIR]),
            id="가정의 근거 중복",
        ),
        pytest.param(
            _with({**GUESS, "evidence": []}, omitted=[{**PAIR, "sentenceIndex": 0}]),
            id="받칠 것이 있는데 가정만",
        ),
        pytest.param(
            _with(
                {**GUESS, "evidence": []},
                OPINION,
                omitted=[{**PAIR, "sentenceIndex": 0}],
            ),
            id="가정 · 생각만",
        ),
        pytest.param(
            _with(FIRST, GUESS, omitted=[{**PAIR, "placeholder": "[결과]", "sentenceIndex": None}]),
            id="placeholder 만 있음",
        ),
    ],
)
def test_supposed_rules_retry_then_fail(client_with, bad):
    llm = FakeLLM(bad, bad)
    r = client_with(llm, max_attempts=2).post("/draft", json=BODY)
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "LLM_INVALID_OUTPUT"
    assert llm.calls == 2


def test_supposed_korean_count_needs_specific():
    """「두 번」 같은 한글 수량도 숫자 대조와 같은 기준으로 숫자다 — specific 이 true 여야 한다."""
    text = "발표 전에 팀원 앞에서 두 번 미리 해 봤습니다."
    with pytest.raises(ValueError, match="specific"):
        _check(_with(FIRST, {**GUESS, "text": text}, omitted=[PAIR]))
    _check(_with(FIRST, {**GUESS, "text": text, "specific": True}, omitted=[PAIR]))


def test_prompt_and_schema_know_supposed():
    from app.draft import LLM_JSON_SCHEMA, SYSTEM_PROMPT

    item = LLM_JSON_SCHEMA["properties"]["sentences"]["items"]
    assert "SUPPOSED" in item["properties"]["kind"]["enum"]
    assert "specific" in item["required"]
    assert LLM_JSON_SCHEMA["properties"]["omitted"]["maxItems"] == 10
    for must in ("SUPPOSED", "specific", "지원처", "주제문", "또한"):
        assert must in SYSTEM_PROMPT


# ---- 첫 줄 소제목(S-86) ----


@pytest.mark.parametrize(
    "title",
    [
        None,  # 문장이 있는데 없음
        "   ",  # 공백뿐
        "채팅 지연을\n푼 일",  # 줄바꿈
        "가" * 41,  # 40자 넘음
        "3초 폴링을 구독으로 바꾼 일",  # 숫자
        "세 번 고친 채팅 지연",  # 한글 수사도 숫자
    ],
)
def test_bad_title_retried_then_fails(client_with, title):
    bad = {**GOOD, "title": title}
    llm = FakeLLM(bad, bad)
    r = client_with(llm, max_attempts=2).post("/draft", json=BODY)
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "LLM_INVALID_OUTPUT"
    assert llm.calls == 2


def test_bad_title_retried_then_ok(client_with):
    llm = FakeLLM({**GOOD, "title": None}, GOOD)
    r = client_with(llm).post("/draft", json=BODY)
    assert r.status_code == 200
    assert r.json()["title"] == TITLE
    assert llm.calls == 2


def test_title_at_limit_ok(client_with):
    r = client_with(FakeLLM({**GOOD, "title": "가" * 40})).post("/draft", json=BODY)
    assert r.status_code == 200


def test_title_must_be_null_without_sentences(client_with):
    bare = _exp(105, "매주 풀 문제를 고른다", "ME", status="NEEDS_REVIEW")
    body = {"question": QUESTION, "experiences": [bare]}
    bad = {**_with(omitted=[LOOSE]), "title": "문제 고르기"}
    r = client_with(FakeLLM(bad, bad), max_attempts=2).post("/draft", json=body)
    assert r.status_code == 502
    ok = client_with(FakeLLM(_with(omitted=[LOOSE]))).post("/draft", json=body)
    assert ok.status_code == 200
    assert ok.json()["title"] is None


def test_title_in_llm_schema():
    from app.draft import LLM_JSON_SCHEMA

    assert "title" in LLM_JSON_SCHEMA["required"]
    assert LLM_JSON_SCHEMA["properties"]["title"] == {"type": ["string", "null"]}


def test_card_steps_and_example_in_prompt_and_schema():
    # 구조 카드 실험 판(AI#90) — 단계 이름 · 고른 카드의 예시 하나가 실리고, 스키마가 문장마다 step 을 묻는다
    body = {**GUIDED, "question": {**QUESTION, "frame": "CAUSE_FIX"}, "guidelines": []}
    question, exps, gs = check_request(DraftRequest.model_validate(body), load_settings())
    user = build_draft_message(question, exps, gs, card_mode="example")
    assert "단계 이름: 「한 줄 요약」 「걸림돌」" in user
    assert user.count("<카드 예시>") == 1
    assert "[원인 확인 · 가정]" in user
    assert "학생회" not in user  # 다른 카드의 예시는 싣지 않는다
    item = schema_for(gs, "CAUSE_FIX", "example")["properties"]["sentences"]["items"]
    assert item["properties"]["step"]["enum"] == CARD_STEPS["CAUSE_FIX"]
    assert "step" in item["required"]
    assert "step" not in LLM_JSON_SCHEMA["properties"]["sentences"]["items"]["properties"]


def test_product_card_prompt_is_unchanged_and_step_never_reaches_response(client_with):
    # 제품 판은 off(AI#90 채점) — 카드를 켜도 예전 사용자 메시지 · 스키마. LLM 이 step 을 돌려줘도 응답에 싣지 않는다
    assert CARD_MODE == "off"
    seen = {}

    class Spy(FakeLLM):
        async def generate_json(self, system, user, schema, deadline):
            seen["user"], seen["schema"] = user, schema
            return await super().generate_json(system, user, schema, deadline)

    steps = [{**s, "step": "한 줄 요약"} for s in GOOD["sentences"]]
    body = {**GUIDED, "question": {**QUESTION, "frame": "CAUSE_FIX"}, "guidelines": []}
    r = client_with(Spy({**GOOD, "sentences": steps})).post("/draft", json=body)
    assert r.status_code == 200
    assert all("step" not in s for s in r.json()["sentences"])
    assert "단계 이름" not in seen["user"] and "<카드 예시>" not in seen["user"]
    assert seen["schema"] is LLM_JSON_SCHEMA


@pytest.mark.parametrize("frame", ["STAR", "CARL", "PREP"])
def test_old_frames_get_no_card_steps(frame):
    body = {**GUIDED, "question": {**QUESTION, "frame": frame}, "guidelines": []}
    question, exps, gs = check_request(DraftRequest.model_validate(body), load_settings())
    assert "단계 이름" not in build_draft_message(question, exps, gs)
    assert schema_for(gs, question.frame) is LLM_JSON_SCHEMA


def test_card_mode_off_is_the_old_card_prompt():
    body = {**GUIDED, "question": {**QUESTION, "frame": "ALIGN"}, "guidelines": []}
    question, exps, gs = check_request(DraftRequest.model_validate(body), load_settings())
    off = build_draft_message(question, exps, gs, card_mode="off")
    assert off.endswith(f"<구성 틀>\nALIGN: {FRAME_STEPS['ALIGN']}\n</구성 틀>")
    assert schema_for(gs, "ALIGN", "off") is LLM_JSON_SCHEMA
    steps = build_draft_message(question, exps, gs, card_mode="steps")
    assert "단계 이름" in steps and "<카드 예시>" not in steps


def test_card_examples_follow_card_steps():
    # 예시의 단계는 카드 단계 이름 그대로 · 카드 순서대로
    assert set(CARD_EXAMPLES) == set(CARD_STEPS)
    for frame, lines in CARD_EXAMPLES.items():
        idx = [CARD_STEPS[frame].index(step) for step, _, _ in lines]
        assert idx == sorted(idx), frame
        assert {kind for _, kind, _ in lines} <= {"사실", "가정", "생각"}
