"""/draft — LLM 은 가짜로 바꿔 계약 모양 · 근거 · 숫자 검사 · 오류 처리만 본다(네트워크 없음).

요청 · 응답 예시는 ETC docs/contracts/ai-api.md 「POST /draft」 그대로다.
"""

import pytest

from app.citations import numbers, unsupported_numbers
from app.config import load_settings
from app.draft import backings, build_draft_message, check_output, check_request
from app.match import build_user_message, experience_chars
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
GOOD = {
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
    return {"sentences": list(sentences), "omitted": list(omitted)}


def test_contract_example(client_with):
    llm = FakeLLM(GOOD)
    r = client_with(llm).post("/draft", json=BODY)
    assert r.status_code == 200
    assert r.json() == {**GOOD, "model": "fake-model"}
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
        pytest.param(_with(SECOND, omitted=[LOOSE] * 6), id="omitted 6개"),
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
    """보완 301 의 「3초」는 근거 201 만 붙인 문장을 받치지 못한다(기간 2025-03 의 3 과 겹치지 않게 기간을 뺀다)."""
    exp = {**EXP_101, "period": None}
    out = _with(
        {"kind": "FACT", "text": "앱의 3초 폴링 코드를 바꿨습니다.", "evidence": FIRST["evidence"]}
    )
    with pytest.raises(ValueError, match="숫자"):
        check_output(LLMDraftOutput.model_validate(out), [MatchExperience.model_validate(exp)])


def test_supplement_question_does_not_back_numbers():
    """보완의 질문은 사실이 아니다 — 답에 있는 숫자만 받친다."""
    exp = _exp(101, "본문", "ME", supplements=[(301, "3초 폴링을 어떻게 바꿨나요?", "구독으로")])
    exp["period"] = None
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
    ],
)
def test_numbers(text, want):
    assert numbers(text) == want


def test_unsupported_numbers():
    assert unsupported_numbers("1200명 중 3명", "1,200명이 왔다") == ["3"]
    assert unsupported_numbers("세 번", "") == []  # 한글 수사는 못 잡는다 — 채점으로


@pytest.mark.parametrize(
    "experiences",
    [
        pytest.param([], id="0개"),
        pytest.param([_exp(i, "본문", "ME", [(200 + i, "발췌")]) for i in (1, 2, 3, 4)], id="4개"),
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
    question, exps = check_request(DraftRequest.model_validate(BODY), load_settings())
    assert seen["user"] == build_user_message(question, exps)
    assert "<매칭 단서>" not in seen["user"]


def test_null_job_role_and_empty_hints_are_unchanged():
    body = {
        "question": {**QUESTION, "jobRole": None, "requirements": []},
        "experiences": [{**EXP_101, "gaps": []}],
    }
    question, exps = check_request(DraftRequest.model_validate(body), load_settings())
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
