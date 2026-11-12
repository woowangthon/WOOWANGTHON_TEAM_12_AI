"""/match — LLM 은 가짜로 바꿔 계약 모양 · ID 검사 · 오류 처리만 본다(네트워크 없음).

요청 · 응답 예시는 ETC docs/contracts/ai-api.md 「POST /match」 그대로다.
"""

import json

import pytest

from tests.conftest import FakeLLM


def _exp(id, content, actor, citations, supplements=(), status="SOURCE_CONFIRMED"):
    return {
        "id": id,
        "activityName": "바꿔요 — 캠퍼스 중고거래 앱",
        "period": {"start": "2025-03", "end": "2025-06", "ongoing": False},
        "content": content,
        "competencies": ["COLLABORATION"],
        "actor": actor,
        "sourceStatus": status,
        "citations": [{"id": cid, "quote": q} for cid, q in citations],
        "supplements": [{"id": sid, "question": q, "answer": a} for sid, q, a in supplements],
    }


EXPERIENCES = [
    _exp(
        101,
        "채팅 메시지가 늦게 도착하는 문제를 폴링에서 WebSocket 으로 바꿔 해결했다",
        "ME",
        [
            (
                201,
                "채팅 메시지가 늦게 도착하는 문제가 있었다. 내가 폴링을 WebSocket 으로 바꿔 해결했다.",
            )
        ],
        [
            (
                301,
                "폴링을 WebSocket 으로 바꿀 때 직접 고친 부분은 어디였나요?",
                "서버에 STOMP 엔드포인트를 만들고, 앱의 3초 폴링 코드를 구독 방식으로 바꿨다",
            )
        ],
    ),
    _exp(
        102,
        "매주 금요일 스프린트 회고를 했다",
        "TEAM",
        [(202, "팀은 매주 금요일 스프린트 회고를 했다.")],
    ),
    _exp(103, "서비스를 AWS 에 배포했다", "TEAM", [(203, "배포는 팀원이 맡아 AWS 에 올렸다.")]),
]

BODY = {
    "question": {
        "organization": "OO 연합 동아리",
        "text": "팀 프로젝트에서 맡은 역할과, 팀과 함께 일한 방식을 적어 주세요.",
    },
    "experiences": EXPERIENCES,
}

GOOD = {
    "questionType": "COLLABORATION",
    "requirements": [
        {"text": "팀에서 내가 맡은 역할", "status": "SUPPORTED", "experienceIds": [101]},
        {"text": "팀과 함께 일한 방식", "status": "PARTIAL", "experienceIds": [102]},
    ],
    "recommendations": [
        {
            "experienceId": 102,
            "reason": "팀이 매주 회고로 일하는 방식을 맞춘 사례다",
            "emphasis": "정기 회고로 진행을 함께 점검한 방식",
            "evidence": [{"type": "CITATION", "id": 202}],
            "gaps": ["회고에서 내가 맡은 몫이 자료에 없다"],
        },
        {
            "experienceId": 101,
            "reason": "팀에서 실시간 채팅 기능을 직접 맡아 고친 역할이 드러난다",
            "emphasis": "서버 엔드포인트와 앱 코드를 모두 맡은 범위",
            "evidence": [{"type": "CITATION", "id": 201}, {"type": "SUPPLEMENT", "id": 301}],
            "gaps": [],
        },
    ],
    # 102 는 TEAM — 내 몫을 묻는다(S-20). 101 의 「직접 고친 부분」은 보완 301 에 있어 다시 묻지 않는다(S-25)
    "clarifications": [
        {"experienceId": 102, "question": "매주 회고에서 직접 맡은 일이 있었다면 무엇이었나요?"}
    ],
}


def test_match_contract_shape(client_with):
    llm = FakeLLM(GOOD)
    res = client_with(llm).post("/match", json=BODY)
    assert res.status_code == 200
    data = res.json()
    assert data["questionType"] == "COLLABORATION"
    assert data["requirements"] == GOOD["requirements"]
    assert data["recommendations"] == GOOD["recommendations"]
    assert data["clarifications"] == GOOD["clarifications"]
    assert data["model"] == "fake-model"
    assert llm.calls == 1


def test_prompt_uses_content_not_tags(client_with):
    # 판정은 본문 · 근거 · 보완으로 — 역량 태그는 LLM 에 넘기지 않는다(SPEC F-06)
    seen = {}

    class Spy(FakeLLM):
        async def generate_json(self, system, user, schema, deadline):
            seen["user"] = user
            return await super().generate_json(system, user, schema, deadline)

    client_with(Spy(GOOD)).post("/match", json=BODY)
    user = seen["user"]
    assert "COLLABORATION" not in user
    assert "[경험 101]" in user and "근거 201" in user
    assert "보완 301" in user and "STOMP" in user  # 보완도 근거로 쓸 수 있게 싣는다(S-25)
    assert "수행 주체: TEAM" in user
    assert "OO 연합 동아리" in user


@pytest.mark.parametrize("extra", [{}, {"jobRole": None}, {"postingLines": []}])
def test_absent_job_context_preserves_both_prompts(client_with, extra):
    from app.match import SYSTEM_PROMPT, build_user_message
    from app.schemas import MatchExperience, MatchQuestion

    seen = []

    class Spy(FakeLLM):
        async def generate_json(self, system, user, schema, deadline):
            seen.append((system, user))
            return await super().generate_json(system, user, schema, deadline)

    response = client_with(Spy(GOOD)).post(
        "/match", json={**BODY, "question": {**BODY["question"], **extra}}
    )
    assert response.status_code == 200
    expected = build_user_message(
        MatchQuestion.model_validate(BODY["question"]),
        [MatchExperience.model_validate(e) for e in EXPERIENCES],
    )
    assert seen == [(SYSTEM_PROMPT, expected)]
    assert "<지원 맥락>" not in expected


POSTING_LINE = {"section": "REQUIRED", "text": "Spring 기반 API 개발", "ksa": "SKILL"}


@pytest.mark.parametrize(
    "extra",
    [
        {"jobRole": "BACKEND"},
        {"postingLines": [POSTING_LINE]},
        {"jobRole": "BACKEND", "postingLines": [POSTING_LINE, {**POSTING_LINE, "ksa": None}]},
    ],
)
def test_optional_job_context_is_separate_and_response_unchanged(client_with, extra):
    from app.match import JOB_CONTEXT_RULE, SYSTEM_PROMPT

    seen = []

    class Spy(FakeLLM):
        async def generate_json(self, system, user, schema, deadline):
            seen.append((system, user))
            return await super().generate_json(system, user, schema, deadline)

    fake = Spy(GOOD)
    response = client_with(fake).post(
        "/match", json={**BODY, "question": {**BODY["question"], **extra}}
    )
    assert response.status_code == 200
    assert response.json() == {**GOOD, "model": "fake-model"}
    assert fake.calls == 1
    system, user = seen[0]
    assert system == SYSTEM_PROMPT + JOB_CONTEXT_RULE
    assert user.index("</문항>") < user.index("<지원 맥락>")
    assert user.index("</지원 맥락>") < user.index("<경험 목록>")
    if "jobRole" in extra:
        assert "지원 직무: BACKEND" in user
    if "postingLines" in extra:
        assert "[REQUIRED · SKILL] Spring 기반 API 개발" in user


@pytest.mark.parametrize(
    "extra",
    [
        {"jobRole": "CHEF"},
        {"postingLines": None},
        {"postingLines": [POSTING_LINE] * 41},
        {"postingLines": [{**POSTING_LINE, "section": "OTHER"}]},
        {"postingLines": [{**POSTING_LINE, "ksa": "OTHER"}]},
        {"postingLines": [{**POSTING_LINE, "text": "  "}]},
        {"postingLines": [{**POSTING_LINE, "text": "가" * 301}]},
    ],
)
def test_invalid_job_context_is_rejected_before_llm(client_with, extra):
    llm = FakeLLM()
    response = client_with(llm).post(
        "/match", json={**BODY, "question": {**BODY["question"], **extra}}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"
    assert llm.calls == 0


def test_job_context_accepts_contract_limits_without_changing_413_count(client_with):
    from app.match import experience_chars
    from app.schemas import MatchExperience

    # 계약의 413 합계는 기존 문항 · 경험만 센다. 공고 줄은 별도 40줄 × 300자 한도다.
    limit = len(BODY["question"]["text"]) + experience_chars(
        [MatchExperience.model_validate(e) for e in EXPERIENCES]
    )
    body = {
        **BODY,
        "question": {
            **BODY["question"],
            "postingLines": [{**POSTING_LINE, "text": "가" * 300, "ksa": None}] * 40,
        },
    }
    llm = FakeLLM(GOOD)
    assert client_with(llm, max_match_chars=limit).post("/match", json=body).status_code == 200
    llm = FakeLLM()
    response = client_with(llm, max_match_chars=limit - 1).post("/match", json=body)
    assert response.status_code == 413
    assert llm.calls == 0


def test_job_context_cannot_supply_evidence_ids(client_with):
    # 공고에 붙은 id 는 경험의 근거 ID가 될 수 없다.
    llm = FakeLLM(
        _bad(
            lambda out: out["recommendations"][0]["evidence"].append(
                {"type": "CITATION", "id": 999}
            )
        )
    )
    response = client_with(llm, max_attempts=1).post(
        "/match",
        json={
            **BODY,
            "question": {**BODY["question"], "postingLines": [{**POSTING_LINE, "id": 999}]},
        },
    )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "LLM_INVALID_OUTPUT"


def test_no_experiences_all_not_found(client_with):
    out = {
        "questionType": "GROWTH",
        "requirements": [{"text": "해외 봉사 경험", "status": "NOT_FOUND", "experienceIds": []}],
        "recommendations": [],
        "clarifications": [],
    }
    res = client_with(FakeLLM(out)).post("/match", json={**BODY, "experiences": []})
    assert res.status_code == 200
    data = res.json()
    assert data["recommendations"] == [] and data["clarifications"] == []
    assert data["requirements"][0]["status"] == "NOT_FOUND"
    assert data["questionType"] == "GROWTH"  # 경험이 [] 여도 붙는다(S-41)


def test_no_experiences_cannot_be_supported(client_with):
    # 경험이 없는데 SUPPORTED 면 ID 가 요청에 없다 — 다시 부르고 오류
    llm = FakeLLM(GOOD, GOOD)
    res = client_with(llm, max_attempts=2).post("/match", json={**BODY, "experiences": []})
    assert res.status_code == 502
    assert res.json()["error"]["code"] == "LLM_INVALID_OUTPUT"


@pytest.mark.parametrize(
    "body",
    [
        {"experiences": []},
        {"question": None, "experiences": []},
        {"question": {"text": ""}, "experiences": []},
        {"question": {"text": "  \n"}, "experiences": []},
        {"question": {"organization": "OO"}, "experiences": []},
        {"question": {"text": "문항"}},
        {"question": {"text": "문항"}, "experiences": None},
        {"question": {"text": "문항"}, "experiences": [EXPERIENCES[0], EXPERIENCES[0]]},
        {"question": {"text": "문항"}, "experiences": [{"id": 1}]},
        {"question": {"text": "문항"}, "experiences": [{**EXPERIENCES[0], "actor": "나"}]},
    ],
)
def test_invalid_request(client_with, body):
    llm = FakeLLM()
    res = client_with(llm).post("/match", json=body)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_REQUEST"
    assert llm.calls == 0


def test_too_long(client_with):
    llm = FakeLLM()
    body = {**BODY, "question": {"text": "가" * 5}}
    # 문항 5 + 경험 본문 · 근거 · 보완 합이 한도를 넘는다
    res = client_with(llm, max_match_chars=50).post("/match", json=body)
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "TEXT_TOO_LONG"
    assert llm.calls == 0


def test_user_entered_without_citations_ok(client_with):
    exp = {**EXPERIENCES[0], "id": 141, "sourceStatus": "USER_ENTERED", "period": None}
    exp |= {"citations": [], "supplements": [], "activityName": None}
    out = {
        "questionType": "COLLABORATION",
        "requirements": [{"text": "역할", "status": "PARTIAL", "experienceIds": [141]}],
        "recommendations": [
            {"experienceId": 141, "reason": "r", "emphasis": "e", "evidence": [], "gaps": []}
        ],
        "clarifications": [{"experienceId": 141, "question": "친구들이 쓴 뒤 무엇을 고쳤나요?"}],
    }
    res = client_with(FakeLLM(out)).post("/match", json={**BODY, "experiences": [exp]})
    assert res.status_code == 200


def _bad(mutate):
    data = json.loads(json.dumps(GOOD))
    mutate(data)
    return data


def _rec(**kw):
    rec = {"experienceId": 103, "reason": "r", "emphasis": "e", "evidence": [], "gaps": []}
    return rec | kw


BAD_OUTPUTS = [
    "not json",
    "",
    {"requirements": []},
    # 요구사항: 0개 · 7개 · 요청에 없는 경험 · NOT_FOUND 인데 경험 · SUPPORTED 인데 경험 없음 · 중복 · 값 밖
    _bad(lambda d: d.update(requirements=[])),
    _bad(lambda d: d.update(requirements=d["requirements"] * 4)),
    _bad(lambda d: d["requirements"][0].update(experienceIds=[999])),
    _bad(lambda d: d["requirements"][0].update(status="NOT_FOUND")),
    _bad(lambda d: d["requirements"][0].update(experienceIds=[])),
    _bad(lambda d: d["requirements"][0].update(experienceIds=[101, 101])),
    _bad(lambda d: d["requirements"][0].update(status="FOUND")),
    _bad(lambda d: d["requirements"][0].update(text=" ")),
    # 추천: 4개 · 요청에 없는 경험 · 같은 경험 두 번 · 빈 이유 · 빈 강조점 · 빈 부족한 점
    _bad(lambda d: d["recommendations"].extend([_rec(), _rec(experienceId=999)])),
    _bad(lambda d: d["recommendations"][0].update(experienceId=999)),
    _bad(lambda d: d["recommendations"].append(_rec(experienceId=101))),
    _bad(lambda d: d["recommendations"][0].update(reason="")),
    _bad(lambda d: d["recommendations"][0].update(emphasis="  ")),
    _bad(lambda d: d["recommendations"][0].update(gaps=[""])),
    # 근거: 다른 경험의 근거 · 없는 근거 · 근거 ID 를 보완이라고 · 없는 보완 · 중복 · 값 밖
    _bad(lambda d: d["recommendations"][0].update(evidence=[{"type": "CITATION", "id": 201}])),
    _bad(lambda d: d["recommendations"][0].update(evidence=[{"type": "CITATION", "id": 999}])),
    _bad(lambda d: d["recommendations"][1].update(evidence=[{"type": "SUPPLEMENT", "id": 201}])),
    _bad(lambda d: d["recommendations"][1].update(evidence=[{"type": "SUPPLEMENT", "id": 302}])),
    _bad(lambda d: d["recommendations"][0]["evidence"].append({"type": "CITATION", "id": 202})),
    _bad(lambda d: d["recommendations"][0].update(evidence=[{"type": "QUOTE", "id": 202}])),
    # 되묻기: 4개(잘라 내지 않는다) · 추천하지 않은 경험(103) · 요청에 없는 경험 · 빈 질문 · 칸 없음
    _bad(lambda d: d.update(clarifications=d["clarifications"] * 4)),
    _bad(lambda d: d["clarifications"][0].update(experienceId=103)),
    _bad(lambda d: d["clarifications"][0].update(experienceId=999)),
    _bad(lambda d: d["clarifications"][0].update(question=" ")),
    _bad(lambda d: d.pop("clarifications")),
    # 문항 유형: 목록 밖 · 소문자 · 빠짐 · null · 배열(하나만)
    _bad(lambda d: d.update(questionType="TEAMWORK")),
    _bad(lambda d: d.update(questionType="collaboration")),
    _bad(lambda d: d.pop("questionType")),
    _bad(lambda d: d.update(questionType=None)),
    _bad(lambda d: d.update(questionType=["COLLABORATION"])),
]


@pytest.mark.parametrize("bad", BAD_OUTPUTS)
def test_invalid_output_retries_then_errors(client_with, bad):
    llm = FakeLLM(bad, bad)
    res = client_with(llm, max_attempts=2).post("/match", json=BODY)
    assert res.status_code == 502
    assert res.json()["error"]["code"] == "LLM_INVALID_OUTPUT"
    assert llm.calls == 2


def test_invalid_then_valid_recovers(client_with):
    bad = _bad(lambda d: d["recommendations"][0].update(experienceId=999))
    llm = FakeLLM(bad, GOOD)
    res = client_with(llm, max_attempts=2).post("/match", json=BODY)
    assert res.status_code == 200
    assert llm.calls == 2


def test_three_clarifications_ok(client_with):
    out = _bad(
        lambda d: d.update(
            clarifications=d["clarifications"] * 2
            + [{"experienceId": 101, "question": "고친 뒤 정상 동작을 어떻게 확인했나요?"}]
        )
    )
    res = client_with(FakeLLM(out)).post("/match", json=BODY)
    assert res.status_code == 200
    assert len(res.json()["clarifications"]) == 3


def test_zero_recommendations_ok(client_with):
    out = {
        "questionType": "OTHER",
        "requirements": [{"text": "해외 봉사", "status": "NOT_FOUND", "experienceIds": []}],
        "recommendations": [],
        "clarifications": [],
    }
    res = client_with(FakeLLM(out)).post("/match", json=BODY)
    assert res.status_code == 200
    assert res.json()["recommendations"] == []


def test_question_type_in_llm_schema():
    # Structured Outputs 스키마에 enum 으로 — 목록은 계약 README 「값 목록」 questionType 12개(S-85 JOB_SKILL)
    from app.match import LLM_JSON_SCHEMA
    from app.schemas import QUESTION_TYPES

    assert "questionType" in LLM_JSON_SCHEMA["required"]
    assert LLM_JSON_SCHEMA["properties"]["questionType"]["enum"] == list(QUESTION_TYPES)
    assert len(QUESTION_TYPES) == 12 and {"OTHER", "JOB_SKILL"} <= set(QUESTION_TYPES)


# ---- 추천 0 ~ 8개 · 활동 5개까지(S-84) ----


def _many(n, activity=lambda i: i):
    """경험 n개 — 경험 i 는 활동 activity(i). activity 가 None 을 주면 activityId 를 싣지 않는다."""
    exps = []
    for i in range(1, n + 1):
        exp = _exp(500 + i, f"본문 {i}", "ME", [(600 + i, f"발췌 {i}")])
        if activity(i) is not None:
            exp["activityId"] = activity(i)
        exps.append(exp)
    return exps


def _recommend_all(exps):
    return {
        "questionType": "COLLABORATION",
        "requirements": [{"text": "역할", "status": "SUPPORTED", "experienceIds": [exps[0]["id"]]}],
        "recommendations": [
            {"experienceId": e["id"], "reason": "r", "emphasis": "e", "evidence": [], "gaps": []}
            for e in exps
        ],
        "clarifications": [],
    }


def _post_many(client_with, exps, n_out=None):
    out = _recommend_all(exps[:n_out] if n_out else exps)
    llm = FakeLLM(out, out)
    res = client_with(llm, max_attempts=2).post("/match", json={**BODY, "experiences": exps})
    return res, llm


def test_eight_recommendations_in_five_activities_ok(client_with):
    exps = _many(8, activity=lambda i: (i + 1) // 2)  # 활동 1 ~ 4 에 두 줄씩
    res, llm = _post_many(client_with, exps)
    assert res.status_code == 200
    assert len(res.json()["recommendations"]) == 8
    assert llm.calls == 1


def test_five_activities_ok(client_with):
    res, _ = _post_many(client_with, _many(5))
    assert res.status_code == 200


def test_nine_recommendations_rejected(client_with):
    res, llm = _post_many(client_with, _many(9, activity=lambda i: 1))
    assert res.status_code == 502
    assert res.json()["error"]["code"] == "LLM_INVALID_OUTPUT"
    assert llm.calls == 2


def test_six_activities_rejected(client_with):
    res, llm = _post_many(client_with, _many(6))
    assert res.status_code == 502
    assert res.json()["error"]["code"] == "LLM_INVALID_OUTPUT"
    assert llm.calls == 2


def test_without_activity_id_skips_activity_count(client_with):
    # BE 가 activityId 를 싣기 전 요청 — 400 이 아니고 활동 수 검사를 건너뛴다
    res, _ = _post_many(client_with, _many(8, activity=lambda i: None))
    assert res.status_code == 200


def test_prompt_shows_activity_id(client_with):
    seen = {}

    class Spy(FakeLLM):
        async def generate_json(self, system, user, schema, deadline):
            seen["user"] = user
            return await super().generate_json(system, user, schema, deadline)

    exps = _many(2, activity=lambda i: 7)
    client_with(Spy(_recommend_all(exps))).post("/match", json={**BODY, "experiences": exps})
    assert "(활동 7)" in seen["user"]


def test_material_rules_use_the_same_match_call_without_filtering_candidates(client_with):
    # 실패로 끝난 자료가 없어도 관련된 성공 경험을 코드가 지우지 않는다(S-20).
    from app.match import MATERIAL_RULES

    seen = {}

    class Spy(FakeLLM):
        async def generate_json(self, system, user, schema, deadline):
            seen["system"] = system
            return await super().generate_json(system, user, schema, deadline)

    exp = _exp(501, "첫 시연 오류를 고치고 재시연에 성공했다", "ME", [(601, "재시연 성공")])
    out = _recommend_all([exp])
    out["questionType"] = "FAILURE"
    llm = Spy(out)
    res = client_with(llm).post(
        "/match",
        json={"question": {"text": "실패 경험에서 배운 점은?"}, "experiences": [exp]},
    )
    assert res.status_code == 200
    assert llm.calls == 1
    assert seen["system"].endswith(MATERIAL_RULES)
    assert res.json()["recommendations"][0]["experienceId"] == 501


# ==== 같은 지원서에서 이미 낸 되묻기 · 「딱히 없음」 보완(ETC#354 · AI#138) ====

ASKED = "회고를 진행할 때 직접 맡은 일이 있었다면 무엇이었나요?"


def _with_asked(*questions):
    exps = json.loads(json.dumps(EXPERIENCES))
    exps[1]["askedQuestions"] = list(questions)
    return {**BODY, "experiences": exps}


@pytest.mark.parametrize("asked", [None, []])
def test_no_asked_questions_keeps_prompt(client_with, asked):
    from app.match import ASKED_RULE, NONE_ANSWER_RULE, SYSTEM_PROMPT

    body = _with_asked() if asked == [] else {**BODY}
    if asked is None:
        body["experiences"] = [{**e, "askedQuestions": None} for e in EXPERIENCES]
    llm = FakeLLM(GOOD)
    assert client_with(llm).post("/match", json=body).status_code == 200
    system, user = llm.seen[0]["system"], llm.seen[0]["user"]
    assert system == SYSTEM_PROMPT  # 보통 보완(STOMP)만으론 「없다」 규칙도 붙지 않는다
    assert ASKED_RULE not in system and NONE_ANSWER_RULE not in system
    assert "이미 낸 되묻기" not in user


def test_asked_questions_shown_and_rule_appended(client_with):
    from app.match import ASKED_RULE, SYSTEM_PROMPT

    llm = FakeLLM({**GOOD, "clarifications": []})
    res = client_with(llm).post("/match", json=_with_asked(ASKED))
    assert res.status_code == 200
    system, user = llm.seen[0]["system"], llm.seen[0]["user"]
    assert system == SYSTEM_PROMPT + ASKED_RULE  # 앞부분이 같아 캐시가 산다
    block = user.split("[경험 102]")[1].split("[경험 103]")[0]
    assert "이미 낸 되묻기" in block and ASKED in block
    assert res.json()["clarifications"] == []


@pytest.mark.parametrize(
    "question",
    [ASKED, "회고를 진행할 때, 직접 맡은 일이 있었다면 무엇이었나요", " " + ASKED + " "],
)
def test_same_text_as_asked_retries_then_errors(client_with, question):
    bad = {**GOOD, "clarifications": [{"experienceId": 102, "question": question}]}
    llm = FakeLLM(bad, bad)
    res = client_with(llm, max_attempts=2).post("/match", json=_with_asked(ASKED))
    assert res.status_code == 502
    assert res.json()["error"]["code"] == "LLM_INVALID_OUTPUT"
    assert llm.calls == 2


def test_asked_only_blocks_its_own_experience(client_with):
    # 다른 경험(101)에 낸 같은 글은 막지 않는다 — 이미 낸 되묻기는 경험마다다
    body = _with_asked(ASKED)
    ok = {**GOOD, "clarifications": [{"experienceId": 101, "question": ASKED}]}
    assert client_with(FakeLLM(ok)).post("/match", json=body).status_code == 200


@pytest.mark.parametrize(
    "answer,attached",
    [
        ("딱히 없음", True),
        ("기억 안 남", True),
        ("잘 모르겠어요", True),
        ("따로 맡은 건 없었어요", True),
        ("서버에 STOMP 엔드포인트를 만들었다", False),
        ("품목별로 남은 양을 적었다", False),
        ("시트에 적는 사람이 늘어 스터디가 끊기지 않았다", False),
        ("없" + "가" * 80, False),  # 긴 답은 「없다는 말」이 아니라 사실을 적은 것으로 본다
    ],
)
def test_none_answer_rule_only_for_none_like_supplements(client_with, answer, attached):
    from app.match import NONE_ANSWER_RULE, SYSTEM_PROMPT

    exps = json.loads(json.dumps(EXPERIENCES))
    exps[1]["supplements"] = [{"id": 302, "question": ASKED, "answer": answer}]
    llm = FakeLLM(GOOD)
    assert client_with(llm).post("/match", json={**BODY, "experiences": exps}).status_code == 200
    expected = SYSTEM_PROMPT + NONE_ANSWER_RULE if attached else SYSTEM_PROMPT
    assert llm.seen[0]["system"] == expected


def test_optional_rules_order(client_with):
    from app.match import ASKED_RULE, JOB_CONTEXT_RULE, NONE_ANSWER_RULE, SYSTEM_PROMPT

    body = _with_asked(ASKED)
    body["experiences"][1]["supplements"] = [{"id": 302, "question": "q", "answer": "딱히 없음"}]
    body["question"] = {**BODY["question"], "jobRole": "BACKEND"}
    llm = FakeLLM({**GOOD, "clarifications": []})
    assert client_with(llm).post("/match", json=body).status_code == 200
    assert llm.seen[0]["system"] == SYSTEM_PROMPT + JOB_CONTEXT_RULE + NONE_ANSWER_RULE + ASKED_RULE
