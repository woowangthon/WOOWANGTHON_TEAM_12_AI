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


def test_no_experiences_all_not_found(client_with):
    out = {
        "requirements": [{"text": "해외 봉사 경험", "status": "NOT_FOUND", "experienceIds": []}],
        "recommendations": [],
        "clarifications": [],
    }
    res = client_with(FakeLLM(out)).post("/match", json={**BODY, "experiences": []})
    assert res.status_code == 200
    data = res.json()
    assert data["recommendations"] == [] and data["clarifications"] == []
    assert data["requirements"][0]["status"] == "NOT_FOUND"


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
        "requirements": [{"text": "해외 봉사", "status": "NOT_FOUND", "experienceIds": []}],
        "recommendations": [],
        "clarifications": [],
    }
    res = client_with(FakeLLM(out)).post("/match", json=BODY)
    assert res.status_code == 200
    assert res.json()["recommendations"] == []
