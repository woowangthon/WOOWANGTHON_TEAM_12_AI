"""매칭 시험 자료(samples/match)와 채점 — LLM 없이."""

import re

import pytest

from app.match import check_output, check_request
from app.schemas import (
    QUESTION_TYPES,
    LLMMatchOutput,
    MatchExperience,
    MatchRequest,
    MatchResponse,
)
from scripts.eval_match import (
    load_data,
    load_known,
    load_types,
    nosupp_questions,
    score,
    score_type,
    strip_supplements,
)

EXPERIENCES, QUESTIONS = load_data(None)
EXPS = [MatchExperience.model_validate(e) for e in EXPERIENCES]
BY_ID = {e.id: e for e in EXPS}
Q = {q["id"]: q for q in QUESTIONS}
KNOWN = load_known()


def test_pool_is_a_valid_request():
    from app.config import load_settings

    req = MatchRequest.model_validate(
        {"question": {"text": QUESTIONS[0]["text"]}, "experiences": EXPERIENCES}
    )
    check_request(req, load_settings())  # 400 · 413 이 아니다
    ids = [c.id for e in EXPS for c in e.citations] + [s.id for e in EXPS for s in e.supplements]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("q", QUESTIONS, ids=lambda q: q["id"])
def test_answers_point_to_pool(q):
    good, ok, bad = set(q["good"]), set(q["ok"]), set(q["bad"])
    assert (good | ok | bad) <= set(BY_ID)
    assert not (good & ok) and not (good & bad) and not (ok & bad)
    assert q["kind"] in ("match", "partial", "none")
    if q["kind"] == "none":
        assert not good and not ok
    else:
        assert good


def test_demo_two_questions_pick_different_experiences_of_same_activity():
    # 데모 3 · 6(S-25): 기술 문제 문항과 조율 문항에서 같은 활동(바꿔요)의 다른 경험이 정답이다
    tech, coord = Q["tech-problem"], Q["team-coordination"]
    assert not set(tech["good"]) & set(coord["good"])
    activities = {BY_ID[i].activityName for i in tech["good"] + coord["good"]}
    assert activities == {"바꿔요 — 캠퍼스 중고거래 앱"}
    # 문항 A 의 정답 경험엔 앞에서 답한 보완이 실려 있다
    assert BY_ID[tech["good"][0]].supplements


def _response(requirements, recs, clarifications=(), exps=EXPS):
    out = LLMMatchOutput.model_validate(
        {
            "questionType": "OTHER",
            "requirements": [
                {"text": t, "status": s, "experienceIds": ids} for t, s, ids in requirements
            ],
            "recommendations": recs,
            "clarifications": [{"experienceId": i, "question": q} for i, q in clarifications],
        }
    )
    check_output(out, exps)  # 채점하는 응답도 계약 규칙 안이다
    return MatchResponse(
        questionType=out.questionType,
        requirements=out.requirements,
        recommendations=out.recommendations,
        clarifications=out.clarifications,
        model="fake",
    )


def _rec(exp_id, reason, evidence=()):
    return {
        "experienceId": exp_id,
        "reason": reason,
        "emphasis": "e",
        "evidence": [{"type": t, "id": i} for t, i in evidence],
        "gaps": [],
    }


def test_score_counts():
    data = _response(
        [("기술 문제 해결", "SUPPORTED", [101])],
        [
            _rec(101, "폴링을 WebSocket 으로 바꿔 3초 지연을 없앴다", [("SUPPLEMENT", 301)]),
            # UNKNOWN 을 내 일로 · 원본에 없는 수치
            _rec(114, "내가 DB 커넥션 누수를 찾아 장애를 90% 줄였다"),
            # 팀원이 한 배포 — 엉뚱한 추천
            _rec(103, "팀이 AWS 에 배포했다"),
        ],
    )
    s = score(Q["tech-problem"], EXPS, data)
    assert (s.picked_right, s.picked_total) == (1, 1)
    assert s.wrong_picks == ["tech-problem: 103"]
    assert (s.req_right, s.req_total) == (1, 1)
    assert len(s.made_up_numbers) == 1 and "90" in s.made_up_numbers[0]  # 3 은 보완에 있다
    assert len(s.team_as_mine) == 1 and "114" in s.team_as_mine[0]
    assert (s.supplement_used, s.supplement_recs) == (1, 1)


def test_score_none_question():
    forced = _response(
        [("해외 봉사", "PARTIAL", [131])], [_rec(131, "팀 행사 진행 경험이 봉사와 비슷하다")]
    )
    s = score(Q["overseas-volunteer"], EXPS, forced)
    assert len(s.forced) == 1 and s.req_right == 0
    empty = _response([("해외 봉사", "NOT_FOUND", [])], [])
    s = score(Q["overseas-volunteer"], EXPS, empty)
    assert not s.forced and s.req_right == 1 and s.picked_total == 0


def test_score_partial_allows_zero():
    data = _response([("Python · SQL", "NOT_FOUND", []), ("데이터로 판단", "PARTIAL", [121])], [])
    s = score(Q["data-python-sql"], EXPS, data)
    assert (s.picked_right, s.picked_total, s.req_right) == (1, 1, 1)


def test_known_facts_point_to_pool():
    for fact in KNOWN:
        exp = BY_ID[fact["experienceId"]]
        items = exp.citations if fact["source"] == "CITATION" else exp.supplements
        assert fact["id"] in {x.id for x in items}, fact
        re.compile(fact["pattern"])
        if fact["source"] == "SUPPLEMENT":
            # 그 보완을 만든 질문 자체가 「다시 물음」으로 걸려야 한다
            question = next(x.question for x in exp.supplements if x.id == fact["id"])
            assert re.search(fact["pattern"], question), fact


def test_nosupp_questions_are_those_with_supplement_answers():
    ids = [q["id"] for q in nosupp_questions(EXPERIENCES, QUESTIONS)]
    assert ids == ["tech-problem", "team-role", "data-python-sql"]
    assert all(not e["supplements"] for e in strip_supplements(EXPERIENCES))


def test_score_clarifications_with_supplement():
    data = _response(
        [("기술 문제 해결", "SUPPORTED", [101])],
        [_rec(101, "r", [("SUPPLEMENT", 301)]), _rec(114, "r")],
        [
            (101, "직접 고친 부분은 어디였나요?"),  # 보완 301 에 있다 — 다시 물음
            (101, "지연을 30% 줄이셨죠? 어떻게 쟀나요?"),  # 없는 수치 · 유도 · 두 개를 한 번에
            (114, "더 자세히 알려 주세요."),  # 막연함 · 내 몫을 묻지 않음
        ],
    )
    s = score(Q["tech-problem"], EXPS, data, KNOWN)
    assert len(s.clarifications) == 3 and s.clar_empty == 0
    assert len(s.repeats) == 1 and "직접 고친" in s.repeats[0]
    assert s.repeat_chances == 3  # 101 의 보완 301 · 근거 201, 114 의 근거 214
    assert len(s.leading) == 1 and "30%" in s.leading[0]
    assert len(s.vague) == 1
    assert len(s.compound) == 1
    assert (s.share_asked, s.share_chances) == (0, 1)  # 114 UNKNOWN
    assert s.gap_chances == 0  # 보완이 있는 자료


def test_score_clarifications_without_supplement():
    bare = [MatchExperience.model_validate(e) for e in strip_supplements(EXPERIENCES)]
    data = _response(
        [("기술 문제 해결", "SUPPORTED", [101])],
        [_rec(101, "r"), _rec(114, "r")],
        [
            (101, "직접 고친 부분은 어디였나요?"),  # 보완이 없으니 빈칸을 물은 것
            (114, "그 코드를 직접 고치셨다면 어떤 부분이었나요?"),
        ],
        exps=bare,
    )
    s = score(Q["tech-problem"], bare, data, KNOWN, stripped=True)
    assert not s.repeats and (s.gap_asked, s.gap_chances) == (1, 1)
    assert (s.share_asked, s.share_chances) == (1, 1)
    assert not s.leading and not s.vague


TYPE_QUESTIONS = load_types(None)


@pytest.mark.parametrize("q", QUESTIONS + TYPE_QUESTIONS, ids=lambda q: q["id"])
def test_question_type_answers_in_closed_list(q):
    # 정답 유형도 계약 README 「값 목록」 questionType 안 — 허용 칸에 정답을 겹쳐 적지 않는다
    assert q["questionType"] in QUESTION_TYPES
    assert set(q["questionTypeOk"]) <= set(QUESTION_TYPES)
    assert q["questionType"] not in q["questionTypeOk"]


def test_type_questions_cover_every_type_but_other():
    ids = [q["id"] for q in QUESTIONS + TYPE_QUESTIONS]
    assert len(ids) == len(set(ids))  # --questions 로 같이 고른다
    covered = {q["questionType"] for q in QUESTIONS + TYPE_QUESTIONS}
    assert covered == set(QUESTION_TYPES)


def test_score_question_type():
    q = Q["failure-learning"]  # 정답 FAILURE, 허용 GROWTH
    for got, right, ok in [("FAILURE", 1, 1), ("GROWTH", 0, 1), ("CONFLICT", 0, 0)]:
        data = _response([("실패", "SUPPORTED", [115])], []).model_copy(
            update={"questionType": got}
        )
        s = score(q, EXPS, data)
        assert (s.type_right, s.type_ok, s.type_total) == (right, ok, 1), got
        assert len(s.type_wrong) == 1 - right
        assert score_type(q, data).type_ok == ok


def test_pool_activity_ids_follow_names():
    # 같은 활동 이름 ↔ 같은 activityId(S-84 활동 수를 세는 단서)
    pairs = {(e.activityName, e.activityId) for e in EXPS}
    assert all(e.activityId is not None for e in EXPS)
    assert len({n for n, _ in pairs}) == len(pairs) == len({a for _, a in pairs})


def test_score_counts_activities_and_padding():
    """S-84 — 추천 · 활동 수, 정답 밖 추천, 활동의 경험을 통째로 넣었나."""
    whole = [e.id for e in EXPS if e.activityId == BY_ID[101].activityId]
    data = _response(
        [("기술 문제 해결", "SUPPORTED", [101])],
        [_rec(i, "r") for i in whole],
    )
    s = score(Q["tech-problem"], EXPS, data)
    assert s.rec_count == len(whole)
    assert s.activity_count == 1
    assert len(s.whole_activity) == 1
    labeled = {*Q["tech-problem"]["good"], *Q["tech-problem"]["ok"], *Q["tech-problem"]["bad"]}
    assert len(s.unlabeled) == len([i for i in whole if i not in labeled])
