"""매칭 시험 자료(samples/match)와 채점 — LLM 없이."""

import pytest

from app.match import check_output, check_request
from app.schemas import LLMMatchOutput, MatchExperience, MatchRequest, MatchResponse
from scripts.eval_match import load_data, score

EXPERIENCES, QUESTIONS = load_data(None)
EXPS = [MatchExperience.model_validate(e) for e in EXPERIENCES]
BY_ID = {e.id: e for e in EXPS}
Q = {q["id"]: q for q in QUESTIONS}


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


def _response(requirements, recs):
    out = LLMMatchOutput.model_validate(
        {
            "requirements": [
                {"text": t, "status": s, "experienceIds": ids} for t, s, ids in requirements
            ],
            "recommendations": recs,
        }
    )
    check_output(out, EXPS)  # 채점하는 응답도 계약 규칙 안이다
    return MatchResponse(
        requirements=out.requirements,
        recommendations=out.recommendations,
        clarifications=[],
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
