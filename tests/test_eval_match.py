"""매칭 시험 자료(samples/match)와 채점 — LLM 없이."""

import json
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


def test_ask_kind_follows_reviewer_order():
    # 되묻기 갈래(AI#99) — 내 몫 → 판단 → 확인 · 결과 → 그 밖, 먼저 걸린 것 하나
    from scripts.eval_match import ask_kind

    assert ask_kind("회고에서 직접 맡은 일이 있었다면 무엇이었나요?") == "내 몫"
    assert ask_kind("전체를 바꾸지 않고 재시도를 고른 이유는 무엇인가요?") == "판단 · 원인 · 입장"
    assert ask_kind("고친 뒤 정상 동작을 어떻게 확인했나요?") == "확인 · 결과"
    assert ask_kind("그 프로젝트는 언제 진행했나요?") == "그 밖"


def _exp(content: str) -> MatchExperience:
    return MatchExperience.model_validate(
        {"id": 1, "content": content, "actor": "ME", "citations": [], "supplements": []}
    )


def test_echoes_body_catches_noun_form_restatement():
    # 강조점 본문 되풀이(AI#105) — 운영 화면 예. 조사 · 어미 · 꼴 낱말이 붙어도 되풀이로 친다
    from scripts.eval_match import echoes_body

    assert echoes_body(
        "스터디 일정과 주차별 문제 목록 관리", _exp("스터디 일정과 주차별 문제 목록을 관리했다.")
    )
    assert echoes_body("공연장 대관 일정 조율", _exp("공연장 대관 일정을 조율했다."))
    assert echoes_body("공연장 대관 일정을 조율한 행동", _exp("공연장 대관 일정을 조율했다."))
    assert not echoes_body(
        "손님이 몰린 걸 보고 주문 순서를 바꾼 판단", _exp("카페에서 주문 순서를 바꿨다.")
    )


def test_same_angle_pairs_across_questions():
    from scripts.eval_match import same_angle

    assert same_angle(
        "참여 상황에 맞춰 바꾼 스터디 운영 방식", "참여 감소 뒤 스터디 운영 방식을 바꾼 행동"
    )
    assert not same_angle(
        "메시지 지연 뒤 3초 폴링을 구독 방식으로 바꾼 조치", "채팅 기능에서 직접 맡아 변경한 부분"
    )


def test_score_collects_emphasis_by_experience():
    q = Q["tech-problem"]
    rec = {"experienceId": 101, "reason": "맞다", "emphasis": "가", "evidence": [], "gaps": []}
    data = MatchResponse.model_validate(
        {
            "questionType": "PROBLEM_SOLVING",
            "requirements": [{"text": "문제 해결", "status": "SUPPORTED", "experienceIds": [101]}],
            "recommendations": [rec],
            "clarifications": [],
            "model": "m",
        }
    )
    s = score(q, EXPS, data)
    assert s.emph_by_exp == {101: [("tech-problem", "가")]}


def test_bare_reason_catches_restated_experience():
    # 이유 본문 되풀이(AI#107) — 「…한 경험입니다」 한 문장 · 본문 되풀이는 걸리고, 문항과 잇는 이유는 안 걸린다
    from scripts.eval_match import bare_reason

    exp = _exp(
        "중간 발표 전 일정이 밀리자 남은 기능을 꼭 할 것과 미룰 것으로 나눠 일정표를 다시 짰다"
    )
    assert bare_reason(
        "일정이 밀린 상황에서 기능의 우선순위를 나누고 일정표를 다시 짠 경험입니다.", exp
    )
    assert bare_reason(
        "스터디 참여가 줄어든 뒤 출석 체크를 바꾼 경험이다.", _exp("출석 체크를 바꿨다")
    )
    assert not bare_reason(
        "일정이 밀리자 기능을 나눈 판단이 팀 일정 조율을 보여 줍니다. 다만 팀원과 맞춘 방식은 자료에 없습니다.",
        exp,
    )


# SAMPLE(match-materials): 실제 사람의 기록이 아닌 지어낸 소재 규칙 사례.
def test_material_sample_requests_and_answer_ids(monkeypatch):
    from scripts import eval_match

    monkeypatch.setattr(eval_match, "DATA", eval_match.ROOT / "samples" / "match-materials")
    exps, questions = eval_match.load_data(None)
    assert len(questions) == 6
    assert sorted(q["questionType"] for q in questions) == sorted(
        ["FAILURE", "COLLABORATION", "CONFLICT"] * 2
    )
    for q in questions:
        pool = [e for e in exps if e["id"] in q["pool"]]
        MatchRequest.model_validate({"question": {"text": q["text"]}, "experiences": pool})
        assert set(q["good"] + q["ok"] + q["bad"]) <= set(q["pool"])
        assert set(map(int, q["materialAskPatterns"])) <= set(q["pool"])
        assert set(q["eventualSuccess"]) <= set(q["pool"])
        for pattern in q["materialAskPatterns"].values():
            re.compile(pattern)


def test_material_score_counts_success_first_and_question_alignment():
    q = {
        **Q["failure-learning"],
        "eventualSuccess": [101],
        "materialAskPatterns": {"101": "배웠|배운"},
    }
    data = _response(
        [("실패", "PARTIAL", [101])],
        [_rec(101, "재시도한 경험")],
        [(101, "시연 실패에서 배운 점은 무엇인가요?"), (101, "어떻게 확인했나요?")],
    )
    result = score(q, EXPS, data)
    assert result.success_first == ["failure-learning: 101"]
    assert (result.material_followed, result.material_asks) == (1, 2)
    assert (result.material_covered, result.material_questions) == (1, 1)
    assert len(result.material_missed) == 1


def test_material_score_empty_questions_do_not_count_as_coverage():
    q = {**Q["failure-learning"], "materialAskPatterns": {"101": "배웠"}}
    data = _response([("실패", "PARTIAL", [101])], [_rec(101, "r")])
    result = score(q, EXPS, data)
    assert (result.material_followed, result.material_asks) == (0, 0)
    assert (result.material_covered, result.material_questions) == (0, 1)


@pytest.mark.parametrize(
    "question, follows",
    [
        ("제출 마감을 넘겨 참가하지 못한 경험에서 무엇을 배우셨나요?", True),
        ("첫 시연에서 앱이 멈춘 경험을 통해 무엇을 배우셨나요?", True),
        ("제출하지 못한 경험에서 배우신 점은 무엇인가요?", True),
        ("시연 실패에서 배운 점은 무엇인가요?", True),
        ("그 실패에서 무엇을 배웠나요?", True),
        ("그 실패의 교훈은 무엇인가요?", True),
        ("그 대회는 언제 열렸나요?", False),
        ("마감을 넘겼다는 것을 어떻게 확인했나요?", False),
        ("발표 자료를 어떻게 압축하셨나요?", False),
        ("다음 대회에서는 어떤 기술을 배우고 싶으신가요?", False),
        ("그 공연에서 배우로 출연하셨나요?", False),
    ],
)
def test_material_learning_patterns_score_polite_forms_without_broadening(question, follows):
    from scripts import eval_match

    material_dir = eval_match.ROOT / "samples" / "match-materials"
    experiences = [
        MatchExperience.model_validate(e)
        for e in json.loads((material_dir / "experiences.json").read_text(encoding="utf-8"))
    ]
    questions = json.loads((material_dir / "questions.json").read_text(encoding="utf-8"))
    # 실제 정답 자료와 score 집계까지 통과시켜, 실패 두 문항의 경험 모두를 확인한다.
    for q in questions:
        if q["questionType"] != "FAILURE":
            continue
        for exp_id in map(int, q["materialAskPatterns"]):
            response = _response(
                [("실패", "PARTIAL", [exp_id])],
                [_rec(exp_id, "r")],
                [(exp_id, question)],
                exps=experiences,
            )
            result = score(q, experiences, response)
            assert (result.material_followed, result.material_asks) == (int(follows), 1)
            assert (result.material_covered, result.material_questions) == (int(follows), 1)
            assert len(result.material_missed) == int(not follows)


@pytest.mark.parametrize("job_context", [True, False])
def test_material_evaluation_sends_only_the_question_pool(monkeypatch, tmp_path, job_context):
    import asyncio

    import dotenv

    from app import llm
    from scripts import eval_match
    from tests.conftest import FakeLLM

    seen = []

    class Spy(FakeLLM):
        async def generate_json(self, system, user, schema, deadline):
            seen.append((system, user))
            return await super().generate_json(system, user, schema, deadline)

    fake = Spy(
        {
            "questionType": "OTHER",
            "requirements": [{"text": "요건", "status": "NOT_FOUND", "experienceIds": []}],
            "recommendations": [],
            "clarifications": [],
        }
    )
    monkeypatch.setattr(llm, "build_llm", lambda settings: fake)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    q = {
        **Q["tech-problem"],
        "pool": [101],
        "jobRole": "BACKEND",
        "postingLines": [{"section": "REQUIRED", "text": "Spring API 개발", "ksa": "SKILL"}],
    }
    asyncio.run(
        eval_match.run(
            ["openai:fake"],
            ["generic"],
            ["with"],
            EXPERIENCES,
            [q],
            [],
            tmp_path,
            job_context=job_context,
        )
    )
    assert len(seen) == 1
    from app.match import JOB_CONTEXT_RULE

    assert seen[0][0] == eval_match.GENERIC_PROMPT + (JOB_CONTEXT_RULE if job_context else "")
    assert ("지원 직무: BACKEND" in seen[0][1]) == job_context
    assert ("Spring API 개발" in seen[0][1]) == job_context
    assert "[경험 101]" in seen[0][1]
    assert "[경험 102]" not in seen[0][1]
    assert (tmp_path / "openai_fake" / "generic" / "run-1" / "tech-problem.json").is_file()


def test_context_score_uses_first_activity_and_flags_posting_only_requirements():
    q = {
        **Q["tech-problem"],
        "roleActivityIds": [BY_ID[101].activityId],
        "forbiddenRequirements": ["Spring|트래픽"],
    }
    data = _response([("Spring 트래픽 처리", "SUPPORTED", [101])], [_rec(101, "r")])
    result = score(q, EXPS, data)
    assert (result.role_right, result.role_total) == (1, 1)
    assert result.unrelated_requirements == ["tech-problem: Spring 트래픽 처리"]
    empty = _response([("문제 해결", "NOT_FOUND", [])], [])
    assert score(q, EXPS, empty).role_right == 0
    assert score(Q["tech-problem"], EXPS, data).role_total == 0


# SAMPLE(match-context): 실제 사람의 기록이 아닌 직무 · 공고 맥락 비교 사례.
def test_context_samples_follow_contract(monkeypatch):
    from scripts import eval_match

    monkeypatch.setattr(eval_match, "DATA", eval_match.ROOT / "samples" / "match-context")
    exps, questions = eval_match.load_data(None)
    assert len(questions) == 5
    by_id = {e["id"]: e for e in exps}
    for q in questions:
        MatchRequest.model_validate({"question": q, "experiences": exps})
        assert set(q["good"] + q["ok"] + q["bad"]) <= set(by_id)
        assert set(q.get("roleActivityIds", [])) <= {e["activityId"] for e in exps}
        for pattern in q.get("forbiddenRequirements", []):
            re.compile(pattern)


# SAMPLE(match-strength): 실제 사람의 기록이 아닌 지어낸 장단점 문항 사례(AI#140).
def test_strength_sample_requests_and_answer_ids(monkeypatch):
    from app.match import is_strength_question
    from scripts import eval_match

    monkeypatch.setattr(eval_match, "DATA", eval_match.ROOT / "samples" / "match-strength")
    exps, questions = eval_match.load_data(None)
    ids = {e["id"] for e in exps}
    assert {q["kind"] for q in questions} == {"match", "none"}
    for q in questions:
        pool = [e for e in exps if e["id"] in q.get("pool", ids)]
        MatchRequest.model_validate({"question": {"text": q["text"]}, "experiences": pool})
        assert is_strength_question(q["text"])  # 이 자료는 전부 장단점 규칙이 붙는 문항이다
        assert set(q["good"] + q["ok"] + q["bad"]) <= {e["id"] for e in pool}
        assert set(map(int, q.get("sideByExp", {}))) <= set(q["good"] + q["ok"])
        assert set(q.get("sideByExp", {}).values()) <= {"약점", "강점"}
        for pattern in q.get("materialAskPatterns", {}).values():
            re.compile(pattern)


def test_score_counts_hedged_reasons_and_side_labels():
    q = {**Q["tech-problem"], "sideByExp": {"101": "약점", "104": "강점"}}
    data = _response(
        [("성향", "PARTIAL", [101, 104])],
        [
            _rec(101, "약점 재료 — 확인보다 실행이 앞서는 편으로 볼 수 있습니다."),
            _rec(104, "설계를 맡은 사례입니다. 어떤 강점과 연결될지는 확인이 필요합니다."),
        ],
    )
    result = score(q, EXPS, data)
    assert (result.side_right, result.side_total) == (1, 2)
    assert len(result.side_wrong) == 1 and "104" in result.side_wrong[0]
    assert len(result.hedged) == 1 and "104" in result.hedged[0]
