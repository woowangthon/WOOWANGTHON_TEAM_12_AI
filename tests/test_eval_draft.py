"""초안 시험 자료(samples/draft)와 채점 — LLM 없이."""

import re

import pytest

from app.draft import backings
from app.schemas import LLMDraftOutput, MatchExperience
from scripts.eval_draft import (
    context_of,
    load_experiences,
    load_facts,
    load_pairs,
    picked,
    question,
    score,
    score_essay,
)

EXPERIENCES = load_experiences()
PAIRS = load_pairs()
PAIR = {p["id"]: p for p in PAIRS}


def _exps(pair):
    return [MatchExperience.model_validate(e) for e in picked(pair, EXPERIENCES)]


def _ctx(pair):
    return [MatchExperience.model_validate(e) for e in context_of(pair, EXPERIENCES)]


def _out(*sentences, omitted=()):
    return LLMDraftOutput.model_validate(
        {
            "sentences": [
                {
                    "kind": "FACT" if ev else "OPINION",
                    "text": t,
                    "evidence": [{"type": k, "id": i} for k, i in ev],
                }
                for t, ev in sentences
            ],
            "omitted": list(omitted),
        }
    )


def test_samples_are_sound():
    assert len({p["id"] for p in PAIRS}) == len(PAIRS)
    for pair in PAIRS:
        exps = _exps(pair)
        assert 1 <= len(exps) <= 3
        ctx = _ctx(pair)
        assert len(ctx) <= 12
        assert not {e.id for e in ctx} & {e.id for e in exps}, pair["id"]
        known = backings([*exps, *ctx])
        for want in pair["expect"]:
            assert all(tuple(e) in known for e in want["evidence"]), pair["id"]


def test_missing_facts_are_really_missing():
    """정답이 틀리면 숫자가 틀린다 — 자료에 없다고 적은 사실이 근거 · 보완에 있으면 안 된다."""
    for pair in PAIRS:
        texts = [b.text for b in backings([*_exps(pair), *_ctx(pair)]).values()]
        for m in pair["missing"]:
            hits = [t for t in texts if re.search(m["pattern"], t)]
            assert not hits, (pair["id"], m["fact"], hits)


def test_score_contract_example():
    s = score(
        PAIR["tech-a"],
        _exps(PAIR["tech-a"]),
        _out(
            ("제가 폴링을 WebSocket 으로 바꿔 해결했습니다.", [("CITATION", 201)]),
            ("앱의 3초 폴링 코드를 구독 방식으로 바꿨습니다.", [("SUPPLEMENT", 301)]),
            omitted=[_gap("바꾼 뒤 확인한 방법", "바꾼 뒤 어떻게 확인했나요?", 101)],
        ),
    )
    assert (s.omit_hit, s.omit_chances) == (1, 1)
    assert (s.expect_hit, s.expect_chances) == (1, 1)
    assert not (s.fake_ids or s.bad_numbers or s.invented or s.team_as_mine)


def test_score_counts_violations():
    s = score(
        PAIR["tech-a"],
        _exps(PAIR["tech-a"]),
        _out(
            ("앱의 STOMP 구독으로 지연을 30% 줄였습니다.", [("CITATION", 201)]),
            ("부하 테스트로 확인했습니다.", [("CITATION", 999)]),
        ),
    )
    assert len(s.bad_numbers) == 1
    assert len(s.fake_ids) == 1
    assert len(s.invented) == 2
    assert (s.omit_hit, s.omit_chances) == (0, 1)
    assert (s.expect_hit, s.expect_chances) == (0, 1)


def test_score_counts_unused_experience():
    pair = PAIR["team-way"]
    s = score(pair, _exps(pair), _out(("제가 폴링을 바꿨습니다.", [("CITATION", 201)])))
    assert s.unused == ["team-way: 경험 102"]
    assert s.picked_total == 2


def test_score_counts_order_and_chars():
    pair = PAIR["rich-problem"]
    right = _out(
        ("크롤러가 밤마다 멈춰 문의가 왔습니다.", [("CITATION", 251)]),
        ("원인은 503 을 잡지 못한 코드였습니다.", [("CITATION", 252)]),
        ("발표가 3주 남아 일부만 고치기로 판단했습니다.", [("CITATION", 253)]),
        ("제가 Redis 큐로 다시 긁게 했습니다.", [("CITATION", 254)]),
        ("4주 동안 멈춘 날이 없었습니다.", [("CITATION", 255)]),
        ("실패부터 설계해야 한다는 걸 배웠습니다.", [("SUPPLEMENT", 352)]),
    )
    s = score(pair, _exps(pair), right)
    assert (s.order_hit, s.order_total) == (1, 1)
    assert s.chars == sum(len(x.text) for x in right.sentences)
    swapped = LLMDraftOutput(sentences=right.sentences[::-1], omitted=[])
    assert score(pair, _exps(pair), swapped).order_hit == 0
    short = LLMDraftOutput(sentences=right.sentences[:2], omitted=[])
    assert score(pair, _exps(pair), short).order_miss


def test_score_counts_repeated_sentences():
    pair = PAIR["tech-a"]
    twice = _out(
        ("폴링을 WebSocket 으로 바꿔 문제를 풀었습니다.", [("CITATION", 201)]),
        ("저는 폴링을 WebSocket 으로 바꿔 지연을 없앴습니다.", [("CITATION", 201)]),
    )
    assert len(score(pair, _exps(pair), twice).repeated) == 1
    once = _out(("채팅이 늦게 도착했습니다.", [("CITATION", 201)]))
    assert score(pair, _exps(pair), once).repeated == []


def test_score_counts_once_fact_in_two_sentences():
    pair = PAIR["team-way"]
    twice = _out(
        ("제가 폴링을 바꿔 해결했습니다.", [("CITATION", 201)]),
        ("앱의 폴링 코드를 구독으로 바꿨습니다.", [("SUPPLEMENT", 301)]),
    )
    assert len(score(pair, _exps(pair), twice).repeated) == 1


def test_score_unknown_actor():
    pair = PAIR["unknown-only"]
    s = score(
        pair,
        _exps(pair),
        _out(
            ("서버가 자주 죽었고, 원인은 DB 커넥션을 닫지 않은 코드였습니다.", [("CITATION", 214)]),
            ("해당 코드가 수정됐습니다.", [("CITATION", 214)]),
            ("팀은 코드를 고쳤습니다.", [("CITATION", 214)]),
            ("코드를 고쳤습니다.", [("CITATION", 214)]),
            ("제가 코드를 고쳤습니다.", [("CITATION", 214)]),
        ),
    )
    assert len(s.unknown_actor) == 3
    assert len(s.team_as_mine) == 1


def test_score_team_as_mine():
    pair = PAIR["team-way"]
    s = score(
        pair,
        _exps(pair),
        _out(
            ("제가 매주 금요일 회고를 진행했습니다.", [("CITATION", 202)]),
            ("매주 금요일 회고를 했습니다.", [("CITATION", 202)]),
            ("팀은 매주 금요일 스프린트 회고를 했습니다.", [("CITATION", 202)]),
            ("제가 폴링을 WebSocket 으로 바꿨습니다.", [("CITATION", 201)]),
        ),
    )
    assert len(s.team_as_mine) == 2


def _gap(text, question=None, experience=None):
    return {"text": text, "question": question, "experienceId": experience}


def test_score_counts_question_rules():
    pair = PAIR["tech-a"]
    s = score(
        pair,
        _exps(pair),
        _out(
            ("제가 폴링을 WebSocket 으로 바꿔 해결했습니다.", [("CITATION", 201)]),
            omitted=[
                _gap("바꾼 뒤 확인한 방법", "지연을 30% 줄이셨죠?", 101),
                _gap("배운 점", "더 자세히 알려 주세요?", 101),
                _gap("직접 고친 부분", "직접 고친 부분은 어디였나요?", 101),
                _gap("결과", "어떻게 됐나요? 얼마나 나아졌나요?", 101),
            ],
        ),
        load_facts(),
    )
    assert len(s.asked) == 4
    assert len(s.leading) == 1
    assert len(s.vague) == 1
    assert len(s.known_asked) == 1
    assert len(s.compound) == 1
    assert (s.gap_asked, s.gap_chances) == (1, 1)
    assert s.bad_pairs == []


def test_score_counts_bad_pairs_and_forced_questions():
    pair = PAIR["motivation"]
    s = score(
        pair,
        _exps(pair),
        _out(
            ("크롤러가 밤마다 멈췄습니다.", [("CITATION", 251)]),
            omitted=[
                _gap("지원 동기", "이 부트캠프에 왜 지원했나요?", 151),
                _gap("확인한 방법", "어떻게 확인했나요?", None),
                _gap("확인한 방법", "어떻게 확인했나요?", 999),
            ],
        ),
    )
    assert len(s.forced) == 1
    assert len(s.bad_pairs) == 2
    assert len(s.asked) == 1


def test_loose_gap_needs_no_question():
    pair = PAIR["motivation"]
    s = score(pair, _exps(pair), _out(omitted=[_gap("이 부트캠프에 지원하는 이유")]))
    assert s.forced == []
    assert (s.omit_hit, s.omit_chances) == (1, 1)


def test_filled_pair_adds_supplement_to_request():
    pair = PAIR["tech-a-fill"]
    exp = _exps(pair)[0]
    assert [x.id for x in exp.supplements] == [301, 901]
    assert 901 not in [x.id for x in _exps(PAIR["tech-a"])[0].supplements]


def test_score_counts_supplement_use():
    pair = PAIR["tech-a-fill"]
    used = _out(("두 기기로 도착을 확인했습니다.", [("SUPPLEMENT", 901)]))
    s = score(pair, _exps(pair), used)
    assert (s.fill_hit, s.fill_chances) == (1, 1)
    wrong_ref = _out(("두 기기로 도착을 확인했습니다.", [("CITATION", 201)]))
    assert score(pair, _exps(pair), wrong_ref).fill_hit == 0
    asked_again = _out(
        ("제가 폴링을 바꿨습니다.", [("CITATION", 201)]),
        omitted=[_gap("바꾼 뒤 확인한 방법", "어떻게 확인했나요?", 101)],
    )
    s = score(pair, _exps(pair), asked_again)
    assert len(s.fill_miss) == 1
    assert len(s.fill_reasked) == 1


def test_my_share_from_supplement_is_not_team_as_mine():
    """보완에 사용자가 자기 몫이라 말한 일은 UNKNOWN 경험이어도 「제가」로 쓴다."""
    pair = PAIR["unknown-fill"]
    said = _out(("제가 try-with-resources 로 고쳤습니다.", [("SUPPLEMENT", 904)]))
    s = score(pair, _exps(pair), said)
    assert s.team_as_mine == []
    assert s.unknown_actor == []
    assert s.fill_hit == 1
    leaked = _out(("제가 서버 장애를 고쳤습니다.", [("CITATION", 214)]))
    assert len(score(pair, _exps(pair), leaked).team_as_mine) == 1


def test_known_facts_point_to_real_items():
    by_id = {e["id"]: e for e in EXPERIENCES}
    for fact in load_facts():
        key = "citations" if fact["source"] == "CITATION" else "supplements"
        assert fact["id"] in [x["id"] for x in by_id[fact["experienceId"]][key]]


def test_score_opinions_are_kept_out_of_fact_rules():
    """OPINION 은 근거 규칙에 세지 않는다 — 문장 수 · 주어 · 지어냄이 아니라 생각 칸에 간다."""
    s = score(
        PAIR["tech-a"],
        _exps(PAIR["tech-a"]),
        _out(
            ("제가 폴링을 WebSocket 으로 바꿔 해결했습니다.", [("CITATION", 201)]),
            ("앞으로도 문제의 원인을 살피고 싶습니다.", []),
        ),
    )
    assert len(s.sentences) == 1
    assert len(s.opinions) == 1
    assert (s.kind_broken, s.opinion_happened, s.opinion_invented) == ([], [], [])


def test_score_catches_fact_hidden_as_opinion():
    """가장 큰 위험 — 자료에 없는 결과를 생각 문장으로 빼면 후보로 잡힌다."""
    s = score(
        PAIR["tech-a"],
        _exps(PAIR["tech-a"]),
        _out(
            ("제가 폴링을 WebSocket 으로 바꿔 해결했습니다.", [("CITATION", 201)]),
            ("메시지 지연을 크게 줄였습니다.", []),
            ("뿌듯함을 느꼈습니다.", []),
        ),
    )
    assert len(s.invented) == 0  # 사실 문장 쪽 칸에는 없다
    assert len(s.opinion_invented) == 2
    assert len(s.opinion_happened) == 2
    assert len(s.opinion_flattery) == 1


def test_score_kind_broken():
    pair = PAIR["tech-a"]
    only = _out(("앞으로 살피겠습니다.", []))
    assert score(pair, _exps(pair), only).kind_broken == ["tech-a: 생각 문장만"]
    many = _out(
        ("제가 해결했습니다.", [("CITATION", 201)]),
        *[(f"살피겠습니다 {i}.", []) for i in "가나다라"],
    )
    assert score(pair, _exps(pair), many).kind_broken == ["tech-a: OPINION 4개"]
    number = _out(("제가 해결했습니다.", [("CITATION", 201)]), ("3년 안에 살피겠습니다.", []))
    assert "숫자" in score(pair, _exps(pair), number).kind_broken[0]


def test_hinted_pairs_make_valid_requests():
    """hints 는 고른 경험의 것이고, 실어 보낸 요청이 /draft 의 400 검사를 통과한다."""
    from app.config import load_settings
    from app.draft import check_request
    from app.schemas import DraftRequest
    from scripts.eval_draft import question

    hinted = [p for p in PAIRS if p.get("hints") or p.get("jobRole")]
    assert hinted
    for pair in hinted:
        assert {int(i) for i in pair.get("hints", {})} <= set(pair["experienceIds"]), pair["id"]
        req = DraftRequest.model_validate(
            {"question": question(pair), "experiences": picked(pair, EXPERIENCES)}
        )
        _, exps, _ = check_request(req, load_settings())
        assert any(e.emphasis for e in exps) == bool(pair.get("hints")), pair["id"]


def test_hint_only_facts_are_not_backed():
    """강조점에만 있는 숫자 · 말을 missing 에 적은 쌍 — 그게 근거에 없어야 채점이 맞다(위 시험이 본다)."""
    assert "90" in PAIR["emph-number"]["hints"]["151"]["emphasis"]
    assert any(m["pattern"].startswith("90") for m in PAIR["emph-number"]["missing"])


def test_score_lead_and_stale_gap():
    pair = PAIR["emph-judgment"]
    early = _out(
        ("크롤러가 밤마다 멈춰 문의가 왔습니다.", [("CITATION", 251)]),
        ("통째로 바꾸지 않고 실패한 학과만 다시 긁기로 했습니다.", [("CITATION", 253)]),
    )
    late = _out(
        ("크롤러가 밤마다 멈춰 문의가 왔습니다.", [("CITATION", 251)]),
        ("사이트가 503 을 주면 크롤러가 죽었습니다.", [("CITATION", 252)]),
        ("통째로 바꾸지 않고 실패한 학과만 다시 긁기로 했습니다.", [("CITATION", 253)]),
    )
    assert (score(pair, _exps(pair), early).lead_hit, score(pair, _exps(pair), late).lead_hit) == (
        1,
        0,
    )
    invented = _out(("백엔드 개발자로 일하며 크롤러를 고쳤습니다.", [("CITATION", 254)]))
    assert score(pair, _exps(pair), invented).hinted_invented

    stale = PAIR["emph-unbacked"]
    s = score(
        stale,
        _exps(stale),
        _out(
            ("제가 폴링을 WebSocket 으로 바꿔 해결했습니다.", [("CITATION", 201)]),
            omitted=[_gap("직접 고친 범위", "직접 고친 부분은 어디였나요?", 101)],
        ),
    )
    assert len(s.stale_reasked) == 1
    assert not score(PAIR["tech-a"], _exps(PAIR["tech-a"]), early).lead_chances


# ---- 글 짜임 · AI 보탬 · 빈칸 · 목표 글자 수 · 보조 재료(S-75 · S-77 · S-78) ----


def _mix(*items, omitted=()):
    """(kind, text, evidence, note) — ELABORATED 와 note 를 쓸 수 있는 _out."""
    return LLMDraftOutput.model_validate(
        {
            "sentences": [
                {
                    "kind": kind,
                    "text": text,
                    "evidence": [{"type": k, "id": i} for k, i in ev],
                    "note": note,
                }
                for kind, text, ev, note in items
            ],
            "omitted": list(omitted),
        }
    )


def _essay(pair_id, data):
    pair = PAIR[pair_id]
    return score_essay(pair, _exps(pair), _ctx(pair), data)


def test_context_pairs_make_valid_requests():
    """보조 재료 쌍도 /draft 의 400 검사를 통과한다 — 개수 · 겹침 · 목표 글자 수 범위."""
    from app.config import load_settings
    from app.draft import check_request
    from app.schemas import DraftRequest

    with_ctx = [p for p in PAIRS if p.get("contextExperienceIds") or p.get("targetLength")]
    assert {"ctx-tech", "ctx-collab", "ctx-unknown", "ctx-two", "target-thin"} <= {
        p["id"] for p in with_ctx
    }
    for pair in with_ctx:
        req = DraftRequest.model_validate(
            {
                "question": question(pair),
                "experiences": picked(pair, EXPERIENCES),
                "contextExperiences": context_of(pair, EXPERIENCES),
            }
        )
        check_request(req, load_settings())


def test_context_of_keeps_pair_order_and_defaults_empty():
    pair = PAIR["ctx-tech"]
    assert [e["id"] for e in context_of(pair, EXPERIENCES)] == pair["contextExperienceIds"]
    assert context_of(PAIR["tech-a"], EXPERIENCES) == []


def test_question_carries_target_length():
    assert question(PAIR["ctx-tech"])["targetLength"] == 600
    assert "targetLength" not in question(PAIR["tech-a"])


def test_score_counts_listing():
    """경험마다 한 문장씩뿐이면 나열이고, 한 경험에 둘 이상이면 아니다."""
    flat = _out(
        ("채팅이 늦게 도착했습니다.", [("CITATION", 201)]),
        ("일정이 밀려 남은 기능을 줄였습니다.", [("CITATION", 206)]),
    )
    s = _essay("ctx-tech", flat)
    assert (len(s.listed), s.list_chances) == (1, 1)
    deep = _out(
        ("채팅이 늦게 도착했습니다.", [("CITATION", 201)]),
        ("앱의 3초 폴링 코드를 구독으로 바꿨습니다.", [("SUPPLEMENT", 301)]),
        ("일정이 밀려 남은 기능을 줄였습니다.", [("CITATION", 206)]),
    )
    s = _essay("ctx-tech", deep)
    assert (s.listed, s.list_chances) == ([], 1)
    # 경험이 하나뿐인 쌍은 나열을 따지지 않는다
    assert _essay("tech-a", flat).list_chances == 0


def test_score_counts_deep_and_shallow():
    deep = _out(
        ("채팅이 늦게 도착했습니다.", [("CITATION", 201)]),
        ("앱의 3초 폴링 코드를 구독으로 바꿨습니다.", [("SUPPLEMENT", 301)]),
    )
    s = _essay("ctx-tech", deep)
    assert (s.deep_hit, s.deep_chances, s.shallow) == (1, 1, [])
    one = _out(("채팅이 늦게 도착했습니다.", [("CITATION", 201)]))
    s = _essay("ctx-tech", one)
    assert (s.deep_hit, s.deep_chances) == (0, 1)
    assert s.shallow == ["ctx-tech: 경험 101 문장 1개"]


def test_deep_not_asked_when_first_has_no_backing():
    pair = PAIR["target-thin"]
    bare = {**EXPERIENCES[0], "id": 9001, "citations": [], "supplements": []}
    s = score_essay(pair, [MatchExperience.model_validate(bare)], [], _out())
    assert (s.deep_hit, s.deep_chances) == (0, 0)


def test_score_prints_first_sentence_kind():
    data = _mix(("ELABORATED", "문제를 먼저 풀어야 했습니다.", [("CITATION", 201)], "풀어 씀"))
    assert _essay("ctx-tech", data).firsts == [
        "ctx-tech: [ELABORATED] 「문제를 먼저 풀어야 했습니다.」"
    ]
    assert _essay("ctx-tech", _out()).firsts == []


def test_score_counts_elaborated():
    data = _mix(
        ("FACT", "폴링을 WebSocket 으로 바꿨습니다.", [("CITATION", 201)], None),
        ("ELABORATED", "문제를 풀어야 했습니다.", [("CITATION", 201)], "상황을 풀어 씀"),
        ("ELABORATED", "지연을 대폭 줄였다고 판단했습니다.", [("CITATION", 201)], "결과를 짐작"),
        ("ELABORATED", "뿌듯함을 느꼈습니다.", [("CITATION", 201)], "감정"),
    )
    s = _essay("ctx-tech", data)
    assert len(s.elaborated) == 3
    assert s.elaborated[0] == "ctx-tech: 「문제를 풀어야 했습니다.」 — note 「상황을 풀어 씀」"
    assert len(s.elab_heavy) == 1  # ELABORATED 3 > FACT 1
    assert len(s.elab_invented) == 1  # missing 「대폭」
    assert len(s.elab_judged) == 2  # 판단했 · 느꼈
    balanced = _mix(
        ("FACT", "폴링을 WebSocket 으로 바꿨습니다.", [("CITATION", 201)], None),
        ("ELABORATED", "문제를 풀어야 했습니다.", [("CITATION", 201)], "풀어 씀"),
    )
    s = _essay("ctx-tech", balanced)
    assert (s.elab_heavy, s.elab_invented, s.elab_judged) == ([], [], [])


def test_unknown_mine_counts_elaborated_and_fact():
    elab = _mix(("ELABORATED", "제가 서버 장애 원인을 찾았습니다.", [("CITATION", 214)], "확인"))
    s = _essay("ctx-unknown", elab)
    assert len(s.unknown_mine) == 1
    assert s.unknown_mine[0].startswith("ctx-unknown: [ELABORATED]")
    # 사용자 몫이라는 말이 없는 문장은 세지 않는다
    other = _mix(("ELABORATED", "서버가 자주 죽었습니다.", [("CITATION", 214)], "풀어 씀"))
    assert _essay("ctx-unknown", other).unknown_mine == []


def test_elaborated_on_unknown_is_not_team_as_mine():
    pair = PAIR["ctx-unknown"]
    text = "제가 서버 장애 원인을 찾았습니다."
    elab = _mix(("ELABORATED", text, [("CITATION", 214)], "확인 필요"))
    s = score(pair, _exps(pair), elab, context=_ctx(pair))
    assert s.team_as_mine == []
    assert s.unknown_actor == []
    fact = _mix(("FACT", text, [("CITATION", 214)], None))
    s = score(pair, _exps(pair), fact, context=_ctx(pair))
    assert len(s.team_as_mine) == 1
    assert len(s.unknown_actor) == 1


def test_elaborated_on_team_experience_is_still_team_as_mine():
    """UNKNOWN 만 예외다 — TEAM 경험을 ELABORATED 로 「제가」 쓰면 그대로 센다."""
    pair = PAIR["ctx-tech"]
    text = "제가 매주 금요일 회고를 진행했습니다."
    elab = _mix(("ELABORATED", text, [("CITATION", 202)], "풀어 씀"))
    s = score(pair, _exps(pair), elab, context=_ctx(pair))
    assert len(s.team_as_mine) == 1


def test_score_lists_marks_with_sentence_index():
    text = "지연을 [줄인 정도]만큼 줄이고 [확인 방법]으로 확인했습니다."
    data = _out(
        ("폴링을 바꿨습니다.", [("CITATION", 201)]),
        (text, [("CITATION", 201)]),
    )
    s = _essay("ctx-tech", data)
    assert s.marks == [
        f"ctx-tech: [1] [줄인 정도] 「{text}」",
        f"ctx-tech: [1] [확인 방법] 「{text}」",
    ]
    assert _essay("ctx-tech", _out(("폴링을 바꿨습니다.", [("CITATION", 201)]))).marks == []


@pytest.mark.parametrize("chars", [399, 400, 450, 500, 501, 900])
def test_score_target_ratio(chars):
    """ctx-collab 의 목표는 500자 — 80 ~ 100%(400 ~ 500자) 면 맞다."""
    pair = PAIR["ctx-collab"]
    assert pair["targetLength"] == 500
    s = _essay("ctx-collab", _out(("가" * chars, [("CITATION", 207)])))
    assert s.target_chances == 1
    if 400 <= chars <= 500:
        assert (s.target_hit, s.target_miss) == (1, [])
    else:
        assert s.target_hit == 0
        assert s.target_miss == [f"ctx-collab: 목표 500자의 {chars / 500:.0%}"]


def test_score_target_absent_not_counted():
    assert _essay("tech-a", _out(("가" * 10, [("CITATION", 201)]))).target_chances == 0


def test_score_context_use():
    used = _out(
        ("채팅이 늦게 도착했습니다.", [("CITATION", 201)]),
        ("앱의 3초 폴링 코드를 구독으로 바꿨습니다.", [("SUPPLEMENT", 301)]),
        ("일정이 밀려 남은 기능을 줄였습니다.", [("CITATION", 206)]),
    )
    s = _essay("ctx-tech", used)
    assert (s.context_used, s.context_chances, s.context_heavy) == (1, 1, [])
    unused = _out(("채팅이 늦게 도착했습니다.", [("CITATION", 201)]))
    s = _essay("ctx-tech", unused)
    assert (s.context_used, s.context_chances) == (0, 1)
    # 보조 재료가 없는 쌍은 따지지 않는다
    assert _essay("tech-a", unused).context_chances == 0


def test_score_context_heavy():
    heavy = _out(
        ("채팅이 늦게 도착했습니다.", [("CITATION", 201)]),
        ("일정이 밀려 남은 기능을 줄였습니다.", [("CITATION", 206)]),
        ("발표 전 기능을 정리했습니다.", [("CITATION", 206)]),
    )
    s = _essay("ctx-tech", heavy)
    assert s.context_used == 1
    assert s.context_heavy == ["ctx-tech: 보조만 2 > 고른 경험 1"]
    # 한 문장이 둘 다 받치면 고른 경험 쪽으로 센다
    both = _out(("채팅과 일정 이야기입니다.", [("CITATION", 201), ("CITATION", 206)]))
    s = _essay("ctx-tech", both)
    assert s.context_used == 1 and s.context_heavy == []


def test_score_accepts_context_evidence():
    """보조 재료의 근거는 가짜 ID 가 아니다 — context 를 안 주면 가짜로 센다."""
    pair = PAIR["ctx-tech"]
    data = _out(
        ("제가 폴링을 바꿨습니다.", [("CITATION", 201)]),
        ("일정이 밀려 남은 기능을 줄였습니다.", [("CITATION", 206)]),
    )
    assert score(pair, _exps(pair), data, context=_ctx(pair)).fake_ids == []
    assert len(score(pair, _exps(pair), data).fake_ids) == 1


def test_score_omitted_may_ask_about_context_experience():
    pair = PAIR["ctx-tech"]
    data = _out(
        ("제가 폴링을 바꿨습니다.", [("CITATION", 201)]),
        omitted=[_gap("일정을 줄인 결과", "줄인 뒤 어떻게 됐나요?", 106)],
    )
    assert score(pair, _exps(pair), data, context=_ctx(pair)).bad_pairs == []
    assert len(score(pair, _exps(pair), data).bad_pairs) == 1


def test_score_without_context_has_empty_new_fields():
    pair = PAIR["tech-a"]
    s = score(pair, _exps(pair), _out(("제가 폴링을 바꿨습니다.", [("CITATION", 201)])))
    assert (s.context_chances, s.context_used, s.target_chances) == (0, 0, 0)
    assert s.elaborated == [] and s.marks == []


def test_kind_broken_counts_elaborated_without_evidence():
    pair = PAIR["tech-a"]
    data = _mix(("ELABORATED", "문제를 풀어야 했습니다.", [], "풀어 씀"))
    assert any("ELABORATED 근거 0개" in x for x in score(pair, _exps(pair), data).kind_broken)
