"""초안 시험 자료(samples/draft)와 채점 — LLM 없이."""

import re

from app.draft import backings
from app.schemas import LLMDraftOutput, MatchExperience
from scripts.eval_draft import load_experiences, load_facts, load_pairs, picked, score

EXPERIENCES = load_experiences()
PAIRS = load_pairs()
PAIR = {p["id"]: p for p in PAIRS}


def _exps(pair):
    return [MatchExperience.model_validate(e) for e in picked(pair, EXPERIENCES)]


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
        known = backings(exps)
        for want in pair["expect"]:
            assert all(tuple(e) in known for e in want["evidence"]), pair["id"]


def test_missing_facts_are_really_missing():
    """정답이 틀리면 숫자가 틀린다 — 자료에 없다고 적은 사실이 근거 · 보완에 있으면 안 된다."""
    for pair in PAIRS:
        texts = [b.text for b in backings(_exps(pair)).values()]
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
        _, exps = check_request(req, load_settings())
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
