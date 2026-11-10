"""구조 카드 초안 시험 자료(samples/draft/cards.json)와 채점 — LLM 없이."""

from app.config import load_settings
from app.draft import CARD_EXAMPLES, CARD_STEPS, FRAME_STEPS, check_request
from app.schemas import DraftRequest, LLMDraftOutput, MatchExperience
from scripts.eval_draft import context_of, load_experiences, picked, question
from scripts.eval_frame_cards import load_cards, plain_of, score_trunk

EXPERIENCES = load_experiences()
CARDS = load_cards()


def test_card_pairs_are_valid_requests():
    assert len({c["id"] for c in CARDS}) == len(CARDS)
    for pair in CARDS:
        assert pair["frame"] in FRAME_STEPS
        for p in (pair, plain_of(pair)):
            req = DraftRequest.model_validate(
                {
                    "question": question(p),
                    "experiences": picked(p, EXPERIENCES),
                    "contextExperiences": context_of(p, EXPERIENCES),
                }
            )
            q, _, _ = check_request(req, load_settings())
            assert q.frame == p.get("frame")
    # 틀 없이 쌍은 frame 만 다르다
    assert "frame" not in plain_of(CARDS[0])


def _sentence(text, citation):
    return {"kind": "FACT", "text": text, "evidence": [{"type": "CITATION", "id": citation}]}


def test_trunk_counts_switches_and_share():
    pair = next(c for c in CARDS if c["id"] == "c-motive")  # 경험 151 · 101
    chosen = [MatchExperience.model_validate(e) for e in picked(pair, EXPERIENCES)]
    mine = {e.id: e.citations[0].id for e in chosen}
    a, b = mine[151], mine[101]
    data = LLMDraftOutput.model_validate(
        {
            "sentences": [
                _sentence("가", a),
                _sentence("나", a),
                _sentence("다", b),
                _sentence("라", a),
            ],
            "omitted": [],
        }
    )
    t = score_trunk(pair, chosen, [], data)
    assert (t.pairs, t.switches, t.share) == (1, 2, 0.75)


def test_in_order_allows_merged_and_repeated_steps():
    from scripts.eval_frame_cards import in_order

    names = CARD_STEPS["CAUSE_FIX"]
    assert in_order("CAUSE_FIX", [names[0], names[2], names[2], names[5]])  # 건너뜀 · 두 문장
    assert not in_order("CAUSE_FIX", [names[0], names[3], names[2]])  # 뒤로 감
    assert not in_order("CAUSE_FIX", [names[0], "없는 단계"])
    skill = CARD_STEPS["SKILL_STEPS"]
    assert in_order("SKILL_STEPS", [skill[0], skill[4], skill[0], skill[2]])  # 둘째 역량
    assert not in_order("SKILL_STEPS", [skill[0], skill[4], skill[1]])


def test_score_steps_counts_copy_joins_and_bare_changes():
    from scripts.eval_frame_cards import score_steps

    card = next(c for c in CARDS if c["frame"] == "CAUSE_FIX")
    names = CARD_STEPS["CAUSE_FIX"]
    copied = CARD_EXAMPLES["CAUSE_FIX"][1][2]  # 예시 문장 그대로
    data = LLMDraftOutput.model_validate(
        {
            "sentences": [
                _sentence("서버 응답이 느렸습니다.", 1),
                _sentence(copied, 1),
                _sentence("그래서 로그를 봤습니다.", 1),
            ],
            "omitted": [],
        }
    )
    s = score_steps(card, data, [names[0], names[1], names[2]])
    assert (s.pairs, s.ordered, s.missing) == (1, 1, 3)
    assert len(s.copied) == 1
    assert (s.joins, s.gaps) == (1, 2)
    assert (len(s.bare), s.changes) == (1, 2)  # 걸림돌로 넘어간 문장만 이음말 없음
    # step 이 없는 판도 예시 베끼기 · 이음말은 센다
    assert score_steps(card, data, None).pairs == 0
    assert len(score_steps(card, data, None).copied) == 1
