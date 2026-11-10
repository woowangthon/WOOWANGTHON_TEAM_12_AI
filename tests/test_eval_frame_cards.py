"""구조 카드 초안 시험 자료(samples/draft/cards.json)와 채점 — LLM 없이."""

from app.config import load_settings
from app.draft import FRAME_STEPS, check_request
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


def test_joins_count_sentences_opened_by_connectives():
    from scripts.eval_frame_cards import joins_of

    data = LLMDraftOutput.model_validate(
        {
            "sentences": [
                _sentence("서버가 느렸습니다.", 1),
                _sentence("그래서 로그를 봤습니다.", 1),
                _sentence("캐시를 붙였습니다.", 1),
                _sentence("이 일로 측정부터 하게 됐습니다.", 1),
            ],
            "omitted": [],
        }
    )
    assert joins_of(data) == (2, 3)
