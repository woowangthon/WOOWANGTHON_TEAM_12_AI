"""지침 · 틀 초안 시험 자료(samples/draft/guided.json)와 채점 — LLM 없이."""

from app.config import load_settings
from app.draft import check_request
from app.schemas import DraftRequest, LLMDraftOutput
from scripts.eval_draft import guidelines, load_experiences, load_pairs, picked, question
from scripts.eval_draft_guided import load_guided, score_style

EXPERIENCES = load_experiences()
GUIDED = load_guided()


def test_guided_pairs_are_valid_requests():
    bases = {p["id"] for p in load_pairs()}
    assert len({g["id"] for g in GUIDED}) == len(GUIDED)
    for pair in GUIDED:
        assert pair["base"] in bases
        assert "beats" not in pair  # 기본 순서 정답은 틀을 켠 쌍에 쓰지 않는다
        req = DraftRequest.model_validate(
            {
                "question": question(pair),
                "experiences": picked(pair, EXPERIENCES),
                "guidelines": guidelines(pair),
            }
        )
        q, _, gs = check_request(req, load_settings())
        assert q.frame == pair["frame"]
        assert [g.id for g in gs] == [g["id"] for g in pair["guidelines"]]
    # 근거와 부딪치는 「수치로」 지침이 하나는 있다
    assert any(g.get("expect") == "not" for p in GUIDED for g in p["guidelines"])


def _out(sentences, results):
    return LLMDraftOutput.model_validate(
        {
            "sentences": [
                {"kind": "FACT", "text": t, "evidence": [{"type": "CITATION", "id": 201}]}
                for t in sentences
            ],
            "omitted": [],
            "guidelines": results,
        }
    )


def test_score_style():
    pair = next(p for p in GUIDED if p["id"] == "g-tech-number")
    good = _out(
        ["채팅 메시지가 늦게 도착했습니다.", "제가 폴링을 WebSocket 으로 바꿨습니다."],
        [
            {"id": 905, "applied": False, "note": "자료에 수치가 없어 넣지 않았어요"},
            {"id": 906, "applied": True, "note": None},
        ],
    )
    s = score_style(pair, good)
    assert not s.broken
    assert (s.refused, s.refuse_chances, s.applied, s.apply_chances) == (1, 1, 1, 1)
    assert (s.order_hit, s.order_total) == (1, 1)

    bad = _out(
        ["제가 폴링을 WebSocket 으로 바꿨습니다.", "채팅 메시지가 늦게 도착했습니다."],
        [
            {"id": 906, "applied": True, "note": None},
            {"id": 905, "applied": True, "note": None},
        ],
    )
    s = score_style(pair, bad)
    assert s.broken and s.refused == 0 and s.order_hit == 0
