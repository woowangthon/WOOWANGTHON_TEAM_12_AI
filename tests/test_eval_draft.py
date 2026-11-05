"""초안 시험 자료(samples/draft)와 채점 — LLM 없이."""

import re

from app.draft import backings
from app.schemas import LLMDraftOutput, MatchExperience
from scripts.eval_draft import load_experiences, load_pairs, picked, score

EXPERIENCES = load_experiences()
PAIRS = load_pairs()
PAIR = {p["id"]: p for p in PAIRS}


def _exps(pair):
    return [MatchExperience.model_validate(e) for e in picked(pair, EXPERIENCES)]


def _out(*sentences, omitted=()):
    return LLMDraftOutput.model_validate(
        {
            "sentences": [
                {"text": t, "evidence": [{"type": k, "id": i} for k, i in ev]}
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
            omitted=["바꾼 뒤 확인한 방법"],
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
