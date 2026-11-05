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
