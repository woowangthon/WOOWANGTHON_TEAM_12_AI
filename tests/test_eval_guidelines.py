"""지침 정리 시험 자료(samples/guidelines)와 채점 — LLM 없이."""

import re

from scripts.eval_guidelines import load_texts, score

TEXTS = load_texts()
TEXT = {t["id"]: t for t in TEXTS}


def test_samples_are_sound():
    assert len({t["id"] for t in TEXTS}) == len(TEXTS)
    for t in TEXTS:
        assert t["text"].strip() and len(t["text"]) <= 20000, t["id"]
        for e in t["expect"]:
            # 모범 줄은 자기 정규식에 걸리고 금지 정규식엔 안 걸려야 채점이 맞다
            assert re.search(e["pattern"], e["line"]), (t["id"], e["line"])
            assert not any(re.search(f["pattern"], e["line"]) for f in t["forbidden"]), e["line"]
    assert any(not t["expect"] for t in TEXTS)  # 쓰는 법이 없는 글이 하나는 있다


def test_score_contract_example():
    lines = [
        "첫 문장에 무엇을 했고 어떻게 됐는지 먼저 쓴다",
        "형용사로 꾸미지 않고 한 행동을 쓴다",
        "팀 프로젝트는 내가 맡은 몫을 분명히 쓴다",
        "문단 하나에 이야기 하나만 쓴다",
    ]
    s = score(TEXT["contract"], lines)
    assert (s.kept, s.chances) == (4, 4)
    assert not s.added and not s.forbidden and not s.unmerged


def test_score_catches_added_forbidden_unmerged():
    s = score(
        TEXT["contract"],
        [
            "결론부터 쓴다",
            "첫 문장에 결과를 쓴다",
            "경험을 조금 부풀려 돋보이게 쓴다",
            "STAR 기법을 쓴다",
        ],
    )
    assert len(s.forbidden) == 1
    assert len(s.added) == 2  # 부풀림 줄 · STAR 줄은 글의 쓰는 법이 아니다
    assert len(s.unmerged) == 1
    assert s.kept == 1


def test_empty_text_and_cap():
    assert score(TEXT["review-only"], []).empty_hit == 1
    assert score(TEXT["review-only"], ["감사 인사를 쓴다"]).empty_hit == 0
    assert score(TEXT["many-tips"], []).chances == 10
