"""새 경험 후보 시험 자료(samples/new-facts)와 채점 — LLM 없이."""

from app.ground_check import split_sentences
from app.new_facts import remaining_sentences
from app.schemas import NewFactsRequest, NewFactsResponse
from scripts.eval_new_facts import (
    NEW_KINDS,
    SKIP_KINDS,
    answer_text,
    load_activities,
    load_answers,
    load_experiences,
    request_body,
    score,
)

EXPERIENCES = load_experiences()
ACTIVITIES = load_activities()
ANSWERS = load_answers()


def test_samples_are_sound():
    assert len({a["id"] for a in ANSWERS}) == len(ANSWERS)
    names = {a["name"] for a in ACTIVITIES}
    assert {e["activityName"] for e in EXPERIENCES} <= names
    ids = {a["id"] for a in ACTIVITIES}
    for answer in ANSWERS:
        assert split_sentences(answer_text(answer)) == [s["text"] for s in answer["sentences"]]
        req = NewFactsRequest.model_validate(request_body(answer, EXPERIENCES, ACTIVITIES))
        left = remaining_sentences(split_sentences(req.text), req.excludedSentences)
        for s in answer["sentences"]:
            assert (s["text"] in left) != bool(s.get("excluded")), s["text"]
            if s["want"] == "NEW":
                assert s["kind"] in NEW_KINDS
                assert s["activityId"] is None or s["activityId"] in ids
            else:
                assert s["want"] == "SKIP" and s["kind"] in SKIP_KINDS
                assert (s["kind"] == "draft") == bool(s.get("excluded"))


def _got(*candidates):
    return NewFactsResponse.model_validate(
        {
            "candidates": [
                {"sentence": t, "content": c, "actor": a, "activityId": i}
                for t, c, a, i in candidates
            ],
            "model": None,
        }
    )


def test_score_counts_each_mistake():
    answer = next(a for a in ANSWERS if a["id"] == "moyeo")
    texts = [s["text"] for s in answer["sentences"]]
    s = score(
        answer,
        _got(
            (texts[1], "FCM 알림으로 출석이 늘었다", "ME", 9),  # 찾음 · 다시 씀
            (texts[2], "추천 로직을 만들었다", "ME", 9),  # 잘못 고름 mine
            (texts[4], "헬스 체크를 붙였다", "TEAM", None),  # 찾음 · 주체 · 활동 다름
        ),
    )
    assert s.wrong == {"mine": 1}
    assert (s.found, s.chances) == (2, 2)
    assert (s.actor_ok, s.activity_ok) == (1, 1)
    assert len(s.off) == 1
    assert len(s.restated) == 1
    assert s.missed == []
