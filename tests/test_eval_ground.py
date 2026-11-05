"""근거 점검 시험 자료(samples/ground)와 채점 — LLM 없이."""

from app.draft import backings
from app.ground_check import split_sentences
from app.schemas import GroundCheckResponse, MatchExperience
from scripts.eval_ground import KINDS, answer_text, load_answers, load_experiences, score

EXPERIENCES = [MatchExperience.model_validate(e) for e in load_experiences()]
ANSWERS = load_answers()


def test_samples_are_sound():
    assert len({a["id"] for a in ANSWERS}) == len(ANSWERS)
    known = backings(EXPERIENCES)
    for answer in ANSWERS:
        assert split_sentences(answer_text(answer)) == [s["text"] for s in answer["sentences"]]
        for s in answer["sentences"]:
            if s["want"] == "UNSUPPORTED":
                assert s["kind"] in KINDS
            if s["want"] in ("SOURCE", "USER_INPUT"):
                assert all(tuple(e) in known for e in s["evidence"]), s["text"]


def _got(answer, *statuses):
    sentences = []
    for s, status in zip(answer["sentences"], statuses, strict=True):
        evidence = s.get("evidence", []) if status in ("SOURCE", "USER_INPUT") else []
        note = "문장의 숫자(80)가 받치는 근거 · 보완에 없어요" if status == "UNSUPPORTED" else None
        sentences.append(
            {
                "text": s["text"],
                "status": status,
                "evidence": [{"type": t, "id": i} for t, i in evidence],
                "note": note,
            }
        )
    return GroundCheckResponse.model_validate({"sentences": sentences, "model": None})


def test_score_counts_each_mistake():
    answer = next(a for a in ANSWERS if a["id"] == "contract")
    s = score(answer, _got(answer, "UNSUPPORTED", "USER_INPUT", "UNSUPPORTED", "OPINION", "SOURCE"))
    assert len(s.false_alarm) == 1  # 받쳐진 첫 문장을 근거 없음으로
    assert s.caught == {"number": 1} and s.chances == {"number": 1, "mine": 1}
    assert s.by_code == 1
    assert len(s.to_opinion) == 1  # 「제가 맡아」를 생각으로 뺐다
    assert len(s.opinion_wrong) == 1
    assert (s.evidence_hit, s.evidence_chances) == (1, 1)
