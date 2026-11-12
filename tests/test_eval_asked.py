"""scripts/eval_asked.py — 자료가 요청 모양에 맞나 · 전후 요청 · 채점(LLM 안 부름)."""

from app.schemas import DraftRequest, MatchRequest
from scripts.eval_asked import Score, load_data, request_body, score_case


def test_cases_are_valid_requests():
    data = load_data()
    for case in data["cases"]:
        assert case["target"] in case["experienceIds"]
        for mode in ("before", "after"):
            body = request_body(case, data["experiences"], mode)
            model = MatchRequest if case["endpoint"] == "match" else DraftRequest
            model.model_validate(body)


def test_before_drops_asked_questions():
    data = load_data()
    case = next(c for c in data["cases"] if c.get("asked"))
    before = request_body(case, data["experiences"], "before")
    after = request_body(case, data["experiences"], "after")
    assert all("askedQuestions" not in e for e in before["experiences"])
    assert any(e.get("askedQuestions") for e in after["experiences"])


def test_score_counts_reask_and_claim():
    data = load_data()
    case = next(c for c in data["cases"] if c["id"] == "none-role")
    record = {
        "response": {
            "clarifications": [{"experienceId": 502, "question": "직접 맡은 일이 있었나요?"}],
            "recommendations": [
                {
                    "experienceId": 502,
                    "reason": "제가 일정을 조율했다",
                    "emphasis": "조율",
                    "evidence": [],
                    "gaps": ["내 몫이 없다고 답했다"],
                }
            ],
        }
    }
    s = Score()
    score_case(case, record, s)
    assert len(s.reasked) == 1 and len(s.claimed) == 1 and s.gaps == 1
