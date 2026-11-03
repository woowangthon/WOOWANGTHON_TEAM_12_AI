"""시험 자료 정답(samples/answers)과 채점 — LLM 없이."""

import pytest

from app.citations import SourceIndex
from app.schemas import ExtractResponse
from scripts.eval_extract import ANSWERS, SAMPLES, load_answers, score

ALL = load_answers(None)


def test_every_sample_has_answer():
    samples = {p.stem for p in SAMPLES.glob("*.md")} - {"README"}
    assert samples == {p.stem for p in ANSWERS.glob("*.json")}


@pytest.mark.parametrize("stem", sorted(ALL))
def test_anchors_are_in_source(stem):
    source, answer = ALL[stem]
    index = SourceIndex(source)
    anchors = [e["anchor"] for a in answer["activities"] for e in a["experiences"]]
    for anchor in anchors + answer.get("notExperiences", []):
        assert index.find(anchor), f"{stem}: 원본에 없는 정답 구절 「{anchor}」"
    for act in answer["activities"]:
        for exp in act["experiences"]:
            assert exp["actor"] and set(exp["actor"]) <= {"ME", "TEAM", "UNKNOWN"}


def _exp(content, actor, *quotes, status="SOURCE_CONFIRMED"):
    return {
        "content": content,
        "competencies": ["IMPLEMENTATION"],
        "actor": actor,
        "sourceStatus": status,
        "citations": [{"quote": q, "matched": True, "location": None} for q in quotes],
    }


def test_score_counts():
    source, answer = ALL["team-retro"]
    period = {"start": "2025-03", "end": "2025-06", "ongoing": False}
    data = ExtractResponse.model_validate(
        {
            "model": "fake",
            "activities": [
                {
                    "name": "모여봄",
                    "period": period,
                    "experiences": [
                        # 팀 일을 ME 로 — 틀림
                        _exp("스탠드업을 했다", "ME", "우리는 매주 월요일 30분씩 스탠드업을 했다."),
                        # 원본에 없는 수치
                        _exp(
                            "알림 서버를 맡아 50% 빨라졌다",
                            "ME",
                            "나는 로그인과 알림 서버를 맡았다.",
                        ),
                    ],
                },
                {
                    "name": "모여봄 회고",  # 같은 활동이 둘로 갈림
                    "period": period,
                    "experiences": [_exp("시안을 고쳤다", "TEAM", "시안을 세 번 고쳤다")],
                },
            ],
        }
    )
    s = score("team-retro", source, answer, data)
    assert (s.experiences, s.quotes, s.matched) == (3, 3, 3)
    assert len(s.made_up_numbers) == 1 and "50" in s.made_up_numbers[0]
    assert len(s.actor_wrong) == 1 and s.actor_wrong_me == 1
    assert (s.required, s.found) == (10, 3)
    assert (s.activities_expected, s.activities_got) == (1, 2)
    assert (s.mixed, s.split) == (0, 1)
