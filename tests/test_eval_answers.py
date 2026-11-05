"""시험 자료 정답(samples/answers)과 채점 — LLM 없이."""

import pytest

from app.citations import SourceIndex
from app.schemas import ACTIVITY_CATEGORIES, ExtractResponse
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
    keys = [k for a in answer["activities"] for k in a["keys"]]
    for anchor in (
        anchors + keys + answer.get("notExperiences", []) + answer.get("notActivities", [])
    ):
        assert index.find(anchor), f"{stem}: 원본에 없는 정답 구절 「{anchor}」"
    for act in answer["activities"]:
        for exp in act["experiences"]:
            assert exp["actor"] and set(exp["actor"]) <= {"ME", "TEAM", "UNKNOWN"}
    for act in answer["activities"]:
        assert act["category"] and set(act["category"]) <= {*ACTIVITY_CATEGORIES, None}


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


def test_score_activities_without_experiences():
    # MOC 모양 — 링크만 있는 활동은 이름(keys)으로 찾고, 공부 노트를 활동으로 만들면 센다
    source, answer = ALL["obsidian-moc"]
    none = {"start": None, "end": None, "ongoing": False}
    data = ExtractResponse.model_validate(
        {
            "model": "fake",
            "activities": [
                {"name": "코딩동아리 브릿지 7기", "period": none, "experiences": []},
                {"name": "캡스톤 디자인", "period": none, "experiences": []},
                {
                    "name": "브릿지 해커톤",
                    "period": none,
                    # 노트 제목을 경험으로 — 경험 아닌 것
                    "experiences": [_exp("해커톤 회고를 썼다", "ME", "250520-브릿지 해커톤 회고")],
                },
                {"name": "스프링 입문 강의", "period": none, "experiences": []},
            ],
        }
    )
    s = score("obsidian-moc", source, answer, data)
    assert s.activities_required == 10
    assert s.activities_found == 3
    assert len(s.not_activities) == 1 and "스프링" in s.not_activities[0]
    assert len(s.not_experiences) == 1


def test_score_categories():
    # 맞음 · 판단이 갈리는 정답 · 억지로 고름 · 정답 null · 놓친 null 을 나눠 센다
    source, answer = ALL["category-list"]
    none = {"start": None, "end": None, "ongoing": False}
    picks = [
        ("공공데이터 활용 공모전", "PROJECT"),  # 맞음
        ("하늘 3기", "CLUB"),  # 판단이 갈려 CLUB 도 맞음
        ("둥지", "STUDY"),  # 가늠이 안 되는데 골랐다 — 틀림
        ("오로라", None),  # 정답이 null — 맞음
        ("TOEIC 900", None),  # 고를 수 있는데 비웠다 — 틀림
    ]
    data = ExtractResponse.model_validate(
        {
            "model": "fake",
            "activities": [
                {"name": n, "period": none, "category": c, "experiences": []} for n, c in picks
            ],
        }
    )
    s = score("category-list", source, answer, data)
    assert (s.categories, s.category_right, s.category_null) == (5, 3, 2)
    assert len(s.category_wrong) == 2 and "둥지" in s.category_wrong[0]
