"""같은 활동 제안 시험 자료(samples/same)와 채점 — LLM 없이."""

import pytest

from app.config import load_settings
from app.extract import check_request
from app.schemas import ExtractRequest
from scripts.eval_same import Score, build_request, load_inputs, score_item

INPUTS = load_inputs(None)


def test_enough_inputs_of_every_shape():
    # 이슈 완료 조건 — 계약 예시 · 같은 활동 · 다른 활동 · 빈 목록 · 100개쯤인 목록
    ids = [i["id"] for i in INPUTS]
    assert len(ids) == len(set(ids))
    answers = [a for i in INPUTS for a in i["answers"]]
    assert sum(a["same"] is not None for a in answers) >= 5
    assert sum(a["same"] is None and not a.get("maybe") for a in answers) >= 4
    assert any(not build_request(i)["existingActivities"] for i in INPUTS)
    assert any(len(build_request(i)["existingActivities"]) >= 100 for i in INPUTS)
    contract = next(i for i in INPUTS if i["id"] == "contract-example")
    assert [a["same"] for a in contract["answers"]] == [7, 8]


@pytest.mark.parametrize("item", INPUTS, ids=lambda i: i["id"])
def test_inputs_obey_the_request_rules(item):
    req = build_request(item)
    check_request(ExtractRequest.model_validate(req), load_settings())
    assert item["why"].strip()
    existing = [a["id"] for a in req["existingActivities"]]
    assert len(existing) == len(set(existing))
    for want in item["answers"]:
        assert {want["same"], *want.get("maybe", [])} - {None} <= set(existing)
        # 정답 활동의 이름 조각이 원본에 있어야 LLM 활동 이름으로 찾을 수 있다
        assert want.get("whole") or any(k in req["text"] for k in want["keys"]), want["keys"]
    # 대상은 목록에 있어야 한다(없으면 400)
    assert item.get("target") in {None, *existing}


def _act(name, same):
    return {"name": name, "sameAsActivityId": same}


def test_score_counts():
    item = next(i for i in INPUTS if i["id"] == "resume-years-only")  # 14 · 15 · 13 · null
    s = Score()
    score_item(
        s,
        item,
        {
            "activities": [
                _act("데이터분석 학회 DASH", 14),  # 잡음
                _act("길동무", None),  # 놓침
                _act("카페 아르바이트", 7),  # 틀리게 붙임
                _act("고등학교 방송부", None),
                _act("어디에도 없는 활동", 8),  # 정답에 없는 활동이 id — 틀리게 붙임
            ]
        },
    )
    assert (s.caught, s.expected_same) == (1, 3)
    assert len(s.missed) == 1 and "길동무" in s.missed[0]
    assert len(s.wrong) == 2
    assert not s.not_found and not s.id_on_empty
    assert len(s.suggestions) == 3


def test_score_maybe_and_empty_list():
    item = next(i for i in INPUTS if i["id"] == "no-service-name")  # 7, maybe null
    s = Score()
    score_item(s, item, {"activities": [_act("캡스톤 발표", None)]})
    assert (s.caught, s.expected_same, len(s.missed), len(s.wrong)) == (0, 1, 0, 0)

    empty = next(i for i in INPUTS if i["id"] == "empty-list")
    s = Score()
    score_item(s, empty, {"activities": [_act("아무 활동", 7)]})
    assert len(s.id_on_empty) == 1 and len(s.wrong) == 1


def _item(item_id):
    return next(i for i in INPUTS if i["id"] == item_id)


def test_target_inputs_cover_the_issue():
    # 이슈 완료 조건 — 대상을 찾아야 함 · 활동 여럿인 이력서 · 이름 없는 README · 대상이 자료에 없음
    targets = [i for i in INPUTS if "target" in i]
    assert len(targets) >= 10
    assert all(build_request(i)["targetActivityId"] == i["target"] for i in targets)
    assert len(_item("target-resume-many")["answers"]) == 4
    absent = _item("target-resume-absent")
    assert all(a["same"] != absent["target"] for a in absent["answers"])
    assert _item("target-readme-no-name")["answers"][0]["whole"]


def test_score_target_caught_and_leaned():
    s = Score()
    score_item(
        s,
        _item("target-resume-many"),
        {
            "activities": [
                _act("데이터분석 학회 DASH", 14),  # 대상 찾음
                _act("길동무", 14),  # 대상 쪽으로 쏠림
                _act("카페 아르바이트", 13),
                _act("고등학교 방송부", None),
            ]
        },
    )
    assert (s.target_caught, s.target_expected, len(s.target_missed)) == (1, 1, 0)
    assert len(s.target_wrong) == 1 and "길동무" in s.target_wrong[0]


def test_score_target_missed_and_whole():
    item = _item("target-readme-no-name")
    s = Score()
    score_item(s, item, {"activities": [_act("여행 앱", 15)]})  # 이름과 상관없이 잡는다
    assert (s.target_caught, len(s.target_missed), s.not_found) == (1, 0, [])
    s = Score()
    score_item(s, item, {"activities": [_act("여행 앱", None)]})
    assert (s.target_caught, len(s.target_missed)) == (0, 1)
    s = Score()
    score_item(s, item, {"activities": []})  # 활동을 못 뽑아도 대상 놓침이다
    assert (s.target_caught, len(s.target_missed), len(s.not_found)) == (0, 1, 1)


def test_score_target_maybe_null_is_not_missed():
    s = Score()
    score_item(s, _item("target-no-clue"), {"activities": [_act("회의록", None)]})
    assert (s.target_caught, s.target_missed, s.wrong) == (0, [], [])


def test_score_target_on_other_activity_counts_as_leaning():
    s = Score()
    score_item(s, _item("target-wrong-file"), {"activities": [_act("하루가계", 13)]})
    assert len(s.target_wrong) == 1 and len(s.wrong) == 1
