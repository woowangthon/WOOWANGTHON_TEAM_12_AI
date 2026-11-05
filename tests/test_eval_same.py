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
        assert any(k in req["text"] for k in want["keys"]), want["keys"]


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
