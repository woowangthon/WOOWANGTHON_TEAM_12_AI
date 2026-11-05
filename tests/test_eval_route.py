"""가르기 시험 자료(samples/route)와 채점 — LLM 없이."""

import json

import pytest

from app.route import MAX_ROUTE_CHARS, check_request
from app.schemas import ENTRANCE_KINDS, RouteRequest
from scripts.eval_route import judge, load_inputs, score_dir

INPUTS = load_inputs(None)


def test_enough_inputs_of_every_kind():
    # 이슈 완료 조건 — 자료 · 문항 · 공고 · 한 줄 질문 · 애매한 것 섞어 20개 이상
    assert len(INPUTS) >= 20
    for kind in ENTRANCE_KINDS:
        assert sum(i["answer"] == kind for i in INPUTS) >= 4, kind
    ids = [i["id"] for i in INPUTS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("item", INPUTS, ids=lambda i: i["id"])
def test_input_is_a_valid_request(item):
    assert item["answer"] in ENTRANCE_KINDS
    assert set(item.get("alsoOk", ())) <= set(ENTRANCE_KINDS) - {item["answer"]}
    assert item["why"].strip()
    req = RouteRequest(text=item["text"], totalLength=item["totalLength"])
    check_request(req)  # 400 이 아니다
    assert len(item["text"]) <= MAX_ROUTE_CHARS <= max(MAX_ROUTE_CHARS, item["totalLength"])


def test_judge():
    item = {"answer": "APPLICATION", "alsoOk": ["UNSURE"]}
    assert judge(item, "APPLICATION") == "right"
    assert judge(item, "UNSURE") == "right"
    assert judge(item, "ASK") == "wrong"
    strict = {"answer": "ASK"}
    assert judge(strict, "UNSURE") == "asked"
    assert judge(strict, "SOURCE") == "wrong"


def test_score_dir(tmp_path):
    items = [
        {"id": "a", "answer": "SOURCE", "heldOut": True},
        {"id": "b", "answer": "ASK"},
        {"id": "c", "answer": "APPLICATION"},
        {"id": "d", "answer": "UNSURE", "heldOut": True},
        {"id": "missing", "answer": "UNSURE"},
    ]
    records = {
        "a": {"response": {"kind": "SOURCE", "reason": "회고"}, "seconds": 1.0},
        "b": {"response": {"kind": "UNSURE", "reason": "애매"}, "seconds": 2.0},
        "c": {"response": {"kind": "ASK", "reason": "질문"}, "seconds": 3.0},
        "d": {"error": "LLM_INVALID_OUTPUT: x", "seconds": 4.0},
    }
    for name, record in records.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")
    s = score_dir(tmp_path, items)
    assert (s.total, s.right) == (4, 1)
    assert len(s.asked) == 1 and s.asked[0].startswith("b:")
    assert len(s.wrong) == 1 and s.wrong[0].startswith("c:")
    assert len(s.errors) == 1
    assert s.held_out == [1, 2]
    assert s.by_answer == {"SOURCE": [1, 1], "ASK": [0, 1], "APPLICATION": [0, 1], "UNSURE": [0, 1]}
