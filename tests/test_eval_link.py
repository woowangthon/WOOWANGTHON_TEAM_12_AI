"""이어짐 제안 시험 자료(samples/link)와 채점 — LLM 없이."""

import json

import pytest

from app.config import load_settings
from app.link_suggest import check_request
from app.schemas import LinkSuggestionsRequest
from scripts.eval_link import invented_quotes, is_causal, leans_on_both, load_inputs, score_dir

INPUTS = load_inputs(None)


def test_enough_inputs_of_every_shape():
    # 이슈 완료 조건 — 계약 예시 포함 · 이어짐이 없는 후보만 주는 자료
    assert len(INPUTS) >= 10
    assert sum(not i["linked"] and not i["maybe"] for i in INPUTS) >= 3
    assert sum(len(i["linked"]) >= 2 for i in INPUTS) >= 1
    assert any(len(i["request"]["candidates"]) == 20 for i in INPUTS)
    contract = next(i for i in INPUTS if i["id"] == "contract-example")
    assert contract["request"]["from"]["name"] == "알고리즘 스터디"
    assert contract["linked"] == [7]
    ids = [i["id"] for i in INPUTS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("item", INPUTS, ids=lambda i: i["id"])
def test_inputs_obey_the_request_rules(item):
    check_request(LinkSuggestionsRequest.model_validate(item["request"]), load_settings())
    assert item["why"].strip()
    ids = {c["id"] for c in item["request"]["candidates"]}
    assert {*item["linked"], *item["maybe"]} <= ids
    assert not set(item["linked"]) & set(item["maybe"])


@pytest.mark.parametrize(
    ("reason", "causal"),
    [
        ("「직접 서비스를 만들어 보고 싶어졌다」와 「바꿔요」가 닿아 보여요", False),
        ("계기가 이 활동으로 이어져 보여요", False),
        ("이 마음이 이어졌을 수도 있어 보여요", False),
        ("계기가 앱 개발로 이어진 것으로 보인다", False),
        ("「발표가 무서웠기 때문에」와 「토론 동아리」가 닿아 보여요", False),
        ("서비스를 만들고 싶었기 때문에 바꿔요를 시작했어요", True),
        ("이 마음이 바꿔요로 이어졌어요", True),
        ("스터디 덕분에 앱을 만들었어요", True),
        ("서비스를 만들고 싶어서 시작했어요", False),  # 정규식 밖 — 사람이 읽고 센다
        ("이 경험이 바꿔요의 계기가 됐어요", True),
    ],
)
def test_is_causal(reason, causal):
    assert is_causal(reason) is causal


ORIGIN = {
    "name": "알고리즘 스터디",
    "note": "직접 서비스를 만들어 보고 싶어졌다",
    "experiences": [{"id": 1, "content": "매주 풀 문제를 고른다"}],
}
CANDIDATE = {"name": "바꿔요", "note": None, "experiences": [{"id": 2, "content": "앱을 만들었다"}]}


def test_date_only_reason_does_not_lean_on_anything():
    assert not leans_on_both("바로 다음에 시작한 활동이에요", ORIGIN, CANDIDATE)
    assert leans_on_both("「직접 서비스를」과 「앱을 만들었다」가 닿아 보여요", ORIGIN, CANDIDATE)


def test_invented_quotes():
    assert invented_quotes("「앱을 만들었다」가 닿아요", ORIGIN, [CANDIDATE]) == []
    assert invented_quotes("「앱을  만들었다.」가 닿아요", ORIGIN, [CANDIDATE]) == []
    assert invented_quotes("「창업을 꿈꿨다」가 닿아요", ORIGIN, [CANDIDATE]) == ["창업을 꿈꿨다"]


def _write(path, name, record):
    (path / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


def test_score_dir(tmp_path):
    body = {"from": ORIGIN, "candidates": [{**CANDIDATE, "id": 7}, {**CANDIDATE, "id": 9}]}
    items = [
        {"id": "a", "request": body, "linked": [7], "maybe": []},
        {"id": "b", "request": body, "linked": [], "maybe": []},
        {"id": "c", "request": body, "linked": [7], "maybe": []},
    ]
    good = "「직접 서비스를」과 「앱을 만들었다」가 닿아 보여요"
    _write(
        tmp_path,
        "a",
        {
            "response": {
                "suggestions": [
                    {"toActivityId": 7, "reason": good},
                    {"toActivityId": 9, "reason": "바로 다음에 시작했기 때문이에요"},
                ]
            },
            "seconds": 1.0,
        },
    )
    _write(
        tmp_path,
        "b",
        {"response": {"suggestions": [{"toActivityId": 7, "reason": good}]}, "seconds": 2.0},
    )
    _write(tmp_path, "c", {"error": "LLM_INVALID_OUTPUT: x", "seconds": 3.0})

    s = score_dir(tmp_path, items)
    assert (s.total, s.linked, s.found, s.suggested) == (3, 2, 1, 3)
    assert len(s.forced) == 2
    assert len(s.zero_failed) == 1
    assert len(s.causal) == 1
    assert len(s.unsupported) == 1
    assert len(s.errors) == 1
