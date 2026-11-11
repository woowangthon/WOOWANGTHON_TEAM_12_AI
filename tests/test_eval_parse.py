"""지원 나누기 시험 자료(samples/parse)와 채점 — LLM 없이."""

import json

import pytest

from app.parse_application import check_written
from app.schemas import (
    JOB_ROLES,
    ORGANIZATION_TYPES,
    LLMParseApplicationOutput,
    ParseApplicationRequest,
)
from scripts.eval_parse import load_inputs, same_question, same_questions, score_dir

INPUTS = load_inputs(None)


def test_enough_inputs_of_every_shape():
    # 이슈 완료 조건 — 시험 글 10개 이상(문항 하나만 · 여러 문항 공고 · 동아리 · 채용 · 문항이 없는 공고)
    assert len(INPUTS) >= 10
    assert sum(i["questions"] == [] for i in INPUTS) >= 3
    assert sum(len(i["questions"]) == 1 for i in INPUTS) >= 3
    assert sum(len(i["questions"]) >= 2 for i in INPUTS) >= 3
    for kind in ("CLUB", "JOB"):
        assert any(kind in i["organizationType"] for i in INPUTS), kind
    ids = [i["id"] for i in INPUTS]
    assert len(ids) == len(set(ids))


def test_postings_cover_questionless_pages_and_menu_noise():
    postings = [item for item in INPUTS if "postingLines" in item]
    assert sum(not item["questions"] for item in postings) >= 5
    assert {"직행", "사람인", "원티드", "회사 채용 페이지"} <= {
        item["sourceShape"] for item in postings
    }
    assert any("메뉴 잡음" in item["sourceShape"] for item in postings)
    for item in postings:
        assert item["noise"]
        assert all(noise in item["text"] for noise in item["noise"])
        assert not any(line["text"] in item["noise"] for line in item["postingLines"])


@pytest.mark.parametrize("item", INPUTS, ids=lambda i: i["id"])
def test_answer_obeys_the_response_rules(item):
    # 정답도 응답 규칙을 지켜야 한다 — 글에 적혀 있고 · 순서대로 · 목록 안의 값
    ParseApplicationRequest(text=item["text"])
    assert item["why"].strip()
    assert set(item["organizationType"]) <= {*ORGANIZATION_TYPES, None}
    assert set(item["jobRole"]) <= {*JOB_ROLES, None}
    for organization in item["organization"]:
        out = LLMParseApplicationOutput(
            organization=organization,
            organizationType=item["organizationType"][0],
            jobRole=item["jobRole"][0],
            questions=item["questions"],
            postingLines=item.get("postingLines", []),
        )
        check_written(out, item["text"])


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("지원 동기를 적어 주세요.", "1. 지원 동기를 적어 주세요.", True),
        ("지원 동기를 적어 주세요.", "지원 동기를 적어 주세요. (500자)", True),
        ("지원 동기를 적어 주세요.", "지원 동기를 적어 주세요. (공백 포함 700자 이내)", True),
        ("지원 동기를\n적어 주세요.", "지원 동기를 적어 주세요.", True),
        ("지원 동기를 적어 주세요.", "지원 동기를 써 주세요.", False),
        ("지원 동기를 적어 주세요. (선택)", "지원 동기를 적어 주세요.", False),
    ],
)
def test_same_question(a, b, same):
    assert same_question(a, b) is same


def test_same_questions_keeps_order():
    assert same_questions(["가", "나"], ["가", "나"])
    assert not same_questions(["나", "가"], ["가", "나"])
    assert not same_questions(["가"], ["가", "나"])


def test_score_dir(tmp_path):
    base = {"organization": [None], "organizationType": [None], "jobRole": [None]}
    items = [
        {**base, "id": "right", "questions": ["가"]},
        {**base, "id": "invented", "questions": []},
        {**base, "id": "field", "questions": [], "jobRole": ["BACKEND"]},
        {**base, "id": "error", "questions": []},
        {**base, "id": "missing", "questions": []},
    ]
    nulls = {"organization": None, "organizationType": None, "jobRole": None}
    records = {
        "right": {"response": {**nulls, "questions": ["1. 가"]}, "seconds": 1.0},
        "invented": {"response": {**nulls, "questions": ["자격 요건"]}, "seconds": 2.0},
        "field": {"response": {**nulls, "questions": []}, "seconds": 3.0},
        "error": {"error": "LLM_INVALID_OUTPUT: 위반", "seconds": 4.0},
    }
    for name, record in records.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")

    s = score_dir(tmp_path, items)

    assert s.total == 4
    assert s.questions == 2
    assert len(s.invented) == 1
    assert s.right["jobRole"] == 2
    assert s.right["organization"] == 3
    assert len(s.errors) == 1
    assert s.seconds == [1.0, 2.0, 3.0, 4.0]


def test_posting_metrics_count_missing_errors_and_unknown_noise(tmp_path):
    line = {"section": "REQUIRED", "text": "Java 사용 경험", "ksa": "SKILL"}
    base = {"organization": [None], "organizationType": [None], "jobRole": [None], "questions": []}
    items = [{**base, "id": name, "postingLines": [line]} for name in ("got", "old", "error")]
    response = {"organization": None, "organizationType": None, "jobRole": None, "questions": []}
    records = {
        "got": {
            "response": {
                **response,
                "postingLines": [
                    {**line, "text": "- Java  사용 경험", "section": "STACK"},
                    {**line, "text": "Java 사용 경험"},  # 중복도 재현율을 올리지 못한다
                    {**line, "text": "인재풀 등록"},
                ],
            }
        },
        "old": {"response": response},  # 옛 응답도 분모에는 남는다
        "error": {"error": "LLM_INVALID_OUTPUT"},
    }
    for name, record in records.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")
    score = score_dir(tmp_path, items)
    assert score.posting_expected == 3
    assert score.posting_found == 1
    assert score.posting_returned == 3
    assert score.posting_noise == 2
    assert score.posting_section == 0
    assert score.posting_ksa == 1
    assert len(score.errors) == 1
