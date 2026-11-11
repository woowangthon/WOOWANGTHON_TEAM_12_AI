"""공고 기준 경험 찾기 시험 자료(samples/posting-match)와 채점 — LLM 없이."""

import json

import pytest

from app.config import load_settings
from app.posting_match import check_request
from app.schemas import PostingMatchRequest
from scripts.eval_posting_match import DUMMIES, load_data, request_body, score_dir

DUMMY_EXPS, POSTINGS = load_data()
STATUSES = {"SUPPORTED", "PARTIAL", "NOT_FOUND"}
ALL_IDS = {e["id"] for e in DUMMY_EXPS["mixed"]}


def test_two_postings_two_dummies():
    # 이슈 완료 조건 — 백엔드 · 마케팅 공고 각 1건 × 개발 경험만 / 섞임
    assert {p["jobRole"] for p in POSTINGS} == {"BACKEND", "MARKETING"}
    assert set(DUMMIES) == {"dev", "mixed"}
    dev = DUMMY_EXPS["dev"]
    assert dev and {e["id"] for e in dev} < ALL_IDS
    marketing = next(p for p in POSTINGS if p["id"] == "marketing")
    # 개발만 × 마케팅에 NOT_FOUND 만 맞는 줄이 있고, 분야와 상관없이 받치는 줄도 있다
    assert sum(p["dev"] == {"expect": "NOT_FOUND", "ok": []} for p in marketing["lines"]) >= 4
    assert any(p["dev"]["expect"] != "NOT_FOUND" for p in marketing["lines"])


@pytest.mark.parametrize("posting", POSTINGS, ids=lambda p: p["id"])
def test_postings_obey_the_request_rules(posting):
    for dummy in DUMMIES:
        body = request_body(posting, DUMMY_EXPS[dummy])
        check_request(PostingMatchRequest.model_validate(body), load_settings())
    for line in posting["lines"]:
        assert set(line["bad"]) <= ALL_IDS
        for dummy in DUMMIES:
            answer = line[dummy]
            assert answer["expect"] in STATUSES
            assert set(answer["ok"]) <= STATUSES - {answer["expect"]}


def test_score_counts_forced_and_bad(tmp_path):
    marketing = next(p for p in POSTINGS if p["id"] == "marketing")
    lines = [
        {"id": p["id"], "status": p["dev"]["expect"], "experienceIds": []}
        for p in marketing["lines"]
    ]
    for j in lines:
        if j["status"] != "NOT_FOUND":
            j["experienceIds"] = [106]
    lines[0] |= {
        "status": "PARTIAL",
        "experienceIds": [101],
    }  # 2001 SNS 줄에 WebSocket — 억지 · 엉뚱
    res = {
        "lines": lines,
        "clarifications": [
            {"lineId": 2001, "experienceId": 101, "question": "사용자가 30% 늘었죠?"}
        ],
        "model": "x",
    }
    (tmp_path / "marketing-dev.json").write_text(
        json.dumps({"response": res, "seconds": 1.0}), encoding="utf-8"
    )
    s = score_dir(tmp_path, DUMMY_EXPS, POSTINGS)
    assert s.cases == 1
    assert s.exact == len(lines) - 1
    assert s.dev_marketing_forced == 1 and len(s.forced) == 1
    assert len(s.bad_picks) == 1
    assert len(s.made_up_numbers) == 1 and len(s.leading) == 1
