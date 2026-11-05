"""꺼내기 시험 자료(samples/ask)와 채점 — LLM 없이."""

import json

import pytest

from app.ask import check_request
from app.config import load_settings
from app.schemas import AskRequest
from scripts.eval_ask import claims_mine, load_data, score_dir

EXPERIENCES, QUERIES = load_data(None)
IDS = {e["id"] for e in EXPERIENCES}
ACTOR = {e["id"]: e["actor"] for e in EXPERIENCES}


def test_enough_queries_of_every_shape():
    # 이슈 완료 조건 — 10개 이상 · 맞는 경험이 없는 질문 포함 · 계약 예시
    assert len(QUERIES) >= 10
    assert sum(q["kind"] == "none" for q in QUERIES) >= 3
    assert sum(bool(q.get("allowZero")) for q in QUERIES) >= 1  # 팀원이 한 일만 있는 질문
    assert any(ACTOR[i] != "ME" for q in QUERIES for i in q["good"])  # 팀 · 모름 경험이 정답인 질문
    contract = next(q for q in QUERIES if q["id"] == "contract-example")
    assert contract["query"] == "동아리에서 의견 모아 본 적 있었나?"
    ids = [q["id"] for q in QUERIES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("q", QUERIES, ids=lambda q: q["id"])
def test_queries_obey_the_request_rules(q):
    check_request(AskRequest(query=q["query"], experiences=EXPERIENCES), load_settings())
    assert q["kind"] in ("match", "none")
    marked = q["good"] + q["ok"] + q["bad"]
    assert set(marked) <= IDS
    assert len(marked) == len(set(marked))
    if q["kind"] == "none":
        assert not q["good"] and not q["ok"]
    else:
        assert q["good"] or q.get("allowZero")


@pytest.mark.parametrize(
    ("reason", "mine"),
    [
        ("팀이 매주 회고로 의견을 모은 자리다. 회고를 내가 이끌었는지는 자료에 없다", False),
        ("팀이 매주 회고를 했다", False),
        ("팀원이 AWS 에 배포했다. 사용자가 직접 배포한 것은 아니다", False),
        ("내가 매주 회고를 이끌며 의견을 모았다", True),
        ("서버 장애를 직접 고친 경험이다. 누가 고쳤는지는 확인이 필요하다", True),
    ],
)
def test_claims_mine(reason, mine):
    assert claims_mine(reason) is mine


def _write(path, qid, recs=None, error=None):
    record = {"seconds": 3.0}
    if error:
        record["error"] = error
    else:
        record["response"] = {"recommendations": recs, "model": "m"}
    (path / f"{qid}.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


def _rec(exp_id, reason, evidence=()):
    return {"experienceId": exp_id, "reason": reason, "evidence": list(evidence)}


def test_score_dir(tmp_path):
    queries = [q for q in QUERIES if q["id"] in ("contract-example", "deploy", "ml-model")]
    queries += [q for q in QUERIES if q["id"] in ("excel-data", "overseas-volunteer")]
    _write(tmp_path, "contract-example", [_rec(102, "내가 회고를 이끌었다")])
    _write(tmp_path, "deploy", [])
    _write(tmp_path, "ml-model", [_rec(112, "팀이 추천 로직을 만들었다")])
    _write(
        tmp_path,
        "excel-data",
        [_rec(121, "엑셀로 30개 품목을 정리했다", [{"type": "SUPPLEMENT", "id": 321}])],
    )
    _write(tmp_path, "overseas-volunteer", error="LLM_INVALID_OUTPUT: x")
    s = score_dir(tmp_path, EXPERIENCES, queries)
    assert s.total == 5
    assert (s.first_right, s.match_total) == (3, 3)  # 102 good · deploy 0개 허용 · 121 good
    assert (s.none_zero, s.none_total) == (0, 1)
    assert len(s.forced) == 1
    assert len(s.team_as_mine) == 1
    assert len(s.made_up_numbers) == 1  # 30
    assert (s.supplement_used, s.supplement_recs) == (1, 1)
    assert len(s.errors) == 1
