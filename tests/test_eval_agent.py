"""차곡이 시험 자료(samples/agent)와 채점 — LLM 없이."""

import asyncio
import json

import pytest

from scripts.eval_agent import (
    EXPERIENCES,
    EvalMcp,
    Score,
    expected_paste,
    format_score,
    load_requests,
    load_tools,
    score_dir,
    score_one,
)

REQUESTS = load_requests()
TOOLS = load_tools()
EXPERIENCE_LIST = json.loads(EXPERIENCES.read_text(encoding="utf-8"))
IDS = {e["id"] for e in EXPERIENCE_LIST}


def test_requests_cover_every_shape():
    # 이슈 완료 조건 — 경험 찾기 · 지원 만들어 줘 · 맞는 경험 없음 · 자료 쌓아 줘
    ids = [r["id"] for r in REQUESTS]
    assert len(ids) == len(set(ids))
    assert sum(r["end"] == "done" and bool(r.get("mustFind")) for r in REQUESTS) >= 3
    assert sum(r["end"] == "confirm:create_application" for r in REQUESTS) >= 2
    assert sum(r["id"].startswith("none-") for r in REQUESTS) >= 3
    assert sum(r["end"] == "confirm:add_source" for r in REQUESTS) >= 3


@pytest.mark.parametrize("r", REQUESTS, ids=lambda r: r["id"])
def test_requests_point_at_real_data(r):
    assert set(r.get("mustFind", [])) <= IDS
    if r["end"] == "confirm:add_source":
        assert expected_paste(r)
        assert expected_paste(r) in r["text"]


def test_tools_match_the_contract():
    assert {t.name: t.read_only for t in TOOLS} == {
        "search_experiences": True,
        "list_activities": True,
        "get_activity": True,
        "create_application": False,
        "add_source": False,
    }


def _call(name, arguments):
    return asyncio.run(EvalMcp(TOOLS, EXPERIENCE_LIST).call(name, arguments))


def test_eval_mcp_search_is_any_word():
    out = _call("search_experiences", {"query": "WebSocket 없는낱말"})
    assert out.result["total"] == 1
    assert out.result["experiences"][0]["activity"]["name"].startswith("바꿔요")
    assert _call("search_experiences", {"query": "없는낱말"}).result == {
        "total": 0,
        "experiences": [],
    }


def test_eval_mcp_refuses_writes_and_unknown_activity():
    mcp = EvalMcp(TOOLS, EXPERIENCE_LIST)
    assert asyncio.run(mcp.call("add_source", {"text": "x"})).error["code"] == "CONFIRM_REQUIRED"
    assert mcp.writes == ["add_source"]
    assert _call("get_activity", {"activityId": 999}).error["code"] == "ACTIVITY_NOT_FOUND"
    assert _call("get_activity", {"activityId": 1}).result["experiences"]
    assert len(_call("list_activities", {}).result["activities"]) >= 3


def _record(*events, seconds=3.0, retries=0, writes=()):
    return {"seconds": seconds, "events": list(events), "retries": retries, "writes": list(writes)}


def test_score_right_end_and_found():
    score = Score()
    request = {"id": "r", "end": "done", "mustFind": [101]}
    found = {"type": "tool_result", "result": {"experiences": [{"id": 101}]}}
    score_one(score, request, _record({"type": "tool_call"}, found, {"type": "done"}))
    assert (score.right_end, score.last_rule, score.must_found) == (1, 1, 1)
    assert score.tool_calls == [1]


def test_score_catches_wrong_end_writes_and_error():
    score = Score()
    confirm = {"type": "confirm", "tool": "add_source", "arguments": {"text": "요약"}, "label": "?"}
    request = {"id": "r", "end": "done"}
    score_one(score, request, _record(confirm, writes=["add_source"]))
    assert score.wrong == ["r: confirm:add_source"]
    assert score.writes == 1
    score_one(score, request, _record({"type": "error", "code": "LLM_ERROR", "message": "x"}))
    assert len(score.errors) == 1


def test_score_catches_events_after_last_and_changed_paste():
    score = Score()
    request = {
        "id": "r",
        "end": "confirm:add_source",
        "text": "쌓아 줘: 원문",
        "pasteMark": "쌓아 줘:",
    }
    confirm = {"type": "confirm", "tool": "add_source", "arguments": {"text": "고친 글"}}
    score_one(score, request, _record({"type": "done"}, confirm))
    assert score.last_rule == 0
    assert (score.paste_same, score.paste_total) == (0, 1)


def test_score_team_as_mine_candidate():
    score = Score()
    said = {"type": "message", "text": "내가 AWS 에 배포했어요"}
    score_one(score, {"id": "r", "end": "done", "notMine": True}, _record(said, {"type": "done"}))
    assert len(score.team_as_mine) == 1
    assert "말 안의 팀 일 1" in format_score("m", score)


def test_score_dir_rescores_saved_runs(tmp_path):
    model = tmp_path / "openai-m"
    model.mkdir()
    record = _record({"type": "message", "text": "찾았어요"}, {"type": "done"})
    path = model / "find-websocket-1.json"
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    summary = score_dir(tmp_path, REQUESTS)
    assert "끝 맞음 1/1" in summary
    assert "찾았어요" in summary
