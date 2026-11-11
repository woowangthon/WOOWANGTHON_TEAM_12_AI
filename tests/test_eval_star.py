"""scripts/eval_star.py — 시험 자료가 맞게 짜였나 · 채점이 세는 것(LLM 안 부름)."""

import json

from app.schemas import STAR_SLOTS, StarRequest
from scripts.eval_star import load_data, report, request_body, score_dir


def test_samples_are_valid():
    activities = load_data()
    exps = [e for a in activities for e in a["experiences"]]
    assert len(exps) == 20
    assert len({e["id"] for e in exps}) == 20
    for a in activities:
        StarRequest.model_validate(request_body(a))
    for e in exps:
        keys = {f"C{c['id']}" for c in e["citations"]} | {f"P{s['id']}" for s in e["supplements"]}
        assert set(e["expect"]) == set(STAR_SLOTS)
        for name, expect in e["expect"].items():
            assert expect is None or set(expect) <= keys, (e["id"], name)
        for sentence in e["learned"]:  # 근거에 그대로 있는 문장
            assert any(sentence in c["quote"] for c in e["citations"]), e["id"]
        if e["actor"] != "ME":  # TEAM · UNKNOWN 의 T 는 근거로 못 채운다
            assert not any(k.startswith("C") for k in e["expect"]["T"] or [])


def _slot(name, kind=None, sid=None, text=None, q=None):
    return {"slot": name, "sourceType": kind, "sourceId": sid, "text": text, "question": q}


def test_score_counts(tmp_path):
    activities = [a for a in load_data() if a["id"] == "retro"]
    slots = {
        e["id"]: [_slot(n, q="그 뒤 무엇이 달라졌나요?" if n == "R" else None) for n in STAR_SLOTS]
        for e in activities[0]["experiences"]
    }
    # 202(TEAM) — R 을 근거의 배운 점 문장으로 · T 를 근거로. 201 은 같은 근거의 결과 사실만(세지 않는다)
    slots[202][3] = _slot("R", "CITATION", 2121, "솔직하게 말하는 문화가 중요하다는 걸")
    slots[201][3] = _slot("R", "CITATION", 2111, "고쳤다.")
    slots[202][1] = _slot("T", "CITATION", 2121, "우리 팀은 매주 금요일 KPT 로 회고를 했다")
    response = {"experiences": [{"id": i, "slots": s} for i, s in slots.items()], "model": "m"}
    run = tmp_path / "m" / "tuned" / "run-1"
    run.mkdir(parents=True)
    record = {"response": response, "dropped": ["경험 201 S: x"], "attempt": 2, "seconds": 1.0}
    (run / "retro.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    s = score_dir(run, activities)
    assert s.filled == 3 and s.verbatim == 3
    assert len(s.made_up_learning) == 1 and len(s.team_t) == 1
    assert len(s.dropped) == 1 and s.retried == 1
    assert len(s.questions) == 3 and s.fits == 3
    assert "| m · tuned |" in report(tmp_path, activities)
