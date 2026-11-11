"""scripts/eval_capture.py 채점 — 저장한 응답만으로(LLM 없음). 자료 정답이 그림 · 스크립트와 맞는지도 본다."""

import json

from scripts.eval_capture import DATA, load_cases, report, request_body, score_dir

CASE = {
    "id": "c1",
    "files": [],
    "organization": ["OO 동아리"],
    "organizationType": ["CLUB"],
    "jobRole": [None],
    "questions": ["지원 동기를 적어 주세요. (300자)", "협업 경험을 적어 주세요."],
    "postingLines": [
        {"section": "REQUIRED", "text": "주 1회 모임 참여 가능자"},
        {"section": "PREFERRED", "text": "GitHub 사용 경험"},
    ],
    "traps": ["활동비 월 20만 원"],
}


def write(path, case_id, record):
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{case_id}.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


def response(**kw):
    base = {
        "organization": "OO 동아리",
        "organizationType": "CLUB",
        "jobRole": None,
        "questions": [],
        "postingLines": [],
        "readFrom": "IMAGE",
        "model": "m",
    }
    return {**base, **kw}


def test_verbatim_misread_and_invented(tmp_path):
    got = response(
        questions=["1. 지원 동기를 적어 주세요. (300자) *", "협업한 경험을 적어 주세요."],
        postingLines=[
            {"section": "REQUIRED", "text": "· 주 1회 모임 참여 가능자", "ksa": None},
            {"section": "REQUIRED", "text": "GitHub 사용 경험", "ksa": None},
            {"section": "PREFERRED", "text": "활동비 월 20만 원", "ksa": None},
        ],
    )
    write(tmp_path, "c1", {"response": got, "seconds": 3.0, "tokens": [1000, 100]})
    s = score_dir(tmp_path, [CASE])
    assert (s.questions, s.questions_expected) == (1, 2)  # 앞 번호 · 끝 「*」는 같게 본다
    assert (s.lines, s.lines_expected) == (2, 2)
    assert s.section == 1  # GitHub 줄은 칸이 틀렸다
    assert len(s.misread) == 1  # 「협업한」 — 글자 다름
    assert len(s.invented) == 1 and s.traps == 1  # 혜택 줄 — 만들어 냄 · 함정
    assert s.right == {"organization": 1, "organizationType": 1, "jobRole": 1}


def test_errors_stay_in_denominator(tmp_path):
    write(tmp_path, "c1", {"error": "IMAGE_UNREADABLE: …", "seconds": 2.0})
    s = score_dir(tmp_path, [CASE])
    assert (s.total, s.questions, s.questions_expected, len(s.errors)) == (1, 0, 2, 1)


def test_report_has_cost_for_known_model(tmp_path):
    run = tmp_path / "openai_gpt-6-luna__high" / "run-1"
    write(run, "c1", {"response": response(), "seconds": 1.0, "tokens": [1_000_000, 0]})
    text = report(tmp_path, [CASE])
    assert "openai_gpt-6-luna__high · run-1" in text
    assert "$0.1000" in text


def test_samples_match_files():
    cases = load_cases(None)
    assert len(cases) == 8
    kinds = [c["kind"] for c in cases]
    assert (kinds.count("poster"), kinds.count("form"), kinds.count("job")) == (3, 3, 2)
    for case in cases:
        body = request_body(case)
        assert 1 <= len(body["images"]) <= 5
        for f in case["files"]:
            assert (DATA / f).read_bytes().startswith(b"\x89PNG")
        assert case["questions"] or case["postingLines"]
