"""채점 스크립트의 --runs — 회차 폴더 찾기 · 보고 표 합치기(AI#111). LLM 을 부르지 않는다."""

import json

from scripts.eval_extract import load_answers
from scripts.eval_extract import report as extract_report
from scripts.eval_runs import merge_cell, merge_runs, run_dirs


def test_merge_cell_numbers_per_run():
    assert merge_cell(["50/51", "49/51"]) == "50 · 49/51"
    assert merge_cell(["3", "3"]) == "3"
    assert merge_cell(["12초", "15초"]) == "12 · 15초"
    assert merge_cell(["0 (0)", "1 (1)"]) == "[0 · 1] ([0 · 1])"
    assert merge_cell(["0/18", "0/20"]) == "0/[18 · 20]"


def test_merge_cell_wraps_when_cell_has_dot():
    assert merge_cell(["1 · 2 · 3", "1 · 4 · 3"]) == "1 · [2 · 4] · 3"


def test_merge_cell_different_shapes():
    assert merge_cell(["-", "1/2"]) == "- · 1/2"


def test_merge_runs_groups_by_name():
    rows = [
        "| m · tuned · run-1 | 50/51 | 0 |",
        "| m · tuned · run-2 | 49/51 | 0 |",
        "| m · generic · run-1 | 40/51 | 2 |",
    ]
    assert merge_runs(rows) == [
        "| m · tuned | 50 · 49/51 | 0 |",
        "| m · generic | 40/51 | 2 |",
    ]


def test_merge_runs_keeps_single_run_rows():
    assert merge_runs(["| m · tuned | 1/2 |"]) == ["| m · tuned | 1/2 |"]


def test_run_dirs_old_and_new_layout(tmp_path):
    flat = tmp_path / "flat"
    flat.mkdir()
    assert run_dirs(flat) == [("", flat)]
    for n in (10, 2, 1):
        (tmp_path / "runs" / f"run-{n}").mkdir(parents=True)
    got = run_dirs(tmp_path / "runs")
    assert [s for s, _ in got] == [" · run-1", " · run-2", " · run-10"]


def test_extract_report_merges_runs(tmp_path):
    samples = load_answers(None)
    stem = next(iter(samples))
    for n, secs in ((1, 3.0), (2, 5.0)):
        folder = tmp_path / "openai_x" / "tuned" / f"run-{n}"
        folder.mkdir(parents=True)
        record = {"error": "LLM_INVALID_OUTPUT: x", "seconds": secs}
        (folder / f"{stem}.json").write_text(json.dumps(record), encoding="utf-8")
    text = extract_report(tmp_path, samples)
    rows = [line for line in text.splitlines() if line.startswith("| openai_x")]
    assert len(rows) == 1
    assert rows[0].startswith("| openai_x · tuned | ")
    assert rows[0].endswith("| 1/1 | 3 · 5초 |")
    assert "### openai_x · tuned · run-1" in text
    assert "### openai_x · tuned · run-2" in text
