"""채점 스크립트의 회차(--runs) — 회차별 저장 폴더를 찾고, 보고 표의 회차별 줄을 한 줄로 합친다(AI#111).

같은 입력에도 LLM 답이 흔들리니 전 · 후를 같은 회차로 견준다. 합친 칸은 회차 값을 차례로 `a · b` 로 적고
(ETC docs/ai-scope.md 의 표 모양 — 「50 · 49/51」), 값이 같은 자리는 한 번만 적는다.
다른 자리가 첫 숫자 하나가 아니거나 칸에 이미 「·」가 있으면 회차 값을 `[a · b]` 로 묶어 어느 자리인지 헷갈리지 않게 한다.
"""

import re
from pathlib import Path

RUN_DIR = re.compile(r"run-(\d+)")
RUN_SUFFIX = re.compile(r" · run-\d+$")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def run_path(base: Path, n: int) -> Path:
    return base / f"run-{n}"


def run_dirs(prompt_dir: Path) -> list[tuple[str, Path]]:
    """(이름에 붙일 꼬리, 폴더) — 회차 폴더(run-N)가 있으면 그것들, 없으면(회차 전 저장본) 그 폴더 하나."""
    runs = [p for p in prompt_dir.iterdir() if p.is_dir() and RUN_DIR.fullmatch(p.name)]
    runs.sort(key=lambda p: int(RUN_DIR.fullmatch(p.name).group(1)))
    return [(f" · {p.name}", p) for p in runs] or [("", prompt_dir)]


def merge_cell(values: list[str]) -> str:
    if len(set(values)) == 1:
        return values[0]
    shapes = {NUMBER.sub("#", v) for v in values}
    if len(shapes) > 1:
        return " · ".join(values)
    shape = shapes.pop()
    columns = list(zip(*(NUMBER.findall(v) for v in values), strict=True))
    differ = [len(set(c)) > 1 for c in columns]
    wrap = "·" in shape or differ != [True] + [False] * (len(differ) - 1)
    parts = shape.split("#")
    out = parts[0]
    for column, d, tail in zip(columns, differ, parts[1:], strict=True):
        if d:
            joined = " · ".join(column)
            out += f"[{joined}]" if wrap else joined
        else:
            out += column[0]
        out += tail
    return out


def merge_runs(rows: list[str]) -> list[str]:
    """`| 이름 · run-N | … |` 줄들을 이름(회차 꼬리 뺀 것)마다 한 줄로. 처음 나온 순서를 지킨다."""
    groups: dict[str, list[list[str]]] = {}
    for row in rows:
        cells = row.split(" | ")
        key = RUN_SUFFIX.sub("", cells[0])
        groups.setdefault(key, []).append([key, *cells[1:]])
    merged = []
    for items in groups.values():
        if len({len(c) for c in items}) > 1:  # 칸 수가 다르면 합치지 않는다
            merged += [" | ".join(c) for c in items]
            continue
        merged.append(" | ".join(merge_cell(list(col)) for col in zip(*items, strict=True)))
    return merged
