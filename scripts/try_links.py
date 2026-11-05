"""실제 공고 링크로 링크 받기(S-46)를 시험한다 — 사이트별 읽힌 수 · 못 읽은 이유 · 문항을 찾은 수(AI#39).

    uv run python scripts/try_links.py                       # samples/links.txt 를 받기만(LLM 안 부름)
    uv run python scripts/try_links.py --parse --env ../WOOWANGTHON_TEAM_12_AI/.env   # 읽힌 것은 나누기까지

링크 목록 samples/links.txt — 한 줄에 「사이트 | 링크」. 받은 본문은 저장하지 않고 화면에도 찍지 않는다
(결과 · 글자 수 · 찾은 문항 수만). --parse 는 읽힌 링크마다 /parse-application 과 같은 나누기를 부른다
(ROUTE_LLM_MODELS — LLM 호출 수 = 읽힌 링크 수, 규칙 위반이면 링크마다 최대 LLM_MAX_ATTEMPTS 번).
"""

import argparse
import asyncio
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.errors import ApiError, UrlError  # noqa: E402
from app.fetch_page import fetch_text  # noqa: E402

LINKS = ROOT / "samples" / "links.txt"


def load_links(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        site, url = (p.strip() for p in line.split("|", 1))
        out.append((site, url))
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--links", type=Path, default=LINKS)
    parser.add_argument("--parse", action="store_true", help="읽힌 링크를 LLM 으로 나눈다")
    parser.add_argument("--env", type=Path, default=ROOT / ".env")
    args = parser.parse_args()

    split = None
    if args.parse:
        from dotenv import load_dotenv

        from app.config import load_settings
        from app.llm import build_llm
        from app.parse_application import parse_application
        from app.schemas import ParseApplicationRequest

        load_dotenv(args.env)  # 값은 화면에 찍지 않는다
        settings = load_settings()
        llm = build_llm(settings, settings.route_models, settings.llm_sync_call_timeout_seconds)

        async def split(text: str):
            return await parse_application(ParseApplicationRequest(text=text), llm, settings)

    rows = []
    for site, url in load_links(args.links):
        started = time.monotonic()
        try:
            text = await asyncio.to_thread(fetch_text, url)
        except UrlError as e:
            rows.append((site, url, f"{e.code}({e.why})", "", ""))
            print(f"{site:10} 못 읽음 {e.code}({e.why}) — {e.message} — {url}", flush=True)
            continue
        took = time.monotonic() - started
        found = ""
        if split is not None:
            try:
                got = await split(text)
                found = f"문항 {len(got.questions)} · {got.organization} · {got.organizationType} · {got.jobRole}"
            except ApiError as e:
                found = f"나누기 오류 {e.code}"
        rows.append((site, url, "OK", len(text), found))
        print(f"{site:10} 읽음 {len(text):5}자 {took:4.1f}s {found} — {url}", flush=True)

    print("\n| 사이트 | 읽힘 / 시도 | 못 읽은 이유 | 문항 찾음 |")
    print("| --- | --- | --- | --- |")
    by: dict[str, list] = defaultdict(list)
    for row in rows:
        by[row[0]].append(row)
    for site, items in by.items():
        ok = [r for r in items if r[2] == "OK"]
        why = sorted({r[2] for r in items if r[2] != "OK"})
        with_q = sum(1 for r in ok if r[4].startswith("문항 ") and not r[4].startswith("문항 0"))
        q = f"{with_q} / {len(ok)}" if split is not None else "—"
        print(f"| {site} | {len(ok)} / {len(items)} | {' · '.join(why) or '—'} | {q} |")


if __name__ == "__main__":
    asyncio.run(main())
