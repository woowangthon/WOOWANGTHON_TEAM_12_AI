"""파일을 진짜 LLM 으로 추출해 보고 결과를 사람이 읽기 좋게 찍는다(손 확인용).

    uv run python scripts/try_extract.py samples/*.md [--json out_dir]

파일 하나에 LLM 을 1번(규칙 위반이면 최대 LLM_MAX_ATTEMPTS 번) 부른다 — .env 의 OPENAI_API_KEY(또는 GEMINI_API_KEY).
결과는 화면에만 찍는다. 실제 노트를 넣었다면 결과를 커밋하지 않는다.
"""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from app.config import load_settings  # noqa: E402
from app.errors import ApiError  # noqa: E402
from app.extract import extract  # noqa: E402
from app.llm import build_llm  # noqa: E402
from app.schemas import ExtractRequest  # noqa: E402


def show(path: Path, data) -> None:
    print(f"\n=== {path.name} ({data.model}) — 활동 {len(data.activities)}개")
    for act in data.activities:
        p = act.period
        print(f"\n■ {act.name}  [{p.start} ~ {'진행 중' if p.ongoing else p.end}]")
        for exp in act.experiences:
            print(f"  - ({exp.actor:<7} · {', '.join(exp.competencies)}) {exp.content}")
            for c in exp.citations:
                print(f"      └ 「{c.quote}」")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, help="응답 JSON 을 이 폴더에 저장")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")  # 값은 화면에 찍지 않는다
    settings = load_settings()
    llm = build_llm(settings)
    for path in args.files:
        req = ExtractRequest(sourceName=path.name, text=path.read_text(encoding="utf-8"))
        try:
            data = await extract(req, llm, settings)
        except ApiError as e:
            print(f"\n=== {path.name} — 오류 {e.code}: {e.message}")
            continue
        show(path, data)
        if args.json:
            args.json.mkdir(parents=True, exist_ok=True)
            out = args.json / f"{path.stem}.json"
            out.write_text(data.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
