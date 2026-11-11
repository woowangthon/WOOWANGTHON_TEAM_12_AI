"""공고 기준 경험 찾기(/posting-match) 규모 시험 — 경험 수 · 줄 수를 키워 **시간만** 잰다(판정은 채점하지 않는다).

    uv run python scripts/scale_posting_match.py --cases 23x17,200x40 --runs 5
    uv run python scripts/scale_posting_match.py --cases 200x40 --runs 3 --model openai:gpt-6-luna

경험은 samples/posting-match 의 「섞임」 23개를 ID · 활동 이름(「(2기)」…)만 바꿔 복제하고, 줄은 시험 공고 17줄 +
지어낸 일반 공고 줄 23개로 40줄까지 채운다. 같은 글의 경험이 겹치니 판정 품질은 이 스크립트로 보지 않는다.
모델 한 번 제한(--call-timeout)을 운영(45초)보다 길게 풀어 「느린 것」과 「응답이 안 오는 것」을 가른다.
다시 부르기 · 폴백 없이 고른 모델 하나만 부른다. LLM 호출 수 = 경우 수 × runs. 키는 --env 에서 읽고 찍지 않는다.
2026-11-11 결과는 ETC docs/ai-scope.md 「공고 기준 경험 찾기 채점」 아래 「규모」.
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_posting_match import load_data  # noqa: E402

# 지어낸 일반 공고 줄 — 40줄 상한(계약)을 채우는 데만 쓴다
EXTRA_LINES = [
    ("DUTY", "신규 서비스 기능 기획 및 요구사항 정리", "SKILL"),
    ("DUTY", "사내 운영 도구 개발 및 유지보수", "SKILL"),
    ("DUTY", "데이터 파이프라인 구축 및 운영", "SKILL"),
    ("DUTY", "고객 문의 대응 및 VOC 분석", "SKILL"),
    ("REQUIRED", "Python 을 활용한 데이터 처리 경험", "SKILL"),
    ("REQUIRED", "REST API 설계 원칙에 대한 이해", "KNOWLEDGE"),
    ("REQUIRED", "문제를 끝까지 파고드는 태도", "ATTITUDE"),
    ("REQUIRED", "문서로 생각을 정리해 공유하는 습관", "ATTITUDE"),
    ("REQUIRED", "Docker 등 컨테이너 환경 이해", "KNOWLEDGE"),
    ("REQUIRED", "Git 을 이용한 협업 경험", "SKILL"),
    ("PREFERRED", "Kafka 등 메시지 큐 사용 경험", "SKILL"),
    ("PREFERRED", "오픈소스 기여 경험", "SKILL"),
    ("PREFERRED", "스타트업 근무 경험", None),
    ("PREFERRED", "동아리 · 학회 운영 경험", "ATTITUDE"),
    ("PREFERRED", "발표 · 세미나 진행 경험", "SKILL"),
    ("PREFERRED", "엑셀 · 스프레드시트로 데이터를 정리한 경험", "SKILL"),
    ("PREFERRED", "모바일 앱 출시 경험", "SKILL"),
    ("PREFERRED", "알고리즘 · 자료구조 학습 경험", "KNOWLEDGE"),
    ("STACK", "Redis", "SKILL"),
    ("STACK", "Kotlin", "SKILL"),
    ("STACK", "MySQL", "SKILL"),
    ("STACK", "Notion · Slack", None),
    ("STACK", "Figma", "SKILL"),
]


def experiences(n: int, base: list[dict]) -> list[dict]:
    """base 를 차례로 복제해 n 개 — 복제마다 경험 · 근거 · 보완 ID 와 활동을 바꾼다."""
    out: list[dict] = []
    k = 0
    while len(out) < n:
        for e in base[: n - len(out)]:
            c = json.loads(json.dumps(e))
            c["id"] += 10000 * k
            c["activityId"] += 100 * k
            if k:
                c["activityName"] = f"{c['activityName']} ({k + 1}기)"
            for x in c["citations"] + c["supplements"]:
                x["id"] += 10000 * k
            out.append(c)
        k += 1
    return out


def lines(n: int, postings: list[dict]) -> list[dict]:
    base = [
        {k: p[k] for k in ("id", "section", "text", "ksa")} for x in postings for p in x["lines"]
    ]
    extra = [
        {"id": 3000 + i, "section": s, "text": t, "ksa": k}
        for i, (s, t, k) in enumerate(EXTRA_LINES)
    ]
    return (base + extra)[:n]


async def run(cases: list[tuple[int, int]], runs: int, model: str, call_timeout: float) -> None:
    from app.config import load_settings, parse_models
    from app.errors import ApiError
    from app.llm import build_llm
    from app.posting_match import posting_match
    from app.schemas import PostingMatchRequest

    dummies, postings = load_data()
    settings = replace(
        load_settings(),
        match_models=parse_models(model),
        llm_timeout_seconds=call_timeout + 5,
        llm_call_timeout_seconds=call_timeout,
        max_attempts=1,
    )
    llm = build_llm(settings, settings.match_models)

    async def one(ne: int, nl: int, r: int) -> str:
        body = {
            "jobRole": "BACKEND",
            "lines": lines(nl, postings),
            "experiences": experiences(ne, dummies["mixed"]),
        }
        req = PostingMatchRequest.model_validate(body)
        started = time.monotonic()
        try:
            res = await posting_match(req, llm, settings)
            got = "".join(j.status[0] for j in res.lines) + f" · 되묻기 {len(res.clarifications)}"
        except ApiError as e:
            got = f"{e.code}: {e.message}"
        return f"{model} · 경험 {ne} × 줄 {nl} · run-{r + 1}: {time.monotonic() - started:.1f}초 — {got}"

    jobs = [one(ne, nl, r) for ne, nl in cases for r in range(runs)]
    for line in await asyncio.gather(*jobs):
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="23x17,200x40", help="쉼표로 「경험수x줄수」(줄 40까지)")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--model", default="openai:gpt-6.1-terra", help="「제공사:모델」 하나")
    parser.add_argument("--call-timeout", type=float, default=120, help="모델 한 번 제한(초)")
    parser.add_argument("--env", type=Path, default=ROOT / ".env", help="키를 읽을 .env")
    args = parser.parse_args()
    cases = [tuple(int(x) for x in c.split("x")) for c in args.cases.split(",")]
    if any(nl > 40 or nl < 1 for _, nl in cases):
        parser.error("줄은 1 ~ 40(계약)")

    from dotenv import load_dotenv

    load_dotenv(args.env)  # 값은 화면에 찍지 않는다
    print(f"LLM 호출 {len(cases) * args.runs}회")
    asyncio.run(run(cases, args.runs, args.model, args.call_timeout))


if __name__ == "__main__":
    main()
