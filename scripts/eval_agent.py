"""시험 부탁으로 차곡이 도구 루프(/agent)를 채점한다 — 프롬프트 · 모델을 바꿀 때 같은 자료 · 같은 숫자로 견준다(AI#41).

    uv run python scripts/eval_agent.py                                  # AGENT_LLM_MODELS 첫 모델로
    uv run python scripts/eval_agent.py --runs 2 --models openai:gpt-6.1-terra,openai:gpt-6-luna
    uv run python scripts/eval_agent.py --rescore eval-runs/agent-<시각>   # 저장한 이벤트만 다시 채점(LLM 안 부름)

LLM 호출 수 = 부탁 수 × 모델 수 × runs × 부탁마다 차례 수(도구 하나에 한 번 + 마지막 말).
이벤트는 --out 폴더(기본 eval-runs/agent-<시각>, 커밋하지 않음)에 저장한다.
모델 폴백 없이 고른 모델만 부른다. 키는 --env 파일(기본 레포의 .env)에서 읽고 화면에 찍지 않는다.

MOCK(agent-eval-mcp): 도구는 실제 BE 가 아니라 이 파일의 EvalMcp 가 답한다 — 채점에만 쓴다. 도구 정의는 BE 에서 받은 것
(samples/agent/tools.json), 경험은 매칭과 같은 지어낸 경험(samples/match/experiences.json), 검색은 계약 「낱말 하나라도」를 따른다.
앱 코드는 이 파일을 부르지 않는다. 실제 BE 와 맞물리는지는 PR 에 적은 로컬 연동으로 따로 확인했다.

자료: samples/agent/requests.json(부탁과 손으로 적은 정답 — 칸의 뜻은 samples/agent/README.md).

숫자의 뜻:
- 끝 맞음: 마지막 이벤트 · 쓰기 도구가 정답(done · confirm 도구)과 같은 일 수
- 마지막 줄 규칙: 마지막 줄이 done · confirm · error 이고 그 뒤에 줄이 없는 일 수
- 쓰기를 MCP 로 부름: 0 이어야 한다 — 코드가 막는다
- 찾음: 찾아야 할 경험이 도구 결과에 나온 일 수 / 그런 일 수
- 원문 그대로: add_source text 가 붙여 넣은 글과 같은 일 수 / add_source 정답인 일 수
- 말 안의 팀 일: 내 일로 말하면 안 되는 부탁(notMine)의 message 에 「내가 · 직접 · 주도 …」가 부정 없이 있는 수 — 후보다
- 다시 부름: 코드 검사에 걸려 다시 부른 수(모델이 처음 낸 답이 규칙을 어긴 수)
- 오류: error 로 끝난 일 수
- 시간 · 도구 호출: 일 하나의 중앙값 · 최대

말의 내용은 message 전문을 summary.md 에 남기니 사람이 읽고 센다.
"""

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent import run_agent  # noqa: E402
from app.mcp_client import McpOutcome, McpToolInfo  # noqa: E402
from app.schemas import AgentRequest  # noqa: E402
from scripts.eval_ask import claims_mine  # noqa: E402

DATA = ROOT / "samples" / "agent"
EXPERIENCES = ROOT / "samples" / "match" / "experiences.json"
LAST_TYPES = ("done", "confirm", "error")


def load_requests(only: list[str] | None = None) -> list[dict]:
    requests = json.loads((DATA / "requests.json").read_text(encoding="utf-8"))
    return [r for r in requests if not only or r["id"] in only]


def load_tools() -> list[McpToolInfo]:
    tools = json.loads((DATA / "tools.json").read_text(encoding="utf-8"))
    return [
        McpToolInfo(t["name"], t["description"], t["inputSchema"], t["readOnly"]) for t in tools
    ]


class EvalMcp:
    """MOCK(agent-eval-mcp): 계약 mcp-api 대로 답하는 가짜 MCP — 채점 전용."""

    def __init__(self, tools: list[McpToolInfo], experiences: list[dict], instructions: str = ""):
        self.tools = tools
        self.instructions = instructions
        self.writes: list[str] = []
        names = list(dict.fromkeys(e["activityName"] for e in experiences))
        self.activity_ids = {name: i + 1 for i, name in enumerate(names)}
        self.experiences = [self._shape(e) for e in experiences]

    def _shape(self, exp: dict) -> dict:
        shaped = {k: v for k, v in exp.items() if k not in ("activityName", "period")}
        shaped["activity"] = {
            "id": self.activity_ids[exp["activityName"]],
            "name": exp["activityName"],
            "period": exp.get("period"),
        }
        return shaped

    def _search(self, arguments: dict) -> dict:
        words = (arguments.get("query") or "").casefold().split()

        def hits(exp: dict) -> int:
            text = json.dumps(exp, ensure_ascii=False).casefold()
            return sum(word in text for word in words)

        found = [e for e in self.experiences if not words or hits(e)]
        found.sort(key=lambda e: -hits(e))
        wanted = arguments.get("competencies")
        if wanted:
            found = [e for e in found if set(wanted) & set(e.get("competencies", []))]
        return {"total": len(found), "experiences": found[: arguments.get("limit", 20)]}

    async def call(self, name: str, arguments: dict[str, Any]) -> McpOutcome:
        tool = next((t for t in self.tools if t.name == name), None)
        if tool is None or not tool.read_only:
            self.writes.append(name)
            error = {"code": "CONFIRM_REQUIRED", "message": "쓰기는 사용자 확인 뒤"}
            return McpOutcome(ok=False, error=error)
        if name == "search_experiences":
            return McpOutcome(ok=True, result=self._search(arguments))
        if name == "list_activities":
            activities = [
                {
                    "id": i,
                    "name": n,
                    "experienceCount": sum(e["activity"]["id"] == i for e in self.experiences),
                }
                for n, i in self.activity_ids.items()
            ]
            return McpOutcome(ok=True, result={"activities": activities})
        wanted = arguments.get("activityId")
        experiences = [e for e in self.experiences if e["activity"]["id"] == wanted]
        if not experiences:
            error = {"code": "ACTIVITY_NOT_FOUND", "message": f"활동 {wanted} 이 없습니다"}
            return McpOutcome(ok=False, error=error)
        name_ = experiences[0]["activity"]["name"]
        return McpOutcome(ok=True, result={"id": wanted, "name": name_, "experiences": experiences})


def expected_paste(request: dict) -> str | None:
    """add_source 정답 글 — 부탁에서 pasteMark 뒤 전부."""
    mark = request.get("pasteMark")
    return request["text"].split(mark, 1)[1].strip() if mark else None


@dataclass
class Score:
    total: int = 0
    right_end: int = 0
    last_rule: int = 0
    writes: int = 0
    must_total: int = 0
    must_found: int = 0
    paste_total: int = 0
    paste_same: int = 0
    team_as_mine: list[str] = field(default_factory=list)
    retries: int = 0
    errors: list[str] = field(default_factory=list)
    wrong: list[str] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    tool_calls: list[int] = field(default_factory=list)


def found_ids(events: list[dict]) -> set[int]:
    ids: set[int] = set()
    for event in events:
        if event["type"] == "tool_result" and isinstance(event.get("result"), dict):
            ids.update(e["id"] for e in event["result"].get("experiences", []))
    return ids


def score_one(score: Score, request: dict, record: dict) -> None:
    rid, events = request["id"], record["events"]
    score.total += 1
    score.seconds.append(record["seconds"])
    score.tool_calls.append(sum(e["type"] == "tool_call" for e in events))
    score.retries += record.get("retries", 0)
    score.writes += len(record.get("writes", []))
    last = events[-1] if events else {"type": ""}
    if last["type"] in LAST_TYPES and all(e["type"] not in LAST_TYPES for e in events[:-1]):
        score.last_rule += 1
    if last["type"] == "error":
        score.errors.append(f"{rid}: {last['code']} {last['message'][:80]}")
    got = last["type"] if last["type"] != "confirm" else f"confirm:{last['tool']}"
    if got in ([request["end"]] if isinstance(request["end"], str) else request["end"]):
        score.right_end += 1
    else:
        score.wrong.append(f"{rid}: {got}")
    if request.get("mustFind"):
        score.must_total += 1
        score.must_found += set(request["mustFind"]) <= found_ids(events)
    paste = expected_paste(request)
    if paste is not None:
        score.paste_total += 1
        score.paste_same += last.get("arguments", {}).get("text") == paste
    if request.get("notMine"):
        for event in events:
            if event["type"] == "message" and claims_mine(event["text"]):
                score.team_as_mine.append(f"{rid}: {event['text']}")


def format_score(name: str, s: Score) -> str:
    must = f"{s.must_found}/{s.must_total}"
    paste = f"{s.paste_same}/{s.paste_total}"
    median = statistics.median(s.seconds) if s.seconds else 0
    lines = [
        f"## {name}",
        f"끝 맞음 {s.right_end}/{s.total} · 마지막 줄 규칙 {s.last_rule}/{s.total} · 쓰기를 MCP 로 부름 {s.writes}"
        f" · 찾음 {must} · 원문 그대로 {paste} · 말 안의 팀 일 {len(s.team_as_mine)}"
        f" · 다시 부름 {s.retries} · 오류 {len(s.errors)}",
        f"시간 중앙 {median:.1f}초 · 최대 {max(s.seconds, default=0):.1f}초"
        f" · 도구 호출 중앙 {statistics.median(s.tool_calls) if s.tool_calls else 0:g}"
        f" · 최대 {max(s.tool_calls, default=0)}",
    ]
    lines += [f"- 틀림 {w}" for w in s.wrong]
    lines += [f"- 오류 {e}" for e in s.errors]
    lines += [f"- 팀 일 후보 {t}" for t in s.team_as_mine]
    return "\n".join(lines)


def score_dir(out: Path, requests: list[dict]) -> str:
    """out/<모델>/<부탁 id>-<run>.json 전부를 채점해 summary 글로."""
    by_id = {r["id"]: r for r in requests}
    parts = []
    for model_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        score = Score()
        said = []
        for path in sorted(model_dir.glob("*.json")):
            rid = path.stem.rsplit("-", 1)[0]
            if rid not in by_id:
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            score_one(score, by_id[rid], record)
            for event in record["events"]:
                if event["type"] == "message":
                    said.append(f"- `{path.stem}` {event['text']}")
                if event["type"] == "confirm":
                    said.append(f"- `{path.stem}` 확인: {event['label']}")
        parts.append(format_score(model_dir.name, score) + "\n\n" + "\n".join(said))
    return "\n\n".join(parts) + "\n"


class CountViolations(logging.Handler):
    """다시 부른 수 — app.agent 의 경고 줄을 센다."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.count += "규칙 위반" in record.getMessage()


async def run_one(request: dict, llm: Any, tools: Any, experiences: Any, settings: Any) -> dict:
    mcp = EvalMcp(tools, experiences, (DATA / "instructions.txt").read_text(encoding="utf-8"))

    @asynccontextmanager
    async def connect(_: str):
        yield mcp

    req = AgentRequest(
        text=request["text"],
        context=request.get("context"),
        mcpUrl="http://eval/mcp/none",
        maxToolCalls=8,
    )
    events: list[dict] = []
    counter = CountViolations()
    logger = logging.getLogger("app.agent")
    logger.addHandler(counter)
    started = time.monotonic()
    try:
        await run_agent(req, llm, connect, settings, events.append)
    finally:
        logger.removeHandler(counter)
    return {
        "seconds": time.monotonic() - started,
        "events": events,
        "writes": mcp.writes,
        "retries": counter.count,
    }


async def run_all(args: argparse.Namespace, requests: list[dict], out: Path) -> None:
    from app.config import load_settings, parse_models
    from app.llm import FallbackLLM, GeminiProvider, OpenAIProvider

    settings = load_settings()
    models = parse_models(args.models) if args.models else settings.agent_models[:1]
    providers = {
        "openai": OpenAIProvider(settings.openai_api_key),
        "gemini": GeminiProvider(settings.gemini_api_key),
    }
    tools = load_tools()
    experiences = json.loads(EXPERIENCES.read_text(encoding="utf-8"))
    for model in models:
        llm = FallbackLLM((model,), providers, settings.llm_call_timeout_seconds)
        model_dir = out / f"{model[0]}-{model[1]}"
        model_dir.mkdir(parents=True, exist_ok=True)
        for run in range(1, args.runs + 1):
            for request in requests:
                record = await run_one(request, llm, tools, experiences, settings)
                path = model_dir / f"{request['id']}-{run}.json"
                path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                last = record["events"][-1]
                print(
                    f"{model[1]} {request['id']}-{run} {last.get('code') or last['type']}"
                    f" {record['seconds']:.1f}s",
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", help="쉼표로 「제공사:모델」 — 기본은 AGENT_LLM_MODELS 첫 모델")
    parser.add_argument("--runs", type=int, default=1, help="같은 부탁을 몇 번씩")
    parser.add_argument("--requests", help="쉼표로 부탁 id — 기본은 전부")
    parser.add_argument("--out", type=Path, help="이벤트 저장 폴더")
    parser.add_argument("--env", type=Path, default=ROOT / ".env", help="키를 읽을 .env")
    parser.add_argument("--rescore", type=Path, help="저장한 이벤트만 다시 채점(LLM 안 부름)")
    args = parser.parse_args()

    requests = load_requests(args.requests.split(",") if args.requests else None)
    if args.rescore:
        print(score_dir(args.rescore, requests))
        return

    from dotenv import load_dotenv

    if not args.env.exists():
        sys.exit(f"{args.env} 이 없습니다 — --env 로 키 파일을 주세요")
    load_dotenv(args.env)  # 값은 화면에 찍지 않는다
    out = args.out or ROOT / "eval-runs" / f"agent-{datetime.now():%Y%m%d-%H%M%S}"
    asyncio.run(run_all(args, requests, out))
    summary = score_dir(out, requests)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print(f"\n{summary}\n저장: {out}")


if __name__ == "__main__":
    main()
