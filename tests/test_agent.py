"""POST /agent — 이벤트 순서 · 쓰기는 confirm · 코드 검사 · 오류 이벤트 · 끊기(계약 ai-api 「POST /agent」)."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from app.agent import agent_lines, llm_tools, run_agent
from app.config import load_settings
from app.errors import LLMError, LLMTimeout
from app.llm import ToolCall, ToolTurn
from app.main import get_mcp_connect
from app.mcp_client import McpOutcome, McpToolInfo, McpUnavailable, read_outcome
from app.schemas import JOB_ROLES, ORGANIZATION_TYPES, AgentRequest

OBJECT = {"type": "object", "properties": {"query": {"type": "string"}}}
APPLICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "organization": {"type": "string"},
        "organizationType": {"type": ["string", "null"], "enum": [*ORGANIZATION_TYPES, None]},
        "jobRole": {"type": ["string", "null"], "enum": [*JOB_ROLES, None]},
        "questions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    "required": ["organization", "questions"],
}
TOOLS = [
    McpToolInfo("search_experiences", "경험 찾기", OBJECT, True),
    McpToolInfo("create_application", "지원 만들기", APPLICATION_SCHEMA, False),
    McpToolInfo(
        "add_source",
        "자료 넣기",
        {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        False,
    ),
]
FOUND = {
    "total": 1,
    "experiences": [{"id": 101, "content": "채팅 지연을 2025년에 고쳤다", "actor": "ME"}],
}
TOKEN_URL = "http://backend:8080/mcp/secret-token"


class FakeMcp:
    instructions = "읽는 규칙"
    tools = TOOLS

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        out = self.outcomes.pop(0) if self.outcomes else McpOutcome(ok=True, result=FOUND)
        if isinstance(out, Exception):
            raise out
        return out


class FakeToolLLM:
    def __init__(self, *turns):
        self.turns = list(turns)
        self.seen = []

    async def generate_turn(self, messages, tools, deadline):
        self.seen.append(list(messages))
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn


def turn(text="", *calls):
    return ToolTurn(text=text, calls=tuple(calls), model="fake-model")


def call(name, n=1, **arguments):
    arguments.setdefault("label", "찾는 중")
    return ToolCall(id=f"llm{n}", name=name, arguments=json.dumps(arguments, ensure_ascii=False))


def connect_to(mcp):
    @asynccontextmanager
    async def connect(url):
        yield mcp

    return connect


def body(**over):
    return {"text": "WebSocket 경험 찾아 줘", "mcpUrl": TOKEN_URL, "maxToolCalls": 8, **over}


@pytest.fixture
def post(client_with):
    def run(llm, mcp, *, settings=None, **over):
        client = client_with(llm, **(settings or {}))
        client.app.dependency_overrides[get_mcp_connect] = lambda: connect_to(mcp)
        response = client.post("/agent", json=body(**over))
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        return [json.loads(line) for line in response.text.splitlines()]

    return run


def test_읽기_도구를_부르고_말한_뒤_done(post):
    llm = FakeToolLLM(
        turn("", call("search_experiences", query="WebSocket")), turn("채팅 경험을 찾았어요.")
    )
    mcp = FakeMcp()
    events = post(llm, mcp)
    assert [e["type"] for e in events] == ["tool_call", "tool_result", "message", "done"]
    assert events[0] == {
        "type": "tool_call",
        "id": "c1",
        "tool": "search_experiences",
        "arguments": {"query": "WebSocket"},  # label 은 MCP 입력이 아니다
        "label": "찾는 중",
    }
    assert events[1]["id"] == "c1"
    assert events[1]["ok"] is True
    assert events[1]["result"] == FOUND
    assert events[1]["error"] is None
    assert events[1]["summary"] == "경험 1개를 찾았어요"
    assert mcp.calls == [("search_experiences", {"query": "WebSocket"})]
    assert "읽는 규칙" in llm.seen[0][0]["content"]


def test_도구_오류는_일을_끝내지_않고_모델이_읽는다(post):
    failed = McpOutcome(ok=False, error={"code": "INVALID_ARGUMENT", "message": "year 형식"})
    llm = FakeToolLLM(
        turn("", call("search_experiences", 1, query="a")),
        turn("", call("search_experiences", 2, query="b")),
        turn("찾았어요."),
    )
    events = post(llm, FakeMcp(failed))
    assert [e["type"] for e in events][:2] == ["tool_call", "tool_result"]
    assert events[1]["ok"] is False
    assert events[1]["result"] is None
    assert events[1]["error"]["code"] == "INVALID_ARGUMENT"
    assert events[-1]["type"] == "done"
    assert "INVALID_ARGUMENT" in llm.seen[1][-1]["content"]


def test_쓰기_도구는_부르지_않고_confirm_으로_끝낸다(post):
    arguments = {
        "organization": "OO 부트캠프",
        "organizationType": "BOOTCAMP",
        "jobRole": None,
        "questions": [{"text": "협업 경험을 적어 주세요."}],
    }
    llm = FakeToolLLM(
        turn(
            "지원을 준비했어요.",
            call("create_application", label="지원을 만들까요?", **arguments),
        ),
    )
    mcp = FakeMcp()
    events = post(llm, mcp, text="OO 부트캠프 지원 만들어 줘. 문항: 협업 경험을 적어 주세요.")
    assert [e["type"] for e in events] == ["message", "confirm"]
    assert events[-1] == {
        "type": "confirm",
        "tool": "create_application",
        "arguments": arguments,
        "label": "지원을 만들까요?",
    }
    assert mcp.calls == []


def test_읽기_표시가_없는_도구는_쓰기로_본다(post):
    mcp = FakeMcp()
    mcp.tools = [McpToolInfo("delete_all", "?", {"type": "object"}, False)]
    events = post(FakeToolLLM(turn("", call("delete_all"))), mcp)
    assert [e["type"] for e in events] == ["confirm"]
    assert mcp.calls == []


def test_add_source_는_부탁에서_코드가_잘라_넣는다(post):
    pasted = "축제 부스를 맡았다.\n\n재고 시트를 만들었다."
    llm = FakeToolLLM(
        turn(
            "",
            call(
                "add_source", label="자료로 넣을까요?", textStart="축제 부스", textEnd="만들었다."
            ),
        ),
    )
    events = post(llm, FakeMcp(), text=f"이거 쌓아 줘\n\n{pasted}\n")
    assert [e["type"] for e in events] == ["confirm"]
    assert events[0]["arguments"] == {
        "text": pasted
    }  # label · textStart · textEnd 는 넘기지 않는다


def test_add_source_구절이_부탁에_없으면_다시_부른다(post):
    llm = FakeToolLLM(
        turn(
            "", call("add_source", label="넣을까요?", textStart="고쳐 쓴 말", textEnd="만들었다.")
        ),
        turn("", call("add_source", label="넣을까요?", textEnd="만들었다.")),
        turn(
            "", call("add_source", label="넣을까요?", textStart="재고", textEnd="부스")
        ),  # 앞뒤가 뒤집힘
        turn("", call("add_source", label="넣을까요?", textStart="부스", textEnd="만들었다.")),
    )
    events = post(
        llm,
        FakeMcp(),
        text="쌓아 줘: 부스를 맡아 재고 시트를 만들었다.",
        settings={"max_attempts": 4},
    )
    assert events[0]["arguments"] == {"text": "부스를 맡아 재고 시트를 만들었다."}


def test_add_source_를_끝까지_틀리면_error(post):
    bad = call("add_source", label="넣을까요?", textStart="없는 글", textEnd="없는 글")
    events = post(FakeToolLLM(turn("", bad), turn("", bad)), FakeMcp(), text="이거 쌓아 줘: 원문")
    assert len(events) == 1
    assert events[-1]["code"] == "LLM_ERROR"


def test_모델에게_add_source_는_처음_끝_구절로_보인다():
    add_source = llm_tools(TOOLS)[2]["parameters"]
    assert set(add_source["properties"]) == {"textStart", "textEnd", "label"}
    assert add_source["required"] == ["textStart", "textEnd", "label"]
    assert TOOLS[2].input_schema["required"] == ["text"]  # BE 정의는 그대로


@pytest.mark.parametrize(
    "arguments",
    [
        {"organization": "OO", "questions": [{"text": "지어낸 문항"}]},
        {
            "organization": "OO",
            "organizationType": "COMPANY",
            "questions": [{"text": "협업 경험"}],
        },  # 스키마 — 값 목록 밖
        {
            "organization": "OO",
            "jobRole": "개발",
            "questions": [{"text": "협업 경험"}],
        },  # 스키마 — 값 목록 밖
        {"questions": [{"text": "협업 경험"}]},  # 스키마 — organization 없음
        {"organization": "OO", "questions": []},  # 스키마 — 문항 1개 이상
    ],
)
def test_create_application_규칙을_어기면_다시_부른다(post, arguments):
    good = {"organization": "OO", "questions": [{"text": "협업 경험"}]}
    llm = FakeToolLLM(
        turn("", call("create_application", label="만들까요?", **arguments)),
        turn("", call("create_application", label="만들까요?", **good)),
    )
    events = post(llm, FakeMcp(), text="OO 지원 만들어 줘. 문항은 협업 경험")
    assert [e["type"] for e in events] == ["confirm"]
    assert events[0]["arguments"] == good


def test_문항은_맥락_문항에서_가져와도_된다(post):
    arguments = {"organization": "OO", "questions": [{"text": "기술 문제 해결 경험"}]}
    events = post(
        FakeToolLLM(turn("", call("create_application", label="만들까요?", **arguments))),
        FakeMcp(),
        text="이걸로 지원 만들어 줘",
        context={"question": {"id": 41, "text": "기술 문제 해결 경험", "organization": "OO"}},
    )
    assert events[-1]["type"] == "confirm"


@pytest.mark.parametrize(
    "bad",
    [
        "가" * 301,  # 300자
        "성능을 73% 줄였어요.",  # 도구 결과 밖 수치
        "「없는 프로젝트」 경험이 맞아요.",  # 도구 결과 밖 이름
    ],
)
def test_message_규칙을_어기면_다시_부르고_내보내지_않는다(post, bad):
    llm = FakeToolLLM(
        turn("", call("search_experiences", query="WebSocket")), turn(bad), turn("찾았어요.")
    )
    events = post(llm, FakeMcp())
    assert [e["type"] for e in events] == ["tool_call", "tool_result", "message", "done"]
    assert events[2]["text"] == "찾았어요."


def test_도구_결과에_있는_수치_이름_찾은_개수는_말해도_된다(post):
    llm = FakeToolLLM(
        turn("", call("search_experiences", query="WebSocket")),
        turn("「채팅 지연」을 2025년에 고친 경험 등 3개를 찾았어요."),
    )
    events = post(llm, FakeMcp())
    assert [e["type"] for e in events][-2:] == ["message", "done"]


def test_label_60자를_넘으면_다시_부른다(post):
    llm = FakeToolLLM(
        turn("", call("search_experiences", label="가" * 61, query="a")),
        turn("", call("search_experiences", query="a")),
        turn("찾았어요."),
    )
    mcp = FakeMcp()
    events = post(llm, mcp)
    assert all(len(e.get("label", "")) <= 60 for e in events)
    assert len(mcp.calls) == 1


def test_없는_도구_빈_답은_다시_부른다(post):
    llm = FakeToolLLM(turn("", call("no_such_tool")), turn(""), turn("끝났어요."))
    events = post(llm, FakeMcp(), settings={"max_attempts": 3})
    assert [e["type"] for e in events] == ["message", "done"]


def test_maxToolCalls_를_넘으면_TOO_MANY_STEPS(post):
    llm = FakeToolLLM(*[turn("", call("search_experiences", i, query="a")) for i in range(3)])
    mcp = FakeMcp()
    events = post(llm, mcp, maxToolCalls=2)
    assert [e["type"] for e in events] == [*["tool_call", "tool_result"] * 2, "error"]
    assert events[-1]["code"] == "TOO_MANY_STEPS"
    assert len(mcp.calls) == 2
    assert [e["id"] for e in events if e["type"] == "tool_call"] == ["c1", "c2"]


def test_LLM_오류_시간_초과는_error_이벤트(post):
    events = post(FakeToolLLM(LLMError("모델 없음")), FakeMcp())
    assert [e["code"] for e in events] == ["LLM_ERROR"]
    events = post(FakeToolLLM(LLMTimeout("시간 초과")), FakeMcp())
    assert [e["code"] for e in events] == ["LLM_TIMEOUT"]


def test_도중에_MCP_가_끊기면_tool_result_뒤_MCP_UNAVAILABLE(post):
    llm = FakeToolLLM(turn("", call("search_experiences", query="a")))
    events = post(llm, FakeMcp(McpUnavailable("code=-1")))
    assert [e["type"] for e in events] == ["tool_call", "tool_result", "error"]
    assert events[1]["ok"] is False
    assert events[-1]["code"] == "MCP_UNAVAILABLE"
    assert "secret-token" not in json.dumps(events)


def test_연결에_실패하면_MCP_UNAVAILABLE_주소를_싣지_않는다(client_with, caplog):
    @asynccontextmanager
    async def refused(url):
        raise ExceptionGroup("task", [RuntimeError(f"401 for {url}")])
        yield

    client = client_with(FakeToolLLM())
    client.app.dependency_overrides[get_mcp_connect] = lambda: refused
    with caplog.at_level(logging.INFO):
        response = client.post("/agent", json=body())
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [e["code"] for e in events] == ["MCP_UNAVAILABLE"]
    assert "secret-token" not in response.text
    assert "secret-token" not in caplog.text


def test_내부_오류는_INTERNAL_ERROR(post):
    class Broken(FakeMcp):
        async def call(self, name, arguments):
            raise RuntimeError("버그")

    llm = FakeToolLLM(turn("", call("search_experiences", query="a")))
    events = post(llm, Broken())
    assert events[-1]["code"] == "INTERNAL_ERROR"


@pytest.mark.parametrize(
    "over",
    [
        {"text": "  "},
        {"text": None},
        {"mcpUrl": ""},
        {"mcpUrl": None},
        {"maxToolCalls": 0},
        {"maxToolCalls": None},
        {"text": "가" * 11},
    ],
)
def test_스트림을_열기_전에_400(client_with, over):
    client = client_with(FakeToolLLM(), max_text_chars=10)
    response = client.post("/agent", json=body(**{"text": "부탁", **over}))
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_모양이_틀린_요청은_토큰을_돌려주지_않는다(client_with):
    response = client_with(FakeToolLLM()).post("/agent", json=body(maxToolCalls="많이"))
    assert response.status_code == 400
    assert "secret-token" not in response.text


def _request(**over):
    return AgentRequest(**body(**over))


def _settings(**over):
    return replace(load_settings(), **over)


def test_일_전체_시간을_넘으면_LLM_TIMEOUT():
    class Stuck:
        async def generate_turn(self, messages, tools, deadline):
            await asyncio.sleep(10)

    events = []
    settings = _settings(agent_timeout_seconds=0.05)
    asyncio.run(run_agent(_request(), Stuck(), connect_to(FakeMcp()), settings, events.append))
    assert [e["code"] for e in events] == ["LLM_TIMEOUT"]


def test_BE_가_끊으면_LLM_대기를_취소한다():
    cancelled = []
    started = asyncio.Event()

    class Stuck:
        async def generate_turn(self, messages, tools, deadline):
            started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

    async def main():
        lines = agent_lines(_request(), Stuck(), connect_to(FakeMcp()), _settings())
        reading = asyncio.create_task(anext(lines))
        await started.wait()
        reading.cancel()  # Starlette 가 스트림을 취소하는 자리
        with pytest.raises(asyncio.CancelledError):
            await reading

    asyncio.run(main())
    assert cancelled == [True]


def test_llm_tools_는_label_만_더하고_원본을_바꾸지_않는다():
    defined = llm_tools(TOOLS)
    assert [d["name"] for d in defined] == [t.name for t in TOOLS]
    assert defined[1]["parameters"]["required"] == ["organization", "questions", "label"]
    assert "label" in defined[0]["parameters"]["properties"]
    assert "label" not in OBJECT["properties"]


def test_read_outcome_계약_오류_모양():
    text = json.dumps({"error": {"code": "ACTIVITY_NOT_FOUND", "message": "없음"}})
    assert read_outcome(True, None, [text]).error["code"] == "ACTIVITY_NOT_FOUND"
    assert read_outcome(True, None, ["뭔가 실패"]).error["code"] == "TOOL_ERROR"
    assert read_outcome(False, {"a": 1}, ["무시"]).result == {"a": 1}
    assert read_outcome(False, None, ['{"a":1}']).result == {"a": 1}


CURRENT = {
    "id": 41,
    "text": "기술 문제 해결 경험",
    "organization": "OO",
    "experiences": [
        {
            "id": 101,
            "activityName": "바꿔요",
            "content": "채팅 지연을 고쳤다",
            "recommended": True,
            "inDraft": True,
        },
        {
            "id": 105,
            "activityName": "알고리즘 스터디",
            "content": "문제 선정을 맡았다",
            "recommended": False,
            "inDraft": True,
        },
    ],
}


def test_지금_걸린_경험을_프롬프트에_싣는다(post):
    """ETC#300 — 「지금 경험 말고」에 되묻지 않게 그 문항의 추천 · 초안 경험을 안다."""
    llm = FakeToolLLM(turn("다른 경험을 찾지 못했어요."))
    post(llm, FakeMcp(), text="지금 경험 말고 다른 거", context={"question": CURRENT})
    prompt = llm.seen[0][1]["content"]
    assert "<지금 걸린 경험>" in prompt
    assert "- [101] 바꿔요 — 채팅 지연을 고쳤다 (추천 · 초안에 씀)" in prompt
    assert "- [105] 알고리즘 스터디 — 문제 선정을 맡았다 (초안에 씀)" in prompt


@pytest.mark.parametrize(
    "question",
    [
        {"id": 41, "text": "기술 문제 해결 경험", "organization": "OO"},  # 이전 BE — 칸 없음
        {**CURRENT, "experiences": []},
    ],
)
def test_걸린_경험이_없으면_목록을_싣지_않는다(post, question):
    llm = FakeToolLLM(turn("찾지 못했어요."))
    events = post(llm, FakeMcp(), context={"question": question})
    assert events[-1]["type"] == "done"
    assert "<지금 걸린 경험>" not in llm.seen[0][1]["content"]


def test_지금_걸린_경험의_활동_이름은_말해도_된다(post):
    llm = FakeToolLLM(turn("「바꿔요」 말고는 쌓인 자료에서 찾지 못했어요."))
    events = post(llm, FakeMcp(), text="지금 경험 말고 다른 거", context={"question": CURRENT})
    assert [e["type"] for e in events] == ["message", "done"]
