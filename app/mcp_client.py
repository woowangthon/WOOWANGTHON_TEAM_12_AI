"""BE MCP 클라이언트 — 차곡이 루프가 도구를 읽고 부르는 자리(ETC ADR-008 · 계약 mcp-api).

도구 정의는 BE 에만 있다. 여기서는 tools/list 결과를 받아 쓸 뿐이다.
주소에 일 하나짜리 토큰이 들어 있으니 주소 · 전송 오류 글을 로그나 이벤트에 싣지 않는다.
"""

import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError

log = logging.getLogger(__name__)

# 내부망의 DB 읽기라 길 일이 없다 — 멈추면 일 전체 시간을 다 먹지 않게 끊는다
CALL_TIMEOUT_SECONDS = 20.0
# JSON-RPC 「없는 도구 · 입력 틀림」 — 모델이 읽고 다시 부를 수 있다. 그 밖의 오류는 연결 문제로 본다
INVALID_PARAMS = -32602


class McpUnavailable(Exception):
    """MCP 에 닿지 못한다 · 토큰이 끊겼다. 메시지에 주소를 넣지 않는다."""


@dataclass(frozen=True)
class McpToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool


@dataclass(frozen=True)
class McpOutcome:
    ok: bool
    result: Any = None
    error: dict[str, str] | None = None


class McpConnection(Protocol):
    instructions: str
    tools: list[McpToolInfo]

    async def call(self, name: str, arguments: dict[str, Any]) -> McpOutcome: ...


McpConnect = Callable[[str], AbstractAsyncContextManager[McpConnection]]


def read_outcome(is_error: bool, structured: Any, texts: list[str]) -> McpOutcome:
    """도구 결과 → 계약 「도구 오류」 모양. isError 는 일을 끝내지 않는다 — 모델이 읽는다."""
    text = "".join(texts)
    if is_error:
        try:
            error = json.loads(text)["error"]
            return McpOutcome(ok=False, error={"code": error["code"], "message": error["message"]})
        except (ValueError, KeyError, TypeError):
            return McpOutcome(ok=False, error={"code": "TOOL_ERROR", "message": text[:300]})
    if structured is not None:
        return McpOutcome(ok=True, result=structured)
    try:
        return McpOutcome(ok=True, result=json.loads(text))
    except ValueError:
        return McpOutcome(ok=True, result={"text": text})


class _Connection:
    def __init__(self, session: ClientSession, instructions: str, tools: list[McpToolInfo]):
        self._session = session
        self.instructions = instructions
        self.tools = tools

    async def call(self, name: str, arguments: dict[str, Any]) -> McpOutcome:
        try:
            result = await self._session.call_tool(
                name, arguments, read_timeout_seconds=CALL_TIMEOUT_SECONDS
            )
        except MCPError as e:
            if e.error.code == INVALID_PARAMS:
                return McpOutcome(
                    ok=False, error={"code": "INVALID_ARGUMENT", "message": e.error.message}
                )
            raise McpUnavailable(f"tools/call 실패 code={e.error.code}") from None
        texts = [c.text for c in result.content if c.type == "text"]
        return read_outcome(result.is_error, result.structured_content, texts)


@asynccontextmanager
async def connect_mcp(url: str) -> AsyncIterator[McpConnection]:
    """initialize → tools/list 까지 끝낸 연결. BE 는 상태가 없어 세션을 지우는 일은 없다."""
    async with streamable_http_client(url) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=CALL_TIMEOUT_SECONDS) as session:
            try:
                init = await session.initialize()
                listing = await session.list_tools()
            except MCPError as e:
                raise McpUnavailable(f"연결 실패 code={e.error.code}") from None
            tools = [
                McpToolInfo(
                    name=t.name,
                    description=t.description or "",
                    input_schema=t.input_schema,
                    # 표시가 없으면 쓰기로 본다 — 저장을 부를 위험이 있는 쪽을 막는다
                    read_only=bool(t.annotations and t.annotations.read_only_hint),
                )
                for t in listing.tools
            ]
            yield _Connection(session, init.instructions or "", tools)
