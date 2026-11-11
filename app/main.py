"""AI 서버 — ETC docs/contracts/ai-api.md. DB 없이 받은 텍스트로 JSON 만 돌려준다(ADR-003)."""

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from app.agent import agent_lines, check_request
from app.ask import ask
from app.config import Settings, load_settings
from app.draft import draft
from app.errors import ApiError
from app.extract import extract
from app.ground_check import ground_check
from app.guidelines import guidelines
from app.link_suggest import suggest_links
from app.llm import JsonLLM, ToolLLM, build_llm
from app.match import match
from app.mcp_client import McpConnect, connect_mcp
from app.new_facts import new_facts
from app.parse_application import parse_application
from app.posting_match import posting_match
from app.route import route
from app.schemas import (
    AgentRequest,
    AskRequest,
    AskResponse,
    DraftRequest,
    DraftResponse,
    ExtractRequest,
    ExtractResponse,
    GroundCheckRequest,
    GroundCheckResponse,
    GuidelinesRequest,
    GuidelinesResponse,
    LinkSuggestionsRequest,
    LinkSuggestionsResponse,
    MatchRequest,
    MatchResponse,
    NewFactsRequest,
    NewFactsResponse,
    ParseApplicationRequest,
    ParseApplicationResponse,
    PostingMatchRequest,
    PostingMatchResponse,
    RouteRequest,
    RouteResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("app")
# MCP SDK 의 httpx2 는 INFO 로 주소를 찍는다 — 주소에 차곡이 토큰이 들어 있다(계약 「로그에 남기지 않는다」)
logging.getLogger("httpx2").setLevel(logging.WARNING)

app = FastAPI(title="woowang-ai")


@lru_cache
def get_settings() -> Settings:
    return load_settings()


@lru_cache
def get_llm() -> JsonLLM:
    return build_llm(get_settings())


@lru_cache
def get_match_llm() -> JsonLLM:
    settings = get_settings()
    return build_llm(settings, settings.match_models)


@lru_cache
def get_route_llm() -> JsonLLM:
    settings = get_settings()
    return build_llm(settings, settings.route_models, settings.llm_sync_call_timeout_seconds)


@lru_cache
def get_vision_llm() -> JsonLLM:
    # /parse-application 캡처(S-92) — 이미지를 받는 모델 순서, 시간은 동기 호출(50초)
    settings = get_settings()
    return build_llm(settings, settings.vision_models, settings.llm_vision_call_timeout_seconds)


@lru_cache
def get_sync_match_llm() -> JsonLLM:
    # /link-suggestions · /ask · /guidelines — 모델은 매칭과 같은 순서, 시간은 동기 호출(계약 — 50초 안에 답하거나 LLM_TIMEOUT)
    settings = get_settings()
    return build_llm(settings, settings.match_models, settings.llm_sync_call_timeout_seconds)


@lru_cache
def get_agent_llm() -> ToolLLM:
    settings = get_settings()
    return build_llm(settings, settings.agent_models)


def get_mcp_connect() -> McpConnect:
    return connect_mcp


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@app.exception_handler(ApiError)
async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
    log.warning("오류 %s: %s", exc.code, exc.message)
    return _error(exc.status, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def _invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
    # 받은 값(input)은 빼고 위치 · 이유만 — 캡처 base64 · 사용자 글이 응답 · BE 로그로 새지 않게(S-92)
    first = [{k: v for k, v in e.items() if k != "input"} for e in exc.errors()[:1]]
    return _error(400, "INVALID_REQUEST", f"요청 모양이 맞지 않습니다: {first}")


@app.exception_handler(Exception)
async def _internal(_: Request, exc: Exception) -> JSONResponse:
    log.exception("처리하지 못한 오류")
    return _error(500, "INTERNAL_ERROR", f"{type(exc).__name__}")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractResponse)
async def post_extract(
    req: ExtractRequest,
    llm: Annotated[JsonLLM, Depends(get_llm)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ExtractResponse:
    return await extract(req, llm, settings)


@app.post("/match", response_model=MatchResponse)
async def post_match(
    req: MatchRequest,
    llm: Annotated[JsonLLM, Depends(get_match_llm)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> MatchResponse:
    return await match(req, llm, settings)


@app.post("/posting-match", response_model=PostingMatchResponse)
async def post_posting_match(
    req: PostingMatchRequest,
    llm: Annotated[JsonLLM, Depends(get_match_llm)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> PostingMatchResponse:
    # BE 비동기 작업이 부른다 — 모델 순서 · 시간은 /match 와 같다(계약)
    return await posting_match(req, llm, settings)


@app.post("/ground-check", response_model=GroundCheckResponse)
async def post_ground_check(
    req: GroundCheckRequest,
    llm: Annotated[JsonLLM, Depends(get_match_llm)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> GroundCheckResponse:
    return await ground_check(req, llm, settings)


@app.post("/new-facts", response_model=NewFactsResponse)
async def post_new_facts(
    req: NewFactsRequest,
    llm: Annotated[JsonLLM, Depends(get_match_llm)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> NewFactsResponse:
    return await new_facts(req, llm, settings)


@app.post("/draft", response_model=DraftResponse)
async def post_draft(
    req: DraftRequest,
    llm: Annotated[JsonLLM, Depends(get_match_llm)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> DraftResponse:
    return await draft(req, llm, settings)


@app.post("/route", response_model=RouteResponse)
async def post_route(
    req: RouteRequest,
    llm: Annotated[JsonLLM, Depends(get_route_llm)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RouteResponse:
    return await route(req, llm, settings)


@app.post("/parse-application", response_model=ParseApplicationResponse)
async def post_parse_application(
    req: ParseApplicationRequest,
    llm: Annotated[JsonLLM, Depends(get_route_llm)],
    vision_llm: Annotated[JsonLLM, Depends(get_vision_llm)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ParseApplicationResponse:
    return await parse_application(req, llm, settings, vision_llm=vision_llm)


@app.post("/link-suggestions", response_model=LinkSuggestionsResponse)
async def post_link_suggestions(
    req: LinkSuggestionsRequest,
    llm: Annotated[JsonLLM, Depends(get_sync_match_llm)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> LinkSuggestionsResponse:
    return await suggest_links(req, llm, settings)


@app.post("/ask", response_model=AskResponse)
async def post_ask(
    req: AskRequest,
    llm: Annotated[JsonLLM, Depends(get_sync_match_llm)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AskResponse:
    return await ask(req, llm, settings)


@app.post("/guidelines", response_model=GuidelinesResponse)
async def post_guidelines(
    req: GuidelinesRequest,
    llm: Annotated[JsonLLM, Depends(get_sync_match_llm)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> GuidelinesResponse:
    return await guidelines(req, llm, settings)


@app.post("/agent")
async def post_agent(
    req: AgentRequest,
    llm: Annotated[ToolLLM, Depends(get_agent_llm)],
    connect: Annotated[McpConnect, Depends(get_mcp_connect)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> StreamingResponse:
    check_request(req, settings)
    return StreamingResponse(
        agent_lines(req, llm, connect, settings),
        media_type="application/x-ndjson",
    )
