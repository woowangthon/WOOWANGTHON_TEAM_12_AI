"""AI 서버 — ETC docs/contracts/ai-api.md. DB 없이 받은 텍스트로 JSON 만 돌려준다(ADR-003)."""

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import Settings, load_settings
from app.errors import ApiError
from app.extract import extract
from app.llm import JsonLLM, build_llm
from app.schemas import ExtractRequest, ExtractResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("app")

app = FastAPI(title="woowang-ai")


@lru_cache
def get_settings() -> Settings:
    return load_settings()


@lru_cache
def get_llm() -> JsonLLM:
    return build_llm(get_settings())


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@app.exception_handler(ApiError)
async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
    log.warning("오류 %s: %s", exc.code, exc.message)
    return _error(exc.status, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def _invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
    return _error(400, "INVALID_REQUEST", f"요청 모양이 맞지 않습니다: {exc.errors()[:1]}")


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
