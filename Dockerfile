# AI 서버 이미지 — ETC ADR-004: OCI 서버 하나에 Docker Compose 로 BE · AI · DB.
# 베이스 이미지가 amd64 · arm64 둘 다라서 OCI ARM 서버에서 그대로 빌드하면 arm64 가 된다.
# 다른 기계에서 ARM 용으로 만들 땐 `docker buildx build --platform linux/arm64 .`
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 의존성 먼저 — 코드만 바뀌면 이 층은 캐시를 쓴다
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app

# 환경 변수(OPENAI_API_KEY 등)는 이미지에 넣지 않는다 — 실행할 때 `--env-file .env` · Compose `env_file`
EXPOSE 8000

# GET /health 는 LLM 을 부르지 않는다(ETC docs/contracts/ai-api.md). slim 이미지엔 curl 이 없어 파이썬으로
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
