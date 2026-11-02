"""계약의 오류 코드 — 응답 모양은 { "error": { "code", "message" } }."""


class ApiError(Exception):
    status = 500
    code = "INTERNAL_ERROR"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidRequest(ApiError):
    status = 400
    code = "INVALID_REQUEST"


class TextTooLong(ApiError):
    status = 413
    code = "TEXT_TOO_LONG"


class LLMError(ApiError):
    status = 502
    code = "LLM_ERROR"


class LLMInvalidOutput(ApiError):
    status = 502
    code = "LLM_INVALID_OUTPUT"


class LLMTimeout(ApiError):
    status = 504
    code = "LLM_TIMEOUT"
