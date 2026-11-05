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


class UrlError(ApiError):
    """링크 받기(S-46) 실패. message 는 화면에 그대로 보인다 — why 는 로그에만 남는 짧은 코드."""

    status = 422

    def __init__(self, message: str, why: str = ""):
        super().__init__(message)
        self.why = why


class UrlBlocked(UrlError):
    code = "URL_BLOCKED"


class UrlUnreadable(UrlError):
    code = "URL_UNREADABLE"
