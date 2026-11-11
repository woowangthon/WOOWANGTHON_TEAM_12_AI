"""/parse-application 캡처(images, S-92 · AI#121) — LLM 은 가짜로 바꿔 요청 검사 · 응답 모양 · 로그만 본다.

계약: ETC docs/contracts/ai-api.md 「POST /parse-application」 「캡처 받기」 · entrance-api `images`.
"""

import asyncio
import base64
import logging
import time

import pytest

from app.llm import FallbackLLM, Image, OpenAIProvider
from app.parse_application import IMAGE_RULE, MAX_IMAGE_CHARS, SYSTEM_PROMPT, UNREADABLE_MESSAGE
from tests.conftest import FakeLLM

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 32


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


PNG_IMAGE = {"mediaType": "image/png", "data": b64(PNG)}
BODY = {"images": [PNG_IMAGE]}
Q1 = "우리 동아리에 지원한 동기를 적어 주세요."
GOOD = {
    "organization": "OO 연합 동아리 12기",
    "organizationType": "CLUB",
    "jobRole": None,
    "questions": [Q1],
    "postingLines": [],
}
EMPTY = {
    "organization": None,
    "organizationType": None,
    "jobRole": None,
    "questions": [],
    "postingLines": [],
}


def post(client_with, *outputs, body=None, **settings):
    llm = FakeLLM(*outputs)
    return llm, client_with(llm, **settings).post("/parse-application", json=body or BODY)


def test_contract_shape_reads_from_image(client_with):
    llm, res = post(client_with, GOOD)
    assert res.status_code == 200
    assert res.json() == {**GOOD, "readFrom": "IMAGE", "model": "fake-model"}
    # 이미지는 base64 그대로 모델에 간다 · 이미지 규칙은 프롬프트 끝에만 붙는다
    seen = llm.seen[0]
    assert seen["images"] == [Image(media_type="image/png", data=PNG_IMAGE["data"])]
    assert seen["system"] == SYSTEM_PROMPT + IMAGE_RULE
    assert "캡처 이미지 1장" in seen["user"]


def test_text_prompt_is_unchanged_without_images(client_with):
    # 글 · 링크는 프롬프트가 그대로다 — 앞부분이 같아야 캐시가 산다(AGENTS.md 「프롬프트를 고칠 때」 3)
    llm = FakeLLM({**EMPTY})
    res = client_with(llm).post("/parse-application", json={"text": Q1})
    assert res.status_code == 200
    assert res.json()["readFrom"] == "TEXT"
    assert llm.seen[0]["system"] == SYSTEM_PROMPT
    assert llm.seen[0]["images"] == []


def test_no_written_check_for_images(client_with):
    # 대조할 원문 글이 없다 — 화면에서 사용자가 캡처와 나란히 확인한다(readFrom: IMAGE)
    out = {**GOOD, "postingLines": [{"section": "REQUIRED", "text": "Git 사용 경험", "ksa": None}]}
    _, res = post(client_with, out)
    assert res.status_code == 200
    assert res.json()["postingLines"] == out["postingLines"]


def test_several_formats_and_five_images(client_with):
    images = [
        PNG_IMAGE,
        {"mediaType": "image/jpeg", "data": b64(JPEG)},
        {"mediaType": "image/webp", "data": b64(WEBP)},
        PNG_IMAGE,
        PNG_IMAGE,
    ]
    llm, res = post(client_with, GOOD, body={"images": images})
    assert res.status_code == 200
    assert [i.media_type for i in llm.seen[0]["images"]] == [i["mediaType"] for i in images]
    assert "캡처 이미지 5장" in llm.seen[0]["user"]


@pytest.mark.parametrize(
    "images",
    [
        [],
        [PNG_IMAGE] * 6,
        [{"mediaType": "image/gif", "data": b64(b"GIF89a" + b"\x00" * 8)}],
        [{"mediaType": "image/png"}],
        [{"mediaType": "image/png", "data": ""}],
        [{"mediaType": "image/png", "data": "이건 base64 가 아니다"}],
        [{"mediaType": "image/png", "data": "data:image/png;base64," + b64(PNG)}],
        [{"mediaType": "image/png", "data": b64(JPEG)}],
        [{"mediaType": "image/webp", "data": b64(b"RIFF\x00\x00\x00\x00AVI " + b"\x00" * 8)}],
        [PNG_IMAGE, {"mediaType": "image/jpeg", "data": b64(PNG)}],
    ],
)
def test_invalid_image_does_not_call_llm(client_with, images):
    llm = FakeLLM()
    res = client_with(llm).post("/parse-application", json={"images": images})
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_IMAGE"
    assert llm.calls == 0


def test_image_size_limit(client_with):
    # 한 장 4MB — base64 글자 수 기준(계약 entrance-api)
    raw_size = MAX_IMAGE_CHARS // 4 * 3
    at_limit = {"mediaType": "image/png", "data": b64(PNG + b"\x00" * (raw_size - len(PNG)))}
    assert len(at_limit["data"]) == MAX_IMAGE_CHARS
    _, res = post(client_with, GOOD, body={"images": [at_limit]})
    assert res.status_code == 200
    over = {"mediaType": "image/png", "data": b64(PNG + b"\x00" * (raw_size - len(PNG) + 1))}
    llm = FakeLLM()
    res = client_with(llm).post("/parse-application", json={"images": [over]})
    assert res.json()["error"]["code"] == "INVALID_IMAGE"
    assert "4MB" in res.json()["error"]["message"]
    assert llm.calls == 0


@pytest.mark.parametrize(
    "body",
    [
        {"text": Q1, "images": [PNG_IMAGE]},
        {"url": "https://example.com/1", "images": [PNG_IMAGE]},
        {"images": None},
    ],
)
def test_only_one_of_text_url_images(client_with, body):
    llm = FakeLLM()
    res = client_with(llm).post("/parse-application", json=body)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_REQUEST"
    assert llm.calls == 0


def test_nothing_read_is_unreadable(client_with):
    # 모든 칸이 비면 빈 결과를 200 으로 넘기지 않는다 — 붙여넣기로 돌아간다(계약 「캡처 받기」)
    llm, res = post(client_with, {**EMPTY, "organizationType": "CLUB"})
    assert res.status_code == 422
    assert res.json()["error"] == {"code": "IMAGE_UNREADABLE", "message": UNREADABLE_MESSAGE}
    assert llm.calls == 1  # 규칙 위반이 아니다 — 다시 부르지 않는다


@pytest.mark.parametrize(
    "out",
    [
        {**EMPTY, "organization": "OO 연합 동아리"},
        {**EMPTY, "questions": [Q1]},
        {**EMPTY, "postingLines": [{"section": "DUTY", "text": "서버 개발", "ksa": "SKILL"}]},
    ],
)
def test_any_field_read_is_ok(client_with, out):
    _, res = post(client_with, out)
    assert res.status_code == 200


@pytest.mark.parametrize(
    "bad",
    [
        {**GOOD, "questions": [Q1, Q1]},
        {**GOOD, "questions": [Q1, Q1.replace(" ", "  ")]},
        {**GOOD, "organizationType": "COMPANY"},
        {**GOOD, "questions": [""]},
        "읽은 글입니다",
    ],
)
def test_invalid_output_retries_then_fails(client_with, bad):
    llm, res = post(client_with, bad, bad, max_attempts=2)
    assert res.status_code == 502
    assert res.json()["error"]["code"] == "LLM_INVALID_OUTPUT"
    assert llm.calls == 2


def test_sync_deadline(client_with):
    seen = {}

    class Spy(FakeLLM):
        async def generate_json(self, system, user, schema, deadline, images=()):
            seen["left"] = deadline - time.monotonic()
            return await super().generate_json(system, user, schema, deadline, images)

    res = client_with(Spy(GOOD), llm_timeout_seconds=100, llm_sync_timeout_seconds=50).post(
        "/parse-application", json=BODY
    )
    assert res.status_code == 200
    assert 45 < seen["left"] <= 50


def test_image_and_read_text_are_not_logged(client_with, caplog):
    caplog.set_level(logging.DEBUG)
    secret = "연락처 010-0000-0000 이 찍힌 캡처"
    out = {**GOOD, "questions": [secret]}
    _, res = post(client_with, out)
    assert res.status_code == 200
    _, res = post(client_with, {**out, "questions": [secret, secret]}, EMPTY, max_attempts=2)
    assert res.status_code == 422
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "source=images" in logged
    assert PNG_IMAGE["data"] not in logged
    assert secret not in logged
    assert GOOD["organization"] not in logged


def test_request_shape_error_does_not_echo_image(client_with):
    # 모양 오류(INVALID_REQUEST) 문구에 받은 값을 싣지 않는다 — BE 가 message 를 로그에 남길 수 있다
    data = b64(PNG) * 3
    res = client_with(FakeLLM()).post(
        "/parse-application", json={"images": [{"mediaType": ["image/png"], "data": data}]}
    )
    assert res.status_code == 400
    assert data not in res.text
    assert "image/png" not in res.text


def test_openai_puts_images_as_data_urls(monkeypatch):
    sent = {}

    class Message:
        refusal = None
        content = "{}"

    async def fake_create(model, **kwargs):
        sent.update(kwargs)
        return Message()

    provider = OpenAIProvider("test-key", image_detail="low")
    monkeypatch.setattr(provider, "_create", fake_create)
    images = [Image("image/png", "QUJD"), Image("image/jpeg", "REVG")]
    asyncio.run(provider.call("m", "sys", "user", {"type": "object"}, images))
    content = sent["messages"][1]["content"]
    assert content == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD", "detail": "low"}},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,REVG", "detail": "low"}},
        {"type": "text", "text": "user"},
    ]
    # 글만 부를 땐 예전과 같다
    asyncio.run(provider.call("m", "sys", "user", {"type": "object"}))
    assert sent["messages"][1]["content"] == "user"


def test_fallback_passes_images_only_when_given():
    calls = []

    class Provider:
        name = "openai"
        available = True

        async def call(self, model, system, user, schema, *images):
            calls.append(images)
            return "{}"

    llm = FallbackLLM((("openai", "m"),), {"openai": Provider()}, 5.0)
    image = Image("image/png", "QUJD")
    asyncio.run(llm.generate_json("s", "u", {}, time.monotonic() + 5, images=[image]))
    asyncio.run(llm.generate_json("s", "u", {}, time.monotonic() + 5))
    assert calls == [([image],), ()]


def test_image_repr_hides_data():
    assert "QUJD" not in repr(Image("image/png", "QUJD"))
