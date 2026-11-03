"""근거 대조(S-04) — 「원본에 있다」의 기준."""

import pytest

from app.citations import Match, SourceIndex

SOURCE = """# 모여봄 회고

## 잘한 점

- 우리는 매주 월요일 30분씩 스탠드업을 했다.
- **나는** 로그인과 알림 서버를 맡았다.
  FCM 으로 모임 시작 알림을 보내게 했다.
| 기술 | `Kotlin` · Room |
"""


@pytest.mark.parametrize(
    ("quote", "lines"),
    [
        ("우리는 매주 월요일 30분씩 스탠드업을 했다.", (5, 5)),
        ("- 우리는 매주 월요일 30분씩 스탠드업을 했다.", (5, 5)),  # 목록 기호를 남겨도
        ("매주 월요일 30분씩", (5, 5)),  # 문장 일부
        ("나는 로그인과 알림 서버를 맡았다.", (6, 6)),  # 강조 기호를 빼도
        ("우리는  매주\n월요일 30 분씩", (5, 5)),  # 공백 · 줄바꿈이 달라도
        ("알림 서버를 맡았다. FCM 으로 모임 시작 알림을", (6, 7)),  # 줄을 넘는 발췌
        ("Kotlin · Room", (8, 8)),  # 표 · 코드 기호
        ("모여봄 회고", (1, 1)),
        ("ＦＣＭ 으로", (7, 7)),  # 전각 글자
    ],
)
def test_found(quote, lines):
    assert SourceIndex(SOURCE).find(quote) == Match(*lines)


@pytest.mark.parametrize(
    "quote",
    [
        "나는 백엔드 API 설계를 전부 맡았다.",  # 없는 문장
        "우리는 매주 월요일 1시간씩 스탠드업을 했다.",  # 수치를 바꿈
        "내가 로그인과 알림 서버를 맡았다.",  # 주어를 바꿈
        "fcm 으로 모임 시작 알림을 보내게 했다.",  # 대소문자도 원문 그대로여야
        "로그인과 알림 서버를 맡았다. 스탠드업을 했다.",  # 떨어진 곳을 이어 붙임
        "- **  ",  # 기호만
    ],
)
def test_not_found(quote):
    assert SourceIndex(SOURCE).find(quote) is None


def test_decomposed_hangul_matches():
    # macOS 에서 온 파일처럼 자모가 분리된(NFD) 원본도 같은 글자로 본다
    import unicodedata

    nfd = unicodedata.normalize("NFD", SOURCE)
    assert SourceIndex(nfd).find("나는 로그인과 알림 서버를 맡았다.") == Match(6, 6)


def test_crlf_lines():
    assert SourceIndex("첫 줄\r\n둘째 줄\r\n셋째 줄").find("셋째 줄") == Match(3, 3)
