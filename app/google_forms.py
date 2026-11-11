"""S-94 — 공개 구글폼의 내장 JSON만 읽는다. JavaScript 실행 · LLM 호출 없음."""

import json
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from app.errors import UrlUnreadable

UNREADABLE_MESSAGE = "로그인이 필요한 폼이에요 — 문항 화면을 캡처해 넣어 주세요"
_DATA = re.compile(r"\b(?:var\s+)?FB_PUBLIC_LOAD_DATA_\s*=\s*")
# 서술형 칸이어도 인적 사항을 묻는 라벨은 지원 문항이 아니다.
_PERSONAL = re.compile(
    r"^(?:(?:본인|지원자)의?\s*)?(?:이름|성명|성함|학번|학과|전공|학년|학교(?:명)?|소속|연락처|전화번호|휴대폰(?:\s*번호)?|"
    r"핸드폰(?:\s*번호)?|이메일(?:\s*주소)?|메일\s*주소|생년월일|주소|성별|"
    r"name|full\s*name|student\s*(?:id|number)|e-?mail(?:\s*address)?|phone(?:\s*number)?)"
    r"(?:\s*[/,:：·(\[*].*|\s*(?:을|를)?\s*(?:입력|작성|기입|적어|알려|남겨|써).*|\s*)$",
    re.I,
)


@dataclass
class GoogleForm:
    organization: str | None
    questions: list[str]


def is_form_url(url: str) -> bool:
    parts = urlsplit(url)
    return parts.hostname == "forms.gle" or (
        parts.hostname == "docs.google.com"
        and re.fullmatch(r"/forms/(?:[^/]+/)*viewform/?", parts.path) is not None
    )


def unreadable() -> UrlUnreadable:
    return UrlUnreadable(UNREADABLE_MESSAGE, why="google-form")


def parse_form(html: str) -> GoogleForm:
    """FB_PUBLIC_LOAD_DATA_ 의 form[1] 에 items · title. 구조가 다르면 추측하지 않는다."""
    soup = BeautifulSoup(html, "html.parser")
    try:
        data = None
        for script in soup.find_all("script"):
            match = _DATA.search(script.string or "")
            if match:
                data, _ = json.JSONDecoder().raw_decode(script.string[match.end() :])
                break
        if not isinstance(data, list) or len(data) < 2 or not isinstance(data[1], list):
            raise ValueError
        form = data[1]
        if len(form) < 9:
            raise ValueError
        title = form[8]
        if title is not None and not isinstance(title, str):
            raise ValueError
        items = form[1] if form[1] is not None else []
        if not isinstance(items, list):
            raise ValueError
        questions = []
        for item in items:
            if not isinstance(item, list) or len(item) < 4 or type(item[3]) is not int:
                raise ValueError
            # 0: 단답형 · 1: 장문형. 객관식 · 파일 · 섹션/설명 등은 읽지 않는다.
            if item[3] not in (0, 1):
                continue
            question, description = item[1:3]
            if not isinstance(question, str) or not question.strip():
                raise ValueError
            if description is not None and not isinstance(description, str):
                raise ValueError
            label = re.sub(r"^(?:Q?\d+[.)：:]?\s*)", "", question.strip(), flags=re.I)
            if _PERSONAL.search(label):
                continue
            questions.append(question + ("\n" + description if description else ""))
        return GoogleForm(title if title and title.strip() else None, questions)
    except (ValueError, TypeError, RecursionError):
        raise unreadable() from None
