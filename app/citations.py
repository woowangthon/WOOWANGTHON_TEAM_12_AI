"""근거 대조 (SPEC F-02 S-04) — LLM 이 단 발췌문이 원본에 정말 있는지 코드로 확인한다.

「원본에 있다」의 기준(ETC docs/ai-scope.md §2 에도 적는다):
- 두 쪽 다 유니코드 NFKC 로 맞춘 뒤(전각 · 반각, 자모 분리 등)
- 공백 · 줄바꿈과 마크다운 기호(IGNORED)를 빼고
- 발췌문이 원본의 이어진 부분 문자열이면 찾은 것이다. 글자 · 맞춤법 · 대소문자는 그대로 비교한다.
뜻이 맞는지(인용은 있지만 해석이 틀림)는 보지 않는다 — 사람이 본다(F-04).

숫자 대조(SPEC S-27 · S-45) — 글 속 숫자가 받치는 글에 있는지 코드로 확인한다. 계약 ai-api 「숫자 대조」.
- 기간은 날짜로 쓴 숫자만 받친다 — 기간 2025-03 은 「2025년 3월」은 받치고 「3초」는 받치지 않는다.
- 한글 수사는 「세 번」처럼 단위가 붙은 2 ~ 49 만 읽는다. 「한」은 안 본다 — 「한 번 더」 「한 달」은 대개 수가 아니다.
  「두 번째」 「두 가지」 「두세 번」과 한자어 수(「삼 년」)도 안 본다.
"""

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

# 마크다운 목록 · 제목 · 강조 · 인용 · 표 기호. LLM 이 발췌하며 빼거나 남겨도 같은 문장으로 본다
IGNORED = frozenset("#*_`~>|-")

# 연속된 숫자 하나 — 3 · 80 · 1,200 · 3.5. 쉼표는 세 자리 끊기일 때만 숫자 안이다(「3,4주차」는 둘)
_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?!\d)|\d+(?:\.\d+)?")

# 날짜로 쓴 숫자 — 2025년 · 2025학년도 · 2025-03 · 2025.03 · 3월. 「3년」 「3개월」은 날짜가 아니다
_DATE = re.compile(r"(?<![\d.,])(?:(\d{4})(?:\s*학?년|[-./](\d{1,2})(?!\d))|(\d{1,2})\s*월)")
_PERIOD = re.compile(r"(\d{4})(?:-(\d{1,2}))?")

_UNITS = {"두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9}
_ONES = {"한": 1, **_UNITS}
# 「쉰」은 안 본다 — 「쉰 시간」은 대개 쉬었다는 말이다
_TENS = {"열": 10, "스물": 20, "서른": 30, "마흔": 40}
_KOREAN = {
    **_UNITS,
    "열": 10,
    "스무": 20,
    "서른": 30,
    "마흔": 40,
    **{ten + one: a + b for ten, a in _TENS.items() for one, b in _ONES.items()},
}
# 수사 + 단위. 앞이 한글이면 수사가 아니고(「두세 번」), 뒤에는 조사만 붙는다
# (「두 개발자」 「세 번역」 「두 번째」를 수로 읽지 않게)
_KOREAN_NUMBER = re.compile(
    rf"(?<![가-힣])({'|'.join(sorted(_KOREAN, key=len, reverse=True))})\s*"
    r"(?:번|차례|명|사람|개월|개|달|주일|주|해|시간|살|곳|배)"
    r"(?=[을를이가은는도만의에로과와으나보부까밖뿐씩쯤간]|[^가-힣]|$)"
)


def _normalize(text: str) -> tuple[str, list[int]]:
    """비교용 문자열과, 그 글자마다 원본에서 몇 번째 줄(1부터)이었는지."""
    chars: list[str] = []
    lines: list[int] = []
    line = 1
    for ch in unicodedata.normalize("NFKC", text):
        if ch == "\n":
            line += 1
        elif not ch.isspace() and ch not in IGNORED:
            chars.append(ch)
            lines.append(line)
    return "".join(chars), lines


def normalize(text: str) -> str:
    return _normalize(text)[0]


@dataclass(frozen=True)
class Match:
    start_line: int
    end_line: int


class SourceIndex:
    """원본 하나를 한 번만 맞춰 두고 발췌문 여러 개를 대조한다."""

    def __init__(self, text: str):
        self._text, self._lines = _normalize(text)

    def find(self, quote: str) -> Match | None:
        """원본에서 찾으면 줄 범위(처음 나온 곳), 못 찾으면 None. 기호만 있는 발췌문은 못 찾은 것."""
        needle = normalize(quote)
        if not needle:
            return None
        at = self._text.find(needle)
        if at < 0:
            return None
        return Match(self._lines[at], self._lines[at + len(needle) - 1])


def numbers(text: str) -> set[str]:
    """글 속 숫자들. NFKC 로 맞추고, 1,200 은 1200 으로, 03 은 3 으로, 「세 번」은 3 으로 본다.

    글자가 아니라 숫자 단위로 비교한다 — 「3초」가 「30초」 안의 글자로 맞았다고 치면 안 된다.
    """
    text = unicodedata.normalize("NFKC", text)
    found = {str(_KOREAN[word]) for word in _KOREAN_NUMBER.findall(text)}
    for raw in _NUMBER.findall(text):
        value = raw.replace(",", "")
        found.add(value if "." in value else str(int(value)))
    return found


def _without_period_dates(text: str, periods: Iterable[str]) -> str:
    """text 에서 기간과 맞는 날짜를 지운다 — 남은 숫자만 받치는 글과 맞춘다."""
    matched = [m for m in map(_PERIOD.match, periods) if m]
    years = {m[1] for m in matched}
    months = {int(m[2]) for m in matched if m[2]}

    def drop(date: re.Match[str]) -> str:
        year, month = date[1], date[2] or date[3]
        backed = (year is None or year in years) and (month is None or int(month) in months)
        return " " if backed else date[0]

    return _DATE.sub(drop, unicodedata.normalize("NFKC", text))


def unsupported_numbers(text: str, source: str, periods: Iterable[str] = ()) -> list[str]:
    """text 의 숫자 중 source 에 없는 것. periods(「2025-03」)는 날짜로 쓴 숫자만 받친다."""
    return sorted(numbers(_without_period_dates(text, periods)) - numbers(source))
