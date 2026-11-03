"""근거 대조 (SPEC F-02 S-04) — LLM 이 단 발췌문이 원본에 정말 있는지 코드로 확인한다.

「원본에 있다」의 기준(ETC docs/ai-scope.md §2 에도 적는다):
- 두 쪽 다 유니코드 NFKC 로 맞춘 뒤(전각 · 반각, 자모 분리 등)
- 공백 · 줄바꿈과 마크다운 기호(IGNORED)를 빼고
- 발췌문이 원본의 이어진 부분 문자열이면 찾은 것이다. 글자 · 맞춤법 · 대소문자는 그대로 비교한다.
뜻이 맞는지(인용은 있지만 해석이 틀림)는 보지 않는다 — 사람이 본다(F-04).
"""

import unicodedata
from dataclasses import dataclass

# 마크다운 목록 · 제목 · 강조 · 인용 · 표 기호. LLM 이 발췌하며 빼거나 남겨도 같은 문장으로 본다
IGNORED = frozenset("#*_`~>|-")


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
