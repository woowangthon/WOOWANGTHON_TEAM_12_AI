"""공고 링크 하나를 그때 한 번 받아 본문 글을 뽑는다 (SPEC R-04 F-10 / S-46).

규칙은 ETC docs/contracts/ai-api.md 「링크 받기」 표 그대로다 — 공개 주소만(리다이렉트마다 다시) ·
robots.txt `*` 규칙 · 15초 · 2MB · HTML 만 · 200자 미만이면 못 읽음 · 20000자에서 자름.
받은 본문 · 결과는 저장 · 캐시하지 않고, 로그에는 링크와 결과 코드만 남긴다(본문은 남기지 않는다).
방식은 PM 의 공고 수집기(hq scout net.py · save.py)와 같다 — 브라우저 모양 UA · 인코딩 순서 · bs4.

못 읽으면 지어내지 않고 UrlBlocked · UrlUnreadable(422) — message 는 화면에 그대로 보이는 한국어 한 줄이라
내부 주소 · 스택 · 원문을 담지 않는다.
"""

import asyncio
import ipaddress
import json
import logging
import re
import socket
import time
from dataclasses import dataclass
from urllib import robotparser
from urllib.parse import parse_qsl, urljoin, urlsplit

import requests
from bs4 import BeautifulSoup
from charset_normalizer import from_bytes
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.exceptions import LocationParseError
from urllib3.util import parse_url

from app.errors import UrlBlocked, UrlUnreadable
from app.google_forms import GoogleForm, is_form_url, parse_form, unreadable

log = logging.getLogger(__name__)

FETCH_SECONDS = 15
MAX_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 3
MIN_CHARS = 200
MAX_CHARS = 20000

# scout net.py 와 같다 — 파이썬 기본 UA 를 막는 사이트가 있다(사람인 robots.txt 가 400).
# robots 판단은 `*` 규칙으로 한다 — 「사람인 척」이 아니라 「일반 수집기」 규칙을 지키는 쪽
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)
HTML_TYPES = ("text/html", "application/xhtml+xml")
DROP_TAGS = [
    "script",
    "style",
    "noscript",
    "template",
    "nav",
    "header",
    "footer",
    "form",
    "iframe",
    "svg",
    "button",
    "aside",
]

# 화면에 그대로 보이는 말(계약 — 내부 주소 · 스택 · 원문 없음)
MSG_NOT_PUBLIC = "공개된 인터넷 주소만 읽을 수 있어요"
MSG_NO_HOST = "주소를 찾지 못했어요 — 링크를 확인해 주세요"
MSG_ROBOTS = "이 사이트는 자동으로 읽는 것을 막아 두었어요"
MSG_ROBOTS_UNKNOWN = "사이트의 읽기 규칙(robots.txt)을 확인하지 못했어요"
MSG_REDIRECTS = "다른 주소로 너무 여러 번 넘어가는 링크예요"
MSG_TIMEOUT = "페이지가 제때 답하지 않았어요"
MSG_NO_RESPONSE = "페이지를 받지 못했어요(응답 없음)"
MSG_NOT_HTML = "웹 페이지가 아니에요 — PDF · 이미지는 읽지 않아요"
MSG_TOO_BIG = "페이지가 너무 커요(2MB 넘음)"
MSG_TOO_SHORT = "글이 거의 없는 페이지예요 — 화면을 그림이나 스크립트로 그리는 사이트일 수 있어요"


def msg_status(status: int) -> str:
    return f"페이지가 열리지 않았어요(HTTP {status})"


@dataclass
class Raw:
    """응답 하나 — 리다이렉트는 따라가지 않은 그대로. 시험은 _open 을 이 모양으로 바꿔 끼운다."""

    status: int
    content_type: str = ""
    location: str | None = None
    body: bytes = b""
    too_big: bool = False


def _public_ip(addr: str) -> bool:
    ip = ipaddress.ip_address(addr.split("%")[0])
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


class _PrivatePeer(Exception):
    """연결된 상대가 공개 주소가 아니다 — 요청을 보내기 전에 끊었다."""


class _PublicOnly:
    """소켓이 붙은 순간 상대 IP 를 본다 — 검사 뒤 DNS 답이 바뀌어도(리바인딩) 내부로는 요청을 보내지 않는다."""

    def _new_conn(self):
        sock = super()._new_conn()
        if not _public_ip(sock.getpeername()[0]):
            sock.close()
            raise _PrivatePeer()
        return sock


class _PublicHTTPConnection(_PublicOnly, HTTPConnection):
    pass


class _PublicHTTPSConnection(_PublicOnly, HTTPSConnection):
    pass


class _PublicHTTPPool(HTTPConnectionPool):
    ConnectionCls = _PublicHTTPConnection


class _PublicHTTPSPool(HTTPSConnectionPool):
    ConnectionCls = _PublicHTTPSConnection


class _PublicOnlyAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = {
            "http": _PublicHTTPPool,
            "https": _PublicHTTPSPool,
        }


# ──────────────────────────────────────────────
# 네트워크 — 시험이 바꿔 끼우는 두 자리
# ──────────────────────────────────────────────
def _resolve(host: str) -> list[str]:
    """호스트가 풀리는 IP 전부. 못 풀면 OSError."""
    return list({info[4][0] for info in socket.getaddrinfo(host, None)})


def _open(url: str, timeout: float, deadline: float) -> Raw:
    """GET 한 번(리다이렉트 안 따라감). 본문은 MAX_BYTES 까지만 읽는다. 실패하면 requests 예외.
    연결된 상대가 공개 주소가 아니면 _PrivatePeer. 프록시 환경 변수는 따르지 않는다."""
    headers = {"User-Agent": UA, "Accept-Language": "ko-KR,ko;q=0.9"}
    with requests.Session() as session:
        session.trust_env = False
        session.mount("http://", _PublicOnlyAdapter())
        session.mount("https://", _PublicOnlyAdapter())
        return _read(session, url, headers, timeout, deadline)


def _read(
    session: requests.Session, url: str, headers: dict, timeout: float, deadline: float
) -> Raw:
    with session.get(
        url, headers=headers, timeout=timeout, allow_redirects=False, stream=True
    ) as r:
        raw = Raw(r.status_code, r.headers.get("content-type", ""), r.headers.get("location"))
        length = r.headers.get("content-length", "")
        if length.isdigit() and int(length) > MAX_BYTES:
            raw.too_big = True
            return raw
        body = bytearray()
        for chunk in r.iter_content(64 * 1024):
            body += chunk
            if len(body) > MAX_BYTES:
                raw.too_big = True
                break
            if time.monotonic() > deadline:
                raise requests.Timeout("본문 받기 시간 초과")
        raw.body = bytes(body)
        return raw


# ──────────────────────────────────────────────
# 주소 · robots
# ──────────────────────────────────────────────
# 주소에 쓰지 않는 글자 — 역슬래시 · 공백 · 제어 문자. 해석기마다 다르게 읽는다
_UNSAFE_URL = re.compile(r"[\\\x00-\x20\x7f]")


def _connect_host(url: str) -> str | None:
    """requests(urllib3)가 실제로 연결할 호스트 — 검사도 이 호스트로 한다."""
    try:
        host = parse_url(url).host
    except LocationParseError:
        return None
    return host.strip("[]").lower() if host else None


def _same_host(checked: str, connect: str | None) -> bool:
    if connect is None:
        return False
    if checked == connect:
        return True
    try:
        return checked.encode("idna").decode("ascii") == connect
    except UnicodeError:
        return False


def check_url(url: str) -> str:
    """http(s) 이고, 검사하는 호스트와 연결할 호스트가 같은가 — 다르게 읽힐 수 있는 주소는 받지 않는다. 호스트를 돌려준다."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise UrlUnreadable(MSG_NOT_PUBLIC, why="scheme")
    if (
        _UNSAFE_URL.search(url)
        or "@" in parts.netloc
        or not _same_host(parts.hostname, _connect_host(url))
    ):
        raise UrlUnreadable(MSG_NOT_PUBLIC, why="url")
    return parts.hostname


def check_public(url: str) -> None:
    """주소 모양(check_url)을 보고, 호스트가 풀리는 IP 가 모두 공개 주소인가(SSRF — AI 서버는 BE · DB 와 같은 내부망)."""
    host = check_url(url)
    try:
        addrs = _resolve(host)
    except (OSError, UnicodeError):
        raise UrlUnreadable(MSG_NO_HOST, why="dns") from None
    if not addrs:
        raise UrlUnreadable(MSG_NO_HOST, why="dns")
    if not all(_public_ip(addr) for addr in addrs):
        raise UrlUnreadable(MSG_NOT_PUBLIC, why="private")


class _Request:
    """요청 하나 안의 상태 — 15초 마감과 호스트별 robots(한 요청 안에서 한 번)."""

    def __init__(self) -> None:
        self.deadline = time.monotonic() + FETCH_SECONDS
        self.robots: dict[str, robotparser.RobotFileParser | None] = {}

    def get(self, url: str) -> Raw:
        """공개 주소인지 보고 GET 한 번."""
        check_public(url)
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise UrlUnreadable(MSG_TIMEOUT, why="timeout")
        try:
            return _open(url, left, self.deadline)
        except _PrivatePeer:
            raise UrlUnreadable(MSG_NOT_PUBLIC, why="private") from None
        except requests.Timeout:
            raise UrlUnreadable(MSG_TIMEOUT, why="timeout") from None
        except requests.RequestException:
            raise UrlUnreadable(MSG_NO_RESPONSE, why="connect") from None

    def follow(self, url: str, *, robots: bool) -> tuple[str, Raw]:
        """리다이렉트를 3번까지 따라간다 — 넘어갈 때마다 주소(와 robots)를 다시 본다."""
        for _ in range(MAX_REDIRECTS + 1):
            check_url(url)
            if robots:
                self.check_robots(url)
            raw = self.get(url)
            if raw.status in (301, 302, 303, 307, 308) and raw.location:
                url = urljoin(url, raw.location)
                continue
            return url, raw
        raise UrlUnreadable(MSG_REDIRECTS, why="redirects")

    def check_robots(self, url: str) -> None:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self.robots:
            self.robots[origin] = self._load_robots(origin)
        rules = self.robots[origin]
        if rules is not None and not rules.can_fetch("*", url):
            raise UrlBlocked(MSG_ROBOTS, why="robots")

    def _load_robots(self, origin: str) -> robotparser.RobotFileParser | None:
        """규칙 없음이면 None. 4xx 는 규칙 없음, 5xx · 못 받음은 받지 않는다(조심하는 쪽)."""
        try:
            _, raw = self.follow(f"{origin}/robots.txt", robots=False)
        except UrlUnreadable as e:
            if e.why in ("private", "url"):
                raise
            raise UrlUnreadable(MSG_ROBOTS_UNKNOWN, why=f"robots-{e.why}") from None
        if 400 <= raw.status < 500:
            return None
        if raw.status >= 300 or raw.too_big:
            raise UrlUnreadable(MSG_ROBOTS_UNKNOWN, why=f"robots-{raw.status}")
        text = decode(raw.body, raw.content_type)
        # robots.txt 자리에 HTML 을 주는 사이트가 있다 — 규칙 줄이 없으면 규칙 없음(scout 와 같다)
        if not re.search(r"(?im)^\s*user-agent\s*:", text):
            return None
        rules = robotparser.RobotFileParser()
        rules.parse(text.splitlines())
        return rules


# ──────────────────────────────────────────────
# 인코딩 · 본문 뽑기
# ──────────────────────────────────────────────
_META_CHARSET = re.compile(rb"""<meta[^>]+charset=["']?([\w-]+)""", re.I)


def decode(raw: bytes, content_type: str = "") -> str:
    """헤더 → <meta charset> → 추정 순(계약). EUC-KR 은 상위집합 cp949 로 읽는다(scout 와 같다)."""
    m = re.search(r"charset=([\w-]+)", content_type or "", re.I)
    enc = m.group(1) if m else None
    if not enc:
        m2 = _META_CHARSET.search(raw[:4096])
        enc = m2.group(1).decode("ascii") if m2 else None
    if enc:
        if enc.lower() in ("euc-kr", "ks_c_5601-1987", "euckr"):
            enc = "cp949"
        try:
            return raw.decode(enc, errors="replace")
        except LookupError:
            pass
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        best = from_bytes(raw).best()
        return str(best) if best is not None else raw.decode("utf-8", errors="replace")


def _lines(text: str) -> str:
    out = [" ".join(ln.split()) for ln in text.splitlines()]
    return "\n".join(ln for ln in out if ln)


def _posting_json(soup: BeautifulSoup) -> str:
    """페이지에 실린 JSON-LD JobPosting 의 제목 · 회사 · 본문(HTML) — 화면은 스크립트로 그려도 여기 글이 있는 사이트가 있다."""
    parts: list[str] = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except ValueError:
            continue
        items = data if isinstance(data, list) else data.get("@graph", [data])
        for item in items:
            if not isinstance(item, dict) or item.get("@type") != "JobPosting":
                continue
            org = item.get("hiringOrganization")
            if isinstance(org, dict) and isinstance(org.get("name"), str):
                parts.append(org["name"])
            for key in ("title", "description", "qualifications", "responsibilities"):
                if isinstance(item.get(key), str):
                    parts.append(BeautifulSoup(item[key], "html.parser").get_text("\n"))
    return _lines("\n".join(parts))


def extract_text(html: str) -> str:
    """script · style · nav · header · footer · form 따위를 빼고 글자만. 공고 JSON 이 있으면 앞에 붙인다."""
    soup = BeautifulSoup(html, "html.parser")
    posting = _posting_json(soup)
    for tag in soup(DROP_TAGS):
        tag.decompose()
    # nav 대신 div + ARIA role로 만든 사이트 메뉴도 본문 글로 보지 않는다.
    for tag in soup.select('[role="navigation"], [role="menubar"], [role="menu"]'):
        tag.decompose()
    body = _lines(soup.get_text("\n"))
    if posting and squeeze(posting) not in squeeze(body):
        return f"{posting}\n\n{body}" if body else posting
    return body


def squeeze(text: str) -> str:
    return " ".join(text.split())


# ──────────────────────────────────────────────
# 입구
# ──────────────────────────────────────────────
def detail_url(url: str) -> str | None:
    """사이트별 손질 — 공고 본문을 iframe 으로 따로 싣는 사이트의 본문 주소(2026-11-05 시험에서 찾음).

    사람인 · 잡코리아는 공고 페이지엔 제목 · 모집 요강만 있고 상세 요강(문항이 있는 자리)이 iframe 이다.
    """
    parts = urlsplit(url)
    host = parts.hostname or ""
    query = dict(parse_qsl(parts.query))
    if host.endswith("saramin.co.kr") and parts.path in (
        "/zf_user/jobs/relay/view",
        "/zf_user/jobs/view",
    ):
        if query.get("rec_idx", "").isdigit():
            return (
                f"{parts.scheme}://{parts.netloc}/zf_user/jobs/relay/view-detail"
                f"?rec_idx={query['rec_idx']}&rec_seq=0"
            )
    m = re.fullmatch(r"/Recruit/GI_Read/(\d+)", parts.path)
    if host.endswith("jobkorea.co.kr") and m:
        return f"{parts.scheme}://{parts.netloc}/Recruit/GI_Read_Comt_Ifrm?Gno={m.group(1)}"
    return None


def _read_page(req: "_Request", url: str) -> tuple[str, str]:
    """최종 주소와 HTML. 2xx · HTML · 2MB 가 아니면 UrlUnreadable."""
    final_url, raw = req.follow(url, robots=True)
    if raw.status in (401, 403) and (is_form_url(url) or is_form_url(final_url)):
        raise unreadable()
    if not 200 <= raw.status < 300:
        raise UrlUnreadable(msg_status(raw.status), why=f"http-{raw.status}")
    ctype = raw.content_type.split(";")[0].strip().lower()
    if ctype not in HTML_TYPES:
        raise UrlUnreadable(MSG_NOT_HTML, why="not-html")
    if raw.too_big:
        raise UrlUnreadable(MSG_TOO_BIG, why="too-big")
    return final_url, decode(raw.body, raw.content_type)


def _read_html(req: "_Request", url: str) -> str:
    return extract_text(_read_page(req, url)[1])


def fetch_text(url: str) -> str:
    """링크 → 뽑은 글(최대 20000자). 못 읽으면 UrlBlocked · UrlUnreadable."""
    req = _Request()
    text = _read_html(req, url)
    return _page_text(req, url, text)


def _page_text(req: "_Request", url: str, text: str) -> str:
    detail = detail_url(url)
    if detail is not None:
        # 상세 본문을 앞에 — 못 받으면 공고 페이지 글만 쓴다(그 글도 실제 페이지라 지어낸 것이 아니다)
        try:
            text = f"{_read_html(req, detail)}\n\n{text}"
        except (UrlBlocked, UrlUnreadable) as e:
            log.info("상세 본문 못 받음 url=%s result=%s(%s)", detail, e.code, e.why)
    if len(squeeze(text)) < MIN_CHARS:
        raise UrlUnreadable(MSG_TOO_SHORT, why="too-short")
    return text[:MAX_CHARS]


def fetch_application(url: str) -> str | GoogleForm:
    """최종 주소가 구글폼이면 script JSON, 일반 공고면 기존 본문 뽑기를 쓴다."""
    req = _Request()
    final_url, html = _read_page(req, url)
    if is_form_url(final_url) and urlsplit(final_url).hostname == "docs.google.com":
        return parse_form(html)
    if is_form_url(url):
        # 로그인 등 다른 페이지로 넘어갔다. 그 글을 LLM 에 넘기지 않는다.
        raise unreadable()
    return _page_text(req, url, extract_text(html))


async def read_url(url: str) -> str | GoogleForm:
    """요청 스레드를 막지 않게 따로 받는다. 로그에는 링크와 결과 코드만."""
    started = time.monotonic()
    try:
        text = await asyncio.wait_for(asyncio.to_thread(fetch_application, url), FETCH_SECONDS + 1)
    except TimeoutError:
        log.info("링크 읽기 url=%s result=timeout", url)
        raise UrlUnreadable(MSG_TIMEOUT, why="timeout") from None
    except (UrlBlocked, UrlUnreadable) as e:
        log.info("링크 읽기 url=%s result=%s(%s)", url, e.code, e.why)
        raise
    log.info(
        "링크 읽기 url=%s result=OK chars=%d elapsed=%.1fs",
        url,
        len(text) if isinstance(text, str) else sum(map(len, text.questions)),
        time.monotonic() - started,
    )
    return text
