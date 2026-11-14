"""링크 받기(S-46) — 네트워크(_resolve · _open)를 가짜 웹으로 바꿔 계약 「링크 받기」 표만 본다.

실제 사이트 시험은 scripts/try_links.py(손으로).
"""

import http.server
import threading

import pytest

from app import fetch_page
from app.errors import UrlUnreadable
from app.fetch_page import Raw
from tests.conftest import FakeLLM

PUBLIC = "93.184.216.34"
QUESTION = "우리 회사에 지원한 동기와 입사 후 이루고 싶은 목표를 구체적으로 적어 주세요. (700자)"
PAGE = (
    "<html><head><title>OO 테크 채용</title><script>var x = '지원';</script></head><body>"
    "<nav>홈 | 채용 | 로그인</nav><header>OO 테크</header>"
    "<main><h1>OO 테크 백엔드 신입 채용</h1>"
    + "<p>주요 업무: 결제 서버를 만들고 운영합니다. 대용량 트래픽을 다룹니다.</p>"
    * 6
    + f"<h2>자기소개서 문항</h2><ol><li>{QUESTION}</li></ol></main>"
    "<footer>Copyright OO 테크</footer></body></html>"
)
HTML = "text/html; charset=utf-8"
GOOD = {
    "organization": "OO 테크",
    "organizationType": "JOB",
    "jobRole": "BACKEND",
    "questions": [QUESTION],
    "postingLines": [],
}
EMPTY = {
    "organization": None,
    "organizationType": None,
    "jobRole": None,
    "questions": [],
    "postingLines": [],
}


class Web:
    """가짜 웹 — hosts: 호스트 → IP 들, pages: 주소 → Raw(또는 예외). 받은 주소는 seen 에 쌓인다."""

    def __init__(self) -> None:
        self.hosts: dict[str, list[str]] = {"example.com": [PUBLIC], "other.com": [PUBLIC]}
        self.pages: dict[str, object] = {"https://example.com/robots.txt": Raw(404)}
        self.seen: list[str] = []

    def resolve(self, host: str) -> list[str]:
        if host not in self.hosts:
            raise OSError("no such host")
        return self.hosts[host]

    def open(self, url: str, timeout: float, deadline: float) -> Raw:
        self.seen.append(url)
        got = self.pages.get(url, Raw(404))
        if isinstance(got, Exception):
            raise got
        assert isinstance(got, Raw)
        return got


@pytest.fixture
def web(monkeypatch):
    w = Web()
    monkeypatch.setattr(fetch_page, "_resolve", w.resolve)
    monkeypatch.setattr(fetch_page, "_open", w.open)
    return w


def page(html: str = PAGE, ctype: str = HTML, **kw) -> Raw:
    return Raw(200, ctype, body=html.encode("utf-8"), **kw)


def post(client_with, url, *outputs):
    llm = FakeLLM(*outputs)
    res = client_with(llm).post("/parse-application", json={"url": url})
    return llm, res


def assert_unreadable(res, llm, code="URL_UNREADABLE"):
    assert res.status_code == 422
    assert res.json()["error"]["code"] == code
    assert llm.calls == 0


def test_reads_page_and_parses_like_text(client_with, web):
    web.pages["https://example.com/job/1"] = page()
    _, res = post(client_with, "https://example.com/job/1", GOOD)
    assert res.status_code == 200
    assert res.json() == {**GOOD, "readFrom": "URL", "model": "fake-model"}
    assert web.seen == ["https://example.com/robots.txt", "https://example.com/job/1"]


def test_written_check_uses_extracted_text(client_with, web):
    # 「적혀 있나」의 기준은 뽑은 글 — 페이지에 없는 문항은 위반으로 다시 부른 뒤 오류
    web.pages["https://example.com/job/1"] = page()
    made_up = {**GOOD, "questions": ["본인의 강점을 적어 주세요."]}
    _, res = post(client_with, "https://example.com/job/1", made_up, made_up)
    assert res.json()["error"]["code"] == "LLM_INVALID_OUTPUT"


def test_extract_drops_script_nav_header_footer():
    text = fetch_page.extract_text(PAGE)
    assert QUESTION in text
    assert "var x" not in text
    assert "로그인" not in text
    assert "Copyright" not in text


def test_extract_drops_aria_navigation_but_keeps_posting():
    html = (
        '<div role="navigation"><div role="menu">Engineering Design Marketing</div></div>'
        "<main><h1>노을상자 채용</h1><p>B2B 브랜드 캠페인 기획 및 운영</p></main>"
        '<div role="menubar">로그인 인재풀</div>'
    )
    text = fetch_page.extract_text(html)
    assert "B2B 브랜드 캠페인 기획 및 운영" in text
    assert "Engineering" not in text
    assert "인재풀" not in text


def test_menu_only_page_is_unreadable_not_filled_in(client_with, web):
    web.pages["https://example.com/menu"] = page(
        '<div role="navigation">' + "Engineering Design Marketing " * 30 + "</div>"
    )
    llm, res = post(client_with, "https://example.com/menu")
    assert_unreadable(res, llm)


def test_job_posting_json_is_used_when_page_is_drawn_by_script():
    desc = "<p>" + "모집 요강입니다. " * 30 + f"</p><p>{QUESTION}</p>"
    html = (
        '<html><body><div id="root"></div><script type="application/ld+json">'
        '{"@type": "JobPosting", "title": "백엔드 개발자", "hiringOrganization": {"name": "OO 테크"},'
        f' "description": "{desc}"}}</script></body></html>'
    )
    text = fetch_page.extract_text(html)
    assert "OO 테크" in text
    assert QUESTION in text


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.0.0.5",
        "192.168.0.1",
        "169.254.169.254",
        "::1",
        "fd00::1",
        "::ffff:10.0.0.1",
    ],
)
def test_private_address_is_refused(client_with, web, ip):
    web.hosts["internal.example"] = [ip]
    llm, res = post(client_with, "http://internal.example/admin")
    assert_unreadable(res, llm)
    assert web.seen == []
    assert "internal" not in res.json()["error"]["message"]


def test_any_private_ip_among_many_is_refused(client_with, web):
    web.hosts["mixed.example"] = [PUBLIC, "10.0.0.1"]
    llm, res = post(client_with, "https://mixed.example/job")
    assert_unreadable(res, llm)
    assert web.seen == []


def test_redirect_to_private_is_refused(client_with, web):
    web.hosts["internal.example"] = ["10.0.0.5"]
    web.pages["https://example.com/go"] = Raw(302, location="http://internal.example/secret")
    llm, res = post(client_with, "https://example.com/go")
    assert_unreadable(res, llm)
    assert not any("internal.example" in u for u in web.seen)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1\\@example.com/job",
        "http://user@example.com/job",
        "http://example.com\t/job",
        "http://exa mple.com/job",
    ],
)
def test_url_read_differently_by_parsers_is_refused(client_with, web, url):
    # 검사하는 호스트와 연결할 호스트가 같아야 한다 — 다르게 읽힐 수 있는 주소는 받지 않는다
    llm, res = post(client_with, url)
    assert_unreadable(res, llm)
    assert web.seen == []


def test_redirect_read_differently_by_parsers_is_refused(client_with, web):
    web.pages["https://example.com/go"] = Raw(302, location="http://127.0.0.1\\@example.com/")
    llm, res = post(client_with, "https://example.com/go")
    assert_unreadable(res, llm)
    assert res.json()["error"]["message"] == fetch_page.MSG_NOT_PUBLIC
    assert web.seen == ["https://example.com/robots.txt", "https://example.com/go"]


def test_connection_to_private_peer_is_cut_before_request(monkeypatch):
    # 검사 땐 공개 IP 로 풀리던 이름이 연결 땐 내부 IP 로 풀리는 경우(리바인딩) — 실제 연결로 본다
    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(fetch_page, "_resolve", lambda host: [PUBLIC])
        with pytest.raises(UrlUnreadable) as e:
            fetch_page._Request().get(f"http://localhost:{server.server_address[1]}/job")
        assert e.value.why == "private"
        assert hits == []
    finally:
        server.shutdown()


def test_redirect_is_followed_and_robots_checked_again(client_with, web):
    web.pages["https://example.com/go"] = Raw(301, location="https://other.com/job")
    web.pages["https://other.com/robots.txt"] = Raw(
        200, "text/plain", body=b"User-agent: *\nDisallow: /job\n"
    )
    llm, res = post(client_with, "https://example.com/go")
    assert_unreadable(res, llm, "URL_BLOCKED")
    assert "https://other.com/job" not in web.seen


def test_relative_redirect_is_followed(client_with, web):
    web.pages["https://example.com/go"] = Raw(302, location="/job/1")
    web.pages["https://example.com/job/1"] = page()
    _, res = post(client_with, "https://example.com/go", GOOD)
    assert res.status_code == 200


def test_too_many_redirects(client_with, web):
    for i in range(5):
        web.pages[f"https://example.com/{i}"] = Raw(302, location=f"/{i + 1}")
    llm, res = post(client_with, "https://example.com/0")
    assert_unreadable(res, llm)
    assert "https://example.com/4" not in web.seen


def test_robots_disallow_is_blocked(client_with, web):
    web.pages["https://example.com/robots.txt"] = Raw(
        200,
        "text/plain",
        body=b"User-agent: Googlebot\nAllow: /\n\nUser-agent: *\nDisallow: /job/\n",
    )
    web.pages["https://example.com/job/1"] = page()
    llm, res = post(client_with, "https://example.com/job/1")
    assert_unreadable(res, llm, "URL_BLOCKED")
    assert res.json()["error"]["message"] == fetch_page.MSG_ROBOTS
    assert "https://example.com/job/1" not in web.seen


def test_robots_allowing_other_path_is_read(client_with, web):
    web.pages["https://example.com/robots.txt"] = Raw(
        200, "text/plain", body=b"User-agent: *\nDisallow: /admin\n"
    )
    web.pages["https://example.com/job/1"] = page()
    _, res = post(client_with, "https://example.com/job/1", GOOD)
    assert res.status_code == 200


def test_robots_404_means_no_rules(client_with, web):
    web.pages["https://example.com/job/1"] = page()
    _, res = post(client_with, "https://example.com/job/1", GOOD)
    assert res.status_code == 200


def test_robots_html_page_means_no_rules(client_with, web):
    web.pages["https://example.com/robots.txt"] = page("<html>not found</html>")
    web.pages["https://example.com/job/1"] = page()
    _, res = post(client_with, "https://example.com/job/1", GOOD)
    assert res.status_code == 200


@pytest.mark.parametrize(
    "robots",
    [Raw(503), fetch_page.requests.ConnectionError("down"), fetch_page.requests.Timeout()],
)
def test_robots_5xx_or_unreachable_is_not_read(client_with, web, robots):
    web.pages["https://example.com/robots.txt"] = robots
    web.pages["https://example.com/job/1"] = page()
    llm, res = post(client_with, "https://example.com/job/1")
    assert_unreadable(res, llm)
    assert "https://example.com/job/1" not in web.seen


@pytest.mark.parametrize("ctype", ["application/pdf", "image/png", "application/json", ""])
def test_not_html_is_unreadable(client_with, web, ctype):
    web.pages["https://example.com/a"] = page(ctype=ctype)
    llm, res = post(client_with, "https://example.com/a")
    assert_unreadable(res, llm)


def test_xhtml_is_read(client_with, web):
    web.pages["https://example.com/a"] = page(ctype="application/xhtml+xml")
    _, res = post(client_with, "https://example.com/a", GOOD)
    assert res.status_code == 200


def test_over_2mb_is_unreadable(client_with, web):
    web.pages["https://example.com/a"] = page(too_big=True)
    llm, res = post(client_with, "https://example.com/a")
    assert_unreadable(res, llm)
    assert res.json()["error"]["message"] == fetch_page.MSG_TOO_BIG


@pytest.mark.parametrize("status", [403, 404, 500])
def test_error_status_is_unreadable(client_with, web, status):
    web.pages["https://example.com/a"] = Raw(status, HTML, body=PAGE.encode())
    llm, res = post(client_with, "https://example.com/a")
    assert_unreadable(res, llm)


def test_short_text_is_unreadable(client_with, web):
    # JS 로 그리는 페이지 — 글이 200자 미만이면 LLM 을 부르지 않는다
    web.pages["https://example.com/a"] = page(
        '<html><body><div id="root"></div><script>' + "render();" * 500 + "</script></body></html>"
    )
    llm, res = post(client_with, "https://example.com/a")
    assert_unreadable(res, llm)
    assert res.json()["error"]["message"] == fetch_page.MSG_TOO_SHORT


def test_unknown_host_and_connection_error(client_with, web):
    llm, res = post(client_with, "https://nowhere.example/a")
    assert_unreadable(res, llm)
    web.pages["https://example.com/a"] = fetch_page.requests.ConnectionError("x")
    llm, res = post(client_with, "https://example.com/a")
    assert_unreadable(res, llm)


def test_long_text_is_cut_not_refused(client_with, web):
    web.pages["https://example.com/a"] = page(
        "<html><body><p>" + "가" * 30000 + "</p></body></html>"
    )
    _, res = post(client_with, "https://example.com/a", EMPTY)
    assert res.status_code == 200
    assert len(fetch_page.fetch_text("https://example.com/a")) == fetch_page.MAX_CHARS


def test_euc_kr_page_from_meta_charset(web):
    html = '<html><head><meta charset="euc-kr"></head><body><p>' + "지원 동기를 적어 주세요. " * 20
    body = (html + "</p></body></html>").encode("cp949")
    web.pages["https://example.com/a"] = Raw(200, "text/html", body=body)
    assert "지원 동기를 적어 주세요." in fetch_page.fetch_text("https://example.com/a")


def test_detail_iframe_body_comes_first(web):
    # 사람인 · 잡코리아 — 공고 페이지엔 요강만, 문항이 있는 상세 본문은 iframe 주소에 따로 있다
    web.hosts["www.saramin.co.kr"] = [PUBLIC]
    base = "https://www.saramin.co.kr/zf_user/jobs/relay"
    web.pages[f"{base}/view?rec_idx=123"] = page(
        "<html><body><h1>사람인 공고 제목</h1></body></html>"
    )
    web.pages[f"{base}/view-detail?rec_idx=123&rec_seq=0"] = page()
    text = fetch_page.fetch_text(f"{base}/view?rec_idx=123")
    assert text.index(QUESTION) < text.index("사람인 공고 제목")


def test_detail_failure_keeps_main_page(web):
    web.hosts["www.jobkorea.co.kr"] = [PUBLIC]
    web.pages["https://www.jobkorea.co.kr/Recruit/GI_Read/9"] = page()
    text = fetch_page.fetch_text("https://www.jobkorea.co.kr/Recruit/GI_Read/9")
    assert QUESTION in text
    assert "https://www.jobkorea.co.kr/Recruit/GI_Read_Comt_Ifrm?Gno=9" in web.seen


def test_detail_body_that_is_an_image_is_unreadable(web):
    # AI#145 — 상세 본문이 이미지 한 장이고 글은 직종 분류 · 주소뿐인 공고(사람인 신세계센트럴 rec_idx=55455220 모양)
    web.hosts["www.saramin.co.kr"] = [PUBLIC]
    base = "https://www.saramin.co.kr/zf_user/jobs/relay"
    web.pages[f"{base}/view?rec_idx=124"] = page("<html><body><h1>공고 제목</h1>" + "<p>모집 요강</p>" * 60 + "</body></html>")
    taxonomy = "".join(f"<p>선택 : 영업·판매·무역 &gt; 직무·직업 &gt; 영업{i}</p>" for i in range(80))
    web.pages[f"{base}/view-detail?rec_idx=124&rec_seq=0"] = page(
        f"<html><body><img src='//img.example/poster.png'>{taxonomy}<p>근무지 주소 서울</p></body></html>"
    )
    with pytest.raises(UrlUnreadable) as e:
        fetch_page.fetch_text(f"{base}/view?rec_idx=124")
    assert e.value.why == "image-body"
    assert e.value.message == fetch_page.MSG_IMAGE_BODY


def test_detail_body_with_images_and_text_is_read(web):
    # 글 본문에 아이콘 · 단계 이미지가 섞인 보통 공고(클리오 모양)는 그대로 읽는다
    web.hosts["www.saramin.co.kr"] = [PUBLIC]
    base = "https://www.saramin.co.kr/zf_user/jobs/relay"
    web.pages[f"{base}/view?rec_idx=125"] = page("<html><body><h1>공고 제목</h1></body></html>")
    web.pages[f"{base}/view-detail?rec_idx=125&rec_seq=0"] = page(
        "<html><body><img src='//img.example/top.png'><p>" + "브랜드 상품 전략을 세우고 신제품을 기획합니다. " * 20 + "</p></body></html>"
    )
    assert "신제품을 기획합니다" in fetch_page.fetch_text(f"{base}/view?rec_idx=125")


def test_short_text_detail_with_a_logo_is_read(web):
    # 회사 이미지 한 장 + 짧은 글 본문(잡코리아 50359824 — 틀 글 뺀 글 398자)은 이미지 본문이 아니다
    web.hosts["www.jobkorea.co.kr"] = [PUBLIC]
    web.pages["https://www.jobkorea.co.kr/Recruit/GI_Read/10"] = page("<html><body><h1>공고 제목</h1></body></html>")
    body = "모집부서 생산팀 신입 자격요건 공업고등학교 졸업 위험물자격증 우대 근무지 경남 양산시 전형절차 서류 면접 " * 4
    web.pages["https://www.jobkorea.co.kr/Recruit/GI_Read_Comt_Ifrm?Gno=10"] = page(
        f"<html><body><img src='//img.example/logo.png'><p>{body}</p></body></html>"
    )
    assert 200 <= len(fetch_page.squeeze(body)) < 300
    assert "위험물자격증" in fetch_page.fetch_text("https://www.jobkorea.co.kr/Recruit/GI_Read/10")


def test_detail_url_only_for_known_sites():
    assert fetch_page.detail_url("https://example.com/zf_user/jobs/view?rec_idx=1") is None
    assert fetch_page.detail_url("https://www.saramin.co.kr/zf_user/jobs/view?rec_idx=x") is None
