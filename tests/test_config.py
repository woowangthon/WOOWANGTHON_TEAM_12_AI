"""모델 한 번 제한 — /extract 만 따로(AI#153)."""

import pytest

from app import main


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.delenv("LLM_EXTRACT_CALL_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("LLM_CALL_TIMEOUT_SECONDS", "45")
    main.get_settings.cache_clear()
    main.get_llm.cache_clear()
    main.get_match_llm.cache_clear()
    yield monkeypatch
    main.get_settings.cache_clear()
    main.get_llm.cache_clear()
    main.get_match_llm.cache_clear()


def test_extract_llm_uses_extract_call_timeout(fresh):
    assert main.get_llm().call_timeout == 65
    # 매칭은 그대로 45초
    assert main.get_match_llm().call_timeout == 45


def test_extract_call_timeout_from_env(fresh):
    fresh.setenv("LLM_EXTRACT_CALL_TIMEOUT_SECONDS", "50")
    assert main.get_llm().call_timeout == 50
