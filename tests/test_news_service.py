from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.models.schemas import NewsArticle
from app.services.news_service import (
    NewsCreditBudgetError,
    NewsDataError,
    NewsService,
)


def _make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    kwargs: dict[str, Any] = dict(
        NEWS_API_KEY="test-key",
        cache_dir=tmp_path / "cache",
        sentiment_enabled=False,
        news_min_request_interval_seconds=0.0,
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


def _sample_newsdata_payload() -> dict[str, Any]:
    return {
        "status": "success",
        "totalResults": 1,
        "results": [
            {
                "article_id": "nd-1",
                "title": "Fed signals rate cut next quarter",
                "description": "Policymakers hint at easier monetary policy.",
                "link": "https://example.com/fed-cut",
                "source_name": "Example Wire",
                "pubDate": "2026-04-17 14:02:00",
                "symbol": ["JPM", "GS"],
            }
        ],
    }


def test_fetch_news_parses_results_and_tags_source(monkeypatch, tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    service = NewsService(settings)

    calls: list[tuple[str, dict[str, str]]] = []

    def fake_call(self: NewsService, path: str, params: dict[str, str]) -> dict[str, Any]:
        calls.append((path, params))
        return _sample_newsdata_payload()

    monkeypatch.setattr(NewsService, "_call", fake_call)

    articles = service.fetch_news("Fed cuts interest rates", ["JPM", "GS"])

    assert {call[0] for call in calls} == {"/market", "/latest"}
    assert len(articles) == 1
    article = articles[0]
    assert article.title == "Fed signals rate cut next quarter"
    assert article.url == "https://example.com/fed-cut"
    assert article.source == "Example Wire"
    assert set(article.tickers) == {"JPM", "GS"}
    # Returned by both endpoints -> source_type should merge to "both".
    assert article.source_type == "both"
    assert article.time_published == datetime(2026, 4, 17, 14, 2, 0)


def test_populate_ticker_tags_detects_ticker_in_text(tmp_path: Path) -> None:
    service = NewsService(_make_settings(tmp_path))
    article = NewsArticle(
        article_id="x",
        title="NVDA beats earnings",
        summary="Analysts applaud guidance.",
        tickers=[],
    )
    service._populate_ticker_tags([article], ["NVDA", "AMD"])
    assert article.tickers == ["NVDA"]


def test_build_query_string_drops_stopwords() -> None:
    q = NewsService._build_query_string("The Fed is cutting interest rates")
    tokens = set(q.split(" OR "))
    assert "fed" in tokens
    assert "cutting" in tokens
    assert "interest" in tokens
    assert "rates" in tokens
    assert "the" not in tokens
    assert "is" not in tokens


def test_fetch_news_requires_api_key(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path, NEWS_API_KEY="")
    service = NewsService(settings)
    with pytest.raises(NewsDataError):
        service.fetch_news("Fed cut", ["JPM"])


def test_credit_budget_blocks_excess_calls(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path, news_window_credit_limit=1)
    service = NewsService(settings)
    service._record_credit()  # pretend one call already spent
    with pytest.raises(NewsCreditBudgetError):
        service._enforce_credit_budget()


def test_market_failure_falls_back_to_latest(monkeypatch, tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    service = NewsService(settings)

    seen_paths: list[str] = []

    def fake_call(self: NewsService, path: str, params: dict[str, str]) -> dict[str, Any]:
        seen_paths.append(path)
        if path == "/market":
            raise NewsDataError("boom")
        return _sample_newsdata_payload()

    monkeypatch.setattr(NewsService, "_call", fake_call)

    articles = service.fetch_news("Fed cut", ["JPM"])
    assert "/market" in seen_paths
    # Fallback to /latest happened for the ticker-anchored call, plus the
    # event-anchored call also hits /latest, so we expect at least two /latest calls.
    assert seen_paths.count("/latest") >= 2
    assert len(articles) == 1
