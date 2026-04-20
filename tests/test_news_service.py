from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.services.news_service import (
    NewsCreditBudgetError,
    NewsDataError,
    NewsService,
)


def _make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    kwargs: dict[str, Any] = dict(
        NEW_NEWS="test-token",
        cache_dir=tmp_path / "cache",
        marketaux_min_request_interval_seconds=0.0,
        # Tests use fixed payload dates; disable the freshness cutoff unless
        # a specific test overrides it.
        max_article_age_days=0,
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


def _article_payload(
    uuid: str,
    entity_symbol: str,
    entity_name: str | None = None,
    highlights: list[str] | None = None,
) -> dict[str, Any]:
    """Minimal MarketAux-style article payload. Sentiment numbers on
    entities/highlights are irrelevant because we ignore them end-to-end."""
    entity: dict[str, Any] = {
        "symbol": entity_symbol,
        "name": entity_name or f"{entity_symbol} Corp",
        "type": "equity",
        "match_score": 40.0,
    }
    if highlights is not None:
        entity["highlights"] = [
            {"highlight": text, "sentiment": 0.0} for text in highlights
        ]
    return {
        "uuid": uuid,
        "title": f"Headline for {uuid}",
        "description": f"Body text for {uuid}.",
        "snippet": f"Body text for {uuid}.",
        "url": f"https://example.com/{uuid}",
        "source": "Example Wire",
        "published_at": "2025-01-27T16:27:16.000000Z",
        "entities": [entity],
    }


def test_fetch_news_makes_one_call_per_ticker(monkeypatch, tmp_path: Path) -> None:
    """Simplified flow: one MarketAux call per ticker, constraining both
    symbols AND search."""
    settings = _make_settings(tmp_path)
    service = NewsService(settings)

    seen: list[dict[str, str]] = []

    def fake_call(self: NewsService, path: str, params: dict[str, str]) -> dict[str, Any]:
        seen.append(dict(params))
        ticker = params["symbols"]
        return {
            "data": [
                _article_payload(
                    f"{ticker}-1",
                    ticker,
                    f"{ticker} Corp",
                    highlights=[f"{ticker} context sentence."],
                )
            ]
        }

    monkeypatch.setattr(NewsService, "_call", fake_call)

    articles = service.fetch_news("AI chip export curbs", ["NVDA", "AMD"])

    assert len(seen) == 2
    symbols = sorted(p["symbols"] for p in seen)
    assert symbols == ["AMD", "NVDA"]
    for params in seen:
        assert params["filter_entities"] == "true"
        assert params["must_have_entities"] == "true"
        assert "search" in params and "chip" in params["search"]
        # No bull/bear sentiment filtering -- we score sentiment ourselves.
        assert "sentiment_gte" not in params
        assert "sentiment_lte" not in params

    ids = {a.article_id for a in articles}
    assert ids == {"NVDA-1", "AMD-1"}
    nvda = next(a for a in articles if a.article_id == "NVDA-1")
    # Aliases = [symbol, entity.name] so SentimentService can match either.
    assert nvda.ticker_aliases["NVDA"] == ["NVDA", "NVDA Corp"]
    assert nvda.ticker_snippets["NVDA"] == ["NVDA context sentence."]


def test_fetch_news_merges_article_returned_for_multiple_tickers(
    monkeypatch, tmp_path: Path
) -> None:
    """An article that tags multiple requested tickers shouldn't duplicate --
    metadata for each ticker should merge onto the single NewsArticle."""
    settings = _make_settings(tmp_path)
    service = NewsService(settings)

    def fake_call(self: NewsService, path: str, params: dict[str, str]) -> dict[str, Any]:
        ticker = params["symbols"]
        # Same UUID for both tickers; entity switches depending on the query.
        return {
            "data": [
                _article_payload(
                    "shared-1",
                    ticker,
                    f"{ticker} Corp",
                    highlights=[f"From {ticker} call: sentence."],
                )
            ]
        }

    monkeypatch.setattr(NewsService, "_call", fake_call)

    articles = service.fetch_news("AI rally", ["NVDA", "AMD"])
    assert len(articles) == 1
    merged = articles[0]
    assert set(merged.tickers) == {"NVDA", "AMD"}
    assert merged.ticker_snippets["NVDA"] == ["From NVDA call: sentence."]
    assert merged.ticker_snippets["AMD"] == ["From AMD call: sentence."]


def test_entity_parser_harvests_symbol_aliases_and_snippets() -> None:
    entities = [
        {
            "symbol": "NVDA.US",
            "name": "NVIDIA Corporation",
            "type": "equity",
            "match_score": 75.0,
            # Any sentiment fields here must be ignored -- text only.
            "sentiment_score": 0.42,
            "highlights": [
                {"highlight": "Nvidia fell 15% Monday", "sentiment": -0.9},
                {"highlight": "<em>NVDA</em> shares recovered", "sentiment": 0.6},
            ],
        },
        {
            "symbol": "GS",
            "name": "Goldman Sachs Group Inc",
            "type": "equity",
            "match_score": 20.0,
        },
        {
            "symbol": "SomeOrg",
            "name": "Some Org",
            "type": "organization",
            "match_score": 50.0,
        },
    ]
    tickers, match_scores, aliases, snippets = NewsService._extract_entity_info(entities)
    assert tickers == ["NVDA", "GS"]
    assert match_scores["NVDA"] == pytest.approx(0.75)
    assert match_scores["GS"] == pytest.approx(0.20)
    assert "SomeOrg" not in match_scores
    assert aliases["NVDA"] == ["NVDA", "NVIDIA Corporation"]
    assert snippets["NVDA"] == [
        "Nvidia fell 15% Monday",
        "<em>NVDA</em> shares recovered",
    ]
    assert snippets["GS"] == []


def test_build_search_clause_drops_stopwords() -> None:
    clause = NewsService._build_search_clause("The Fed is cutting interest rates")
    tokens = {tok.strip() for tok in clause.split("|")}
    assert {"fed", "cutting", "interest", "rates"}.issubset(tokens)
    assert "the" not in tokens and "is" not in tokens


def test_fetch_news_requires_api_token(tmp_path: Path) -> None:
    service = NewsService(_make_settings(tmp_path, NEW_NEWS=""))
    with pytest.raises(NewsDataError):
        service.fetch_news("Fed cut", ["JPM"])


def test_call_budget_blocks_excess_calls(tmp_path: Path) -> None:
    service = NewsService(_make_settings(tmp_path, marketaux_minute_call_limit=1))
    service._record_call()
    with pytest.raises(NewsCreditBudgetError):
        service._enforce_call_budget()


def test_rate_limited_response_raises_budget_error(monkeypatch, tmp_path: Path) -> None:
    service = NewsService(_make_settings(tmp_path))

    class FakeResponse:
        status_code = 429
        text = "rate limited"

        def json(self) -> dict[str, Any]:
            return {"error": {"code": "rate_limit_reached", "message": "slow down"}}

    monkeypatch.setattr(
        "app.services.news_service.requests.get", lambda *a, **kw: FakeResponse()
    )
    with pytest.raises(NewsCreditBudgetError):
        service._call("/news/all", {"api_token": "x", "limit": "3"})


def test_published_window_is_forwarded(monkeypatch, tmp_path: Path) -> None:
    settings = _make_settings(
        tmp_path,
        marketaux_published_after="2025-01-27",
        marketaux_published_before="2025-02-02",
    )
    service = NewsService(settings)
    captured: list[dict[str, str]] = []

    def fake_call(self: NewsService, path: str, params: dict[str, str]) -> dict[str, Any]:
        captured.append(dict(params))
        return {"data": []}

    monkeypatch.setattr(NewsService, "_call", fake_call)
    service.fetch_news("DeepSeek launch", ["NVDA"])
    assert captured and captured[0]["published_after"] == "2025-01-27"
    assert captured[0]["published_before"] == "2025-02-02"


def test_article_timestamp_parsed(monkeypatch, tmp_path: Path) -> None:
    service = NewsService(_make_settings(tmp_path))

    def fake_call(self: NewsService, path: str, params: dict[str, str]) -> dict[str, Any]:
        return {"data": [_article_payload("t-1", "NVDA", "NVIDIA Corporation")]}

    monkeypatch.setattr(NewsService, "_call", fake_call)
    articles = service.fetch_news("AI", ["NVDA"])
    assert articles[0].time_published is not None
    assert articles[0].time_published.replace(tzinfo=None) == datetime(2025, 1, 27, 16, 27, 16)


def test_freshness_cutoff_applied_server_and_client_side(
    monkeypatch, tmp_path: Path
) -> None:
    """With ``max_article_age_days`` set, we both (a) forward a computed
    ``published_after`` to MarketAux and (b) drop articles whose parsed
    timestamp is older than the cutoff (defends against stale cache hits)."""
    settings = _make_settings(tmp_path, max_article_age_days=30)
    service = NewsService(settings)

    fresh = _article_payload("fresh", "NVDA", "NVIDIA Corporation")
    fresh["published_at"] = datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    stale = _article_payload("stale", "NVDA", "NVIDIA Corporation")
    stale["published_at"] = "2021-03-16T17:03:33.000000Z"

    captured: list[dict[str, str]] = []

    def fake_call(self: NewsService, path: str, params: dict[str, str]) -> dict[str, Any]:
        captured.append(dict(params))
        return {"data": [fresh, stale]}

    monkeypatch.setattr(NewsService, "_call", fake_call)
    articles = service.fetch_news("AI", ["NVDA"])

    assert captured and "published_after" in captured[0]
    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=30)).date().isoformat()
    assert captured[0]["published_after"] == cutoff_date
    assert [a.article_id for a in articles] == ["fresh"]
