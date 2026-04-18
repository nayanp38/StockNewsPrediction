from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import Settings
from app.models.schemas import NewsArticle
from app.services.sentiment_service import SentimentService


def _settings(tmp_path: Path) -> Settings:
    return Settings(NEWS_API_KEY="k", cache_dir=tmp_path / "cache")


def test_score_to_label_mappings(tmp_path: Path) -> None:
    svc = SentimentService(_settings(tmp_path))
    assert svc._score_to_label(-0.9) == "Bearish"
    assert svc._score_to_label(-0.2) == "Somewhat-Bearish"
    assert svc._score_to_label(0.0) == "Neutral"
    assert svc._score_to_label(0.2) == "Somewhat-Bullish"
    assert svc._score_to_label(0.9) == "Bullish"


def test_score_from_finbert_output(tmp_path: Path) -> None:
    svc = SentimentService(_settings(tmp_path))
    score, label = svc._score_and_label_from_output([
        {"label": "positive", "score": 0.8},
        {"label": "negative", "score": 0.1},
        {"label": "neutral", "score": 0.1},
    ])
    assert round(score, 2) == 0.70
    assert label == "Bullish"


def test_apply_scores_propagates_to_tagged_tickers(tmp_path: Path) -> None:
    svc = SentimentService(_settings(tmp_path))
    article = NewsArticle(
        article_id="a1",
        title="Banks downgraded across the board",
        summary="Credit outlook deteriorates.",
        tickers=["JPM", "GS"],
    )
    svc._apply_scores(article, {"score": -0.5, "label": "Bearish"})
    assert article.overall_sentiment_score == -0.5
    assert article.overall_sentiment_label == "Bearish"
    assert article.ticker_sentiment == {"JPM": -0.5, "GS": -0.5}


def test_annotate_uses_cache_and_skips_model(tmp_path: Path, monkeypatch) -> None:
    svc = SentimentService(_settings(tmp_path))
    article = NewsArticle(
        article_id="cache-hit",
        title="Some title",
        summary="Some summary",
        tickers=["AAPL"],
    )
    svc._store_cached(svc._cache_path_for(article), {"score": 0.42, "label": "Bullish"})

    def explode(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("Model should not load when cache hits")

    monkeypatch.setattr(svc, "_get_pipeline", explode)
    svc.annotate([article])

    assert article.overall_sentiment_score == 0.42
    assert article.overall_sentiment_label == "Bullish"
    assert article.ticker_sentiment == {"AAPL": 0.42}


def test_annotate_disabled_is_noop(tmp_path: Path) -> None:
    settings = Settings(
        NEWS_API_KEY="k",
        cache_dir=tmp_path / "cache",
        sentiment_enabled=False,
    )
    svc = SentimentService(settings)
    article = NewsArticle(article_id="x", title="t", summary="s", tickers=["AAA"])
    svc.annotate([article])
    assert article.overall_sentiment_score == 0.0
    assert article.overall_sentiment_label == "Neutral"
