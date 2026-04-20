from __future__ import annotations

from app.models.schemas import NewsArticle, RetrievedArticle
from app.services.pipeline_service import PipelineService


def _retrieved(article_id: str, tickers: list[str], nvda_relevance: float | None, similarity: float = 0.5) -> RetrievedArticle:
    article = NewsArticle(article_id=article_id, title=article_id, summary=article_id, tickers=tickers)
    ticker_relevance: dict[str, float] = {}
    if nvda_relevance is not None:
        ticker_relevance["NVDA"] = nvda_relevance
    return RetrievedArticle(
        article=article, similarity_score=similarity, ticker_relevance=ticker_relevance
    )


def test_supporting_for_ticker_returns_only_tagged_articles_ordered_by_relevance() -> None:
    tagged_high = _retrieved("tagged-high", tickers=["NVDA"], nvda_relevance=0.6)
    tagged_low = _retrieved("tagged-low", tickers=["NVDA"], nvda_relevance=0.4)
    untagged = _retrieved("untagged", tickers=["AMD"], nvda_relevance=0.95)

    ordered = PipelineService._supporting_for_ticker(
        [tagged_low, untagged, tagged_high], "NVDA"
    )
    assert [r.article.article_id for r in ordered] == ["tagged-high", "tagged-low"]


def test_supporting_for_ticker_empty_when_nothing_tags_ticker() -> None:
    amd_only = _retrieved("a", tickers=["AMD"], nvda_relevance=0.9)
    assert PipelineService._supporting_for_ticker([amd_only], "NVDA") == []
