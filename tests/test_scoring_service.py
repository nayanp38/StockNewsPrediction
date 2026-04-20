from __future__ import annotations

import pytest

from app.models.schemas import NewsArticle, RetrievedArticle
from app.services.scoring_service import ScoringService


def _retrieved(
    article_id: str,
    tickers: list[str],
    similarity: float,
    ticker_relevance: dict[str, float] | None = None,
    ticker_sentiment: dict[str, float] | None = None,
    ticker_match_score: dict[str, float] | None = None,
) -> RetrievedArticle:
    article = NewsArticle(
        article_id=article_id,
        title=article_id,
        summary=article_id,
        tickers=tickers,
        ticker_sentiment=ticker_sentiment or {},
        ticker_match_score=ticker_match_score or {},
    )
    return RetrievedArticle(
        article=article,
        similarity_score=similarity,
        ticker_relevance=ticker_relevance or {t: similarity for t in tickers},
    )


def test_compute_overall_and_ticker_scores() -> None:
    articles = [
        _retrieved(
            "1",
            tickers=["NVDA", "AMD"],
            similarity=0.8,
            ticker_relevance={"NVDA": 0.9, "AMD": 0.75},
            ticker_sentiment={"NVDA": 0.5, "AMD": 0.3},
        ),
        _retrieved(
            "2",
            tickers=["NVDA"],
            similarity=0.6,
            ticker_relevance={"NVDA": 0.5},
            ticker_sentiment={"NVDA": -0.4},
        ),
    ]
    service = ScoringService()
    overall = service.compute_overall_semantic_score(articles)
    scores = service.compute_ticker_scores(articles, ["NVDA", "AMD"])

    assert round(overall, 2) == 0.70
    nvda = next(s for s in scores if s.ticker == "NVDA")
    amd = next(s for s in scores if s.ticker == "AMD")
    # NVDA has a positive and a negative contributor; AMD has only the positive one.
    assert amd.sentiment_score > nvda.sentiment_score


def test_articles_without_ticker_are_ignored_for_that_ticker() -> None:
    """If NVDA isn't in an article's tickers, it should not contribute to
    NVDA's semantic or sentiment score -- not even as a neutral 0 dilution."""
    nvda_negative = _retrieved(
        "nvda-bear",
        tickers=["NVDA"],
        similarity=0.7,
        ticker_sentiment={"NVDA": -0.8},
        ticker_match_score={"NVDA": 0.6},
    )
    amd_only = _retrieved(
        "amd-only",
        tickers=["AMD"],
        similarity=0.9,  # more similar to the event overall
        ticker_sentiment={"AMD": 0.5},
        ticker_match_score={"AMD": 0.6},
    )
    service = ScoringService()
    scores = service.compute_ticker_scores([amd_only, nvda_negative], ["NVDA"])
    nvda = scores[0]
    # Only nvda_negative contributed; NVDA's sentiment should be close to -0.8.
    assert nvda.sentiment_score == pytest.approx(-0.8, abs=1e-6)


def test_sentiment_uses_match_score_weighting() -> None:
    """Two articles with equal similarity but different match_scores: the
    high-match_score article should dominate."""
    central = _retrieved(
        "central",
        tickers=["NVDA"],
        similarity=0.7,
        ticker_sentiment={"NVDA": -0.5},
        ticker_match_score={"NVDA": 0.9},
    )
    peripheral = _retrieved(
        "peripheral",
        tickers=["NVDA"],
        similarity=0.7,
        ticker_sentiment={"NVDA": 0.2},
        ticker_match_score={"NVDA": 0.1},
    )
    service = ScoringService()
    scores = service.compute_ticker_scores([central, peripheral], ["NVDA"])
    assert scores[0].sentiment_score < 0


def test_ticker_with_no_retrieved_articles_returns_zero() -> None:
    article = _retrieved("x", tickers=["AMD"], similarity=0.5)
    service = ScoringService()
    scores = service.compute_ticker_scores([article], ["NVDA"])
    assert scores[0].ticker == "NVDA"
    assert scores[0].semantic_score == 0.0
    assert scores[0].sentiment_score == 0.0
