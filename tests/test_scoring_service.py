from __future__ import annotations

from app.config import Settings
from app.models.schemas import NewsArticle, RetrievedArticle
from app.services.scoring_service import ScoringService


def test_compute_overall_and_ticker_scores() -> None:
    articles = [
        RetrievedArticle(
            article=NewsArticle(
                article_id="1",
                title="AI chip demand rises",
                summary="Strong spending supports semiconductor companies",
                tickers=["NVDA", "AMD"],
                overall_sentiment_score=0.4,
                ticker_sentiment={"NVDA": 0.5, "AMD": 0.3},
            ),
            similarity_score=0.8,
            cluster_id=0,
            ticker_relevance={"NVDA": 0.9, "AMD": 0.75},
        ),
        RetrievedArticle(
            article=NewsArticle(
                article_id="2",
                title="Export controls tighten",
                summary="Restrictions pressure overseas growth",
                tickers=["NVDA"],
                overall_sentiment_score=-0.3,
                ticker_sentiment={"NVDA": -0.4},
            ),
            similarity_score=0.6,
            cluster_id=1,
            ticker_relevance={"NVDA": 0.5, "AMD": 0.2},
        ),
    ]

    service = ScoringService()
    overall = service.compute_overall_semantic_score(articles)
    scores = service.compute_ticker_scores(articles, ["NVDA", "AMD"])

    assert round(overall, 2) == 0.70
    assert scores[0].ticker == "NVDA"
    assert scores[0].combined_score > scores[1].combined_score


def test_sentiment_drives_combined_score_sign() -> None:
    articles = [
        RetrievedArticle(
            article=NewsArticle(
                article_id="neg",
                title="Banks face downgrade",
                summary="Negative outlook weighs on bank names",
                tickers=["GS"],
                overall_sentiment_score=-0.5,
                ticker_sentiment={"GS": -0.6},
            ),
            similarity_score=0.7,
            cluster_id=0,
            ticker_relevance={"GS": 0.5},
        ),
        RetrievedArticle(
            article=NewsArticle(
                article_id="weak_pos",
                title="Unrelated retail blurb",
                summary="Some weak positivity elsewhere",
                tickers=["GS"],
                overall_sentiment_score=0.2,
                ticker_sentiment={"GS": 0.2},
            ),
            similarity_score=0.2,
            cluster_id=1,
            ticker_relevance={"GS": 0.3},
        ),
    ]

    service = ScoringService()
    scores = service.compute_ticker_scores(articles, ["GS"])

    # Similarity-weighted: the stronger-similarity negative article should dominate.
    assert scores[0].ticker == "GS"
    assert scores[0].sentiment_score < 0


def test_untagged_sentiment_is_dampened() -> None:
    # The article is bullish for PDC only; NVDA is not tagged on it.
    article = RetrievedArticle(
        article=NewsArticle(
            article_id="promo",
            title="Perpetuals.com launches AI platform",
            summary="Very bullish for PDC",
            tickers=["PDC"],
            overall_sentiment_score=0.8,
            ticker_sentiment={"PDC": 0.8},
        ),
        similarity_score=0.5,
        cluster_id=0,
        ticker_relevance={"NVDA": 0.5, "PDC": 0.5},
    )

    settings = Settings(ALPHAVANTAGE_API_KEY="", untagged_sentiment_weight=0.25)
    service = ScoringService(settings)
    scores = service.compute_ticker_scores([article], ["NVDA"])

    # 0.25 * overall_sentiment (0.8) = 0.2 for NVDA since ticker isn't tagged.
    assert scores[0].ticker == "NVDA"
    assert abs(scores[0].sentiment_score - 0.2) < 1e-6
