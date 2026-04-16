from __future__ import annotations

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
