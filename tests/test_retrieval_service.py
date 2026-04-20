from __future__ import annotations

from app.config import Settings
from app.models.schemas import NewsArticle
from app.services.retrieval_service import RetrievalService


class FakeEmbeddingService:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed_text(self, text: str) -> list[float]:
        return self.vectors[text]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


def test_retrieve_ranks_by_similarity_and_respects_floor() -> None:
    strong = NewsArticle(article_id="s", title="AI export policy", summary="Controls expand", tickers=["NVDA"])
    weak = NewsArticle(article_id="w", title="Retail blurb", summary="Unrelated", tickers=["NVDA"])
    vectors = {
        "AI export policy update": [1.0, 0.0],
        strong.combined_text: [0.9, 0.1],
        weak.combined_text: [0.1, 0.9],
    }
    settings = Settings(
        ALPHAVANTAGE_API_KEY="",
        top_k_articles=1,
        cluster_count=2,
        similarity_floor=0.5,
    )
    service = RetrievalService(settings, FakeEmbeddingService(vectors))
    retrieved = service.retrieve(
        "AI export policy update", [strong, weak], ["NVDA"], top_k=1
    )
    ids = {item.article.article_id for item in retrieved}
    assert ids == {"s"}


def test_ticker_relevance_uses_only_ticker_scoped_sentiment() -> None:
    """A ticker's relevance uses its OWN ticker_sentiment only; it must not
    pick up sentiment from other tickers via the article's overall score."""
    article = NewsArticle(
        article_id="a",
        title="AMD earnings miss",
        summary="AMD-specific commentary.",
        tickers=["AMD"],
        overall_sentiment_score=-0.9,
        ticker_sentiment={"AMD": -0.9},
    )
    vectors = {
        "AI chip policy": [1.0, 0.0],
        article.combined_text: [0.9, 0.1],
    }
    settings = Settings(
        ALPHAVANTAGE_API_KEY="",
        top_k_articles=1,
        cluster_count=1,
        similarity_floor=0.0,
        sentiment_relevance_weight=0.1,
    )
    service = RetrievalService(settings, FakeEmbeddingService(vectors))
    retrieved = service.retrieve("AI chip policy", [article], ["NVDA"], top_k=1)
    nvda_rel = retrieved[0].ticker_relevance["NVDA"]
    # NVDA not in ticker_sentiment -> sentiment contribution is 0 -> relevance == similarity.
    assert nvda_rel == retrieved[0].similarity_score


def test_cluster_diversity_spreads_supporting_articles() -> None:
    """With two clusters of very similar vectors, retrieval picks one from
    each before reaching for a second item from the same cluster."""
    a = NewsArticle(article_id="a", title="rate cut 1", summary="", tickers=["JPM"])
    b = NewsArticle(article_id="b", title="rate cut 2", summary="", tickers=["JPM"])
    c = NewsArticle(article_id="c", title="chip export", summary="", tickers=["JPM"])
    vectors = {
        "rate cut": [1.0, 0.0],
        a.combined_text: [0.99, 0.0],
        b.combined_text: [0.98, 0.0],
        c.combined_text: [0.95, 0.2],  # high similarity too but in a different cluster
    }
    settings = Settings(
        ALPHAVANTAGE_API_KEY="",
        top_k_articles=2,
        cluster_count=2,
        similarity_floor=0.0,
    )
    service = RetrievalService(settings, FakeEmbeddingService(vectors))
    retrieved = service.retrieve("rate cut", [a, b, c], ["JPM"], top_k=2)
    ids = {r.article.article_id for r in retrieved}
    # The diversity selector should pull one from each cluster (a from {a,b} and c)
    # instead of taking both a and b.
    assert "c" in ids
