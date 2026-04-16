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


def test_retrieve_ranks_and_filters_top_articles() -> None:
    articles = [
        NewsArticle(article_id="1", title="AI demand", summary="Chip spending surges", tickers=["NVDA"]),
        NewsArticle(article_id="2", title="Retail slump", summary="Consumer names weaken", tickers=["TGT"]),
        NewsArticle(article_id="3", title="GPU regulation", summary="Restrictions may hurt exports", tickers=["NVDA", "AMD"]),
    ]
    vectors = {
        "Future AI export ban": [1.0, 0.0],
        articles[0].combined_text: [0.9, 0.1],
        articles[1].combined_text: [0.1, 0.8],
        articles[2].combined_text: [0.8, 0.2],
    }
    settings = Settings(ALPHAVANTAGE_API_KEY="", top_k_articles=2, cluster_count=2)
    service = RetrievalService(settings, FakeEmbeddingService(vectors))

    retrieved = service.retrieve("Future AI export ban", articles, ["NVDA", "AMD"], top_k=2)

    assert len(retrieved) == 2
    assert retrieved[0].article.article_id in {"1", "3"}
    assert all("NVDA" in item.ticker_relevance for item in retrieved)


def test_similarity_floor_filters_weak_articles() -> None:
    strong = NewsArticle(article_id="s", title="AI export policy", summary="Chip export controls expand", tickers=["NVDA"])
    weaker_strong = NewsArticle(article_id="ws", title="Chip policy brief", summary="Policy changes ahead", tickers=["NVDA"])
    weak = NewsArticle(article_id="w", title="Retail blurb", summary="Unrelated content", tickers=["NVDA"])

    vectors = {
        "AI export policy update": [1.0, 0.0],
        strong.combined_text: [0.9, 0.1],
        weaker_strong.combined_text: [0.6, 0.3],
        weak.combined_text: [0.1, 0.9],
    }
    settings = Settings(
        ALPHAVANTAGE_API_KEY="",
        top_k_articles=2,
        cluster_count=2,
        similarity_floor=0.5,
    )
    service = RetrievalService(settings, FakeEmbeddingService(vectors))

    retrieved = service.retrieve(
        "AI export policy update",
        [strong, weaker_strong, weak],
        ["NVDA"],
        top_k=2,
    )

    ids = {item.article.article_id for item in retrieved}
    assert "w" not in ids


def test_ticker_only_articles_are_penalized_in_ranking() -> None:
    topic_hit = NewsArticle(
        article_id="topic",
        title="AI chip policy",
        summary="Policy affects chip sector",
        tickers=["NVDA"],
        source_type="topic",
    )
    ticker_only = NewsArticle(
        article_id="ticker",
        title="NVDA brief note",
        summary="Brief note mentioning NVDA",
        tickers=["NVDA"],
        source_type="ticker",
    )

    vectors = {
        "AI chip policy": [1.0, 0.0],
        topic_hit.combined_text: [0.8, 0.1],
        ticker_only.combined_text: [0.8, 0.1],
    }
    settings = Settings(
        ALPHAVANTAGE_API_KEY="",
        top_k_articles=2,
        cluster_count=2,
        similarity_floor=0.0,
    )
    service = RetrievalService(settings, FakeEmbeddingService(vectors))

    retrieved = service.retrieve(
        "AI chip policy",
        [topic_hit, ticker_only],
        ["NVDA"],
        top_k=2,
    )

    topic_rel = next(r for r in retrieved if r.article.article_id == "topic").ticker_relevance["NVDA"]
    ticker_rel = next(r for r in retrieved if r.article.article_id == "ticker").ticker_relevance["NVDA"]
    assert topic_rel > ticker_rel
