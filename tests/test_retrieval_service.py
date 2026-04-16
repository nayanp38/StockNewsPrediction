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
