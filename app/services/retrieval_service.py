from __future__ import annotations

import math

import numpy as np
from sklearn.cluster import KMeans

from app.config import Settings
from app.models.schemas import NewsArticle, RetrievedArticle
from app.services.embedding_service import EmbeddingService


class RetrievalService:
    def __init__(self, settings: Settings, embedding_service: EmbeddingService) -> None:
        self.settings = settings
        self.embedding_service = embedding_service

    def retrieve(self, query: str, articles: list[NewsArticle], tickers: list[str], top_k: int | None = None) -> list[RetrievedArticle]:
        if not articles:
            return []

        normalized_tickers = [ticker.upper() for ticker in tickers]
        query_vector = np.array(self.embedding_service.embed_text(query))
        article_vectors = np.array(self.embedding_service.embed_texts([article.combined_text for article in articles]))
        similarities = article_vectors @ query_vector

        ranked = sorted(
            zip(articles, article_vectors, similarities, strict=False),
            key=lambda item: float(item[2]),
            reverse=True,
        )
        candidate_count = min(len(ranked), max((top_k or self.settings.top_k_articles) * 3, self.settings.top_k_articles))
        candidates = ranked[:candidate_count]

        cluster_assignments = self._cluster([vector for _, vector, _ in candidates])
        retrieved: list[RetrievedArticle] = []
        for idx, (article, _, similarity) in enumerate(candidates):
            ticker_relevance = {
                ticker: self._ticker_relevance(article, ticker, float(similarity))
                for ticker in normalized_tickers
            }
            retrieved.append(
                RetrievedArticle(
                    article=article,
                    similarity_score=float(similarity),
                    cluster_id=cluster_assignments[idx] if cluster_assignments else None,
                    ticker_relevance=ticker_relevance,
                )
            )

        return self._select_diverse_top_k(retrieved, top_k or self.settings.top_k_articles)

    def _cluster(self, vectors: list[np.ndarray]) -> list[int]:
        if len(vectors) < 2:
            return [0] * len(vectors)
        cluster_count = min(self.settings.cluster_count, len(vectors))
        if cluster_count < 2:
            return [0] * len(vectors)
        model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        return model.fit_predict(np.array(vectors)).tolist()

    @staticmethod
    def _ticker_relevance(article: NewsArticle, ticker: str, similarity: float) -> float:
        direct_mention_bonus = 0.15 if ticker in article.tickers else 0.0
        sentiment_score = article.ticker_sentiment.get(ticker, article.overall_sentiment_score)
        return similarity + direct_mention_bonus + 0.1 * sentiment_score

    @staticmethod
    def _select_diverse_top_k(retrieved: list[RetrievedArticle], top_k: int) -> list[RetrievedArticle]:
        if len(retrieved) <= top_k:
            return retrieved

        by_cluster: dict[int, list[RetrievedArticle]] = {}
        for item in retrieved:
            cluster_id = item.cluster_id or 0
            by_cluster.setdefault(cluster_id, []).append(item)

        selected: list[RetrievedArticle] = []
        cluster_ids = sorted(by_cluster.keys(), key=lambda cid: -max(article.similarity_score for article in by_cluster[cid]))

        while len(selected) < top_k:
            made_progress = False
            for cluster_id in cluster_ids:
                cluster_articles = by_cluster[cluster_id]
                if cluster_articles:
                    selected.append(cluster_articles.pop(0))
                    made_progress = True
                    if len(selected) >= top_k:
                        break
            if not made_progress:
                break

        return sorted(selected, key=lambda item: item.similarity_score, reverse=True)


def logistic_confidence(score: float) -> float:
    return 1.0 / (1.0 + math.exp(-4.0 * score))
