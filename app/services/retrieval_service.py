from __future__ import annotations

import math

import numpy as np
from sklearn.cluster import KMeans

from app.config import Settings
from app.models.schemas import NewsArticle, RetrievedArticle
from app.services.embedding_service import EmbeddingService


class RetrievalService:
    """Semantic re-ranking over articles already pre-filtered by MarketAux.

    Every article in ``articles`` has been returned by a per-ticker call that
    constrained both ``symbols`` and ``search``, so it is guaranteed to tag at
    least one requested ticker AND match the query. This stage:

    1. Embeds the query and each article with MiniLM.
    2. Ranks by cosine similarity, drops weak matches via ``similarity_floor``.
    3. Clusters candidate vectors with KMeans and round-robins the top of
       each cluster to keep the supporting list diverse.
    4. Computes a per-ticker ``ticker_relevance`` = similarity + a small
       sentiment tiebreaker. Ticker-scoped sentiment only; we never fall
       back to article-level aggregates because they mix unrelated entities.
    """

    def __init__(self, settings: Settings, embedding_service: EmbeddingService) -> None:
        self.settings = settings
        self.embedding_service = embedding_service

    def retrieve(
        self,
        query: str,
        articles: list[NewsArticle],
        tickers: list[str],
        top_k: int | None = None,
    ) -> list[RetrievedArticle]:
        if not articles:
            return []

        normalized_tickers = [t.upper() for t in tickers]
        query_vector = np.array(self.embedding_service.embed_text(query))
        article_vectors = np.array(
            self.embedding_service.embed_texts([a.combined_text for a in articles])
        )
        similarities = article_vectors @ query_vector

        ranked = sorted(
            zip(articles, article_vectors, similarities, strict=False),
            key=lambda item: float(item[2]),
            reverse=True,
        )

        effective_top_k = top_k or self.settings.top_k_articles
        candidate_count = min(
            len(ranked), max(effective_top_k * 3, self.settings.top_k_articles)
        )
        candidates = ranked[:candidate_count]

        # Keep the floor but never return empty -- if almost everything is
        # weakly aligned we still surface the best few.
        strong = [
            item for item in candidates if float(item[2]) >= self.settings.similarity_floor
        ]
        if len(strong) >= effective_top_k:
            candidates = strong

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

        return self._select_diverse_top_k(retrieved, effective_top_k)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _cluster(self, vectors: list[np.ndarray]) -> list[int]:
        if len(vectors) < 2:
            return [0] * len(vectors)
        cluster_count = min(self.settings.cluster_count, len(vectors))
        if cluster_count < 2:
            return [0] * len(vectors)
        model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        return model.fit_predict(np.array(vectors)).tolist()

    def _ticker_relevance(
        self, article: NewsArticle, ticker: str, similarity: float
    ) -> float:
        """Cosine similarity with a small signed nudge from ticker-scoped
        sentiment. Since MarketAux already guaranteed the article tags the
        ticker, we don't need a mention bonus here."""
        sentiment = article.ticker_sentiment.get(ticker, 0.0)
        return similarity + self.settings.sentiment_relevance_weight * sentiment

    @staticmethod
    def _select_diverse_top_k(
        retrieved: list[RetrievedArticle], top_k: int
    ) -> list[RetrievedArticle]:
        if len(retrieved) <= top_k:
            return retrieved

        by_cluster: dict[int, list[RetrievedArticle]] = {}
        for item in retrieved:
            by_cluster.setdefault(item.cluster_id or 0, []).append(item)

        selected: list[RetrievedArticle] = []
        cluster_ids = sorted(
            by_cluster.keys(),
            key=lambda cid: -max(a.similarity_score for a in by_cluster[cid]),
        )
        while len(selected) < top_k:
            made_progress = False
            for cid in cluster_ids:
                bucket = by_cluster[cid]
                if bucket:
                    selected.append(bucket.pop(0))
                    made_progress = True
                    if len(selected) >= top_k:
                        break
            if not made_progress:
                break

        return sorted(selected, key=lambda item: item.similarity_score, reverse=True)


def logistic_confidence(score: float) -> float:
    """Map a non-negative conviction score to [0.5, 1.0).

    The slope of 1.5 keeps the curve honest: at ``score=0`` (no news signal)
    confidence is 0.5, and a strongly convinced signal (``score=2``) lands
    around 0.95. Previously we used slope=4, which saturated at 0.93 for
    ``score=0.65`` -- that made weak predictions look confidently traded.
    """
    return 1.0 / (1.0 + math.exp(-1.5 * score))
