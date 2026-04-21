from __future__ import annotations

import numpy as np

from app.config import Settings
from app.models.schemas import RetrievedArticle, TickerScore


class ScoringService:
    SEMANTIC_WEIGHT = 0.5
    SENTIMENT_WEIGHT = 0.5

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings

    @property
    def _untagged_sentiment_weight(self) -> float:
        if self.settings is None:
            return 0.25
        return float(self.settings.untagged_sentiment_weight)

    def compute_overall_semantic_score(self, retrieved_articles: list[RetrievedArticle]) -> float:
        if not retrieved_articles:
            return 0.0
        return float(np.mean([article.similarity_score for article in retrieved_articles]))

    def compute_ticker_scores(self, retrieved_articles: list[RetrievedArticle], tickers: list[str]) -> list[TickerScore]:
        scores: list[TickerScore] = []
        similarities = np.array(
            [max(article.similarity_score, 0.0) for article in retrieved_articles],
            dtype=float,
        )
        weight_sum = float(similarities.sum())
        weights = (similarities / weight_sum) if weight_sum > 0 else None

        untagged_weight = self._untagged_sentiment_weight
        for ticker in [ticker.upper() for ticker in tickers]:
            semantic_components = np.array(
                [article.ticker_relevance.get(ticker, 0.0) for article in retrieved_articles],
                dtype=float,
            )
            sentiment_components = np.array(
                [
                    self._resolve_ticker_sentiment(article, ticker, untagged_weight)
                    for article in retrieved_articles
                ],
                dtype=float,
            )

            if semantic_components.size == 0:
                semantic_score = 0.0
                sentiment_score = 0.0
            elif weights is not None:
                # Similarity-weighted so articles that actually match the event drive the score.
                semantic_score = float(np.dot(weights, semantic_components))
                sentiment_score = float(np.dot(weights, sentiment_components))
            else:
                semantic_score = float(semantic_components.mean())
                sentiment_score = float(sentiment_components.mean())

            combined_score = self.SEMANTIC_WEIGHT * semantic_score + self.SENTIMENT_WEIGHT * sentiment_score
            scores.append(
                TickerScore(
                    ticker=ticker,
                    semantic_score=semantic_score,
                    sentiment_score=sentiment_score,
                    combined_score=combined_score,
                )
            )
        return scores

    @staticmethod
    def _resolve_ticker_sentiment(
        retrieved: RetrievedArticle,
        ticker: str,
        untagged_weight: float,
    ) -> float:
        article = retrieved.article
        if ticker in article.ticker_sentiment:
            return float(article.ticker_sentiment[ticker])
        # Ticker was not tagged on this article; dampen overall sentiment so
        # unrelated bullish/bearish stories do not dominate the target ticker.
        return untagged_weight * float(article.overall_sentiment_score)
