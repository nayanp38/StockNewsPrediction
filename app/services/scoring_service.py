from __future__ import annotations

import numpy as np

from app.models.schemas import RetrievedArticle, TickerScore


class ScoringService:
    SEMANTIC_WEIGHT = 0.5
    SENTIMENT_WEIGHT = 0.5

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

        for ticker in [ticker.upper() for ticker in tickers]:
            semantic_components = np.array(
                [article.ticker_relevance.get(ticker, 0.0) for article in retrieved_articles],
                dtype=float,
            )
            sentiment_components = np.array(
                [
                    article.article.ticker_sentiment.get(ticker, article.article.overall_sentiment_score)
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
