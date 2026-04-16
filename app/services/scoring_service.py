from __future__ import annotations

import numpy as np

from app.models.schemas import RetrievedArticle, TickerScore


class ScoringService:
    def compute_overall_semantic_score(self, retrieved_articles: list[RetrievedArticle]) -> float:
        if not retrieved_articles:
            return 0.0
        return float(np.mean([article.similarity_score for article in retrieved_articles]))

    def compute_ticker_scores(self, retrieved_articles: list[RetrievedArticle], tickers: list[str]) -> list[TickerScore]:
        scores: list[TickerScore] = []
        for ticker in [ticker.upper() for ticker in tickers]:
            semantic_components = [article.ticker_relevance.get(ticker, 0.0) for article in retrieved_articles]
            sentiment_components = [
                article.article.ticker_sentiment.get(ticker, article.article.overall_sentiment_score)
                for article in retrieved_articles
            ]
            semantic_score = float(np.mean(semantic_components)) if semantic_components else 0.0
            sentiment_score = float(np.mean(sentiment_components)) if sentiment_components else 0.0
            combined_score = 0.7 * semantic_score + 0.3 * sentiment_score
            scores.append(
                TickerScore(
                    ticker=ticker,
                    semantic_score=semantic_score,
                    sentiment_score=sentiment_score,
                    combined_score=combined_score,
                )
            )
        return scores
