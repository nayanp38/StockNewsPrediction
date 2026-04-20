from __future__ import annotations

import numpy as np

from app.config import Settings
from app.models.schemas import RetrievedArticle, TickerScore


class ScoringService:
    """Aggregates retrieved articles into per-ticker semantic + sentiment scores.

    For each ticker we look ONLY at articles that tag it -- every other
    article is irrelevant to that ticker's score, so including it would
    dilute both the semantic and sentiment signals.
    """

    SEMANTIC_WEIGHT = 0.5
    SENTIMENT_WEIGHT = 0.5

    # When MarketAux omits ``match_score`` for an entity we assume moderate
    # centrality so the article contributes but doesn't dominate.
    _DEFAULT_MATCH_SCORE = 0.5
    # Floor on |sentiment| weight so an exactly-neutral article still carries
    # a fraction of its similarity signal forward.
    _MIN_SENTIMENT_WEIGHT = 0.05

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings

    def compute_overall_semantic_score(
        self, retrieved_articles: list[RetrievedArticle]
    ) -> float:
        if not retrieved_articles:
            return 0.0
        return float(np.mean([a.similarity_score for a in retrieved_articles]))

    def compute_ticker_scores(
        self, retrieved_articles: list[RetrievedArticle], tickers: list[str]
    ) -> list[TickerScore]:
        scores: list[TickerScore] = []

        for ticker in [t.upper() for t in tickers]:
            ticker_articles = [
                r for r in retrieved_articles if ticker in r.article.tickers
            ]

            semantic_score = self._semantic_score(ticker_articles, ticker)
            sentiment_score = self._sentiment_score(ticker_articles, ticker)
            combined = (
                self.SEMANTIC_WEIGHT * semantic_score
                + self.SENTIMENT_WEIGHT * sentiment_score
            )
            scores.append(
                TickerScore(
                    ticker=ticker,
                    semantic_score=semantic_score,
                    sentiment_score=sentiment_score,
                    combined_score=combined,
                )
            )
        return scores

    # ------------------------------------------------------------------
    # Per-ticker aggregations
    # ------------------------------------------------------------------
    def _semantic_score(
        self, ticker_articles: list[RetrievedArticle], ticker: str
    ) -> float:
        if not ticker_articles:
            return 0.0
        values = [r.ticker_relevance.get(ticker, r.similarity_score) for r in ticker_articles]
        weights = [max(r.similarity_score, 0.0) for r in ticker_articles]
        return self._weighted_mean(values, weights)

    def _sentiment_score(
        self, ticker_articles: list[RetrievedArticle], ticker: str
    ) -> float:
        """Confidence-weighted mean of ticker-local FinBERT scores.

        Confidence weight folds three orthogonal signals:
          * similarity  -> is the article about the event?
          * match_score -> is the ticker central to the article?
          * |sentiment| -> does FinBERT have conviction about direction?
        """
        values: list[float] = []
        weights: list[float] = []
        for retrieved in ticker_articles:
            sentiment = retrieved.article.ticker_sentiment.get(ticker)
            if sentiment is None:
                continue
            sim = max(retrieved.similarity_score, 0.0)
            match = retrieved.article.ticker_match_score.get(
                ticker, self._DEFAULT_MATCH_SCORE
            )
            values.append(float(sentiment))
            weights.append(
                sim * match * max(abs(sentiment), self._MIN_SENTIMENT_WEIGHT)
            )
        if not values:
            return 0.0
        return self._weighted_mean(values, weights)

    @staticmethod
    def _weighted_mean(values: list[float], weights: list[float]) -> float:
        arr_values = np.asarray(values, dtype=float)
        arr_weights = np.asarray(weights, dtype=float)
        weight_sum = float(arr_weights.sum())
        if weight_sum > 0:
            return float(np.dot(arr_weights / weight_sum, arr_values))
        if arr_values.size == 0:
            return 0.0
        return float(arr_values.mean())
