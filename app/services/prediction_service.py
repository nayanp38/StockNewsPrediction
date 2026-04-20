from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

from app.config import Settings
from app.models.schemas import RetrievedArticle, TickerPrediction, TickerScore
from app.services.market_data_service import MarketDataService
from app.services.retrieval_service import logistic_confidence


class PredictionService:
    FEATURE_COLUMNS = [
        "return_1d",
        "return_5d",
        "return_20d",
        "volatility_20d",
        "volume_change_5d",
        "sma_10_gap",
        "sma_30_gap",
        "benchmark_return_5d",
        "relative_strength_5d",
    ]

    def __init__(self, settings: Settings, market_data_service: MarketDataService) -> None:
        self.settings = settings
        self.market_data_service = market_data_service

    def predict_for_ticker(
        self,
        ticker_score: TickerScore,
        overall_semantic_score: float,
        supporting_articles: list[RetrievedArticle],
    ) -> TickerPrediction:
        """Predict a news-driven move for a single ticker.

        Both direction AND magnitude come from news:

            signed_move = sentiment * news_trust * |regression_move|

        * ``sentiment``        -- ticker-scoped FinBERT in [-1, 1]; the sole
                                  source of direction. No momentum fallback.
        * ``news_trust``       -- blend of per-ticker and event-wide semantic
                                  relevance in [0, 1]; dampens moves when
                                  coverage is weakly related to the query.
        * ``|regression_move|``-- RandomForest output on the ticker's price
                                  history, used ONLY as a per-ticker
                                  volatility calibrator (so a high-vol name
                                  moves harder than a low-vol name at equal
                                  conviction). Its sign is ignored.

        Coverage dampener: if only one article contributed ticker-scoped
        sentiment, we scale the move by sqrt(1/3) so a single outlier can't
        produce a large prediction.
        """
        ticker = ticker_score.ticker
        current_price = self.market_data_service.latest_close(ticker)
        regression_move = self._predict_price_move(ticker)

        sentiment_signal = float(ticker_score.sentiment_score)
        semantic_strength = max(0.0, min(1.0, float(ticker_score.semantic_score)))
        event_relevance = max(0.0, min(1.0, float(overall_semantic_score)))
        news_trust = 0.7 * semantic_strength + 0.3 * event_relevance

        scored_count = sum(
            1 for a in supporting_articles if ticker in a.article.ticker_sentiment
        )
        coverage_dampener = (min(scored_count, 3) / 3.0) ** 0.5

        signed_percent_move = (
            sentiment_signal * news_trust * abs(regression_move) * coverage_dampener
        )
        predicted_price = current_price * (1.0 + signed_percent_move)
        direction = "UP" if signed_percent_move >= 0 else "DOWN"
        # Confidence reflects conviction in the NEWS signal only; weak news
        # -> low confidence, regardless of how jumpy the price history is.
        confidence = logistic_confidence(
            3.0 * abs(sentiment_signal) * news_trust * coverage_dampener
        )

        explanation = self._build_explanation(
            ticker_score=ticker_score,
            overall_semantic_score=overall_semantic_score,
            signed_percent_move=signed_percent_move,
            scored_count=scored_count,
            supporting_articles=supporting_articles,
        )

        return TickerPrediction(
            ticker=ticker,
            direction=direction,
            predicted_percent_move=signed_percent_move * 100.0,
            predicted_price=predicted_price,
            current_price=current_price,
            confidence=confidence,
            semantic_score=ticker_score.semantic_score,
            explanation=explanation,
            supporting_articles=supporting_articles[:3],
        )

    def _predict_price_move(self, ticker: str) -> float:
        features = self.market_data_service.build_feature_frame(ticker)
        train_x = features[self.FEATURE_COLUMNS]
        train_y = features["target_pct_move"]

        model = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=4)
        model.fit(train_x, train_y)
        latest_features = train_x.iloc[[-1]]
        prediction = float(model.predict(latest_features)[0])
        return max(min(prediction, 0.25), -0.25)

    @staticmethod
    def _build_explanation(
        ticker_score: TickerScore,
        overall_semantic_score: float,
        signed_percent_move: float,
        scored_count: int,
        supporting_articles: list[RetrievedArticle],
    ) -> str:
        top_titles = ", ".join(
            a.article.title for a in supporting_articles[:2] if a.article.title
        )
        move_direction = "up" if signed_percent_move >= 0 else "down"
        coverage_note = (
            f"based on {scored_count} ticker-scoped article(s)"
            if scored_count
            else "no article contained ticker-scoped text to score, so the move is ~0"
        )
        return (
            f"{ticker_score.ticker} points {move_direction}: "
            f"ticker-scoped sentiment {ticker_score.sentiment_score:+.3f}, "
            f"relevance {ticker_score.semantic_score:.3f}, "
            f"event coverage {overall_semantic_score:.3f}, {coverage_note}. "
            f"Top supporting: {top_titles or 'no strong article titles available'}."
        )
