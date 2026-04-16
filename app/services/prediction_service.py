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
        current_price = self.market_data_service.latest_close(ticker_score.ticker)
        regression_move = self._predict_price_move(ticker_score.ticker)
        directional_signal = 0.65 * ticker_score.combined_score + 0.35 * overall_semantic_score

        magnitude = abs(regression_move) * (0.6 + min(abs(directional_signal), 1.0))
        signed_percent_move = magnitude if directional_signal >= 0 else -magnitude
        predicted_price = current_price * (1.0 + signed_percent_move)
        direction = "UP" if signed_percent_move >= 0 else "DOWN"
        confidence = logistic_confidence(abs(directional_signal))
        explanation = self._build_explanation(ticker_score, overall_semantic_score, signed_percent_move, supporting_articles)

        return TickerPrediction(
            ticker=ticker_score.ticker,
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
        predicted_move: float,
        supporting_articles: list[RetrievedArticle],
    ) -> str:
        top_titles = ", ".join(article.article.title for article in supporting_articles[:2] if article.article.title)
        move_direction = "positive" if predicted_move >= 0 else "negative"
        return (
            f"{ticker_score.ticker} shows a {move_direction} event signal because the ticker semantic score "
            f"({ticker_score.semantic_score:.3f}) and aggregated event relevance ({overall_semantic_score:.3f}) "
            f"point in the same direction. Top supporting coverage: {top_titles or 'no strong article titles available'}."
        )
