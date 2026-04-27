from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

from app.config import Settings
from app.models.schemas import RetrievedArticle, TickerPrediction, TickerScore
from app.services.bayesian_model_service import BayesianModelService
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
        self.bayesian_service = BayesianModelService()
        self._artifact: dict | None = None

    def predict_for_ticker(
        self,
        ticker_score: TickerScore,
        overall_semantic_score: float,
        supporting_articles: list[RetrievedArticle],
    ) -> TickerPrediction:
        current_price = self.market_data_service.latest_close(ticker_score.ticker)
        regression_move = self._predict_price_move(ticker_score.ticker)
        bayesian = self._predict_bayesian_posterior(
            ticker=ticker_score.ticker,
            ticker_score=ticker_score,
            overall_semantic_score=overall_semantic_score,
            supporting_articles=supporting_articles,
        )
        if bayesian is not None:
            expected = max(min(bayesian.expected_return, 0.25), -0.25)
            predicted_price = current_price * (1.0 + expected)
            direction = "UP" if bayesian.prob_up >= self.settings.bayesian_direction_threshold else "DOWN"
            confidence = max(bayesian.prob_up, 1.0 - bayesian.prob_up)
            explanation = (
                f"Bayesian posterior for {ticker_score.ticker}: "
                f"E[r_t+1]={expected:.4f}, P(up)={bayesian.prob_up:.3f}, "
                f"80% CI=[{bayesian.lower_80:.4f}, {bayesian.upper_80:.4f}]."
            )
            return TickerPrediction(
                ticker=ticker_score.ticker,
                direction=direction,
                predicted_percent_move=expected * 100.0,
                predicted_price=predicted_price,
                current_price=current_price,
                confidence=confidence,
                semantic_score=ticker_score.semantic_score,
                sentiment_score=ticker_score.sentiment_score,
                combined_score=ticker_score.combined_score,
                explanation=explanation,
                supporting_articles=supporting_articles[:3],
            )

        # Direction is driven by news sentiment (signed). Semantic relevance scales
        # how much we trust that news signal. Base drift from the price-only
        # regression is only used when news evidence is too weak to be informative.
        sentiment_signal = float(ticker_score.sentiment_score)
        semantic_strength = max(0.0, min(1.0, float(ticker_score.semantic_score)))
        event_relevance = max(0.0, min(1.0, float(overall_semantic_score)))
        news_trust = 0.7 * semantic_strength + 0.3 * event_relevance

        news_directional_signal = sentiment_signal * news_trust
        directional_signal = news_directional_signal
        if abs(news_directional_signal) < 0.05:
            directional_signal = news_directional_signal + (1.0 - news_trust) * regression_move

        magnitude = abs(regression_move) * (0.6 + min(abs(sentiment_signal), 1.0) + 0.4 * semantic_strength)
        signed_percent_move = magnitude if directional_signal >= 0 else -magnitude
        predicted_price = current_price * (1.0 + signed_percent_move)
        direction = "UP" if signed_percent_move >= 0 else "DOWN"
        confidence = logistic_confidence(abs(directional_signal) + 0.5 * news_trust)
        explanation = self._build_explanation(ticker_score, overall_semantic_score, signed_percent_move, supporting_articles)

        return TickerPrediction(
            ticker=ticker_score.ticker,
            direction=direction,
            predicted_percent_move=signed_percent_move * 100.0,
            predicted_price=predicted_price,
            current_price=current_price,
            confidence=confidence,
            semantic_score=ticker_score.semantic_score,
            sentiment_score=ticker_score.sentiment_score,
            combined_score=ticker_score.combined_score,
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

    def _predict_bayesian_posterior(
        self,
        *,
        ticker: str,
        ticker_score: TickerScore,
        overall_semantic_score: float,
        supporting_articles: list[RetrievedArticle],
    ):
        if not self.settings.bayesian_enabled:
            return None
        artifact = self._get_artifact()
        if artifact is None:
            if self.settings.bayesian_shadow_mode:
                return None
            return None
        features = self.market_data_service.build_feature_frame(ticker).iloc[-1]
        source_counts = {"topic": 0.0, "ticker": 0.0, "both": 0.0}
        sentiments: list[float] = []
        for item in supporting_articles:
            sentiments.append(float(item.article.overall_sentiment_score))
            source = item.article.source_type
            if source in source_counts:
                source_counts[source] += 1.0
        n = max(float(len(supporting_articles)), 1.0)
        mean_sentiment = sum(sentiments) / n if sentiments else float(ticker_score.sentiment_score)
        dispersion = (
            (sum((s - mean_sentiment) ** 2 for s in sentiments) / n) ** 0.5 if sentiments else 0.0
        )
        relevance_values = [max(a.similarity_score, 0.0) for a in supporting_articles]
        weight_sum = sum(relevance_values)
        effective_n = (weight_sum * weight_sum / sum((w * w for w in relevance_values))) if weight_sum > 0 else 0.0
        feature_map = {
            "return_1d": float(features["return_1d"]),
            "return_5d": float(features["return_5d"]),
            "return_20d": float(features["return_20d"]),
            "volatility_20d": float(features["volatility_20d"]),
            "volume_change_5d": float(features["volume_change_5d"]),
            "sma_10_gap": float(features["sma_10_gap"]),
            "sma_30_gap": float(features["sma_30_gap"]),
            "benchmark_return_5d": float(features["benchmark_return_5d"]),
            "relative_strength_5d": float(features["relative_strength_5d"]),
            "event_sentiment_mean": mean_sentiment,
            "event_relevance_mean": float(overall_semantic_score),
            "evidence_count": float(len(supporting_articles)),
            "evidence_effective_n": float(effective_n),
            "sentiment_dispersion": float(dispersion),
            "source_topic_ratio": source_counts["topic"] / n,
            "source_ticker_ratio": source_counts["ticker"] / n,
            "source_both_ratio": source_counts["both"] / n,
        }
        return self.bayesian_service.infer(artifact, feature_map=feature_map, ticker=ticker, sector="UNKNOWN")

    def _get_artifact(self) -> dict | None:
        if self._artifact is not None:
            return self._artifact
        path = Path(self.settings.bayesian_artifact_path)
        if not path.exists():
            return None
        try:
            self._artifact = self.bayesian_service.load_artifact(path)
            return self._artifact
        except Exception:
            return None

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
