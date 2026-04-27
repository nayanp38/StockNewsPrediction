from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.services.market_data_service import MarketDataService


@dataclass(frozen=True)
class DatasetBuildResult:
    rows: int
    output_path: Path


class BayesianDatasetService:
    """Builds a T+1 supervised dataset from historical event records."""

    def __init__(self, market_data_service: MarketDataService) -> None:
        self.market_data_service = market_data_service

    def build_from_jsonl(self, input_path: Path, output_path: Path) -> DatasetBuildResult:
        records = self._read_jsonl(input_path)
        rows: list[dict[str, Any]] = []
        for record in records:
            event_time = self._parse_event_time(record)
            if event_time is None:
                continue
            event_text = str(record.get("event_text") or "").strip()
            if not event_text:
                continue
            predictions = record.get("predictions") or []
            if not isinstance(predictions, list):
                continue
            for pred in predictions:
                if not isinstance(pred, dict):
                    continue
                ticker = str(pred.get("ticker") or "").strip().upper()
                if not ticker:
                    continue
                row = self._build_row(
                    ticker=ticker,
                    event_time=event_time,
                    event_text=event_text,
                    prediction_payload=pred,
                )
                if row is not None:
                    rows.append(row)

        dataset = pd.DataFrame(rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_path, index=False)
        return DatasetBuildResult(rows=len(dataset), output_path=output_path)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if isinstance(payload, dict):
                    entries.append(payload)
        return entries

    @staticmethod
    def _parse_event_time(record: dict[str, Any]) -> datetime | None:
        raw = record.get("event_time") or record.get("timestamp") or record.get("created_at")
        if raw is None:
            return None
        text = str(raw).strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None

    def _build_row(
        self,
        *,
        ticker: str,
        event_time: datetime,
        event_text: str,
        prediction_payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        features = self.market_data_service.build_feature_frame(ticker)
        history_at_event = features.loc[features.index <= pd.Timestamp(event_time)]
        if history_at_event.empty:
            return None
        event_row = history_at_event.iloc[-1]
        close_now = float(event_row["Close"])
        future_prices = features.loc[features.index > history_at_event.index[-1], "Close"]
        if future_prices.empty:
            return None
        close_t1 = float(future_prices.iloc[0])
        target_return_t1 = close_t1 / close_now - 1.0

        article_stats = self._article_features(prediction_payload.get("supporting_articles"))
        sentiment_score = float(prediction_payload.get("sentiment_score") or 0.0)
        semantic_score = float(prediction_payload.get("semantic_score") or 0.0)

        return {
            "event_time": event_time.isoformat(),
            "event_text": event_text,
            "ticker": ticker,
            "sector": "UNKNOWN",
            "return_1d": float(event_row["return_1d"]),
            "return_5d": float(event_row["return_5d"]),
            "return_20d": float(event_row["return_20d"]),
            "volatility_20d": float(event_row["volatility_20d"]),
            "volume_change_5d": float(event_row["volume_change_5d"]),
            "sma_10_gap": float(event_row["sma_10_gap"]),
            "sma_30_gap": float(event_row["sma_30_gap"]),
            "benchmark_return_5d": float(event_row["benchmark_return_5d"]),
            "relative_strength_5d": float(event_row["relative_strength_5d"]),
            "event_sentiment_mean": sentiment_score,
            "event_relevance_mean": semantic_score,
            "evidence_count": article_stats["evidence_count"],
            "evidence_effective_n": article_stats["effective_n"],
            "sentiment_dispersion": article_stats["sentiment_dispersion"],
            "source_topic_ratio": article_stats["source_topic_ratio"],
            "source_ticker_ratio": article_stats["source_ticker_ratio"],
            "source_both_ratio": article_stats["source_both_ratio"],
            "target_return_t1": target_return_t1,
        }

    @staticmethod
    def _article_features(raw_articles: Any) -> dict[str, float]:
        if not isinstance(raw_articles, list) or not raw_articles:
            return {
                "evidence_count": 0.0,
                "effective_n": 0.0,
                "sentiment_dispersion": 0.0,
                "source_topic_ratio": 0.0,
                "source_ticker_ratio": 0.0,
                "source_both_ratio": 0.0,
            }
        sentiments: list[float] = []
        weights: list[float] = []
        source_counts = {"topic": 0, "ticker": 0, "both": 0}
        for item in raw_articles:
            if not isinstance(item, dict):
                continue
            article = item.get("article") if isinstance(item.get("article"), dict) else {}
            sentiment = float(article.get("overall_sentiment_score") or 0.0)
            similarity = max(float(item.get("similarity_score") or 0.0), 0.0)
            source_type = str(article.get("source_type") or "topic")
            if source_type in source_counts:
                source_counts[source_type] += 1
            sentiments.append(sentiment)
            weights.append(similarity)
        if not sentiments:
            return {
                "evidence_count": 0.0,
                "effective_n": 0.0,
                "sentiment_dispersion": 0.0,
                "source_topic_ratio": 0.0,
                "source_ticker_ratio": 0.0,
                "source_both_ratio": 0.0,
            }
        n = float(len(sentiments))
        weight_sum = sum(weights)
        effective_n = (weight_sum * weight_sum / sum((w * w for w in weights))) if weight_sum > 0 else 0.0
        mean = sum(sentiments) / n
        variance = sum((x - mean) ** 2 for x in sentiments) / n
        return {
            "evidence_count": n,
            "effective_n": float(effective_n),
            "sentiment_dispersion": float(math.sqrt(variance)),
            "source_topic_ratio": source_counts["topic"] / n,
            "source_ticker_ratio": source_counts["ticker"] / n,
            "source_both_ratio": source_counts["both"] / n,
        }
