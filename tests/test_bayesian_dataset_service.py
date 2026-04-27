from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.services.bayesian_dataset_service import BayesianDatasetService


class _FakeMarketDataService:
    def build_feature_frame(self, ticker: str) -> pd.DataFrame:
        index = pd.to_datetime(["2026-01-14", "2026-01-15", "2026-01-16"])
        return pd.DataFrame(
            {
                "Close": [100.0, 101.0, 99.0],
                "return_1d": [0.0, 0.01, -0.02],
                "return_5d": [0.0, 0.02, -0.01],
                "return_20d": [0.0, 0.04, -0.01],
                "volatility_20d": [0.1, 0.11, 0.12],
                "volume_change_5d": [0.0, 0.1, 0.2],
                "sma_10_gap": [0.0, 0.01, -0.01],
                "sma_30_gap": [0.0, 0.02, -0.01],
                "benchmark_return_5d": [0.0, 0.005, 0.004],
                "relative_strength_5d": [0.0, 0.015, -0.014],
            },
            index=index,
        )


def test_build_dataset_from_jsonl(tmp_path: Path) -> None:
    service = BayesianDatasetService(_FakeMarketDataService())  # type: ignore[arg-type]
    event_file = tmp_path / "events.jsonl"
    payload = {
        "event_time": "2026-01-15T14:30:00Z",
        "event_text": "Export controls on AI chips",
        "predictions": [
            {
                "ticker": "NVDA",
                "semantic_score": 0.4,
                "sentiment_score": -0.2,
                "supporting_articles": [
                    {
                        "similarity_score": 0.5,
                        "article": {"overall_sentiment_score": -0.4, "source_type": "topic"},
                    }
                ],
            }
        ],
    }
    event_file.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    output = tmp_path / "dataset.csv"
    result = service.build_from_jsonl(event_file, output)
    assert result.output_path == output
    assert output.exists()
    frame = pd.read_csv(output)
    assert "target_return_t1" in frame.columns
