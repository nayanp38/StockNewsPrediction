from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.services.bayesian_model_service import BayesianModelService


def test_train_save_load_and_infer(tmp_path: Path) -> None:
    model = BayesianModelService()
    rows = []
    for idx in range(30):
        rows.append(
            {
                "ticker": "AAA" if idx % 2 == 0 else "BBB",
                "sector": "TECH",
                "return_1d": 0.001 * idx,
                "return_5d": 0.002 * idx,
                "return_20d": 0.003 * idx,
                "volatility_20d": 0.01 + 0.0001 * idx,
                "volume_change_5d": 0.02,
                "sma_10_gap": 0.01,
                "sma_30_gap": 0.02,
                "benchmark_return_5d": 0.005,
                "relative_strength_5d": 0.001,
                "event_sentiment_mean": 0.2 if idx % 2 == 0 else -0.2,
                "event_relevance_mean": 0.4,
                "evidence_count": 4.0,
                "evidence_effective_n": 2.4,
                "sentiment_dispersion": 0.3,
                "source_topic_ratio": 0.5,
                "source_ticker_ratio": 0.25,
                "source_both_ratio": 0.25,
                "target_return_t1": 0.01 if idx % 2 == 0 else -0.01,
            }
        )
    data_path = tmp_path / "train.csv"
    artifact_path = tmp_path / "artifact.json"
    pd.DataFrame(rows).to_csv(data_path, index=False)

    model.train_and_save(data_path, artifact_path)
    artifact = model.load_artifact(artifact_path)
    stats = model.infer(
        artifact=artifact,
        feature_map={k: float(rows[0][k]) for k in model.ALL_COLUMNS},
        ticker="AAA",
        sector="TECH",
    )
    assert 0.0 <= stats.prob_up <= 1.0
    assert stats.lower_95 <= stats.upper_95
    assert isinstance(json.loads(artifact_path.read_text()), dict)
