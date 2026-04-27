from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge


@dataclass(frozen=True)
class PosteriorStats:
    expected_return: float
    stddev: float
    prob_up: float
    lower_80: float
    upper_80: float
    lower_95: float
    upper_95: float


class BayesianModelService:
    PRIOR_COLUMNS = [
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
    LIKELIHOOD_COLUMNS = [
        "event_sentiment_mean",
        "event_relevance_mean",
        "evidence_count",
        "evidence_effective_n",
        "sentiment_dispersion",
        "source_topic_ratio",
        "source_ticker_ratio",
        "source_both_ratio",
    ]
    ALL_COLUMNS = PRIOR_COLUMNS + LIKELIHOOD_COLUMNS

    def train_and_save(self, dataset_path: Path, artifact_path: Path) -> None:
        data = pd.read_csv(dataset_path)
        required = set(self.ALL_COLUMNS + ["target_return_t1", "ticker", "sector"])
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")
        data = data.dropna(subset=self.ALL_COLUMNS + ["target_return_t1"])
        if data.empty:
            raise ValueError("Dataset is empty after dropping NaN values.")

        means = data[self.ALL_COLUMNS].mean()
        stds = data[self.ALL_COLUMNS].std(ddof=0).replace(0.0, 1.0)
        x = ((data[self.ALL_COLUMNS] - means) / stds).to_numpy()
        y = data["target_return_t1"].to_numpy(dtype=float)

        model = BayesianRidge(fit_intercept=True)
        model.fit(x, y)
        predictions = model.predict(x)
        residual = y - predictions

        global_alpha = float(residual.mean())
        sector_alpha = data.assign(_res=residual).groupby("sector")["_res"].mean().to_dict()
        ticker_alpha = data.assign(_res=residual).groupby("ticker")["_res"].mean().to_dict()

        artifact = {
            "feature_columns": self.ALL_COLUMNS,
            "means": {k: float(v) for k, v in means.to_dict().items()},
            "stds": {k: float(v) for k, v in stds.to_dict().items()},
            "coef": [float(c) for c in model.coef_],
            "intercept": float(model.intercept_),
            "alpha_precision": float(model.alpha_),
            "lambda_precision": float(model.lambda_),
            "sigma": float(np.sqrt(max(np.var(residual), 1e-8))),
            "global_alpha": global_alpha,
            "sector_alpha": {str(k): float(v) for k, v in sector_alpha.items()},
            "ticker_alpha": {str(k): float(v) for k, v in ticker_alpha.items()},
        }
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    def load_artifact(self, artifact_path: Path) -> dict:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        required = {
            "feature_columns",
            "means",
            "stds",
            "coef",
            "intercept",
            "sigma",
            "global_alpha",
            "sector_alpha",
            "ticker_alpha",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"Artifact missing keys: {missing}")
        return payload

    def infer(self, artifact: dict, feature_map: dict[str, float], ticker: str, sector: str = "UNKNOWN") -> PosteriorStats:
        cols = artifact["feature_columns"]
        means = artifact["means"]
        stds = artifact["stds"]
        coef = np.array(artifact["coef"], dtype=float)
        intercept = float(artifact["intercept"])
        sigma = max(float(artifact["sigma"]), 1e-6)

        vec = []
        for col in cols:
            value = float(feature_map.get(col, 0.0))
            centered = (value - float(means[col])) / max(float(stds[col]), 1e-8)
            vec.append(centered)
        x = np.array(vec, dtype=float)
        base_mu = intercept + float(np.dot(coef, x))

        global_alpha = float(artifact["global_alpha"])
        sector_alpha = float(artifact["sector_alpha"].get(sector, global_alpha))
        ticker_alpha = float(artifact["ticker_alpha"].get(ticker, sector_alpha))
        # Empirical-Bayes shrinkage: ticker to sector to global.
        hierarchy_offset = 0.7 * ticker_alpha + 0.2 * sector_alpha + 0.1 * global_alpha

        mu = base_mu + hierarchy_offset
        prob_up = 1.0 - self._normal_cdf(0.0, mu, sigma)
        lower_80, upper_80 = mu - 1.2816 * sigma, mu + 1.2816 * sigma
        lower_95, upper_95 = mu - 1.96 * sigma, mu + 1.96 * sigma
        return PosteriorStats(
            expected_return=mu,
            stddev=sigma,
            prob_up=prob_up,
            lower_80=lower_80,
            upper_80=upper_80,
            lower_95=lower_95,
            upper_95=upper_95,
        )

    @staticmethod
    def _normal_cdf(value: float, mean: float, stddev: float) -> float:
        z = (value - mean) / max(stddev, 1e-12)
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
