from __future__ import annotations

import pandas as pd
import yfinance as yf

from app.config import Settings


class MarketDataService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def fetch_history(self, ticker: str, period: str | None = None) -> pd.DataFrame:
        history = yf.Ticker(ticker).history(period=period or self.settings.historical_period, auto_adjust=False)
        if history.empty:
            raise ValueError(f"No market history found for {ticker}")
        return history

    def latest_close(self, ticker: str) -> float:
        history = self.fetch_history(ticker, period="1mo")
        return float(history["Close"].dropna().iloc[-1])

    def build_feature_frame(self, ticker: str) -> pd.DataFrame:
        price_history = self.fetch_history(ticker)
        benchmark_history = self.fetch_history(self.settings.benchmark_ticker)
        features = self._engineer_features(price_history, benchmark_history)
        horizon = self.settings.regression_horizon_days
        features["target_pct_move"] = features["Close"].shift(-horizon) / features["Close"] - 1.0
        features = features.dropna()
        if features.empty:
            raise ValueError(f"Insufficient data to build features for {ticker}")
        return features

    @staticmethod
    def _engineer_features(price_history: pd.DataFrame, benchmark_history: pd.DataFrame) -> pd.DataFrame:
        df = price_history.copy()
        bench = benchmark_history[["Close"]].rename(columns={"Close": "BenchmarkClose"})
        df = df.join(bench, how="inner")

        df["return_1d"] = df["Close"].pct_change(1)
        df["return_5d"] = df["Close"].pct_change(5)
        df["return_20d"] = df["Close"].pct_change(20)
        df["volatility_20d"] = df["return_1d"].rolling(20).std()
        df["volume_change_5d"] = df["Volume"] / df["Volume"].rolling(5).mean() - 1.0
        df["sma_10_gap"] = df["Close"] / df["Close"].rolling(10).mean() - 1.0
        df["sma_30_gap"] = df["Close"] / df["Close"].rolling(30).mean() - 1.0
        df["benchmark_return_5d"] = df["BenchmarkClose"].pct_change(5)
        df["relative_strength_5d"] = df["return_5d"] - df["benchmark_return_5d"]
        return df
