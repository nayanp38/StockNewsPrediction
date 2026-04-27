from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class Settings(BaseSettings):
    # Legacy Alpha Vantage key; retained only so existing .env files load without error.
    alpha_vantage_api_key: str = Field(default="", alias="ALPHAVANTAGE_API_KEY")
    alpha_vantage_base_url: str = "https://www.alphavantage.co/query"

    # NewsData.io is the active news provider.
    news_api_key: str = Field(default="", alias="NEWS_API_KEY")
    news_api_base_url: str = "https://newsdata.io/api/1"
    news_articles_per_call: int = 10  # Free plan cap; paid can go up to 50.
    news_cache_ttl_minutes: int = 30
    news_daily_credit_limit: int = 200  # Free plan: 200 credits/day.
    news_window_credit_limit: int = 30  # Free plan: 30 credits / 15 min.
    news_window_seconds: int = 15 * 60
    news_min_request_interval_seconds: float = 1.1
    news_request_timeout_seconds: float = 30.0

    cache_dir: Path = Path("data/cache")
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k_articles: int = 8
    cluster_count: int = 3
    regression_horizon_days: int = 5
    historical_period: str = "2y"
    benchmark_ticker: str = "SPY"
    similarity_floor: float = 0.30
    ticker_only_penalty: float = 0.08
    direct_mention_bonus: float = 0.05
    sentiment_relevance_weight: float = 0.05
    untagged_sentiment_weight: float = 0.25

    # Local sentiment scoring (replaces Alpha Vantage's sentiment fields).
    sentiment_enabled: bool = True
    sentiment_model_name: str = "ProsusAI/finbert"
    sentiment_batch_size: int = 8
    sentiment_max_tokens: int = 256
    sentiment_bearish_threshold: float = -0.35
    sentiment_somewhat_bearish_threshold: float = -0.15
    sentiment_somewhat_bullish_threshold: float = 0.15
    sentiment_bullish_threshold: float = 0.35

    # Bayesian T+1 serving/training controls.
    bayesian_enabled: bool = False
    bayesian_shadow_mode: bool = True
    bayesian_direction_threshold: float = 0.5
    bayesian_artifact_path: str = "data/models/bayesian_t1_artifact.json"
    bayesian_train_dataset_path: str = "data/training/events_t1_dataset.csv"
    bayesian_min_rows: int = 200

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
