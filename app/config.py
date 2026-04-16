from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class Settings(BaseSettings):
    alpha_vantage_api_key: str = Field(default="", alias="ALPHAVANTAGE_API_KEY")
    alpha_vantage_base_url: str = "https://www.alphavantage.co/query"
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

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
