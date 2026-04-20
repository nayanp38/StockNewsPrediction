from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class Settings(BaseSettings):
    # Retained only so existing .env files with legacy keys still load.
    alpha_vantage_api_key: str = Field(default="", alias="ALPHAVANTAGE_API_KEY")
    news_api_key: str = Field(default="", alias="NEWS_API_KEY")

    # MarketAux -- the active news provider.
    marketaux_api_token: str = Field(default="", alias="NEW_NEWS")
    marketaux_base_url: str = "https://api.marketaux.com/v1"
    marketaux_articles_per_call: int = 3  # Free-plan cap; paid plans can raise.
    marketaux_language: str = "en"
    marketaux_cache_ttl_minutes: int = 60
    marketaux_min_request_interval_seconds: float = 1.0
    marketaux_request_timeout_seconds: float = 30.0
    marketaux_daily_call_limit: int = 100
    marketaux_minute_call_limit: int = 60
    # Optional historical window. Empty = MarketAux's full archive.
    marketaux_published_after: str = ""
    marketaux_published_before: str = ""
    # Hard recency cutoff. Articles older than this many days are dropped.
    # Applied both server-side (as a computed ``published_after`` when no
    # explicit override is set) and client-side (post-parse filter, so stale
    # cached payloads don't leak through). 0 = disabled.
    max_article_age_days: int = 365

    cache_dir: Path = Path("data/cache")

    # Retrieval / ranking.
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k_articles: int = 8
    cluster_count: int = 3
    similarity_floor: float = 0.50
    # Small weight so a clearly negative article ranks slightly below a
    # clearly positive one at the same cosine similarity. Direction still
    # comes from sentiment; this is only a tiebreaker in the supporting list.
    sentiment_relevance_weight: float = 0.05

    # Price-only regression fallback.
    regression_horizon_days: int = 5
    historical_period: str = "2y"
    benchmark_ticker: str = "SPY"

    # Ticker-scoped FinBERT.
    sentiment_enabled: bool = True
    sentiment_model_name: str = "ProsusAI/finbert"
    sentiment_batch_size: int = 8
    sentiment_max_tokens: int = 256
    # Ignore aliases shorter than this when matching -- avoids false hits on
    # 1-2 letter tokens that collide with common words.
    sentiment_min_alias_length: int = 3

    # Display thresholds for the signed FinBERT score in [-1, 1].
    sentiment_bearish_threshold: float = -0.35
    sentiment_somewhat_bearish_threshold: float = -0.15
    sentiment_somewhat_bullish_threshold: float = 0.15
    sentiment_bullish_threshold: float = 0.35

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
