from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests

from app.config import Settings
from app.models.schemas import NewsArticle
from app.services.sentiment_service import SentimentService

logger = logging.getLogger(__name__)

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
        "is", "are", "was", "were", "be", "been", "being", "by", "with",
        "from", "as", "that", "this", "these", "those", "it", "its",
    }
)


class NewsDataError(RuntimeError):
    """Raised when NewsData.io returns an error we cannot recover from."""


class NewsCreditBudgetError(NewsDataError):
    """Raised when the configured credit budget would be exceeded."""


class NewsService:
    """Fetches news articles from NewsData.io and annotates them with local sentiment.

    The public surface (`fetch_news(query, tickers, limit)`) matches the legacy
    Alpha Vantage service, so the rest of the pipeline is unchanged.
    """

    _MARKET_PATH = "/market"
    _LATEST_PATH = "/latest"
    _DEFAULT_CATEGORY = "business"
    _TECH_KEYWORDS: frozenset[str] = frozenset(
        {"ai", "chip", "chips", "semiconductor", "semiconductors", "software",
         "cloud", "datacenter", "robotics", "cyber", "cybersecurity"}
    )

    def __init__(
        self,
        settings: Settings,
        sentiment_service: SentimentService | None = None,
    ) -> None:
        self.settings = settings
        self.sentiment_service = sentiment_service or SentimentService(settings)
        self._last_request_at: float | None = None
        self._request_timestamps: deque[float] = deque()
        self._daily_credits_used: int = 0
        self._daily_window_start: float = time.time()

        self._cache_dir: Path = settings.cache_dir / "newsdata"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_news(
        self,
        query: str,
        tickers: list[str],
        limit: int = 50,  # retained for signature compat; actual size from settings
    ) -> list[NewsArticle]:
        if not self.settings.news_api_key:
            raise NewsDataError(
                "NEWS_API_KEY is not set. Add it to your .env to use NewsData.io."
            )

        size = max(1, min(self.settings.news_articles_per_call, 50))
        normalized_tickers = sorted({ticker.upper() for ticker in tickers if ticker})

        ticker_articles = self._fetch_ticker_anchored(query, normalized_tickers, size)
        topic_articles = self._fetch_event_anchored(query, size)

        merged: dict[str, NewsArticle] = {}
        for article in topic_articles:
            article.source_type = "topic"
            merged[article.article_id] = article
        for article in ticker_articles:
            existing = merged.get(article.article_id)
            if existing is not None:
                existing.source_type = "both"
            else:
                article.source_type = "ticker"
                merged[article.article_id] = article

        articles = list(merged.values())
        self._populate_ticker_tags(articles, normalized_tickers)
        self.sentiment_service.annotate(articles)
        return articles

    def _fetch_ticker_anchored(
        self, query: str, tickers: list[str], size: int
    ) -> list[NewsArticle]:
        if not tickers:
            return []
        params = self._base_params(size=size)
        params["symbol"] = ",".join(tickers[:5])
        q = self._build_query_string(query)
        if q:
            params["q"] = q
        payload = self._call_with_fallback(
            primary_path=self._MARKET_PATH,
            primary_params=params,
            fallback_path=self._LATEST_PATH,
            fallback_params=self._latest_params_for_tickers(query, tickers, size),
        )
        return self._parse_articles(payload.get("results", []))

    def _fetch_event_anchored(self, query: str, size: int) -> list[NewsArticle]:
        params = self._base_params(size=size)
        q = self._build_query_string(query)
        if q:
            params["q"] = q
        params["category"] = self._category_for_query(query)
        payload = self._call(self._LATEST_PATH, params)
        return self._parse_articles(payload.get("results", []))

    def _latest_params_for_tickers(
        self, query: str, tickers: list[str], size: int
    ) -> dict[str, str]:
        params = self._base_params(size=size)
        ticker_clause = " OR ".join(tickers[:5])
        query_clause = self._build_query_string(query)
        if query_clause:
            params["q"] = f"({ticker_clause}) AND ({query_clause})"[:512]
        else:
            params["q"] = ticker_clause[:512]
        return params

    def _base_params(self, size: int) -> dict[str, str]:
        return {
            "apikey": self.settings.news_api_key,
            "language": "en",
            "removeduplicate": "1",
            "sort": "relevancy",
            "size": str(size),
        }

    def _category_for_query(self, query: str) -> str:
        tokens = {tok.lower() for tok in re.findall(r"[A-Za-z]+", query)}
        if tokens & self._TECH_KEYWORDS:
            return "business,technology"
        return self._DEFAULT_CATEGORY

    def _call_with_fallback(
        self,
        primary_path: str,
        primary_params: dict[str, str],
        fallback_path: str,
        fallback_params: dict[str, str],
    ) -> dict[str, Any]:
        try:
            return self._call(primary_path, primary_params)
        except NewsCreditBudgetError:
            raise
        except NewsDataError as exc:
            logger.warning(
                "NewsData %s failed (%s); falling back to %s",
                primary_path, exc, fallback_path,
            )
            return self._call(fallback_path, fallback_params)

    def _call(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        cache_key = self._cache_key(path, params)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        self._enforce_credit_budget()
        self._throttle()

        url = self.settings.news_api_base_url.rstrip("/") + path
        safe_params = {k: v for k, v in params.items() if k != "apikey"}
        logger.info("NewsData GET %s params=%s", path, safe_params)

        try:
            response = requests.get(
                url,
                params=params,
                timeout=self.settings.news_request_timeout_seconds,
            )
        except requests.RequestException as exc:
            raise NewsDataError(f"Network error calling {path}: {exc}") from exc
        finally:
            self._last_request_at = time.monotonic()

        self._record_credit()

        if response.status_code == 429:
            raise NewsCreditBudgetError(
                "NewsData.io rate limit hit (HTTP 429). "
                "Free tier: 30 credits / 15 min and 200 / day. Wait and retry."
            )
        if response.status_code >= 400:
            detail = self._extract_error(response)
            raise NewsDataError(
                f"NewsData.io {path} returned HTTP {response.status_code}: {detail}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise NewsDataError(f"Invalid JSON from {path}: {exc}") from exc

        if str(payload.get("status", "")).lower() != "success":
            detail = payload.get("results") or payload.get("message") or payload
            raise NewsDataError(f"NewsData.io {path} error: {detail}")

        self._store_cache(cache_key, payload)
        return payload

    def _enforce_credit_budget(self) -> None:
        now = time.time()

        # 15-minute rolling window
        window = float(self.settings.news_window_seconds)
        while self._request_timestamps and now - self._request_timestamps[0] > window:
            self._request_timestamps.popleft()
        if len(self._request_timestamps) >= self.settings.news_window_credit_limit:
            raise NewsCreditBudgetError(
                "NewsData.io 15-minute credit limit reached; wait before retrying."
            )

        # Daily budget (rolling 24h since first recorded call)
        if now - self._daily_window_start > 24 * 60 * 60:
            self._daily_window_start = now
            self._daily_credits_used = 0
        if self._daily_credits_used >= self.settings.news_daily_credit_limit:
            raise NewsCreditBudgetError(
                "NewsData.io daily credit budget exhausted. Try again tomorrow "
                "or upgrade the plan."
            )

    def _record_credit(self) -> None:
        now = time.time()
        self._request_timestamps.append(now)
        self._daily_credits_used += 1

    def _throttle(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        wait_seconds = self.settings.news_min_request_interval_seconds - elapsed
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    def _cache_key(self, path: str, params: dict[str, str]) -> str:
        # Exclude apikey so the cache is portable across keys.
        keyed = {k: v for k, v in params.items() if k != "apikey"}
        raw = path + "?" + "&".join(f"{k}={keyed[k]}" for k in sorted(keyed))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _load_cache(self, key: str) -> dict[str, Any] | None:
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        ttl = timedelta(minutes=self.settings.news_cache_ttl_minutes)
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        if age > ttl:
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _store_cache(self, key: str, payload: dict[str, Any]) -> None:
        path = self._cache_dir / f"{key}.json"
        try:
            path.write_text(json.dumps(payload))
        except OSError:
            pass

    def _parse_articles(self, results: list[dict[str, Any]]) -> list[NewsArticle]:
        articles: list[NewsArticle] = []
        for item in results:
            article_id = str(item.get("article_id") or item.get("link") or item.get("title") or "").strip()
            if not article_id:
                continue
            title = (item.get("title") or "").strip()
            description = (item.get("description") or item.get("content") or "").strip()
            if not title and not description:
                continue
            summary = description[:2000]

            tickers = self._extract_tickers(item)

            articles.append(
                NewsArticle(
                    article_id=article_id,
                    title=title,
                    summary=summary,
                    url=(item.get("link") or "").strip(),
                    time_published=self._parse_timestamp(item.get("pubDate")),
                    source=(item.get("source_name") or item.get("source_id") or "").strip(),
                    tickers=tickers,
                    # Sentiment is populated later by SentimentService.
                )
            )
        return articles

    def _populate_ticker_tags(
        self, articles: list[NewsArticle], requested_tickers: list[str]
    ) -> None:
        """Ensure requested tickers that appear in an article's text are tagged.

        NewsData.io free tier may not populate the `symbol` field; we do a
        safety pass so downstream scoring can distinguish articles that really
        mention a target ticker from articles retrieved only by free-text query.
        """
        if not requested_tickers:
            return
        patterns = {
            ticker: re.compile(rf"(?<![A-Z0-9]){re.escape(ticker)}(?![A-Z0-9])")
            for ticker in requested_tickers
        }
        for article in articles:
            text = f"{article.title} {article.summary}".upper()
            tagged: list[str] = list(article.tickers)
            for ticker, pattern in patterns.items():
                if pattern.search(text) and ticker not in tagged:
                    tagged.append(ticker)
            article.tickers = tagged

    @staticmethod
    def _extract_tickers(item: dict[str, Any]) -> list[str]:
        raw = item.get("symbol")
        tickers: list[str] = []
        if isinstance(raw, list):
            for value in raw:
                ticker = str(value).strip().upper()
                if ticker and ticker not in tickers:
                    tickers.append(ticker)
        elif isinstance(raw, str) and raw:
            for part in raw.split(","):
                ticker = part.strip().upper()
                if ticker and ticker not in tickers:
                    tickers.append(ticker)
        return tickers

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime | None:
        if not value:
            return None
        text = str(value).strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None

    @staticmethod
    def _extract_error(response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text[:200]
        if isinstance(payload, dict):
            results = payload.get("results")
            if isinstance(results, dict):
                return str(results.get("message") or results)
            return str(payload.get("message") or payload)
        return str(payload)

    @staticmethod
    def _build_query_string(query: str) -> str:
        text = query.strip()
        if not text:
            return ""
        tokens = [
            tok.lower()
            for tok in re.findall(r"[A-Za-z][A-Za-z\-]+", text)
            if tok.lower() not in _STOPWORDS and len(tok) > 2
        ]
        # Deduplicate while preserving order.
        seen: set[str] = set()
        keywords: list[str] = []
        for tok in tokens:
            if tok not in seen:
                seen.add(tok)
                keywords.append(tok)
        if not keywords:
            return ""
        # Use OR so the backend returns any article mentioning at least one
        # meaningful keyword; our semantic re-ranker tightens relevance locally.
        q = " OR ".join(keywords[:8])
        return q[:512]
