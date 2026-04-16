from __future__ import annotations

from datetime import datetime
from typing import Any

import requests

from app.config import Settings
from app.models.schemas import NewsArticle


class NewsService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def fetch_news(self, query: str, tickers: list[str], limit: int = 50) -> list[NewsArticle]:
        articles = self._fetch_by_topic(query, limit=limit)
        ticker_articles = self._fetch_by_tickers(tickers, limit=limit)
        deduped: dict[str, NewsArticle] = {}
        for article in articles + ticker_articles:
            deduped[article.article_id] = article
        return list(deduped.values())

    def _fetch_by_topic(self, query: str, limit: int) -> list[NewsArticle]:
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": self.settings.alpha_vantage_api_key,
            "limit": str(limit),
            "sort": "LATEST",
            "topics": self._topics_from_query(query),
        }
        payload = self._get(params)
        return self._parse_articles(payload.get("feed", []))

    def _fetch_by_tickers(self, tickers: list[str], limit: int) -> list[NewsArticle]:
        if not tickers:
            return []
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": self.settings.alpha_vantage_api_key,
            "limit": str(limit),
            "sort": "LATEST",
            "tickers": ",".join(sorted({ticker.upper() for ticker in tickers})),
        }
        payload = self._get(params)
        return self._parse_articles(payload.get("feed", []))

    def _get(self, params: dict[str, str]) -> dict[str, Any]:
        response = requests.get(self.settings.alpha_vantage_base_url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if "Information" in payload or "Note" in payload:
            raise RuntimeError(payload.get("Information") or payload.get("Note"))
        return payload

    def _parse_articles(self, feed_items: list[dict[str, Any]]) -> list[NewsArticle]:
        articles: list[NewsArticle] = []
        for item in feed_items:
            ticker_sentiment: dict[str, float] = {}
            tickers: list[str] = []
            for sentiment in item.get("ticker_sentiment", []):
                ticker = sentiment.get("ticker", "").upper()
                if not ticker:
                    continue
                tickers.append(ticker)
                try:
                    ticker_sentiment[ticker] = float(sentiment.get("ticker_sentiment_score", 0.0))
                except (TypeError, ValueError):
                    ticker_sentiment[ticker] = 0.0

            articles.append(
                NewsArticle(
                    article_id=item.get("url", "") or item.get("title", ""),
                    title=item.get("title", "").strip(),
                    summary=item.get("summary", "").strip(),
                    url=item.get("url", ""),
                    time_published=self._parse_timestamp(item.get("time_published")),
                    source=item.get("source", ""),
                    overall_sentiment_score=self._safe_float(item.get("overall_sentiment_score")),
                    overall_sentiment_label=item.get("overall_sentiment_label", "Neutral"),
                    tickers=tickers,
                    ticker_sentiment=ticker_sentiment,
                )
            )
        return [article for article in articles if article.title or article.summary]

    @staticmethod
    def _topics_from_query(query: str) -> str:
        words = [word.strip(".,!?").lower() for word in query.split()]
        curated = [word for word in words if len(word) > 4][:5]
        return ",".join(curated) if curated else "economy"

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.strptime(value, "%Y%m%dT%H%M%S")
        except ValueError:
            return None

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
