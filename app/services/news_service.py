from __future__ import annotations

from datetime import datetime
import re
import time
from typing import Any

import requests

from app.config import Settings
from app.models.schemas import NewsArticle


class NewsService:
    _TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
        "economy_monetary": (
            "fed",
            "federal reserve",
            "interest rate",
            "rate cut",
            "rate hike",
            "monetary policy",
            "quantitative easing",
            "inflation",
            "cpi",
            "pce",
        ),
        "economy_macro": (
            "macro",
            "gdp",
            "recession",
            "unemployment",
            "jobs report",
            "consumer spending",
            "economic growth",
            "economic slowdown",
        ),
        "financial_markets": (
            "stocks",
            "equities",
            "bond",
            "bonds",
            "treasury",
            "yield",
            "volatility",
            "risk off",
            "risk-on",
            "market rally",
        ),
        "finance": (
            "bank",
            "banks",
            "banking",
            "lending",
            "loan",
            "credit",
            "financial sector",
            "capital markets",
        ),
        "technology": (
            "ai",
            "artificial intelligence",
            "chip",
            "chips",
            "semiconductor",
            "software",
            "cloud",
            "data center",
        ),
        "mergers_and_acquisitions": (
            "acquisition",
            "acquire",
            "takeover",
            "merger",
            "buyout",
            "m&a",
        ),
        "earnings": (
            "earnings",
            "eps",
            "guidance",
            "revenue",
            "quarterly results",
            "q1",
            "q2",
            "q3",
            "q4",
        ),
        "ipo": (
            "ipo",
            "initial public offering",
            "go public",
            "listing",
        ),
        "energy_transportation": (
            "oil",
            "gas",
            "energy",
            "opec",
            "shipping",
            "airline",
            "transportation",
            "freight",
        ),
        "manufacturing": (
            "manufacturing",
            "factory",
            "industrial output",
            "production",
            "supply chain",
        ),
        "real_estate": (
            "real estate",
            "housing",
            "home sales",
            "mortgage",
            "commercial real estate",
        ),
        "retail_wholesale": (
            "retail",
            "wholesale",
            "consumer demand",
            "ecommerce",
            "same-store sales",
        ),
        "life_sciences": (
            "biotech",
            "pharma",
            "drug trial",
            "fda",
            "healthcare",
            "medical device",
        ),
        "blockchain": (
            "blockchain",
            "bitcoin",
            "ethereum",
            "crypto",
            "token",
            "web3",
        ),
    }
    _DEFAULT_TOPICS = ("economy_macro", "financial_markets", "finance")

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._min_request_interval_seconds = 1.1
        self._last_request_at: float | None = None

    def fetch_news(self, query: str, tickers: list[str], limit: int = 50) -> list[NewsArticle]:
        topic_articles = self._fetch_by_topic(query, limit=limit)
        ticker_articles = self._fetch_by_tickers(tickers, limit=limit)

        deduped: dict[str, NewsArticle] = {}
        for article in topic_articles:
            article.source_type = "topic"
            deduped[article.article_id] = article
        for article in ticker_articles:
            existing = deduped.get(article.article_id)
            if existing is not None:
                existing.source_type = "both"
            else:
                article.source_type = "ticker"
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
        # Alpha Vantage free tier allows 1 request/second bursts.
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            wait_seconds = self._min_request_interval_seconds - elapsed
            if wait_seconds > 0:
                time.sleep(wait_seconds)

        response = requests.get(self.settings.alpha_vantage_base_url, params=params, timeout=30)
        self._last_request_at = time.monotonic()
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
        tokens = re.findall(r"[a-z0-9]+", query.lower())
        if not tokens:
            return ",".join(NewsService._DEFAULT_TOPICS)

        token_set = set(tokens)
        normalized_text = " ".join(tokens)
        scored_topics: list[tuple[str, int]] = []
        for topic, keywords in NewsService._TOPIC_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if " " in keyword:
                    if keyword in normalized_text:
                        score += 1
                elif keyword in token_set:
                    score += 1
            if score > 0:
                scored_topics.append((topic, score))

        if not scored_topics:
            return ",".join(NewsService._DEFAULT_TOPICS)

        scored_topics.sort(key=lambda item: (-item[1], item[0]))
        top_topics = [topic for topic, _ in scored_topics[:3]]
        return ",".join(top_topics)

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
