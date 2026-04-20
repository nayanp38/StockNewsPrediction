from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from app.config import Settings
from app.models.schemas import NewsArticle

# Forward reference to avoid a hard import cycle. SentimentService duck-types
# against this via its `annotate(list[NewsArticle])` method.
SentimentAnnotator = Any  # pragma: no cover

logger = logging.getLogger(__name__)

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
        "is", "are", "was", "were", "be", "been", "being", "by", "with",
        "from", "as", "that", "this", "these", "those", "it", "its",
    }
)


class NewsDataError(RuntimeError):
    """Raised when MarketAux returns an error we cannot recover from."""


class NewsCreditBudgetError(NewsDataError):
    """Raised when the configured call budget would be exceeded."""


class NewsService:
    """Fetches articles from MarketAux, one call per requested ticker.

    Each call constrains BOTH ``symbols=TICKER`` and ``search=<query>``, so
    every article returned is guaranteed to (a) tag the ticker and (b) match
    the event query. We harvest the ticker's aliases and body-level highlight
    snippets so ``SentimentService`` can run FinBERT on ticker-local text.

    MarketAux's aggregate sentiment numbers are deliberately ignored -- they
    mix tone across every entity in the article, which leaks unrelated
    sentiment into the target ticker's score.
    """

    _NEWS_PATH = "/news/all"

    def __init__(
        self,
        settings: Settings,
        sentiment_annotator: SentimentAnnotator | None = None,
    ) -> None:
        self.settings = settings
        self._sentiment_annotator = sentiment_annotator
        self._last_request_at: float | None = None
        self._minute_timestamps: deque[float] = deque()
        self._daily_calls_used: int = 0
        self._daily_window_start: float = time.time()

        self._cache_dir: Path = settings.cache_dir / "marketaux"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch_news(
        self,
        query: str,
        tickers: list[str],
        limit: int = 50,  # retained for signature compat; unused
    ) -> list[NewsArticle]:
        if not self.settings.marketaux_api_token:
            raise NewsDataError(
                "NEW_NEWS is not set. Add your MarketAux API token to .env to "
                "enable news retrieval."
            )

        size = max(1, min(self.settings.marketaux_articles_per_call, 50))
        normalized_tickers = sorted({t.upper() for t in tickers if t})

        merged: dict[str, NewsArticle] = {}
        for ticker in normalized_tickers:
            for article in self._fetch_for_ticker(query=query, ticker=ticker, size=size):
                if not self._is_fresh_enough(article):
                    continue
                existing = merged.get(article.article_id)
                if existing is None:
                    merged[article.article_id] = article
                else:
                    # Same article returned for multiple tickers -- merge
                    # ticker-specific metadata without overwriting.
                    self._merge_article(existing, article)

        articles = list(merged.values())
        if self._sentiment_annotator is not None:
            self._sentiment_annotator.annotate(articles)
        for article in articles:
            self._apply_labels(article)
        return articles

    # ------------------------------------------------------------------
    # HTTP fetch
    # ------------------------------------------------------------------
    def _fetch_for_ticker(
        self, query: str, ticker: str, size: int
    ) -> list[NewsArticle]:
        params = self._base_params(size=size)
        params["symbols"] = ticker
        params["filter_entities"] = "true"
        params["must_have_entities"] = "true"
        search_clause = self._build_search_clause(query)
        if search_clause:
            params["search"] = search_clause
        payload = self._call(self._NEWS_PATH, params)
        return self._parse_articles(payload.get("data", []))

    def _base_params(self, size: int) -> dict[str, str]:
        params: dict[str, str] = {
            "api_token": self.settings.marketaux_api_token,
            "language": self.settings.marketaux_language,
            "limit": str(size),
        }
        if self.settings.marketaux_published_after:
            params["published_after"] = self.settings.marketaux_published_after
        elif self.settings.max_article_age_days > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(
                days=self.settings.max_article_age_days
            )
            params["published_after"] = cutoff.date().isoformat()
        if self.settings.marketaux_published_before:
            params["published_before"] = self.settings.marketaux_published_before
        return params

    def _call(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        cache_key = self._cache_key(path, params)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        self._enforce_call_budget()
        self._throttle()

        url = self.settings.marketaux_base_url.rstrip("/") + path
        safe_params = {k: v for k, v in params.items() if k != "api_token"}
        logger.info("MarketAux GET %s params=%s", path, safe_params)

        try:
            response = requests.get(
                url,
                params=params,
                timeout=self.settings.marketaux_request_timeout_seconds,
            )
        except requests.RequestException as exc:
            raise NewsDataError(f"Network error calling {path}: {exc}") from exc
        finally:
            self._last_request_at = time.monotonic()

        self._record_call()

        if response.status_code == 429:
            raise NewsCreditBudgetError(
                "MarketAux rate limit hit (HTTP 429). Wait ~1 minute and retry."
            )
        if response.status_code == 402:
            raise NewsCreditBudgetError(
                "MarketAux usage limit reached (HTTP 402). Daily free-plan "
                "quota is exhausted; try again tomorrow or upgrade the plan."
            )
        if response.status_code >= 400:
            detail = self._extract_error(response)
            raise NewsDataError(
                f"MarketAux {path} returned HTTP {response.status_code}: {detail}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise NewsDataError(f"Invalid JSON from {path}: {exc}") from exc

        if isinstance(payload, dict) and "error" in payload:
            error = payload["error"]
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise NewsDataError(f"MarketAux {path} error: {message}")

        self._store_cache(cache_key, payload)
        return payload

    # ------------------------------------------------------------------
    # Throttling / budgeting
    # ------------------------------------------------------------------
    def _enforce_call_budget(self) -> None:
        now = time.time()

        while self._minute_timestamps and now - self._minute_timestamps[0] > 60.0:
            self._minute_timestamps.popleft()
        if len(self._minute_timestamps) >= self.settings.marketaux_minute_call_limit:
            raise NewsCreditBudgetError(
                "MarketAux per-minute call limit reached; wait before retrying."
            )

        if now - self._daily_window_start > 24 * 60 * 60:
            self._daily_window_start = now
            self._daily_calls_used = 0
        if self._daily_calls_used >= self.settings.marketaux_daily_call_limit:
            raise NewsCreditBudgetError(
                "MarketAux daily call budget exhausted. Try again tomorrow or "
                "upgrade the plan."
            )

    def _record_call(self) -> None:
        self._minute_timestamps.append(time.time())
        self._daily_calls_used += 1

    def _throttle(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        wait_seconds = self.settings.marketaux_min_request_interval_seconds - elapsed
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    # ------------------------------------------------------------------
    # Disk cache
    # ------------------------------------------------------------------
    def _cache_key(self, path: str, params: dict[str, str]) -> str:
        keyed = {k: v for k, v in params.items() if k != "api_token"}
        raw = path + "?" + "&".join(f"{k}={keyed[k]}" for k in sorted(keyed))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _load_cache(self, key: str) -> dict[str, Any] | None:
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        ttl = timedelta(minutes=self.settings.marketaux_cache_ttl_minutes)
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

    # ------------------------------------------------------------------
    # Article parsing / merging
    # ------------------------------------------------------------------
    def _parse_articles(self, results: list[dict[str, Any]]) -> list[NewsArticle]:
        articles: list[NewsArticle] = []
        for item in results:
            article_id = str(item.get("uuid") or item.get("url") or "").strip()
            if not article_id:
                continue
            title = (item.get("title") or "").strip()
            description = (item.get("description") or item.get("snippet") or "").strip()
            if not title and not description:
                continue

            tickers, match_scores, aliases, snippets = self._extract_entity_info(
                item.get("entities")
            )
            articles.append(
                NewsArticle(
                    article_id=article_id,
                    title=title,
                    summary=description[:2000],
                    url=(item.get("url") or "").strip(),
                    time_published=self._parse_timestamp(item.get("published_at")),
                    source=(item.get("source") or "").strip(),
                    tickers=tickers,
                    ticker_match_score=match_scores,
                    ticker_aliases=aliases,
                    ticker_snippets=snippets,
                )
            )
        return articles

    def _is_fresh_enough(self, article: NewsArticle) -> bool:
        """Drop articles older than ``max_article_age_days``. Articles without
        a parseable timestamp are kept (we'd rather err on recall than reject
        real hits over metadata quirks)."""
        max_age_days = self.settings.max_article_age_days
        if max_age_days <= 0 or article.time_published is None:
            return True
        published = article.time_published
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        return published >= cutoff

    @staticmethod
    def _merge_article(existing: NewsArticle, incoming: NewsArticle) -> None:
        """Union ticker-specific metadata when the same article is returned for
        multiple requested tickers."""
        for ticker in incoming.tickers:
            if ticker not in existing.tickers:
                existing.tickers.append(ticker)
        for ticker, match in incoming.ticker_match_score.items():
            existing.ticker_match_score.setdefault(ticker, match)
        for ticker, aliases in incoming.ticker_aliases.items():
            merged = list(existing.ticker_aliases.get(ticker, []))
            for alias in aliases:
                if alias and alias not in merged:
                    merged.append(alias)
            existing.ticker_aliases[ticker] = merged
        for ticker, snippets in incoming.ticker_snippets.items():
            merged_snips = list(existing.ticker_snippets.get(ticker, []))
            for snippet in snippets:
                if snippet and snippet not in merged_snips:
                    merged_snips.append(snippet)
            existing.ticker_snippets[ticker] = merged_snips

    def _apply_labels(self, article: NewsArticle) -> None:
        article.overall_sentiment_label = self._score_to_label(
            article.overall_sentiment_score
        )

    def _score_to_label(self, score: float) -> str:
        if score <= self.settings.sentiment_bearish_threshold:
            return "Bearish"
        if score <= self.settings.sentiment_somewhat_bearish_threshold:
            return "Somewhat-Bearish"
        if score < self.settings.sentiment_somewhat_bullish_threshold:
            return "Neutral"
        if score < self.settings.sentiment_bullish_threshold:
            return "Somewhat-Bullish"
        return "Bullish"

    @staticmethod
    def _extract_entity_info(
        raw: Any,
    ) -> tuple[
        list[str],
        dict[str, float],
        dict[str, list[str]],
        dict[str, list[str]],
    ]:
        """Return (tickers, match_scores, aliases, snippets) from MarketAux's
        ``entities`` array. Sentiment numbers on entities/highlights are
        ignored -- we only carry TEXT so FinBERT can score it locally."""
        tickers: list[str] = []
        match_scores: dict[str, float] = {}
        aliases: dict[str, list[str]] = {}
        snippets: dict[str, list[str]] = {}
        if not isinstance(raw, list):
            return tickers, match_scores, aliases, snippets
        for entity in raw:
            if not isinstance(entity, dict):
                continue
            entity_type = str(entity.get("type") or "").lower()
            if entity_type and entity_type not in {"equity", "stock", "index"}:
                continue
            raw_symbol = str(entity.get("symbol") or "").strip().upper()
            if not raw_symbol:
                continue
            symbol = raw_symbol.split(".", 1)[0]
            if not symbol:
                continue
            if symbol not in tickers:
                tickers.append(symbol)

            match = entity.get("match_score")
            if match is not None:
                try:
                    raw_match = float(match)
                except (TypeError, ValueError):
                    raw_match = 0.0
                normalised = raw_match / 100.0 if raw_match > 1.0 else raw_match
                match_scores[symbol] = max(0.0, min(1.0, normalised))

            ticker_aliases = aliases.setdefault(symbol, [])
            if symbol not in ticker_aliases:
                ticker_aliases.append(symbol)
            name = str(entity.get("name") or "").strip()
            if name and name not in ticker_aliases:
                ticker_aliases.append(name)

            ticker_snippets = snippets.setdefault(symbol, [])
            highlights = entity.get("highlights")
            if isinstance(highlights, list):
                for snippet in highlights:
                    if not isinstance(snippet, dict):
                        continue
                    text = str(snippet.get("highlight") or "").strip()
                    if text and text not in ticker_snippets:
                        ticker_snippets.append(text)
        return tickers, match_scores, aliases, snippets

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime | None:
        if not value:
            return None
        text = str(value).strip()
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            pass
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def _extract_error(response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text[:200]
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or error)
            return str(payload.get("message") or payload)
        return str(payload)

    @staticmethod
    def _build_search_clause(query: str) -> str:
        """Turn a free-text query into a MarketAux ``search`` clause.

        We OR together the meaningful keywords so MarketAux surfaces any
        article mentioning at least one; semantic re-ranking tightens
        relevance locally via MiniLM.
        """
        text = (query or "").strip()
        if not text:
            return ""
        tokens = [
            tok.lower()
            for tok in re.findall(r"[A-Za-z][A-Za-z\-]+", text)
            if tok.lower() not in _STOPWORDS and len(tok) > 2
        ]
        seen: set[str] = set()
        keywords: list[str] = []
        for tok in tokens:
            if tok not in seen:
                seen.add(tok)
                keywords.append(tok)
        if not keywords:
            return ""
        clause = " | ".join(keywords[:8])
        return clause[:512]
