from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from app.config import Settings
from app.models.schemas import NewsArticle

logger = logging.getLogger(__name__)


class SentimentService:
    """Computes per-article sentiment locally using FinBERT.

    NewsData.io free tier does not expose per-article sentiment, so we replace
    Alpha Vantage's `overall_sentiment_score` / `overall_sentiment_label` /
    `ticker_sentiment` fields with scores derived from a local model.
    """

    _LABEL_ORDER: tuple[str, str, str] = ("positive", "negative", "neutral")

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._cache_dir: Path = settings.cache_dir / "sentiment"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline: Any | None = None

    def annotate(self, articles: list[NewsArticle]) -> None:
        """Fill sentiment fields on `articles` in place."""
        if not self.settings.sentiment_enabled or not articles:
            return

        pending: list[tuple[int, NewsArticle, Path]] = []
        for idx, article in enumerate(articles):
            cache_path = self._cache_path_for(article)
            cached = self._load_cached(cache_path)
            if cached is not None:
                self._apply_scores(article, cached)
                continue
            pending.append((idx, article, cache_path))

        if not pending:
            return

        pipeline = self._get_pipeline()
        if pipeline is None:
            # Model failed to load; leave defaults (neutral) and warn once.
            for _, article, _ in pending:
                article.overall_sentiment_score = 0.0
                article.overall_sentiment_label = "Neutral"
            return

        texts = [self._article_text(article) for _, article, _ in pending]
        batch_size = max(1, int(self.settings.sentiment_batch_size))
        results: list[list[dict[str, Any]]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            outputs = pipeline(
                batch,
                truncation=True,
                max_length=self.settings.sentiment_max_tokens,
                top_k=None,
            )
            # `top_k=None` returns a list of label dicts per input. Older
            # transformers versions return a flat list when the batch size is
            # one, so normalize to list[list[dict]].
            if outputs and isinstance(outputs[0], dict):
                outputs = [outputs]
            results.extend(outputs)

        for (_, article, cache_path), label_scores in zip(pending, results, strict=False):
            score, label = self._score_and_label_from_output(label_scores)
            scored = {"score": score, "label": label}
            self._apply_scores(article, scored)
            self._store_cached(cache_path, scored)

    def _apply_scores(self, article: NewsArticle, scored: dict[str, Any]) -> None:
        article.overall_sentiment_score = float(scored["score"])
        article.overall_sentiment_label = str(scored["label"])
        # FinBERT scores at the article level; propagate to every tagged ticker
        # so downstream scoring has a per-ticker sentiment to work with. The
        # `untagged_sentiment_weight` in ScoringService still dampens this for
        # tickers the article does not explicitly tag.
        if article.tickers:
            article.ticker_sentiment = {
                ticker: article.overall_sentiment_score for ticker in article.tickers
            }

    def _score_and_label_from_output(self, label_scores: list[dict[str, Any]]) -> tuple[float, str]:
        scores: dict[str, float] = {}
        for entry in label_scores:
            raw_label = str(entry.get("label", "")).lower()
            try:
                scores[raw_label] = float(entry.get("score", 0.0))
            except (TypeError, ValueError):
                scores[raw_label] = 0.0

        positive = scores.get("positive", 0.0)
        negative = scores.get("negative", 0.0)
        # Signed score in [-1, 1] so it matches Alpha Vantage's original scale.
        score = max(-1.0, min(1.0, positive - negative))
        return score, self._score_to_label(score)

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

    def _get_pipeline(self) -> Any | None:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            logger.warning("transformers not installed; skipping sentiment scoring")
            return None
        try:
            self._pipeline = hf_pipeline(
                task="text-classification",
                model=self.settings.sentiment_model_name,
                top_k=None,
                truncation=True,
            )
        except Exception as exc:  # noqa: BLE001 - model download/load is best-effort
            logger.warning("Failed to load sentiment model %s: %s", self.settings.sentiment_model_name, exc)
            self._pipeline = None
        return self._pipeline

    def _cache_path_for(self, article: NewsArticle) -> Path:
        digest = hashlib.sha1(
            f"{self.settings.sentiment_model_name}|{article.article_id}|{self._article_text(article)}".encode("utf-8")
        ).hexdigest()
        return self._cache_dir / f"{digest}.json"

    @staticmethod
    def _article_text(article: NewsArticle) -> str:
        text = f"{article.title}\n{article.summary}".strip()
        return text or article.title or article.summary

    @staticmethod
    def _load_cached(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _store_cached(path: Path, payload: dict[str, Any]) -> None:
        try:
            path.write_text(json.dumps(payload))
        except OSError:
            pass
