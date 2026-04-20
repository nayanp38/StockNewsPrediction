"""Ticker-scoped FinBERT sentiment.

For each tagged ticker on an article we:

1. Assemble ticker-local text:
   * MarketAux `entities[].highlights[]` snippets (primary) -- sentences from
     the full article body that explicitly mention this ticker.
   * Alias-matched sentences from the title + summary (secondary) -- catches
     cases where MarketAux did not emit highlights.
2. Run FinBERT on that ticker-local text and write the signed score into
   ``article.ticker_sentiment[ticker]``.

If no ticker-local text is available we leave sentiment unset for that
ticker so downstream scoring excludes the article from the ticker's vote --
we deliberately refuse to score tone we cannot read, and we never carry
tone from unrelated entities into the target ticker.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Sequence

from app.config import Settings
from app.models.schemas import NewsArticle

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\u201c\u2018])")


class SentimentService:
    """FinBERT-based sentiment scorer, scoped per (article, ticker)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pipeline: Any | None = None
        self._initialized: bool = False
        self._load_failed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def annotate(self, articles: Sequence[NewsArticle]) -> None:
        """Fill ``article.ticker_sentiment`` in place for each (article, ticker).

        Safe to call with an empty list. No-ops silently when FinBERT cannot
        be loaded (offline, missing torch, etc.) so the rest of the pipeline
        keeps running with sentiment defaulted to absent.
        """
        if not self.settings.sentiment_enabled or not articles:
            return

        work: list[tuple[NewsArticle, str, str]] = []
        for article in articles:
            for ticker in article.tickers:
                aliases = self._resolve_aliases(article, ticker)
                scoped_text = self._build_ticker_corpus(article, ticker, aliases)
                if not scoped_text:
                    continue
                work.append((article, ticker, scoped_text))

        if not work:
            return

        pipeline = self._get_pipeline()
        if pipeline is None:
            return

        texts = [item[2] for item in work]
        batch_size = max(1, int(self.settings.sentiment_batch_size))
        try:
            outputs = pipeline(texts, batch_size=batch_size, truncation=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FinBERT inference failed: %s", exc)
            return

        if len(outputs) != len(work):
            logger.warning(
                "FinBERT returned %d predictions for %d inputs; skipping sentiment.",
                len(outputs),
                len(work),
            )
            return

        for (article, ticker, _), raw in zip(work, outputs):
            signed = self._signed_score(raw)
            if signed is None:
                continue
            article.ticker_sentiment[ticker] = signed

        for article in articles:
            if article.ticker_sentiment:
                article.overall_sentiment_score = float(
                    sum(article.ticker_sentiment.values())
                    / len(article.ticker_sentiment)
                )

    # ------------------------------------------------------------------
    # Text handling
    # ------------------------------------------------------------------
    def _resolve_aliases(self, article: NewsArticle, ticker: str) -> list[str]:
        aliases = list(article.ticker_aliases.get(ticker, []))
        if ticker not in aliases:
            aliases.append(ticker)
        min_len = max(1, int(self.settings.sentiment_min_alias_length))
        cleaned: list[str] = []
        for alias in aliases:
            trimmed = (alias or "").strip()
            if len(trimmed) < min_len:
                continue
            if trimmed not in cleaned:
                cleaned.append(trimmed)
        return cleaned

    def _build_ticker_corpus(
        self, article: NewsArticle, ticker: str, aliases: list[str]
    ) -> str:
        """Assemble ticker-scoped text: MarketAux snippets + alias sentences."""
        pieces: list[str] = []
        for raw_snippet in article.ticker_snippets.get(ticker, []):
            cleaned = (raw_snippet or "").replace("\n", " ").strip()
            if cleaned and cleaned not in pieces:
                pieces.append(cleaned)

        summary_sentences = self._alias_sentences(
            f"{article.title}. {article.summary}", aliases
        )
        for sentence in summary_sentences:
            if sentence not in pieces:
                pieces.append(sentence)

        if not pieces:
            return ""
        joined = " ".join(pieces)
        return joined[: self.settings.sentiment_max_tokens * 8]

    def _alias_sentences(self, text: str, aliases: Iterable[str]) -> list[str]:
        """Return sentences in ``text`` that mention any alias.

        Matching is word-boundary aware and case-insensitive so e.g.
        "Nvidia fell 17%" matches alias "NVIDIA Corporation" via token
        overlap, and "NVDA shares rose" matches alias "NVDA".
        """
        cleaned = (text or "").replace("\n", " ").strip()
        if not cleaned:
            return []
        patterns = [self._alias_pattern(alias) for alias in aliases]
        patterns = [p for p in patterns if p is not None]
        if not patterns:
            return []
        combined = re.compile("|".join(p.pattern for p in patterns), re.IGNORECASE)

        picked: list[str] = []
        for sentence in self._split_sentences(cleaned):
            stripped = sentence.strip()
            if stripped and combined.search(stripped):
                picked.append(stripped)
        return picked

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = _SENTENCE_SPLIT.split(text)
        return [p for p in (s.strip() for s in parts) if p]

    @staticmethod
    def _alias_pattern(alias: str) -> re.Pattern[str] | None:
        """Build a word-boundary regex from an alias, keeping only alphabetic
        tokens >=3 chars (drops suffixes like "Inc", "Co")."""
        trimmed = (alias or "").strip()
        if not trimmed:
            return None
        tokens = [
            tok
            for tok in re.findall(r"[A-Za-z][A-Za-z]+", trimmed)
            if len(tok) >= 3
        ]
        if not tokens:
            return None
        escaped = [re.escape(tok) for tok in tokens]
        return re.compile(r"\b(?:" + "|".join(escaped) + r")\b")

    # ------------------------------------------------------------------
    # FinBERT loading
    # ------------------------------------------------------------------
    def _get_pipeline(self) -> Any | None:
        if self._initialized:
            return self._pipeline
        self._initialized = True
        try:
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "transformers is not available; ticker sentiment disabled (%s)", exc
            )
            self._load_failed = True
            return None
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.settings.sentiment_model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.settings.sentiment_model_name
            )
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                top_k=None,
                truncation=True,
                max_length=self.settings.sentiment_max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load FinBERT '%s'; ticker sentiment disabled (%s)",
                self.settings.sentiment_model_name,
                exc,
            )
            self._pipeline = None
            self._load_failed = True
        return self._pipeline

    @staticmethod
    def _signed_score(raw: Any) -> float | None:
        """Collapse FinBERT's {positive, neutral, negative} probabilities to
        a single signed score in [-1, 1].

        With ``top_k=None`` FinBERT returns a list of ``{"label", "score"}``
        dicts. We compute ``positive - negative`` and drop the neutral mass.
        """
        if isinstance(raw, list):
            pos = neg = 0.0
            for item in raw:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label") or "").lower()
                score = float(item.get("score") or 0.0)
                if label.startswith("pos"):
                    pos = score
                elif label.startswith("neg"):
                    neg = score
            if pos == 0.0 and neg == 0.0:
                return None
            return max(-1.0, min(1.0, pos - neg))
        if isinstance(raw, dict):
            label = str(raw.get("label") or "").lower()
            score = float(raw.get("score") or 0.0)
            if label.startswith("pos"):
                return max(0.0, min(1.0, score))
            if label.startswith("neg"):
                return -max(0.0, min(1.0, score))
            return 0.0
        return None
