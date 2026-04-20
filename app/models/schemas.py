from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class NewsArticle(BaseModel):
    article_id: str
    title: str
    summary: str
    url: str = ""
    time_published: datetime | None = None
    source: str = ""
    overall_sentiment_score: float = 0.0
    overall_sentiment_label: str = "Neutral"
    # Tickers MarketAux tagged as entities on this article. Every ticker the
    # pipeline retrieves an article for is guaranteed to appear here.
    tickers: list[str] = Field(default_factory=list)
    # FinBERT signed score in [-1, 1] for the sentences mentioning each
    # ticker. Missing key = no ticker-scoped text to score.
    ticker_sentiment: dict[str, float] = Field(default_factory=dict)
    # MarketAux `match_score` per ticker, normalised to [0, 1].
    ticker_match_score: dict[str, float] = Field(default_factory=dict)
    # Symbol + MarketAux `entity.name` per ticker; used by SentimentService
    # to locate ticker-mentioning sentences in the summary.
    ticker_aliases: dict[str, list[str]] = Field(default_factory=dict)
    # Sentences from the FULL article body (MarketAux `highlights[]`) that
    # mention the ticker. FinBERT scores these directly.
    ticker_snippets: dict[str, list[str]] = Field(default_factory=dict)

    @property
    def combined_text(self) -> str:
        return f"{self.title.strip()} {self.summary.strip()}".strip()


class RetrievedArticle(BaseModel):
    article: NewsArticle
    similarity_score: float
    cluster_id: int | None = None
    ticker_relevance: dict[str, float] = Field(default_factory=dict)


class EventRequest(BaseModel):
    event_text: str
    tickers: list[str]
    top_k: int = 8


class TickerScore(BaseModel):
    ticker: str
    semantic_score: float
    sentiment_score: float
    combined_score: float


class TickerPrediction(BaseModel):
    ticker: str
    direction: Literal["UP", "DOWN"]
    predicted_percent_move: float
    predicted_price: float
    current_price: float
    confidence: float
    semantic_score: float
    explanation: str
    supporting_articles: list[RetrievedArticle]


class EventPredictionResponse(BaseModel):
    event_text: str
    overall_semantic_score: float
    predictions: list[TickerPrediction]
