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
    tickers: list[str] = Field(default_factory=list)
    ticker_sentiment: dict[str, float] = Field(default_factory=dict)
    source_type: Literal["topic", "ticker", "both"] = "topic"

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
