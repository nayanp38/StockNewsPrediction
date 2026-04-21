from __future__ import annotations

from app.config import get_settings
from app.services.embedding_service import EmbeddingService
from app.services.market_data_service import MarketDataService
from app.services.news_service import NewsService
from app.services.pipeline_service import PipelineService
from app.services.prediction_service import PredictionService
from app.services.retrieval_service import RetrievalService
from app.services.scoring_service import ScoringService
from app.services.sentiment_service import SentimentService


def build_pipeline() -> PipelineService:
    settings = get_settings()
    embedding_service = EmbeddingService(settings)
    sentiment_service = SentimentService(settings)
    news_service = NewsService(settings, sentiment_service=sentiment_service)
    market_data_service = MarketDataService(settings)
    retrieval_service = RetrievalService(settings, embedding_service)
    scoring_service = ScoringService(settings)
    prediction_service = PredictionService(settings, market_data_service)
    return PipelineService(news_service, retrieval_service, scoring_service, prediction_service)
