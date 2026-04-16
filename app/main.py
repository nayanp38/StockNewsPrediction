from __future__ import annotations

import json

import typer

from app.config import get_settings
from app.models.schemas import EventRequest
from app.services.embedding_service import EmbeddingService
from app.services.market_data_service import MarketDataService
from app.services.news_service import NewsService
from app.services.pipeline_service import PipelineService
from app.services.prediction_service import PredictionService
from app.services.retrieval_service import RetrievalService
from app.services.scoring_service import ScoringService

cli = typer.Typer(help="Predict stock impact from a hypothetical future event.")


def build_pipeline() -> PipelineService:
    settings = get_settings()
    embedding_service = EmbeddingService(settings)
    news_service = NewsService(settings)
    market_data_service = MarketDataService(settings)
    retrieval_service = RetrievalService(settings, embedding_service)
    scoring_service = ScoringService()
    prediction_service = PredictionService(settings, market_data_service)
    return PipelineService(news_service, retrieval_service, scoring_service, prediction_service)


@cli.command("predict-event")
def predict_event(
    event_text: str = typer.Argument(..., help="Hypothetical future event description."),
    tickers: list[str] = typer.Argument(..., help="One or more stock tickers."),
    top_k: int = typer.Option(8, min=1, help="Number of supporting articles to keep."),
) -> None:
    pipeline = build_pipeline()
    response = pipeline.run(
        EventRequest(
            event_text=event_text,
            tickers=tickers,
            top_k=top_k,
        )
    )
    typer.echo(json.dumps(response.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    cli()
