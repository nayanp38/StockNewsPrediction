from __future__ import annotations

from app.models.schemas import EventPredictionResponse, EventRequest
from app.services.news_service import NewsService
from app.services.prediction_service import PredictionService
from app.services.retrieval_service import RetrievalService
from app.services.scoring_service import ScoringService


class PipelineService:
    def __init__(
        self,
        news_service: NewsService,
        retrieval_service: RetrievalService,
        scoring_service: ScoringService,
        prediction_service: PredictionService,
    ) -> None:
        self.news_service = news_service
        self.retrieval_service = retrieval_service
        self.scoring_service = scoring_service
        self.prediction_service = prediction_service

    def run(self, request: EventRequest) -> EventPredictionResponse:
        articles = self.news_service.fetch_news(
            query=request.event_text,
            tickers=request.tickers,
            limit=max(request.top_k * 5, 25),
        )
        retrieved_articles = self.retrieval_service.retrieve(
            query=request.event_text,
            articles=articles,
            tickers=request.tickers,
            top_k=request.top_k,
        )

        overall_semantic_score = self.scoring_service.compute_overall_semantic_score(retrieved_articles)
        ticker_scores = self.scoring_service.compute_ticker_scores(retrieved_articles, request.tickers)
        predictions = []
        for ticker_score in ticker_scores:
            ticker_articles = sorted(
                retrieved_articles,
                key=lambda article: article.ticker_relevance.get(ticker_score.ticker, article.similarity_score),
                reverse=True,
            )
            predictions.append(
                self.prediction_service.predict_for_ticker(
                    ticker_score=ticker_score,
                    overall_semantic_score=overall_semantic_score,
                    supporting_articles=ticker_articles,
                )
            )

        return EventPredictionResponse(
            event_text=request.event_text,
            overall_semantic_score=overall_semantic_score,
            predictions=predictions,
        )
