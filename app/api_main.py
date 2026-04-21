from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.factory import build_pipeline
from app.models.schemas import EventRequest
from app.services.news_service import NewsCreditBudgetError, NewsDataError
from app.services.pipeline_service import PipelineService
from app.ticker_normalize import TickerValidationError, normalize_tickers

logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Event Predictor API", version="0.1.0")

_cors_raw = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
_cors_origins = [origin.strip() for origin in _cors_raw.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: PipelineService | None = None


def get_pipeline() -> PipelineService:
    """Lazy singleton so /api/health does not load FinBERT or embedding models."""
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


@app.get("/api/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/api/predict")
def predict(body: EventRequest) -> dict[str, Any]:
    try:
        tickers = normalize_tickers(body.tickers)
    except TickerValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    event_text = body.event_text.strip()
    if not event_text:
        raise HTTPException(status_code=422, detail="event_text is required.")

    request = EventRequest(event_text=event_text, tickers=tickers, top_k=body.top_k)

    try:
        response = get_pipeline().run(request)
    except NewsCreditBudgetError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except NewsDataError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("predict failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check server logs for details.",
        ) from None

    return response.model_dump(mode="json")
