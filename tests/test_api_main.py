from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api_main import app, get_pipeline
from app.models.schemas import EventPredictionResponse, TickerPrediction


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_predict_rejects_empty_tickers(client: TestClient) -> None:
    response = client.post(
        "/api/predict",
        json={"event_text": "Test event", "tickers": [], "top_k": 8},
    )
    assert response.status_code == 422
    assert "ticker" in response.json()["detail"].lower()


def test_predict_rejects_blank_event(client: TestClient) -> None:
    response = client.post(
        "/api/predict",
        json={"event_text": "   ", "tickers": ["NVDA"], "top_k": 8},
    )
    assert response.status_code == 422


def test_predict_success_monkeypatched(client: TestClient) -> None:
    fake_prediction = TickerPrediction(
        ticker="NVDA",
        direction="UP",
        predicted_percent_move=1.5,
        predicted_price=101.0,
        current_price=100.0,
        confidence=0.7,
        semantic_score=0.5,
        sentiment_score=0.1,
        combined_score=0.3,
        explanation="Synthetic test row.",
        supporting_articles=[],
    )
    fake_response = EventPredictionResponse(
        event_text="Chip news",
        overall_semantic_score=0.4,
        predictions=[fake_prediction],
    )
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = fake_response

    app.dependency_overrides.clear()
    original = get_pipeline

    def _fake_pipeline():
        return mock_pipeline

    # Replace lazy singleton path: patch module-level get_pipeline used by route
    import app.api_main as api_main

    api_main.get_pipeline = _fake_pipeline  # type: ignore[method-assign]
    try:
        response = client.post(
            "/api/predict",
            json={"event_text": "Chip export rules", "tickers": ["NVDA"], "top_k": 8},
        )
    finally:
        api_main.get_pipeline = original  # type: ignore[method-assign]

    assert response.status_code == 200
    data = response.json()
    assert data["event_text"] == "Chip news"
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["ticker"] == "NVDA"
    assert data["predictions"][0]["sentiment_score"] == 0.1
    mock_pipeline.run.assert_called_once()
