from __future__ import annotations

import json
from datetime import datetime

import typer

from app.factory import build_pipeline
from app.models.schemas import EventPredictionResponse, EventRequest
from app.ticker_normalize import TickerValidationError, normalize_tickers

cli = typer.Typer(help="Predict stock impact from a hypothetical future event.")


def _normalize_tickers_cli(raw_tickers: list[str]) -> list[str]:
    try:
        return normalize_tickers(raw_tickers)
    except TickerValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _format_timestamp(value: datetime | None) -> str:
    if value is None:
        return "N/A"
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _format_report(response: EventPredictionResponse) -> str:
    lines: list[str] = []
    lines.append("=" * 84)
    lines.append("STOCK EVENT PREDICTION REPORT")
    lines.append("=" * 84)
    lines.append(f"Event: {response.event_text}")
    lines.append(f"Overall Semantic Score: {response.overall_semantic_score:.3f}")
    lines.append(f"Total Tickers Evaluated: {len(response.predictions)}")

    for index, prediction in enumerate(response.predictions, start=1):
        lines.append("")
        lines.append("-" * 84)
        lines.append(f"{index}. Ticker: {prediction.ticker}")
        lines.append("-" * 84)
        lines.append(f"Direction: {prediction.direction}")
        lines.append(f"Predicted Percent Move: {prediction.predicted_percent_move:+.2f}%")
        lines.append(f"Current Price: ${prediction.current_price:.2f}")
        lines.append(f"Predicted Price: ${prediction.predicted_price:.2f}")
        lines.append(f"Confidence: {prediction.confidence:.3f}")
        lines.append(f"Ticker Semantic Score: {prediction.semantic_score:.3f}")
        lines.append(f"Sentiment Score: {prediction.sentiment_score:.3f}")
        lines.append(f"Combined Score: {prediction.combined_score:.3f}")
        lines.append(f"Explanation: {prediction.explanation}")
        lines.append("")
        lines.append(f"Supporting Articles ({len(prediction.supporting_articles)} shown):")

        for article_index, retrieved in enumerate(prediction.supporting_articles, start=1):
            article = retrieved.article
            ticker_relevance = retrieved.ticker_relevance.get(prediction.ticker, retrieved.similarity_score)
            ticker_sentiment = article.ticker_sentiment.get(prediction.ticker, article.overall_sentiment_score)
            lines.append(f"  {article_index}) {article.title or 'Untitled'}")
            lines.append(f"     Source: {article.source or 'Unknown'} | Published: {_format_timestamp(article.time_published)}")
            lines.append(
                "     Scores: "
                f"similarity={retrieved.similarity_score:.3f}, "
                f"ticker_relevance={ticker_relevance:.3f}, "
                f"ticker_sentiment={ticker_sentiment:.3f}, "
                f"overall_sentiment={article.overall_sentiment_score:.3f}"
            )
            lines.append(f"     Sentiment Label: {article.overall_sentiment_label}")
            lines.append(f"     Tickers Mentioned: {', '.join(article.tickers) if article.tickers else 'None'}")
            lines.append(f"     URL: {article.url or 'N/A'}")
            lines.append(f"     Summary: {article.summary or 'N/A'}")
            lines.append("")

    return "\n".join(lines).rstrip()


@cli.command("predict-event")
def predict_event(
    event_text: str = typer.Argument(..., help="Hypothetical future event description."),
    tickers: list[str] = typer.Argument(..., help="One or more stock tickers."),
    top_k: int = typer.Option(8, min=1, help="Number of supporting articles to keep."),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON instead of formatted report."),
) -> None:
    normalized_tickers = _normalize_tickers_cli(tickers)
    pipeline = build_pipeline()
    response = pipeline.run(
        EventRequest(
            event_text=event_text,
            tickers=normalized_tickers,
            top_k=top_k,
        )
    )
    if json_output:
        typer.echo(json.dumps(response.model_dump(mode="json"), indent=2))
        return
    typer.echo(_format_report(response))


if __name__ == "__main__":
    cli()
