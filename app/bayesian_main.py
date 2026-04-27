from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from app.config import get_settings
from app.services.bayesian_dataset_service import BayesianDatasetService
from app.services.bayesian_model_service import BayesianModelService
from app.services.market_data_service import MarketDataService

cli = typer.Typer(help="Bayesian T+1 tooling.")


@cli.command("build-dataset")
def build_dataset(
    input_jsonl: Path = typer.Option(..., help="Historical event responses JSONL."),
    output_csv: Path = typer.Option(None, help="Output CSV path."),
) -> None:
    settings = get_settings()
    out = output_csv or Path(settings.bayesian_train_dataset_path)
    service = BayesianDatasetService(MarketDataService(settings))
    result = service.build_from_jsonl(input_jsonl, out)
    typer.echo(f"Wrote {result.rows} rows to {result.output_path}")


@cli.command("train")
def train(
    dataset_csv: Path = typer.Option(None, help="Training CSV path."),
    artifact_path: Path = typer.Option(None, help="Output artifact path."),
) -> None:
    settings = get_settings()
    dataset = dataset_csv or Path(settings.bayesian_train_dataset_path)
    artifact = artifact_path or Path(settings.bayesian_artifact_path)
    model_service = BayesianModelService()
    model_service.train_and_save(dataset, artifact)
    typer.echo(f"Saved Bayesian artifact to {artifact}")


@cli.command("evaluate")
def evaluate(
    dataset_csv: Path = typer.Option(None, help="Evaluation CSV path."),
    artifact_path: Path = typer.Option(None, help="Model artifact path."),
) -> None:
    settings = get_settings()
    dataset = pd.read_csv(dataset_csv or Path(settings.bayesian_train_dataset_path))
    model = BayesianModelService()
    artifact = model.load_artifact(artifact_path or Path(settings.bayesian_artifact_path))
    y = dataset["target_return_t1"].to_numpy(dtype=float)
    probs: list[float] = []
    preds: list[float] = []
    for row in dataset.to_dict(orient="records"):
        stats = model.infer(
            artifact=artifact,
            feature_map={k: float(row.get(k, 0.0)) for k in model.ALL_COLUMNS},
            ticker=str(row.get("ticker", "")),
            sector=str(row.get("sector", "UNKNOWN")),
        )
        probs.append(stats.prob_up)
        preds.append(stats.expected_return)
    y_up = (y > 0.0).astype(float)
    brier = float(((pd.Series(probs) - y_up) ** 2).mean())
    hit = float(((pd.Series(probs) >= 0.5) == (y_up == 1.0)).mean())
    mae = float((pd.Series(preds) - y).abs().mean())
    baseline_probs = (0.5 + 0.5 * dataset["event_sentiment_mean"].clip(-1.0, 1.0)).astype(float)
    baseline_brier = float(((baseline_probs - y_up) ** 2).mean())
    baseline_hit = float(((baseline_probs >= 0.5) == (y_up == 1.0)).mean())
    typer.echo(f"Brier: {brier:.5f}")
    typer.echo(f"Directional hit-rate: {hit:.3f}")
    typer.echo(f"MAE(return): {mae:.5f}")
    typer.echo(f"Baseline Brier (sentiment-only): {baseline_brier:.5f}")
    typer.echo(f"Baseline hit-rate (sentiment-only): {baseline_hit:.3f}")


if __name__ == "__main__":
    cli()
