"""Command-line interface: ingest / train / backtest.

trading-ml ingest   --symbols AAPL,MSFT --provider yfinance
trading-ml train    --model technical
trading-ml backtest --model technical --symbols AAPL
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from trading_ml.config import load_config, load_model_config
from trading_ml.models.base import BaseModel

app = typer.Typer(add_completion=False, help="ML models for day-trading equities.")
console = Console()


def _parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _read_base_frames(symbols: list[str], base_timeframe: str) -> dict[str, object]:
    from trading_ml.data import storage

    frames = {}
    for sym in symbols:
        df = storage.read_ohlcv(sym, base_timeframe)
        if not df.empty:
            frames[sym] = df
    return frames


@app.command()
def ingest(
    symbols: str = typer.Option(None, help="Comma-separated symbols (default: config)."),
    provider: str = typer.Option(None, help="yfinance | ibkr (default: config)."),
    timeframes: str = typer.Option(None, help="Comma-separated timeframes (default: config)."),
    no_resume: bool = typer.Option(False, help="Ignore checkpoints and re-download."),
) -> None:
    """Download and store multi-timeframe OHLCV into the partitioned Parquet store."""
    from trading_ml.data.ingestion import ingest as run_ingest

    reports = run_ingest(
        symbols=_parse_csv(symbols),
        timeframes=_parse_csv(timeframes),
        provider_name=provider,
        resume=not no_resume,
    )
    table = Table(title="Ingestion report")
    for col in ("symbol", "timeframe", "rows", "chunks", "errors"):
        table.add_column(col)
    for r in reports:
        table.add_row(r.symbol, r.timeframe, str(r.rows), str(r.chunks), str(len(r.errors)))
    console.print(table)


@app.command()
def train(
    model: str = typer.Option("technical", help="Model to train."),
    symbols: str = typer.Option(None, help="Comma-separated symbols (default: config)."),
    baseline: bool = typer.Option(True, help="Also train the LightGBM baseline."),
    ensemble: bool = typer.Option(False, help="Train the net+baseline ensemble instead."),
) -> None:
    """Train Model 1 (cross-timeframe attention) and, optionally, the baseline."""
    cfg = load_config()
    mcfg = load_model_config(model)
    syms = _parse_csv(symbols) or cfg.data.symbols
    base_tf = mcfg["data"]["base_timeframe"]

    frames = _read_base_frames(syms, base_tf)
    if not frames:
        console.print(f"[red]No stored data for {syms} @ {base_tf}. Run `ingest` first.[/red]")
        raise typer.Exit(1)

    art_dir = Path(cfg.paths.artifacts_dir)
    if ensemble:
        from trading_ml.models.technical.ensemble import EnsembleTechnicalModel

        console.print(f"Training [bold]{model}[/bold] ENSEMBLE on {list(frames)} @ {base_tf} ...")
        ens = EnsembleTechnicalModel(mcfg).fit(frames, logger=_mlflow_logger(cfg))
        ens.save(art_dir / f"{model}_ensemble")
        console.print(f"[green]Saved[/green] {art_dir / f'{model}_ensemble'}")
        return

    from trading_ml.models.technical.module import TechnicalModel

    console.print(f"Training [bold]{model}[/bold] on {list(frames)} @ {base_tf} ...")
    net_model = TechnicalModel(mcfg).fit(frames, logger=_mlflow_logger(cfg))
    net_model.save(art_dir / f"{model}.pt")
    console.print(f"[green]Saved[/green] {art_dir / f'{model}.pt'}")

    if baseline:
        from trading_ml.models.technical.baseline import TechnicalBaseline

        console.print("Training LightGBM baseline ...")
        bl = TechnicalBaseline(mcfg).fit(frames)
        bl.save(art_dir / f"{model}_baseline.joblib")
        console.print(f"[green]Saved[/green] {art_dir / f'{model}_baseline.joblib'}")


@app.command()
def backtest(
    model: str = typer.Option("technical", help="Trained model to backtest."),
    symbols: str = typer.Option(None, help="Comma-separated symbols (default: config)."),
    min_prob: float = typer.Option(0.45, help="Minimum signal probability to trade."),
    use_baseline: bool = typer.Option(False, help="Backtest the LightGBM baseline instead."),
    use_ensemble: bool = typer.Option(False, help="Backtest the net+baseline ensemble."),
) -> None:
    """Backtest a trained model with the risk sizer and print metrics."""
    cfg = load_config()
    mcfg = load_model_config(model)
    syms = _parse_csv(symbols) or cfg.data.symbols
    base_tf = mcfg["data"]["base_timeframe"]
    frames = _read_base_frames(syms, base_tf)
    if not frames:
        console.print(f"[red]No stored data for {syms} @ {base_tf}. Run `ingest` first.[/red]")
        raise typer.Exit(1)

    art_dir = Path(cfg.paths.artifacts_dir)
    trained: BaseModel
    if use_ensemble:
        from trading_ml.models.technical.ensemble import EnsembleTechnicalModel

        trained = EnsembleTechnicalModel.load(art_dir / f"{model}_ensemble")
    elif use_baseline:
        from trading_ml.models.technical.baseline import TechnicalBaseline

        trained = TechnicalBaseline.load(art_dir / f"{model}_baseline.joblib")
    else:
        from trading_ml.models.technical.module import TechnicalModel

        trained = TechnicalModel.load(art_dir / f"{model}.pt")

    signals = trained.predict(frames, min_prob=min_prob)
    console.print(f"Generated {len(signals)} signals.")

    from trading_ml.backtest import run_backtest
    from trading_ml.features.indicators import fill_missing_bars
    from trading_ml.features.multi_timeframe import to_close_labeled
    from trading_ml.models.risk import RiskModel
    from trading_ml.strategy import StrategyEngine

    # Backtest prices must be close-labeled to match signal timestamps.
    price_frames = {
        sym: to_close_labeled(fill_missing_bars(df, base_tf), base_tf) for sym, df in frames.items()
    }
    combined = StrategyEngine(min_prob=min_prob).combine(signals)
    result = run_backtest(
        combined,
        price_frames,
        risk_model=RiskModel(),
        atr_window=mcfg["labeling"]["atr_window"],
        min_prob=min_prob,
    )

    table = Table(title=f"Backtest — {model}{' (baseline)' if use_baseline else ''}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in result.metrics.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)


@app.command()
def evaluate(
    model: str = typer.Option("technical", help="Model config to evaluate."),
    symbols: str = typer.Option(None, help="Comma-separated symbols (default: config)."),
) -> None:
    """Walk-forward cross-validation: honest out-of-sample classification metrics."""
    cfg = load_config()
    mcfg = load_model_config(model)
    syms = _parse_csv(symbols) or cfg.data.symbols
    base_tf = mcfg["data"]["base_timeframe"]
    frames = _read_base_frames(syms, base_tf)
    if not frames:
        console.print(f"[red]No stored data for {syms} @ {base_tf}. Run `ingest` first.[/red]")
        raise typer.Exit(1)

    from trading_ml.models.technical.module import TechnicalModel
    from trading_ml.models.technical.walkforward import walk_forward_validate

    bundle = TechnicalModel(mcfg)._bundle_from_dfs(frames)
    console.print(f"Walk-forward validation on {len(bundle)} samples ...")
    result = walk_forward_validate(bundle, mcfg)

    agg = result["aggregate"]
    table = Table(title=f"Walk-forward — {model} ({int(agg.get('n_folds', 0))} folds)")
    table.add_column("metric")
    table.add_column("mean", justify="right")
    table.add_column("std", justify="right")
    for key in ("balanced_accuracy", "macro_f1", "accuracy"):
        table.add_row(key, f"{agg.get(key + '_mean', 0):.4f}", f"{agg.get(key + '_std', 0):.4f}")
    console.print(table)


@app.command()
def tune(
    model: str = typer.Option("technical", help="Model config to tune."),
    symbols: str = typer.Option(None, help="Comma-separated symbols (default: config)."),
    n_trials: int = typer.Option(None, help="Optuna trials (default: config)."),
) -> None:
    """Optuna hyperparameter search, scored by walk-forward balanced accuracy."""
    cfg = load_config()
    mcfg = load_model_config(model)
    syms = _parse_csv(symbols) or cfg.data.symbols
    base_tf = mcfg["data"]["base_timeframe"]
    frames = _read_base_frames(syms, base_tf)
    if not frames:
        console.print(f"[red]No stored data for {syms} @ {base_tf}. Run `ingest` first.[/red]")
        raise typer.Exit(1)

    from trading_ml.models.technical.tune import tune_hyperparams

    tcfg = mcfg.get("tune", {})
    trials = n_trials if n_trials is not None else tcfg.get("n_trials", 20)
    console.print(f"Tuning {model} over {trials} trials ...")
    out = tune_hyperparams(
        frames, mcfg, n_trials=trials, timeout_seconds=tcfg.get("timeout_seconds", 0)
    )
    console.print(f"[green]Best walk-forward balanced accuracy:[/green] {out['best_value']:.4f}")
    table = Table(title="Best hyperparameters")
    table.add_column("param")
    table.add_column("value", justify="right")
    for k, v in out["best_params"].items():
        table.add_row(k, f"{v:.5g}" if isinstance(v, float) else str(v))
    console.print(table)


def _mlflow_logger(cfg):
    """Return a Lightning MLFlowLogger if mlflow is installed, else None."""
    try:
        from lightning.pytorch.loggers import MLFlowLogger

        return MLFlowLogger(
            experiment_name=cfg.mlflow.experiment,
            tracking_uri=cfg.env.mlflow_tracking_uri,
        )
    except Exception:  # noqa: BLE001 - logging is optional
        return None


if __name__ == "__main__":  # pragma: no cover
    app()
