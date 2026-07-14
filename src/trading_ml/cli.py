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


@app.command(name="ingest-news")
def ingest_news(
    symbols: str = typer.Option(None, help="Comma-separated symbols (default: config)."),
    provider: str = typer.Option(None, help="yfinance | memory (default: config)."),
    lookback: str = typer.Option("30d", help="How far back to fetch news."),
    no_resume: bool = typer.Option(False, help="Ignore checkpoints and re-fetch."),
) -> None:
    """Download and store news items into the partitioned Parquet store (Model 3)."""
    from trading_ml.data.news_ingestion import ingest_news as run_ingest_news

    reports = run_ingest_news(
        symbols=_parse_csv(symbols),
        provider_name=provider,
        lookback=lookback,
        resume=not no_resume,
    )
    table = Table(title="News ingestion report")
    for col in ("symbol", "items", "chunks", "errors"):
        table.add_column(col)
    for r in reports:
        table.add_row(r.symbol, str(r.items), str(r.chunks), str(len(r.errors)))
    console.print(table)


@app.command()
def catalysts(
    symbols: str = typer.Option(None, help="Comma-separated symbols (default: config)."),
    lookback: str = typer.Option("30d", help="News window to scan."),
    nlp: bool = typer.Option(False, help="Use the transformer classifier (needs --extra nlp)."),
) -> None:
    """Detect catalysts from stored news and print them (Model 3)."""
    cfg = load_config()
    syms = _parse_csv(symbols) or cfg.data.symbols
    try:
        signals = _detect_catalysts(syms, lookback=lookback, use_nlp=nlp)
    except ImportError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    actionable = [s for s in signals if s.is_actionable]
    console.print(f"Scanned news for {syms}: {len(actionable)} actionable catalysts.")

    table = Table(title=f"Catalysts ({'nlp' if nlp else 'keyword'})")
    for col in ("timestamp", "symbol", "catalyst", "strength", "confidence"):
        table.add_column(col)
    for s in actionable:
        table.add_row(
            str(s.timestamp),
            s.symbol,
            s.catalyst.value,
            f"{s.strength:.2f}",
            f"{s.confidence:.2f}",
        )
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
    portfolio: bool = typer.Option(False, help="Enable portfolio-level risk constraints."),
    catalysts: bool = typer.Option(False, help="Fuse Model 3 news catalysts into signals."),
    nlp: bool = typer.Option(False, help="Use the NLP catalyst classifier (needs --extra nlp)."),
    costs: bool = typer.Option(
        False, help="Apply realistic trading costs (commission/slippage) from config."
    ),
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
    from trading_ml.strategy import StrategyEngine

    # Backtest prices must be close-labeled to match signal timestamps.
    price_frames = {
        sym: to_close_labeled(fill_missing_bars(df, base_tf), base_tf) for sym, df in frames.items()
    }
    catalyst_signals = None
    if catalysts:
        try:
            catalyst_signals = _detect_catalysts(syms, use_nlp=nlp)
        except ImportError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        n_act = sum(1 for c in catalyst_signals if c.is_actionable)
        console.print(f"Fusing {n_act} actionable catalysts.")
    combined = StrategyEngine(min_prob=min_prob).combine(signals, catalyst_signals)
    risk_model, trail, pm = _build_risk(mcfg, price_frames, portfolio)
    cost_model = _build_costs(mcfg) if costs else None
    result = run_backtest(
        combined,
        price_frames,
        risk_model=risk_model,
        atr_window=mcfg["labeling"]["atr_window"],
        min_prob=min_prob,
        trail_atr_mult=trail,
        portfolio=pm,
        cost_model=cost_model,
    )
    if result.rejections:
        console.print(f"Portfolio rejections: {result.rejections}")

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


def _detect_catalysts(symbols, lookback="30d", use_nlp=False):
    """Read stored news for `symbols` and classify it into CatalystSignals.

    Keyword classifier by default (dependency-free); the transformer classifier
    with ``use_nlp=True`` (requires ``uv sync --extra nlp``).
    """
    import pandas as pd

    from trading_ml.data import news_storage
    from trading_ml.data.ingestion import _parse_window
    from trading_ml.models.events import KeywordCatalystClassifier

    end = pd.Timestamp.now(tz="UTC")
    start = end - _parse_window(lookback)

    if use_nlp:
        from trading_ml.models.events import TransformerCatalystClassifier

        clf = TransformerCatalystClassifier(load_model_config("catalyst"))
    else:
        clf = KeywordCatalystClassifier()

    signals = []
    for sym in symbols:
        items = news_storage.read_news(sym, start=start, end=end)
        if items:
            signals.extend(clf.classify_batch(items))
    return signals


def _build_costs(mcfg):
    """Construct a CostModel from the model config's `costs` section."""
    from trading_ml.backtest.costs import CostModel

    cc = mcfg.get("costs", {})
    return CostModel(
        commission_per_share=cc.get("commission_per_share", 0.0),
        commission_pct=cc.get("commission_pct", 0.0),
        min_commission=cc.get("min_commission", 0.0),
        slippage_bps=cc.get("slippage_bps", 0.0),
        half_spread_bps=cc.get("half_spread_bps", 0.0),
    )


def _build_risk(mcfg, price_frames, portfolio_flag):
    """Construct (RiskModel, trail_atr_mult, PortfolioRiskManager|None) from config."""
    from trading_ml.models.risk import RiskModel

    rc = mcfg.get("risk", {})
    risk_model = RiskModel(
        risk_pct=rc.get("risk_pct", 0.01),
        rr_ratio=rc.get("rr_ratio", 2.0),
        stop_atr_mult=rc.get("stop_atr_mult", 1.5),
        max_position_pct=rc.get("max_position_pct", 0.25),
        method=rc.get("method", "fixed_fractional"),
        kelly_multiplier=rc.get("kelly_multiplier", 0.5),
        target_volatility=rc.get("target_volatility", 0.01),
    )
    trail = rc.get("trail_atr_mult", 0.0) or None

    pcfg = rc.get("portfolio", {})
    pm = None
    if portfolio_flag or pcfg.get("enabled", False):
        import pandas as pd

        from trading_ml.models.risk.portfolio import PortfolioRiskManager, correlation_lookup

        returns = pd.DataFrame(
            {sym: f["close"].pct_change() for sym, f in price_frames.items()}
        ).dropna(how="all")
        pm = PortfolioRiskManager(
            max_concurrent_positions=pcfg.get("max_concurrent_positions", 5),
            max_total_risk_pct=pcfg.get("max_total_risk_pct", 0.06),
            max_symbol_exposure_pct=pcfg.get("max_symbol_exposure_pct", 0.25),
            max_daily_loss_pct=pcfg.get("max_daily_loss_pct", 0.03),
            max_correlation=pcfg.get("max_correlation", 0.8),
            correlation=correlation_lookup(returns) if len(price_frames) > 1 else None,
        )
    return risk_model, trail, pm


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
