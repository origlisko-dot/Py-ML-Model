# Py-ML-Model — ML models for day-trading equities

A modular Python framework for machine-learning day-trading research. It is built
around three models; **Model 1 is fully implemented**, Models 2–3 ship as
well-defined scaffolds behind the same interfaces.

1. **Technical analysis (Model 1, full)** — multi-timeframe candlestick / OHLCV
   analysis (1m → 1d …), with per-timeframe encoders fused by a
   **cross-timeframe attention** network, plus a LightGBM baseline.
2. **Risk management (Model 2, scaffold + basic impl)** — position sizing,
   ATR stops, Risk/Reward targets.
3. **Event-driven (Model 3, scaffold)** — news/catalyst classification interface.

> **Scope:** research & backtesting only. No live order execution is included.
> Use an IBKR **paper** account for any live-data experiments.

## Architecture

```
data      → provider abstraction (IBKR / yfinance), chunked+checkpointed ingestion,
            partitioned Parquet store (symbol/timeframe/year/month) via DuckDB
features  → indicators (missing-bar forward-fill), candlestick patterns,
            strict point-in-time multi-timeframe alignment, triple-barrier labels
models    → technical (full), risk, events — all behind BaseModel/Signal
backtest  → event-driven engine + metrics (Sharpe, Max DD, Win Rate, Profit Factor)
strategy  → combine Model 1 signal → Model 2 sizing → trade plan
```

Technical indicators are implemented directly on pandas (no TA-Lib / pandas-ta
dependency) for deterministic, CI-friendly behaviour.

Four robustness guarantees are baked in (see `config`, `features`, `data`):
IBKR **pacing/throttling + checkpointing**, **no-lookahead** point-in-time alignment,
**lazy loading** to avoid OOM on minute data, and **forward-fill** of missing bars.

## Install

```bash
uv sync                        # core (data + features + CLI)
uv sync --extra ml --extra providers   # add torch/lightning/lightgbm + data providers
```

## Usage

```bash
# 1. Download & store multi-timeframe OHLCV (yfinance needs no broker/gateway)
uv run trading-ml ingest --symbols AAPL,MSFT --provider yfinance

# 2. Train Model 1 (cross-timeframe attention) + LightGBM baseline
uv run trading-ml train --model technical

# 3. Backtest the trained model with the risk sizer
uv run trading-ml backtest --model technical --symbols AAPL
```

IBKR (optional, local): start TWS/IB Gateway on a **paper** account, set `IBKR_*`
in `.env`, then `uv run trading-ml ingest --provider ibkr`.

## Development

```bash
uv run pytest          # unit + robustness tests (no-lookahead, forward-fill, ...)
uv run ruff check
uv run mypy src
```

## Configuration

- `config/default.yaml` — paths, seed, MLflow.
- `config/data.yaml` — symbols, timeframes, provider, IBKR pacing.
- `config/models/technical.yaml` — Model 1 hyperparameters.
- `.env` — secrets / host config (see `.env.example`).
