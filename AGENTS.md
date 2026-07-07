# AGENTS.md

## Cursor Cloud specific instructions

This repo is **`trading-ml`** — a Python (uv-managed) research/backtesting library and
Typer CLI for multi-timeframe ML day-trading. There is **no service, database, web
server, or frontend**; "running the app" means invoking the `trading-ml` CLI. Standard
install / test / lint / usage commands live in `README.md` and `pyproject.toml`; prefer
those rather than duplicating them.

### Environment
- Package manager is **`uv`** (installed at `~/.local/bin`; that dir is on `PATH` via
  `~/.bashrc`). The startup update script runs `uv sync` with all extras.
- **Always sync/run on Python 3.11** (`uv sync --python 3.11`, matching CI). If the
  venv is created on Python 3.12, `uv run mypy src` fails with a numpy stub error
  (`Type statement is only supported in Python 3.12 and greater`). Recreate with
  `rm -rf .venv && uv sync --python 3.11 ...` if that happens.
- CI (`.github/workflows/ci.yml`) installs only `--extra dev` and skips the heavy ML
  extra; model tests self-skip via `pytest.importorskip`. To actually run
  `train`/`backtest`/`evaluate`/`tune` you need `--extra ml --extra providers`
  (and `--extra tune`), which the update script already installs.

### Running the pipeline end to end
- Order matters: `ingest` (writes `data/parquet/`) must run before
  `train`/`backtest`/`evaluate`/`tune`. `train` writes checkpoints to `artifacts/`.
  `data/`, `artifacts/`, and `mlruns/` are gitignored.
- `ingest --provider yfinance` needs internet (works in this env, no auth). IBKR is
  optional and needs a local TWS/IB Gateway paper session (not present here).
- **Gotcha (data span vs. window):** the default `config/models/technical.yaml` uses
  `window: 64` with a `1d` context timeframe, which needs >64 daily bars. yfinance caps
  `5m` history at ~60 calendar days (~40 daily bars), so training on freshly ingested
  intraday data fails with `ValueError: No samples produced across symbols.` For a quick
  end-to-end smoke test on real data, temporarily reduce `window` (e.g. 32) and drop the
  `1d` entry from `context_timeframes` (e.g. `[5m, 15m, 1h]`); revert afterward. The test
  suite is unaffected — it uses synthetic OHLCV fixtures with enough history.
- Re-running `ingest` without `--no-resume` may report `0` new rows when the checkpoint's
  tail window is empty; already-stored data stays usable. Use `--no-resume` to force a
  full re-download.
