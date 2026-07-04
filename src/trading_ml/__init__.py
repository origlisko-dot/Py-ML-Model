"""trading_ml — multi-timeframe ML models for day-trading equities.

Layers:
    data     — provider abstraction, ingestion, partitioned Parquet storage
    features — indicators, candlestick patterns, MTF alignment, labeling
    models   — technical (Model 1, full), risk (Model 2), events (Model 3)
    backtest — engine + metrics
    strategy — signal aggregation across models
"""

__version__ = "0.1.0"
