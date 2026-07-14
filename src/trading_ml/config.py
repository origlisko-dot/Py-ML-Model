"""Configuration loading.

Non-secret defaults live in ``config/*.yaml`` (versioned). Secrets and
host-specific values come from environment variables / ``.env`` (see
``.env.example``) via pydantic-settings.

Usage::

    from trading_ml.config import load_config
    cfg = load_config()          # merges default.yaml + data.yaml + env
    cfg.paths.parquet_dir
    cfg.ibkr.host
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def project_root() -> Path:
    """Repo root — three levels up from this file (src/trading_ml/config.py)."""
    return Path(__file__).resolve().parents[2]


def _config_dir() -> Path:
    return project_root() / "config"


# --------------------------------------------------------------------------- #
# Env-backed settings (secrets / host-specific)                                #
# --------------------------------------------------------------------------- #
class IBKRSettings(BaseSettings):
    """IBKR connection — read from IBKR_* env vars."""

    model_config = SettingsConfigDict(env_prefix="IBKR_", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1


class EnvSettings(BaseSettings):
    """Top-level secrets from the environment / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    news_api_key: str = Field(default="", alias="NEWS_API_KEY")
    mlflow_tracking_uri: str = Field(default="./mlruns", alias="MLFLOW_TRACKING_URI")


# --------------------------------------------------------------------------- #
# YAML-backed config (non-secret defaults)                                     #
# --------------------------------------------------------------------------- #
class Paths(BaseModel):
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    parquet_dir: str = "data/parquet"
    artifacts_dir: str = "artifacts"

    def resolved(self, root: Path | None = None) -> Paths:
        root = root or project_root()
        return Paths(
            data_dir=str(root / self.data_dir),
            raw_dir=str(root / self.raw_dir),
            parquet_dir=str(root / self.parquet_dir),
            artifacts_dir=str(root / self.artifacts_dir),
        )


class MLflowConfig(BaseModel):
    tracking_uri: str = "./mlruns"
    experiment: str = "trading-ml"


class IBKRDataConfig(BaseModel):
    chunk: str = "1W"
    throttle_seconds: float = 10.0
    max_retries: int = 3


class NewsDataConfig(BaseModel):
    """Pacing policy for news ingestion (Model 3)."""

    chunk: str = "7d"
    throttle_seconds: float = 0.0
    max_retries: int = 3


class DataConfig(BaseModel):
    provider: str = "yfinance"
    news_provider: str = "yfinance"
    symbols: list[str] = Field(default_factory=lambda: ["AAPL"])
    timeframes: list[str] = Field(default_factory=lambda: ["1d"])
    history: dict[str, str] = Field(default_factory=dict)
    ibkr: IBKRDataConfig = Field(default_factory=IBKRDataConfig)
    news: NewsDataConfig = Field(default_factory=NewsDataConfig)


class Config(BaseModel):
    """Merged application configuration."""

    seed: int = 42
    paths: Paths = Field(default_factory=Paths)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    # Env-backed sections (not from YAML).
    ibkr: IBKRSettings = Field(default_factory=IBKRSettings)
    env: EnvSettings = Field(default_factory=EnvSettings)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def load_config() -> Config:
    """Load and merge ``default.yaml`` + ``data.yaml`` + environment settings."""
    cfg_dir = _config_dir()
    default = _read_yaml(cfg_dir / "default.yaml")
    data = _read_yaml(cfg_dir / "data.yaml")

    cfg = Config(
        seed=default.get("seed", 42),
        paths=Paths(**default.get("paths", {})),
        mlflow=MLflowConfig(**default.get("mlflow", {})),
        data=DataConfig(**data),
        ibkr=IBKRSettings(),
        env=EnvSettings(),
    )
    # Resolve paths to absolutes anchored at the repo root.
    cfg.paths = cfg.paths.resolved()
    return cfg


def load_model_config(name: str) -> dict[str, Any]:
    """Load a model hyperparameter file, e.g. ``load_model_config("technical")``."""
    return _read_yaml(_config_dir() / "models" / f"{name}.yaml")
