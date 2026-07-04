"""Canonical timeframe handling.

A *timeframe* is a short string like ``"1m"``, ``"5m"``, ``"1h"``, ``"1d"``,
``"1w"``, ``"1M"`` (month). This module is the single source of truth for
converting those labels to pandas offsets, durations, and an ordering, so every
layer (providers, storage, resampling) agrees on their meaning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

# Ordered from finest to coarsest. Used to decide resampling direction and to
# validate that a target timeframe is coarser than its base.
_ORDER: list[str] = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]

_UNIT_TO_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 604800}

_PANDAS_FREQ = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
    "1w": "1W",
    "1M": "1MS",  # month start
}

_PATTERN = re.compile(r"^(\d+)([mhdwM])$")


@dataclass(frozen=True)
class Timeframe:
    """Parsed timeframe with helpers."""

    label: str

    def __post_init__(self) -> None:
        if not _PATTERN.match(self.label):
            raise ValueError(f"Invalid timeframe: {self.label!r}")

    @property
    def pandas_freq(self) -> str:
        """Pandas offset alias for resampling, e.g. ``5min`` for ``5m``."""
        if self.label in _PANDAS_FREQ:
            return _PANDAS_FREQ[self.label]
        n, unit = _PATTERN.match(self.label).groups()  # type: ignore[union-attr]
        alias = {"m": "min", "h": "h", "d": "D", "w": "W", "M": "MS"}[unit]
        return f"{n}{alias}"

    @property
    def seconds(self) -> int:
        """Approximate duration in seconds (month ≈ 30d). ``0`` for month."""
        n, unit = _PATTERN.match(self.label).groups()  # type: ignore[union-attr]
        if unit == "M":
            return 30 * 86400
        return int(n) * _UNIT_TO_SECONDS[unit]

    @property
    def order(self) -> int:
        if self.label in _ORDER:
            return _ORDER.index(self.label)
        # Unknown-but-valid labels ordered by duration.
        return len(_ORDER) + self.seconds

    def is_intraday(self) -> bool:
        return self.seconds < 86400

    def is_coarser_than(self, other: Timeframe) -> bool:
        return self.order > other.order


def parse(label: str) -> Timeframe:
    return Timeframe(label)


def to_freq(label: str) -> str:
    return Timeframe(label).pandas_freq


def sort_labels(labels: list[str]) -> list[str]:
    """Return timeframe labels ordered finest → coarsest."""
    return sorted(labels, key=lambda x: Timeframe(x).order)


def duration_to_timedelta(spec: str) -> pd.Timedelta:
    """Parse a history-window spec like ``"30d"``, ``"60d"``, ``"730d"`` → Timedelta.

    Supports units m/h/d/w. Weeks and days map exactly; months are not accepted
    here (use an explicit day count) to keep windows unambiguous.
    """
    m = re.match(r"^(\d+)([mhdw])$", spec)
    if not m:
        raise ValueError(f"Invalid duration spec: {spec!r}")
    n, unit = int(m.group(1)), m.group(2)
    return pd.Timedelta(seconds=n * _UNIT_TO_SECONDS[unit])
