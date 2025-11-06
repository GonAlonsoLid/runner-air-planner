"""Helpers for loading persisted Madrid air quality datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "madrid_air_quality.csv"


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the persisted dataset into a :class:`pandas.DataFrame`.

    The helper is intentionally tolerant: if the CSV does not exist yet it
    returns an empty dataframe with the expected columns to keep downstream
    consumers simple.
    """

    csv_path = path or DATA_PATH
    if not csv_path.exists():
        return pd.DataFrame(columns=["station", "pollutant", "value", "unit", "is_valid", "timestamp"])
    frame = pd.read_csv(csv_path, parse_dates=["timestamp"])
    return frame


def recent_measurements(limit: int | None = None, *, path: Path | None = None) -> Iterable[dict[str, object]]:
    """Return the most recent measurements as dictionaries."""

    frame = load_dataset(path=path)
    if frame.empty:
        return []
    ordered = frame.sort_values("timestamp", ascending=False)
    if limit is not None:
        ordered = ordered.head(limit)
    return ordered.to_dict(orient="records")
