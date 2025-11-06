"""Utility helpers to load Madrid air quality measurements."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DATA_PATH = DATA_DIR / "madrid_air_quality_raw.csv"

POLLUTANT_FEATURES = ["no2", "no", "nox", "o3", "pm10", "pm25", "so2"]


def load_raw_measurements(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the ingested CSV file exported by the data pipeline."""

    if not path.exists():
        raise FileNotFoundError(
            "The Madrid air quality dataset was not found. Run data_pipeline/ingest_madrid_air.py first."
        )
    frame = pd.read_csv(path)
    if "measurement_time" in frame.columns:
        frame["measurement_time"] = pd.to_datetime(frame["measurement_time"], errors="coerce")
    return frame


def _normalise_pollutant_name(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    replacements = {
        "pm2.5": "pm25",
        "pm2,5": "pm25",
        "nitrogen dioxide": "no2",
        "nitrogen oxide": "no",
        "nitrogen oxides": "nox",
        "ozone": "o3",
        "particulas pm10": "pm10",
        "particulas pm2.5": "pm25",
        "dioxido de azufre": "so2",
        "dióxido de azufre": "so2",
    }
    return replacements.get(lowered, lowered)


def _ensure_numeric_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column not in frame:
            frame[column] = float("nan")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def prepare_station_feature_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long-format measurements into one row per station with pollutant columns."""

    if raw_frame.empty:
        return raw_frame.copy()

    frame = raw_frame.copy()
    frame["pollutant"] = frame["pollutant"].map(_normalise_pollutant_name)
    frame = frame[frame["value"].notna()]
    pivot = (
        frame.pivot_table(
            index=["station_code", "measurement_time"],
            columns="pollutant",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    pivot.columns = [str(column).lower() for column in pivot.columns]
    pivot = pivot.sort_values("measurement_time")
    latest = pivot.groupby("station_code", as_index=False).tail(1)
    latest = latest.reset_index(drop=True)
    return _ensure_numeric_columns(latest, POLLUTANT_FEATURES)


__all__ = ["load_raw_measurements", "prepare_station_feature_frame", "POLLUTANT_FEATURES"]
