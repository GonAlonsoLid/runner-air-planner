"""ETL utilities to download Madrid's real-time air quality dataset."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
import requests

API_DATASET_URL = "https://datos.madrid.es/egob/catalogo/212504-0-calidad-aire-tiempo-real.json"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "madrid_air_quality.csv"


class AirQualityIngestionError(RuntimeError):
    """Raised when the ingestion pipeline cannot be completed."""


def fetch_payload(session: requests.Session | None = None) -> Mapping[str, object]:
    """Fetch the raw JSON payload from Madrid's open data portal.

    The open data portal offers multiple response formats; we rely on the JSON
    representation which includes a ``"@graph"`` key containing a list of
    measurement samples. The function is intentionally light on parameters
    because the portal already returns the most recent measurements.
    """

    http = session or requests.Session()
    try:
        response = http.get(API_DATASET_URL, timeout=30)
        response.raise_for_status()
    except requests.RequestException as error:  # pragma: no cover - network failure branch
        raise AirQualityIngestionError("Could not download Madrid air quality dataset") from error
    return response.json()


def _normalise_record(record: Mapping[str, object]) -> dict[str, object]:
    lower_keys = {str(key).lower(): value for key, value in record.items()}
    timestamp = None
    for candidate in ("fecha", "date", "datetime"):
        raw_value = lower_keys.get(candidate)
        if isinstance(raw_value, str):
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    timestamp = datetime.strptime(raw_value, fmt)
                    break
                except ValueError:
                    continue
        if timestamp:
            break
    if timestamp is None:
        date_raw = lower_keys.get("fecha") or lower_keys.get("date")
        hour_raw = lower_keys.get("hora") or lower_keys.get("hour")
        if isinstance(date_raw, str) and isinstance(hour_raw, (int, str)):
            try:
                date_part = datetime.strptime(date_raw, "%Y-%m-%d").date()
                hour_value = int(str(hour_raw))
                timestamp = datetime.combine(date_part, datetime.min.time()).replace(hour=hour_value)
            except ValueError:
                pass
    value = lower_keys.get("valor") or lower_keys.get("value")
    try:
        numeric_value = float(str(value).replace(",", ".")) if value not in (None, "") else None
    except ValueError:
        numeric_value = None
    return {
        "station": lower_keys.get("estacion") or lower_keys.get("station"),
        "pollutant": lower_keys.get("magnitud") or lower_keys.get("pollutant"),
        "value": numeric_value,
        "unit": lower_keys.get("unidad") or lower_keys.get("unit"),
        "is_valid": lower_keys.get("valido") in {"S", "s", "SI", "Si", "sí", "true", True, 1},
        "timestamp": timestamp,
    }


def _extract_records(payload: Mapping[str, object]) -> Iterable[Mapping[str, object]]:
    graph = payload.get("@graph")
    if isinstance(graph, list):
        return graph
    data = payload.get("data")
    if isinstance(data, list):
        return data
    raise AirQualityIngestionError("Unexpected payload structure returned by Madrid API")


def build_dataframe(payload: Mapping[str, object]) -> pd.DataFrame:
    """Convert the raw JSON payload into a tidy :class:`pandas.DataFrame`."""

    rows = [_normalise_record(item) for item in _extract_records(payload)]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def persist_dataframe(frame: pd.DataFrame, output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    """Persist the dataframe to ``output_path`` and return the resulting path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def run_pipeline(output_path: Path = DEFAULT_OUTPUT_PATH, *, session: requests.Session | None = None) -> Path:
    """Download the dataset, normalise it, and persist it locally."""

    payload = fetch_payload(session=session)
    frame = build_dataframe(payload)
    if frame.empty:
        raise AirQualityIngestionError("Madrid dataset did not contain any measurements")
    return persist_dataframe(frame, output_path=output_path)


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    path = run_pipeline()
    print(f"Saved Madrid air quality dataset to {path}")
