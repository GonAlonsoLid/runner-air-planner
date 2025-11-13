"""Download Madrid's real-time air quality feed and persist it as CSV files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import requests

DEFAULT_OUTPUT = Path("data/madrid_air_quality_raw.csv")
DATASET_URL = "https://datos.madrid.es/egob/catalogo/212531-12751102-calidad-aire-tiempo-real.json"
USER_AGENT = "runner-air-planner/0.2"


@dataclass(slots=True)
class DownloadResult:
    """Summary of the ingestion job."""

    records_fetched: int
    output_path: Path


class MadridAirDownloadError(RuntimeError):
    """Raised when the Madrid open data API cannot be queried."""


def fetch_payload(url: str = DATASET_URL, *, timeout: int = 30) -> Any:
    """Fetch the JSON payload from the Madrid open data endpoint."""

    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as error:  # pragma: no cover - network failure branch
        raise MadridAirDownloadError(f"Madrid open data API returned {error.response.status_code}") from error
    return response.json()


def extract_records(payload: Any) -> Iterable[Mapping[str, Any]]:
    """Extract the iterable of measurement records from the payload."""

    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        # Nuevo formato: records directamente en el payload
        if "records" in payload and isinstance(payload["records"], list):
            return payload["records"]
        # Formato antiguo: @graph
        if "@graph" in payload and isinstance(payload["@graph"], list):
            return payload["@graph"]
        # Formato alternativo: data
        if "data" in payload and isinstance(payload["data"], list):
            return payload["data"]
        # Formato alternativo: result.records
        if "result" in payload and isinstance(payload["result"], Mapping):
            records = payload["result"].get("records")
            if isinstance(records, list):
                return records
    raise MadridAirDownloadError("Unexpected payload structure from Madrid air quality dataset")


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    return str(value)


def _coerce_float(value: Any) -> float | None:
    if value in (None, "", "NA", "nan"):
        return None
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _extract_hour(payload: Mapping[str, Any]) -> time:
    hour_raw = payload.get("hora") or payload.get("hour")
    if hour_raw is not None:
        hour_int = _coerce_int(hour_raw)
        if hour_int is not None:
            return time(hour=max(0, min(23, hour_int)))
    for key, value in payload.items():
        if isinstance(key, str) and key.lower().startswith("h") and len(key) == 3 and key[1:].isdigit():
            if value not in (None, ""):
                hour_int = int(key[1:]) - 1
                return time(hour=max(0, min(23, hour_int)))
    raise MadridAirDownloadError("Unable to determine measurement hour from payload")


def parse_measurement_time(payload: Mapping[str, Any]) -> datetime:
    for key in ("fecha", "date", "datetime", "fecha_medida"):
        value = payload.get(key)
        if value:
            text = str(value)
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
                try:
                    parsed = datetime.strptime(text, fmt)
                    return parsed.replace(tzinfo=None)
                except ValueError:
                    continue
            if len(text) == 8 and text.isdigit():
                base_date = datetime.strptime(text, "%Y%m%d").date()
                return datetime.combine(base_date, _extract_hour(payload))
    year = _coerce_int(payload.get("ano") or payload.get("año") or payload.get("year"))
    month = _coerce_int(payload.get("mes") or payload.get("month"))
    day = _coerce_int(payload.get("dia") or payload.get("day"))
    if None not in (year, month, day):
        return datetime.combine(date(year, month, day), _extract_hour(payload))
    raise MadridAirDownloadError("Unable to parse measurement timestamp from record")


def expand_hourly_records(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand a wide-format record (24 hours per row) into long format (one row per hour)."""
    lowered = {str(key).lower(): value for key, value in record.items()}
    
    # Extraer información común
    station_code = _coerce_str(
        lowered.get("estacion")
        or lowered.get("station")
        or lowered.get("cod_estacion")
        or lowered.get("id")
    )
    sampling_point = _coerce_str(
        lowered.get("punto_muestreo")
        or lowered.get("sampling_point")
        or lowered.get("id_estacion")
    )
    pollutant = _coerce_str(
        lowered.get("magnitud")
        or lowered.get("pollutant")
        or lowered.get("contaminante")
    )
    
    # Extraer fecha
    year = _coerce_int(lowered.get("ano") or lowered.get("año") or lowered.get("year"))
    month = _coerce_int(lowered.get("mes") or lowered.get("month"))
    day = _coerce_int(lowered.get("dia") or lowered.get("day"))
    
    if None in (year, month, day):
        return []  # No podemos procesar sin fecha
    
    # Expandir las 24 horas
    expanded = []
    for hour in range(1, 25):
        hour_key = f"h{hour:02d}"
        valid_key = f"v{hour:02d}"
        
        value_str = lowered.get(hour_key, "")
        valid_str = lowered.get(valid_key, "")
        
        # Convertir valor
        value = _coerce_float(value_str)
        
        # Convertir validez (V = válido, N = no válido)
        is_valid = None
        if valid_str:
            is_valid = str(valid_str).strip().upper() == "V"
        
        # Solo incluir si hay un valor o si está marcado como válido
        if value is not None or is_valid is not None:
            measurement_time = datetime.combine(
                date(year, month, day),
                time(hour=hour - 1)  # H01 = hora 0, H02 = hora 1, etc.
            )
            
            expanded.append({
                "station_code": station_code,
                "sampling_point": sampling_point,
                "pollutant": pollutant.lower() if pollutant else None,
                "measurement_time": measurement_time,
                "value": value,
                "unit": None,  # No viene en este formato
                "is_valid": is_valid,
            })
    
    return expanded


def normalise_record(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize a record, handling both old format (one row per measurement) and new format (24 hours per row)."""
    lowered = {str(key).lower(): value for key, value in record.items()}
    
    # Detectar si es formato nuevo (tiene H01, H02, etc.)
    has_hourly_format = any(key.startswith("h") and len(key) == 3 and key[1:].isdigit() for key in lowered.keys())
    
    if has_hourly_format:
        # Formato nuevo: expandir las 24 horas
        return expand_hourly_records(record)
    
    # Formato antiguo: un registro por medición
    station_code = _coerce_str(
        lowered.get("estacion")
        or lowered.get("station")
        or lowered.get("cod_estacion")
        or lowered.get("id")
    )
    sampling_point = _coerce_str(
        lowered.get("punto_muestreo")
        or lowered.get("sampling_point")
        or lowered.get("id_estacion")
    )
    pollutant = _coerce_str(
        lowered.get("magnitud")
        or lowered.get("pollutant")
        or lowered.get("contaminante")
    )
    measurement_time = parse_measurement_time(lowered)
    value = _coerce_float(lowered.get("valor") or lowered.get("value"))
    unit = _coerce_str(lowered.get("unidad") or lowered.get("unit") or lowered.get("unidades"))
    validity_raw = lowered.get("valido") or lowered.get("validez") or lowered.get("valid")
    is_valid = None
    if validity_raw is not None:
        is_valid = str(validity_raw).strip().lower() in {"s", "si", "sí", "true", "1", "v", "valid"}
    
    return [{
        "station_code": station_code,
        "sampling_point": sampling_point,
        "pollutant": pollutant.lower() if pollutant else None,
        "measurement_time": measurement_time,
        "value": value,
        "unit": unit,
        "is_valid": is_valid,
    }]


def download_latest_measurements() -> pd.DataFrame:
    payload = fetch_payload()
    raw_records = extract_records(payload)
    # normalise_record ahora devuelve una lista (puede expandir 1 registro en 24)
    expanded_records = []
    for record in raw_records:
        expanded_records.extend(normalise_record(record))
    frame = pd.DataFrame.from_records(expanded_records)
    if not frame.empty:
        frame.sort_values("measurement_time", inplace=True)
    return frame


def persist_measurements(output_path: Path = DEFAULT_OUTPUT) -> DownloadResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = download_latest_measurements()
    if not frame.empty:
        frame["measurement_time"] = frame["measurement_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    frame.to_csv(output_path, index=False)
    return DownloadResult(records_fetched=len(frame), output_path=output_path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Madrid's real-time air quality dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination CSV path (default: data/madrid_air_quality_raw.csv)",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    result = persist_measurements(output_path=args.output)
    print(f"Fetched {result.records_fetched} measurements -> {result.output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
