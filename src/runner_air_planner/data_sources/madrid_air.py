"""Utilities to interact with Madrid's real-time air quality dataset."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
import json
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from runner_air_planner.config import AirQualityConfig


class MadridAirQualityClientError(RuntimeError):
    """Raised when the Madrid air quality API cannot be queried successfully."""


@dataclass
class MadridAirQualityMeasurement:
    """Normalized representation of an individual measurement sample."""

    station_code: str | None
    sampling_point: str | None
    pollutant: str | None
    measurement_time: datetime
    value: float | None
    unit: str | None
    is_valid: bool | None
    raw: Mapping[str, Any]


@dataclass
class MadridAirQualityClient:
    """Small helper around the Madrid open data endpoint."""

    settings: AirQualityConfig

    def __post_init__(self) -> None:
        self._headers = {"User-Agent": "runner-air-planner/0.1"}

    def fetch_measurements(self, params: Mapping[str, Any] | None = None) -> list[MadridAirQualityMeasurement]:
        payload = self.fetch_raw_payload(params=params)
        records = [self._normalise_record(item) for item in self._extract_records(payload)]
        return [MadridAirQualityMeasurement(**record) for record in records]

    def fetch_raw_payload(self, params: Mapping[str, Any] | None = None) -> Any:
        url = self._build_dataset_url(params=params)
        request = Request(url, headers=self._headers)
        try:
            with urlopen(request, timeout=self.settings.request_timeout_seconds) as response:
                raw_bytes = response.read()
                encoding = response.headers.get_content_charset() or "utf-8"
        except HTTPError as error:  # pragma: no cover - network failure branch
            raise MadridAirQualityClientError(
                f"Madrid open data endpoint returned {error.code}: {error.reason}"
            ) from error
        except URLError as error:  # pragma: no cover - network failure branch
            raise MadridAirQualityClientError("Failed to reach Madrid open data endpoint") from error

        if self.settings.response_format.lower() == "json":
            try:
                return json.loads(raw_bytes.decode(encoding))
            except json.JSONDecodeError as error:
                raise MadridAirQualityClientError("Received malformed JSON payload from Madrid API") from error
        return raw_bytes.decode(encoding)

    def _build_dataset_url(self, params: Mapping[str, Any] | None = None) -> str:
        resource = f"{self.settings.dataset_id}.{self.settings.response_format}"
        base = urljoin(self.settings.base_url, resource)
        if not params:
            return base
        safe_params = {str(key): str(value) for key, value in params.items()}
        return f"{base}?{urlencode(safe_params)}"

    @staticmethod
    def _extract_records(payload: Any) -> Iterable[Mapping[str, Any]]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, Mapping):
            if "@graph" in payload and isinstance(payload["@graph"], list):
                return payload["@graph"]
            if "data" in payload and isinstance(payload["data"], list):
                return payload["data"]
            if "result" in payload and isinstance(payload["result"], Mapping):
                result = payload["result"]
                records = result.get("records")
                if isinstance(records, list):
                    return records
        raise MadridAirQualityClientError("Unexpected payload structure received from Madrid API")

    @staticmethod
    def _normalise_record(record: Mapping[str, Any]) -> Mapping[str, Any]:
        lowered = {str(key).lower(): value for key, value in record.items()}
        station = MadridAirQualityClient._coerce_str(
            lowered.get("estacion")
            or lowered.get("station")
            or lowered.get("cod_estacion")
            or lowered.get("id")
        )
        sampling_point = MadridAirQualityClient._coerce_str(
            lowered.get("punto_muestreo")
            or lowered.get("sampling_point")
            or lowered.get("id_estacion")
        )
        pollutant = MadridAirQualityClient._coerce_str(
            lowered.get("magnitud")
            or lowered.get("pollutant")
            or lowered.get("contaminante")
        )
        measurement_time = MadridAirQualityClient._parse_measurement_time(lowered)
        value = MadridAirQualityClient._coerce_float(lowered.get("valor") or lowered.get("value"))
        unit = MadridAirQualityClient._coerce_str(
            lowered.get("unidad") or lowered.get("unidades") or lowered.get("unit")
        )
        validity_raw = lowered.get("valido") or lowered.get("validez") or lowered.get("valid")
        is_valid = None
        if validity_raw is not None:
            is_valid = str(validity_raw).strip().lower() in {"s", "si", "sí", "true", "1", "v", "valid"}
        return {
            "station_code": station,
            "sampling_point": sampling_point,
            "pollutant": pollutant,
            "measurement_time": measurement_time,
            "value": value,
            "unit": unit,
            "is_valid": is_valid,
            "raw": record,
        }

    @staticmethod
    def _coerce_str(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        return str(value)

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if value in (None, "", "NA", "nan"):
            return None
        try:
            return float(str(value).replace(",", "."))
        except ValueError:
            return None

    @staticmethod
    def _parse_measurement_time(payload: Mapping[str, Any]) -> datetime:
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
                    hour_value = MadridAirQualityClient._extract_hour(payload)
                    return datetime.combine(base_date, hour_value)
        year = MadridAirQualityClient._coerce_int(payload.get("ano") or payload.get("año") or payload.get("year"))
        month = MadridAirQualityClient._coerce_int(payload.get("mes") or payload.get("month"))
        day = MadridAirQualityClient._coerce_int(payload.get("dia") or payload.get("day"))
        if None not in (year, month, day):
            hour = MadridAirQualityClient._extract_hour(payload)
            return datetime.combine(date(year, month, day), hour)
        raise MadridAirQualityClientError("Unable to determine measurement timestamp from record")

    @staticmethod
    def _extract_hour(payload: Mapping[str, Any]) -> time:
        hour_raw = payload.get("hora") or payload.get("hour")
        if hour_raw is not None:
            hour_int = MadridAirQualityClient._coerce_int(hour_raw)
            if hour_int is not None:
                hour_int = max(0, min(23, hour_int))
                return time(hour=hour_int)
        for key, value in payload.items():
            if isinstance(key, str) and key.lower().startswith("h") and len(key) == 3 and key[1:].isdigit():
                if value not in (None, ""):
                    hour_int = int(key[1:]) - 1
                    hour_int = max(0, min(23, hour_int))
                    return time(hour=hour_int)
        raise MadridAirQualityClientError("Unable to infer hour from record")

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None


__all__ = [
    "MadridAirQualityClient",
    "MadridAirQualityClientError",
    "MadridAirQualityMeasurement",
]
