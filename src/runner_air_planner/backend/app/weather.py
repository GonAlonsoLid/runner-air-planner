"""Client helpers to retrieve current weather conditions for Madrid."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests

WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
MADRID_COORDINATES = {
    "latitude": 40.4168,
    "longitude": -3.7038,
}
TIMEZONE = "Europe/Madrid"
DEFAULT_TIMEOUT = 20

WEATHER_CODE_DESCRIPTIONS = {
    0: "Despejado",
    1: "Predominantemente despejado",
    2: "Parcialmente nublado",
    3: "Cielo cubierto",
    45: "Niebla",
    48: "Niebla con escarcha",
    51: "Llovizna ligera",
    53: "Llovizna moderada",
    55: "Llovizna intensa",
    61: "Lluvia ligera",
    63: "Lluvia moderada",
    65: "Lluvia intensa",
    71: "Nieve ligera",
    73: "Nieve moderada",
    75: "Nieve intensa",
    80: "Chubascos ligeros",
    81: "Chubascos moderados",
    82: "Chubascos intensos",
    95: "Tormenta",
    96: "Tormenta con granizo ligero",
    99: "Tormenta con granizo intenso",
}


class WeatherServiceError(RuntimeError):
    """Raised when the weather service cannot be queried."""


@dataclass(slots=True)
class WeatherReport:
    """Simple container with the latest weather snapshot."""

    observed_at: datetime
    temperature_c: float | None
    relative_humidity: float | None
    wind_speed_kmh: float | None
    weather_code: int | None
    weather_description: str | None


class WeatherClient:
    """Small client to fetch current weather conditions from Open-Meteo."""

    def __init__(self, *, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    def fetch_current_weather(self, *, timeout: int = DEFAULT_TIMEOUT) -> WeatherReport:
        params = {
            **MADRID_COORDINATES,
            "timezone": TIMEZONE,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
        }
        try:
            response = self._session.get(WEATHER_API_URL, params=params, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as error:  # pragma: no cover - network failure guard
            raise WeatherServiceError("No se pudo consultar el servicio meteorológico") from error

        payload = response.json()
        current = payload.get("current")
        if not isinstance(current, dict):
            raise WeatherServiceError("Respuesta inesperada del servicio meteorológico")

        observed_at = self._parse_datetime(current.get("time"))
        temperature = _coerce_float(current.get("temperature_2m"))
        humidity = _coerce_float(current.get("relative_humidity_2m"))
        wind_speed = _coerce_float(current.get("wind_speed_10m"))
        code = _coerce_int(current.get("weather_code"))
        description = WEATHER_CODE_DESCRIPTIONS.get(code)

        if observed_at is None:
            raise WeatherServiceError("No se pudo determinar la hora de la medición meteorológica")

        return WeatherReport(
            observed_at=observed_at,
            temperature_c=temperature,
            relative_humidity=humidity,
            wind_speed_kmh=wind_speed,
            weather_code=code,
            weather_description=description,
        )

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return None


def _coerce_float(value: Any) -> float | None:
    if value in (None, "", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = ["WeatherClient", "WeatherReport", "WeatherServiceError"]
