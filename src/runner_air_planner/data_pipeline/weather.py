"""Client helpers to retrieve current and forecast weather conditions for Madrid."""

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
    precipitation_mm: float | None = None
    cloud_cover: int | None = None


@dataclass(slots=True)
class WeatherForecast:
    """Weather forecast for 1 hour ahead."""

    forecast_time: datetime
    temperature_c: float | None
    relative_humidity: float | None
    wind_speed_kmh: float | None
    weather_code: int | None
    weather_description: str | None
    precipitation_mm: float | None
    cloud_cover: int | None
    probability_precipitation: int | None


class WeatherClient:
    """Client to fetch current and forecast weather conditions from Open-Meteo."""

    def __init__(self, *, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    def fetch_weather_for_location(
        self,
        latitude: float,
        longitude: float,
        *,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> WeatherForecast:
        """Fetch 1-hour ahead weather forecast for a specific location.
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            timeout: Request timeout in seconds
            
        Returns:
            WeatherForecast with 1-hour ahead predictions
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": TIMEZONE,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code,precipitation,cloud_cover,precipitation_probability",
            "forecast_hours": 1,
        }
        
        try:
            response = self._session.get(WEATHER_API_URL, params=params, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as error:
            raise WeatherServiceError(f"No se pudo consultar la predicción meteorológica para {latitude},{longitude}") from error

        payload = response.json()
        hourly = payload.get("hourly")
        if not isinstance(hourly, dict):
            raise WeatherServiceError("Respuesta inesperada del servicio meteorológico")

        times = hourly.get("time", [])
        if not times or len(times) == 0:
            raise WeatherServiceError("No hay datos de predicción disponibles")
        
        forecast_datetime = self._parse_datetime(times[0])
        if forecast_datetime is None:
            raise WeatherServiceError("No se pudo parsear la hora de predicción")

        temperatures = hourly.get("temperature_2m", [])
        humidities = hourly.get("relative_humidity_2m", [])
        wind_speeds = hourly.get("wind_speed_10m", [])
        weather_codes = hourly.get("weather_code", [])
        precipitations = hourly.get("precipitation", [])
        cloud_covers = hourly.get("cloud_cover", [])
        precip_probs = hourly.get("precipitation_probability", [])

        temperature = _coerce_float(temperatures[0] if len(temperatures) > 0 else None)
        humidity = _coerce_float(humidities[0] if len(humidities) > 0 else None)
        wind_speed = _coerce_float(wind_speeds[0] if len(wind_speeds) > 0 else None)
        code = _coerce_int(weather_codes[0] if len(weather_codes) > 0 else None)
        description = WEATHER_CODE_DESCRIPTIONS.get(code) if code is not None else None
        precipitation = _coerce_float(precipitations[0] if len(precipitations) > 0 else None)
        cloud_cover = _coerce_int(cloud_covers[0] if len(cloud_covers) > 0 else None)
        precip_prob = _coerce_int(precip_probs[0] if len(precip_probs) > 0 else None)

        return WeatherForecast(
            forecast_time=forecast_datetime,
            temperature_c=temperature,
            relative_humidity=humidity,
            wind_speed_kmh=wind_speed,
            weather_code=code,
            weather_description=description,
            precipitation_mm=precipitation,
            cloud_cover=cloud_cover,
            probability_precipitation=precip_prob,
        )

    def fetch_current_weather(self, *, timeout: int = DEFAULT_TIMEOUT) -> WeatherReport:
        """Fetch current weather conditions for Madrid.
        
        Args:
            timeout: Request timeout in seconds
            
        Returns:
            WeatherReport with current conditions
            
        Raises:
            WeatherServiceError: If the API request fails
        """
        params = {
            **MADRID_COORDINATES,
            "timezone": TIMEZONE,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code,precipitation,cloud_cover",
        }
        try:
            response = self._session.get(WEATHER_API_URL, params=params, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as error:
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
        description = WEATHER_CODE_DESCRIPTIONS.get(code) if code is not None else None
        precipitation = _coerce_float(current.get("precipitation"))
        cloud_cover = _coerce_int(current.get("cloud_cover"))

        if observed_at is None:
            raise WeatherServiceError("No se pudo determinar la hora de la medición meteorológica")

        return WeatherReport(
            observed_at=observed_at,
            temperature_c=temperature,
            relative_humidity=humidity,
            wind_speed_kmh=wind_speed,
            weather_code=code,
            weather_description=description,
            precipitation_mm=precipitation,
            cloud_cover=cloud_cover,
        )

    def fetch_1hour_forecast(self, *, timeout: int = DEFAULT_TIMEOUT) -> WeatherForecast:
        """Fetch 1-hour ahead weather forecast for Madrid.
        
        Args:
            timeout: Request timeout in seconds
            
        Returns:
            WeatherForecast with 1-hour ahead predictions
            
        Raises:
            WeatherServiceError: If the API request fails
        """
        params = {
            **MADRID_COORDINATES,
            "timezone": TIMEZONE,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code,precipitation,cloud_cover,precipitation_probability",
            "forecast_hours": 1,
        }
        
        try:
            response = self._session.get(WEATHER_API_URL, params=params, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as error:
            raise WeatherServiceError("No se pudo consultar la predicción meteorológica") from error

        payload = response.json()
        hourly = payload.get("hourly")
        if not isinstance(hourly, dict):
            raise WeatherServiceError("Respuesta inesperada del servicio meteorológico")

        # Get the first hour (1 hour ahead)
        times = hourly.get("time", [])
        if not times or len(times) == 0:
            raise WeatherServiceError("No hay datos de predicción disponibles")
        
        # Parse forecast time (first hour in the response)
        forecast_datetime = self._parse_datetime(times[0])
        if forecast_datetime is None:
            raise WeatherServiceError("No se pudo parsear la hora de predicción")

        # Get values for first hour (index 0)
        temperatures = hourly.get("temperature_2m", [])
        humidities = hourly.get("relative_humidity_2m", [])
        wind_speeds = hourly.get("wind_speed_10m", [])
        weather_codes = hourly.get("weather_code", [])
        precipitations = hourly.get("precipitation", [])
        cloud_covers = hourly.get("cloud_cover", [])
        precip_probs = hourly.get("precipitation_probability", [])

        temperature = _coerce_float(temperatures[0] if len(temperatures) > 0 else None)
        humidity = _coerce_float(humidities[0] if len(humidities) > 0 else None)
        wind_speed = _coerce_float(wind_speeds[0] if len(wind_speeds) > 0 else None)
        code = _coerce_int(weather_codes[0] if len(weather_codes) > 0 else None)
        description = WEATHER_CODE_DESCRIPTIONS.get(code) if code is not None else None
        precipitation = _coerce_float(precipitations[0] if len(precipitations) > 0 else None)
        cloud_cover = _coerce_int(cloud_covers[0] if len(cloud_covers) > 0 else None)
        precip_prob = _coerce_int(precip_probs[0] if len(precip_probs) > 0 else None)

        return WeatherForecast(
            forecast_time=forecast_datetime,
            temperature_c=temperature,
            relative_humidity=humidity,
            wind_speed_kmh=wind_speed,
            weather_code=code,
            weather_description=description,
            precipitation_mm=precipitation,
            cloud_cover=cloud_cover,
            probability_precipitation=precip_prob,
        )

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        """Parse ISO format datetime string."""
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        except ValueError:
            return None


def _coerce_float(value: Any) -> float | None:
    """Convert value to float, returning None if not possible."""
    if value in (None, "", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    """Convert value to int, returning None if not possible."""
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = ["WeatherClient", "WeatherReport", "WeatherForecast", "WeatherServiceError"]
