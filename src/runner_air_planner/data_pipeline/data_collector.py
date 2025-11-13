"""Main data collection class that integrates air quality and weather data."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from . import master_data, weather
from .ingest_madrid_air import download_latest_measurements


class DataCollector:
    """Main class to collect and integrate air quality and weather data for ML model.
    
    This class coordinates data collection from:
    - Madrid Air Quality API (air quality measurements by station)
    - Open-Meteo API (weather conditions for Madrid)
    
    It combines these data sources considering:
    - Location (station coordinates vs Madrid city center)
    - Temporal alignment (matching timestamps)
    - Feature engineering for ML model
    - Data quality and validation
    """

    def __init__(
        self,
        *,
        weather_client: weather.WeatherClient | None = None,
        cache_weather_minutes: int = 15,
    ) -> None:
        """Initialize the data collector.
        
        Args:
            weather_client: Optional weather client (creates new one if None)
            cache_weather_minutes: Minutes to cache weather data (default 15)
        """
        self._weather_client = weather_client or weather.WeatherClient()
        self._cache_weather_minutes = cache_weather_minutes
        self._cached_weather: weather.WeatherReport | None = None
        self._cached_weather_time: datetime | None = None

    def collect_air_quality_data(self) -> pd.DataFrame:
        """Download and return latest air quality measurements.
        
        Returns:
            DataFrame with air quality measurements
        """
        return download_latest_measurements()

    def get_weather_data(self, *, force_refresh: bool = False) -> weather.WeatherReport:
        """Get current weather data, with caching.
        
        Args:
            force_refresh: Force refresh even if cache is valid
            
        Returns:
            WeatherReport with current conditions
        """
        now = datetime.now()
        
        # Check cache
        if (
            not force_refresh
            and self._cached_weather is not None
            and self._cached_weather_time is not None
        ):
            cache_age = (now - self._cached_weather_time).total_seconds() / 60
            if cache_age < self._cache_weather_minutes:
                return self._cached_weather
        
        # Fetch fresh data
        try:
            report = self._weather_client.fetch_current_weather()
            self._cached_weather = report
            self._cached_weather_time = now
            return report
        except weather.WeatherServiceError as e:
            # If weather fails but we have cached data, use it
            if self._cached_weather is not None:
                return self._cached_weather
            raise

    def get_weather_forecast(self, *, force_refresh: bool = False) -> weather.WeatherForecast:
        """Get 1-hour ahead weather forecast for ML predictions.
        
        Args:
            force_refresh: Force refresh even if cache is valid
            
        Returns:
            WeatherForecast with 1-hour ahead predictions
        """
        try:
            forecast = self._weather_client.fetch_1hour_forecast()
            return forecast
        except weather.WeatherServiceError as e:
            # If forecast fails, use current weather as fallback
            try:
                current = self.get_weather_data(force_refresh=force_refresh)
                # Convert WeatherReport to WeatherForecast-like object
                return weather.WeatherForecast(
                    forecast_time=datetime.now() + timedelta(hours=1),
                    temperature_c=current.temperature_c,
                    relative_humidity=current.relative_humidity,
                    wind_speed_kmh=current.wind_speed_kmh,
                    weather_code=current.weather_code,
                    weather_description=current.weather_description,
                    precipitation_mm=getattr(current, 'precipitation_mm', None),
                    cloud_cover=getattr(current, 'cloud_cover', None),
                    probability_precipitation=None,
                )
            except Exception:
                # If everything fails, return a minimal forecast
                return weather.WeatherForecast(
                    forecast_time=datetime.now() + timedelta(hours=1),
                    temperature_c=None,
                    relative_humidity=None,
                    wind_speed_kmh=None,
                    weather_code=None,
                    weather_description=None,
                    precipitation_mm=None,
                    cloud_cover=None,
                    probability_precipitation=None,
                )

    def create_ml_dataset(
        self,
        *,
        air_quality_df: pd.DataFrame | None = None,
        weather_report: weather.WeatherReport | None = None,
        weather_forecast: weather.WeatherForecast | None = None,
        use_forecast: bool = True,
        min_records: int = 1000,
    ) -> pd.DataFrame:
        """Create a structured dataset ready for ML model training/prediction.
        
        This method:
        1. Takes air quality measurements by station
        2. Adds weather data (applied to all stations in Madrid)
        3. Adds station metadata (type, location)
        4. Creates temporal features
        5. Creates derived features (synergies between air quality and weather)
        6. Pivots pollutants into columns (one row per station+time)
        
        Args:
            air_quality_df: Optional air quality DataFrame (fetches if None)
            weather_report: Optional weather report (fetches if None)
            min_records: Minimum number of records required (default 1000)
            accumulate_historical: If True, keeps all historical measurements
            
        Returns:
            DataFrame with one row per station+time combination, ready for ML model
        """
        # Get data if not provided
        if air_quality_df is None:
            air_quality_df = self.collect_air_quality_data()
        
        # Get weather data - prefer forecast for ML predictions
        if use_forecast:
            if weather_forecast is None:
                weather_forecast = self.get_weather_forecast()
            # Use forecast data for ML features
            weather_data = weather_forecast
        else:
            if weather_report is None:
                weather_report = self.get_weather_data()
            weather_data = weather_report

        if air_quality_df.empty:
            return pd.DataFrame()

        # Work with a copy
        df = air_quality_df.copy()

        # Parse measurement_time if it's a string
        if df["measurement_time"].dtype == "object":
            df["measurement_time"] = pd.to_datetime(df["measurement_time"])

        # Filter only valid measurements
        df = df[df["is_valid"] == True].copy()
        
        # Pivot pollutants to columns (one row per station+time combination)
        # This keeps all historical measurements, not just the latest
        pollutant_pivot = df.pivot_table(
            index=["station_code", "measurement_time"],
            columns="pollutant",
            values="value",
            aggfunc="mean",  # Use mean if multiple measurements for same station+time
        ).reset_index()

        # Rename columns to use pollutant names
        pollutant_columns = {}
        for col in pollutant_pivot.columns:
            if col == "station_code":
                continue
            try:
                pollutant_code = int(col)
                pollutant_info = master_data.get_pollutant_info(pollutant_code)
                if pollutant_info:
                    # Use normalized name for model (no2, o3, pm10, pm25, etc.)
                    normalized = master_data.normalize_pollutant_code(pollutant_code)
                    pollutant_columns[col] = normalized
                else:
                    pollutant_columns[col] = f"pollutant_{col}"
            except (ValueError, TypeError):
                pollutant_columns[col] = str(col)

        pollutant_pivot = pollutant_pivot.rename(columns=pollutant_columns)

        # Use all measurements (not just latest)
        ml_df = pollutant_pivot.copy()

        # Add station metadata
        station_features = []
        unique_stations = ml_df["station_code"].unique()
        for station_code in unique_stations:
            station_info = master_data.get_station_info(station_code)
            if station_info:
                station_features.append({
                    "station_code": station_code,
                    "station_name": station_info["name"],
                    "station_type": station_info["type"],
                    "station_district": station_info["district"],
                    "station_latitude": station_info["latitude"],
                    "station_longitude": station_info["longitude"],
                })
            else:
                station_features.append({
                    "station_code": station_code,
                    "station_name": f"Estación {station_code}",
                    "station_type": "Unknown",
                    "station_district": "Unknown",
                    "station_latitude": None,
                    "station_longitude": None,
                })

        station_df = pd.DataFrame(station_features)
        ml_df = ml_df.merge(station_df, on="station_code", how="left")

        # Add temporal features
        if "measurement_time" in ml_df.columns:
            ml_df["measurement_time"] = pd.to_datetime(ml_df["measurement_time"])
            ml_df["hour"] = ml_df["measurement_time"].dt.hour
            ml_df["day_of_week"] = ml_df["measurement_time"].dt.dayofweek
            ml_df["month"] = ml_df["measurement_time"].dt.month
            ml_df["is_weekend"] = ml_df["day_of_week"].isin([5, 6]).astype(int)
            ml_df["is_rush_hour"] = (
                ml_df["hour"].isin([7, 8, 9, 18, 19, 20]).astype(int)
            )

        # Add weather features to all rows - use forecast if available
        if weather_data:
            if isinstance(weather_data, weather.WeatherForecast):
                ml_df["weather_temperature_c"] = weather_data.temperature_c
                ml_df["weather_humidity"] = weather_data.relative_humidity
                ml_df["weather_wind_speed_kmh"] = weather_data.wind_speed_kmh
                ml_df["weather_code"] = weather_data.weather_code
                ml_df["weather_description"] = weather_data.weather_description
                ml_df["weather_forecast_time"] = weather_data.forecast_time
                ml_df["weather_precipitation_mm"] = weather_data.precipitation_mm
                ml_df["weather_cloud_cover"] = weather_data.cloud_cover
                ml_df["weather_precipitation_probability"] = weather_data.probability_precipitation
            else:
                # Fallback to current weather
                ml_df["weather_temperature_c"] = weather_data.temperature_c
                ml_df["weather_humidity"] = weather_data.relative_humidity
                ml_df["weather_wind_speed_kmh"] = weather_data.wind_speed_kmh
                ml_df["weather_code"] = weather_data.weather_code
                ml_df["weather_description"] = weather_data.weather_description
                ml_df["weather_observed_at"] = weather_data.observed_at
                ml_df["weather_precipitation_mm"] = getattr(weather_data, 'precipitation_mm', None)
                ml_df["weather_cloud_cover"] = getattr(weather_data, 'cloud_cover', None)

        # Create derived/synergy features (including precipitation features)
        ml_df = self._create_synergy_features(ml_df)
        
        # Ensure precipitation features exist even if weather data doesn't have them
        if "weather_precipitation_mm" not in ml_df.columns:
            ml_df["weather_precipitation_mm"] = 0
        if "weather_precipitation_probability" not in ml_df.columns:
            ml_df["weather_precipitation_probability"] = 0
        if "precipitation_risk" not in ml_df.columns:
            ml_df["precipitation_risk"] = 0
        if "high_precipitation_risk" not in ml_df.columns:
            ml_df["high_precipitation_risk"] = 0

        # Ensure critical pollutant columns exist (fill with NaN if missing)
        critical_pollutants = ["no2", "o3", "pm10", "pm25", "no", "nox", "so2", "co"]
        for pollutant in critical_pollutants:
            if pollutant not in ml_df.columns:
                ml_df[pollutant] = None

        # Sort by station code and time for consistency
        ml_df = ml_df.sort_values(["station_code", "measurement_time"]).reset_index(drop=True)
        
        # Check minimum records requirement
        if len(ml_df) < min_records:
            import warnings
            warnings.warn(
                f"Dataset tiene solo {len(ml_df)} registros. Se requiere al menos {min_records}. "
                "Considera acumular datos de múltiples días o ejecuciones.",
                UserWarning
            )

        return ml_df

    def _create_synergy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features that capture synergies between air quality and weather.
        
        These features help the model understand interactions like:
        - High wind + low pollution = good for running
        - Low wind + high pollution = bad for running
        - Temperature affects O3 formation
        - Weekend + low traffic pollution = better conditions
        """
        df = df.copy()

        # Wind-pollution synergy
        if "weather_wind_speed_kmh" in df.columns:
            wind = df["weather_wind_speed_kmh"].fillna(0)
            
            # Wind helps disperse pollution
            for pollutant in ["no2", "o3", "pm10", "pm25"]:
                if pollutant in df.columns:
                    pollution = df[pollutant].fillna(0)
                    # Higher wind + lower pollution = better
                    df[f"wind_{pollutant}_synergy"] = (
                        wind / (pollution + 1)
                    ).fillna(0)
            
            # Wind strength indicators
            df["wind_strong"] = (wind > 20).astype(int)
            df["wind_weak"] = (wind < 5).astype(int)
            df["wind_moderate"] = ((wind >= 5) & (wind <= 20)).astype(int)

        # Temperature-O3 synergy (O3 forms more in high temperatures)
        if "weather_temperature_c" in df.columns and "o3" in df.columns:
            temp = df["weather_temperature_c"].fillna(0)
            o3 = df["o3"].fillna(0)
            # High temp + high O3 = worse
            df["temp_o3_synergy"] = (temp * o3 / 100).fillna(0)
            df["high_temp_high_o3"] = (
                (temp > 25) & (o3 > 50)
            ).astype(int)

        # Overall air quality index (weighted combination)
        pollutant_weights = {
            "no2": 0.25,
            "o3": 0.25,
            "pm10": 0.20,
            "pm25": 0.20,
            "no": 0.05,
            "so2": 0.05,
        }
        
        aqi = pd.Series(0.0, index=df.index)
        for pollutant, weight in pollutant_weights.items():
            if pollutant in df.columns:
                values = df[pollutant].fillna(0)
                # Normalize to 0-100 scale (rough approximation)
                normalized = (values / 100).clip(0, 1) * 100
                aqi += normalized * weight
        
        df["air_quality_index"] = aqi

        # Running suitability score (preliminary, model will refine)
        suitability = pd.Series(100.0, index=df.index)
        
        # Penalize high pollution
        if "air_quality_index" in df.columns:
            suitability -= df["air_quality_index"] * 0.5
        
        # Bonus for strong wind
        if "wind_strong" in df.columns:
            suitability += df["wind_strong"] * 10
        
        # Penalize weak wind
        if "wind_weak" in df.columns:
            suitability -= df["wind_weak"] * 5
        
        # Penalize bad weather
        if "weather_code" in df.columns:
            bad_weather_codes = [61, 63, 65, 71, 73, 75, 80, 81, 82, 95, 96, 99]
            is_bad_weather = df["weather_code"].isin(bad_weather_codes).astype(int)
            suitability -= is_bad_weather * 20
        
        df["running_suitability_preliminary"] = suitability.clip(0, 100)

        # Station type impact (traffic stations have higher base pollution)
        if "station_type" in df.columns:
            df["is_traffic_station"] = (df["station_type"] == "Tráfico").astype(int)
            df["is_suburban_station"] = (df["station_type"] == "Suburbana").astype(int)

        return df

    def save_ml_dataset(
        self,
        output_path: Path | str,
        *,
        air_quality_df: pd.DataFrame | None = None,
        weather_report: weather.WeatherReport | None = None,
    ) -> Path:
        """Create and save ML-ready dataset to CSV.
        
        Args:
            output_path: Path to save the dataset
            air_quality_df: Optional air quality DataFrame
            weather_report: Optional weather report
            
        Returns:
            Path where dataset was saved
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        ml_df = self.create_ml_dataset(
            air_quality_df=air_quality_df,
            weather_report=weather_report,
        )
        
        # Convert datetime columns to strings for CSV
        for col in ml_df.columns:
            if pd.api.types.is_datetime64_any_dtype(ml_df[col]):
                ml_df[col] = ml_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        
        ml_df.to_csv(output_path, index=False)
        return output_path


__all__ = ["DataCollector"]

