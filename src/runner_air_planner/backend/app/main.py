"""FastAPI application exposing air quality predictions for Madrid stations."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans

from . import storage
from . import weather

app = FastAPI(
    title="Runner Air Planner API",
    version="0.2.0",
    description="Service that clusters Madrid's air quality measurements to suggest running conditions.",
)


class Prediction(BaseModel):
    station_code: str | None = Field(None, description="Station identifier from the Madrid dataset")
    measurement_time: datetime = Field(description="Timestamp of the measurement used for the prediction")
    cluster: int = Field(description="Cluster assigned by the unsupervised model")
    air_quality_label: str = Field(description="Human-readable description of the running conditions")
    pollutants: dict[str, float] = Field(description="Pollutant concentration values used by the model")


class Healthcheck(BaseModel):
    trained: bool
    samples: int
    features: list[str]


class WeatherSnapshot(BaseModel):
    observed_at: datetime = Field(description="Timestamp of the weather observation in Madrid time")
    temperature_c: float | None = Field(
        None, description="Air temperature in degrees Celsius as reported by Open-Meteo"
    )
    relative_humidity: float | None = Field(
        None, description="Relative humidity percentage for the observation"
    )
    wind_speed_kmh: float | None = Field(
        None, description="Wind speed in km/h for the observation"
    )
    weather_description: str | None = Field(
        None, description="Short human-readable label for the reported weather code"
    )


class AirQualityClusteringModel:
    """Simple wrapper around ``KMeans`` with human-readable cluster labels."""

    def __init__(self, *, features: list[str] | None = None, max_clusters: int = 3) -> None:
        self.features = features or storage.POLLUTANT_FEATURES
        self.max_clusters = max_clusters
        self.model: KMeans | None = None
        self.fill_values: dict[str, float] = {}
        self.cluster_labels: dict[int, str] = {}

    def fit(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            raise ValueError("Cannot train air quality model with an empty dataset")

        cleaned = self._prepare_features(frame, fit=True)
        n_clusters = min(self.max_clusters, len(cleaned))
        if n_clusters < 1:
            raise ValueError("Not enough data to build clusters")
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.model.fit(cleaned)
        self._build_cluster_labels()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been trained yet")
        features = self._prepare_features(frame)
        return self.model.predict(features)

    def label_for_cluster(self, cluster: int) -> str:
        if cluster not in self.cluster_labels:
            return "Desconocido"
        return self.cluster_labels[cluster]

    def _prepare_features(self, frame: pd.DataFrame, *, fit: bool = False) -> np.ndarray:
        matrix = frame.copy()
        for column in self.features:
            if column not in matrix:
                matrix[column] = float("nan")
            matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
        if fit or not self.fill_values:
            self.fill_values = {}
            for column in self.features:
                values = matrix[column].to_numpy(dtype=float)
                mean = np.nanmean(values)
                mean_value = float(mean) if not np.isnan(mean) else 0.0
                self.fill_values[column] = mean_value
        for column in self.features:
            matrix[column] = matrix[column].fillna(self.fill_values[column])
        return matrix[self.features].to_numpy(dtype=float)

    def _build_cluster_labels(self) -> None:
        assert self.model is not None
        centroid_scores = self.model.cluster_centers_.sum(axis=1)
        ordering = np.argsort(centroid_scores)
        base_labels = [
            "Excelente para salir a correr",
            "Precaución moderada",
            "Evitar correr al aire libre",
            "Condiciones muy adversas",
            "Condiciones extremas",
        ]
        self.cluster_labels = {}
        for index, cluster in enumerate(ordering):
            label_index = min(index, len(base_labels) - 1)
            self.cluster_labels[int(cluster)] = base_labels[label_index]


@lru_cache(maxsize=1)
def load_feature_frame() -> pd.DataFrame:
    raw = storage.load_raw_measurements()
    return storage.prepare_station_feature_frame(raw)


@lru_cache(maxsize=1)
def load_model() -> AirQualityClusteringModel:
    frame = load_feature_frame()
    model = AirQualityClusteringModel()
    model.fit(frame)
    return model


@lru_cache(maxsize=1)
def get_weather_client() -> weather.WeatherClient:
    return weather.WeatherClient()


@app.get("/health", response_model=Healthcheck)
def healthcheck() -> Healthcheck:
    try:
        frame = load_feature_frame()
    except FileNotFoundError as error:  # pragma: no cover - runtime guard
        raise HTTPException(status_code=500, detail=str(error)) from error

    trained = False
    sample_count = len(frame)
    try:
        load_model()
        trained = True
    except ValueError:
        trained = False
    return Healthcheck(trained=trained, samples=sample_count, features=storage.POLLUTANT_FEATURES)


@app.get("/predictions", response_model=list[Prediction])
def get_predictions() -> list[Prediction]:
    try:
        frame = load_feature_frame()
    except FileNotFoundError as error:  # pragma: no cover - runtime guard
        raise HTTPException(status_code=500, detail=str(error)) from error

    if frame.empty:
        raise HTTPException(status_code=503, detail="No hay datos de calidad del aire disponibles todavía")

    try:
        model = load_model()
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    clusters = model.predict(frame)
    predictions: list[Prediction] = []
    pollutant_columns = [col for col in frame.columns if col in model.features]
    for row, cluster in zip(frame.to_dict(orient="records"), clusters):
        measurement_time = row.get("measurement_time")
        if isinstance(measurement_time, str):
            measurement_time = datetime.fromisoformat(measurement_time)
        pollutants = {
            key: float(row[key])
            for key in pollutant_columns
            if row.get(key) is not None and not pd.isna(row.get(key))
        }
        predictions.append(
            Prediction(
                station_code=row.get("station_code"),
                measurement_time=measurement_time,
                cluster=int(cluster),
                air_quality_label=model.label_for_cluster(int(cluster)),
                pollutants=pollutants,
            )
        )
    return predictions


@app.get("/stations")
def list_stations() -> list[dict[str, Any]]:
    try:
        frame = load_feature_frame()
    except FileNotFoundError as error:  # pragma: no cover - runtime guard
        raise HTTPException(status_code=500, detail=str(error)) from error

    return (
        frame[["station_code", "measurement_time"]]
        .sort_values("station_code")
        .to_dict(orient="records")
    )


@app.get("/weather", response_model=WeatherSnapshot)
def get_current_weather() -> WeatherSnapshot:
    client = get_weather_client()
    try:
        report = client.fetch_current_weather()
    except weather.WeatherServiceError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    return WeatherSnapshot(
        observed_at=report.observed_at,
        temperature_c=report.temperature_c,
        relative_humidity=report.relative_humidity,
        wind_speed_kmh=report.wind_speed_kmh,
        weather_description=report.weather_description,
    )


__all__ = ["app"]
