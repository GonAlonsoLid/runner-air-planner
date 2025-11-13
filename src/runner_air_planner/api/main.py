"""FastAPI backend for Runner Air Planner."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from runner_air_planner.data_pipeline import DataCollector, get_station_info, load_accumulated_dataset
from runner_air_planner.ml.model import MODELS_DIR, RunningSuitabilityModel


app = FastAPI(title="Runner Air Planner API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración
DATASET_PATH = Path("data/ml_dataset_accumulated.csv")
MODEL_PATH = MODELS_DIR / "running_model.pkl"


class PredictionRequest(BaseModel):
    use_realtime: bool = False


class StationData(BaseModel):
    station_code: str
    station_name: str
    latitude: float
    longitude: float
    station_type: str
    aqi: float
    no2: float | None
    o3: float | None
    pm10: float | None
    pm25: float | None
    temperature: float | None
    wind_speed: float | None
    humidity: float | None
    is_good_to_run: bool | None = None
    prob_good: float | None = None


@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "service": "Runner Air Planner API"}


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/data/realtime")
async def get_realtime_data():
    """Get real-time air quality and weather data."""
    try:
        collector = DataCollector()
        air_quality_df = collector.collect_air_quality_data()
        weather_forecast = collector.get_weather_forecast()
        
        if air_quality_df.empty:
            raise HTTPException(status_code=503, detail="No air quality data available")
        
        ml_df = collector.create_ml_dataset(
            air_quality_df=air_quality_df,
            weather_forecast=weather_forecast,
            use_forecast=True,
            min_records=0,
        )
        
        if ml_df.empty:
            raise HTTPException(status_code=503, detail="No processed data available")
        
        # Convert to stations data
        latest_data = ml_df.sort_values("measurement_time", ascending=False).groupby("station_code").first()
        
        stations = []
        for station_code, row in latest_data.iterrows():
            station_info = get_station_info(station_code)
            if not station_info:
                continue
            
            stations.append({
                "station_code": str(station_code),
                "station_name": station_info.get("name", f"Estación {station_code}"),
                "latitude": station_info.get("latitude", 0),
                "longitude": station_info.get("longitude", 0),
                "station_type": station_info.get("type", "Unknown"),
                "aqi": float(row.get("air_quality_index", 0)) if pd.notna(row.get("air_quality_index")) else 0,
                "no2": float(row.get("no2")) if pd.notna(row.get("no2")) else None,
                "o3": float(row.get("o3")) if pd.notna(row.get("o3")) else None,
                "pm10": float(row.get("pm10")) if pd.notna(row.get("pm10")) else None,
                "pm25": float(row.get("pm25")) if pd.notna(row.get("pm25")) else None,
                "temperature": float(row.get("weather_temperature_c")) if pd.notna(row.get("weather_temperature_c")) else None,
                "wind_speed": float(row.get("weather_wind_speed_kmh")) if pd.notna(row.get("weather_wind_speed_kmh")) else None,
                "humidity": float(row.get("weather_humidity")) if pd.notna(row.get("weather_humidity")) else None,
            })
        
        weather_data = None
        if weather_forecast:
            weather_data = {
                "temperature": weather_forecast.temperature_c,
                "humidity": weather_forecast.relative_humidity,
                "wind_speed": weather_forecast.wind_speed_kmh,
                "weather_code": weather_forecast.weather_code,
                "description": weather_forecast.weather_description,
                "precipitation": weather_forecast.precipitation_mm,
                "cloud_cover": weather_forecast.cloud_cover,
                "precipitation_probability": weather_forecast.probability_precipitation,
                "is_forecast": True,
            }
        
        return {
            "stations": stations,
            "weather": weather_data,
            "timestamp": weather_forecast.forecast_time.isoformat() if weather_forecast and weather_forecast.forecast_time else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/historical")
async def get_historical_data():
    """Get historical dataset."""
    try:
        df = load_accumulated_dataset(DATASET_PATH)
        if df.empty:
            return {"stations": [], "count": 0}
        
        latest_data = df.sort_values("measurement_time", ascending=False).groupby("station_code").first()
        
        stations = []
        for station_code, row in latest_data.iterrows():
            station_info = get_station_info(station_code)
            if not station_info:
                continue
            
            stations.append({
                "station_code": str(station_code),
                "station_name": station_info.get("name", f"Estación {station_code}"),
                "latitude": station_info.get("latitude", 0),
                "longitude": station_info.get("longitude", 0),
                "station_type": station_info.get("type", "Unknown"),
                "aqi": float(row.get("air_quality_index", 0)) if pd.notna(row.get("air_quality_index")) else 0,
                "no2": float(row.get("no2")) if pd.notna(row.get("no2")) else None,
                "o3": float(row.get("o3")) if pd.notna(row.get("o3")) else None,
                "pm10": float(row.get("pm10")) if pd.notna(row.get("pm10")) else None,
                "pm25": float(row.get("pm25")) if pd.notna(row.get("pm25")) else None,
                "temperature": float(row.get("weather_temperature_c")) if pd.notna(row.get("weather_temperature_c")) else None,
                "wind_speed": float(row.get("weather_wind_speed_kmh")) if pd.notna(row.get("weather_wind_speed_kmh")) else None,
                "humidity": float(row.get("weather_humidity")) if pd.notna(row.get("weather_humidity")) else None,
            })
        
        return {"stations": stations, "count": len(stations)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict")
async def predict(request: PredictionRequest):
    """Run ML model predictions."""
    try:
        # Load model
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=404, detail="Model not found. Train the model first.")
        
        with open(MODEL_PATH, "rb") as f:
            model: RunningSuitabilityModel = pickle.load(f)
        
        # Get data - use forecast for predictions
        if request.use_realtime:
            collector = DataCollector()
            air_quality_df = collector.collect_air_quality_data()
            weather_forecast = collector.get_weather_forecast()
            df = collector.create_ml_dataset(
                air_quality_df=air_quality_df,
                weather_forecast=weather_forecast,
                use_forecast=True,
                min_records=0,
            )
        else:
            df = load_accumulated_dataset(DATASET_PATH)
        
        if df.empty:
            raise HTTPException(status_code=503, detail="No data available")
        
        # Make predictions
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)
        
        # Add predictions to dataframe
        df["prediction"] = predictions
        df["prob_good"] = probabilities["prob_good"]
        
        # Get latest data per station
        latest_data = df.sort_values("measurement_time", ascending=False).groupby("station_code").first()
        
        results = []
        for station_code, row in latest_data.iterrows():
            station_info = get_station_info(station_code)
            if not station_info:
                continue
            
            results.append({
                "station_code": str(station_code),
                "station_name": station_info.get("name", f"Estación {station_code}"),
                "latitude": station_info.get("latitude", 0),
                "longitude": station_info.get("longitude", 0),
                "station_type": station_info.get("type", "Unknown"),
                "aqi": float(row.get("air_quality_index", 0)) if pd.notna(row.get("air_quality_index")) else 0,
                "no2": float(row.get("no2")) if pd.notna(row.get("no2")) else None,
                "o3": float(row.get("o3")) if pd.notna(row.get("o3")) else None,
                "pm10": float(row.get("pm10")) if pd.notna(row.get("pm10")) else None,
                "pm25": float(row.get("pm25")) if pd.notna(row.get("pm25")) else None,
                "temperature": float(row.get("weather_temperature_c")) if pd.notna(row.get("weather_temperature_c")) else None,
                "wind_speed": float(row.get("weather_wind_speed_kmh")) if pd.notna(row.get("weather_wind_speed_kmh")) else None,
                "humidity": float(row.get("weather_humidity")) if pd.notna(row.get("weather_humidity")) else None,
                "is_good_to_run": bool(row.get("prediction") == 1),
                "prob_good": float(row.get("prob_good")) if pd.notna(row.get("prob_good")) else None,
            })
        
        # Sort by probability
        results.sort(key=lambda x: x["prob_good"] or 0, reverse=True)
        
        return {
            "predictions": results,
            "total": len(results),
            "good_count": sum(1 for r in results if r["is_good_to_run"]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

