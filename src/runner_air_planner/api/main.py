"""FastAPI backend for Runner Air Planner."""

from __future__ import annotations

import asyncio
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd

from runner_air_planner.data_pipeline import (
    DataCollector,
    get_station_info,
    load_accumulated_dataset,
    weather,
)
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


async def _get_weather_forecast_with_fallback(collector: DataCollector) -> weather.WeatherForecast | None:
    """Get weather forecast with fallback to current weather if forecast fails."""
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, collector.get_weather_forecast),
            timeout=10.0
        )
    except (asyncio.TimeoutError, Exception):
        # Fallback to current weather
        try:
            weather_report = await loop.run_in_executor(
                None, collector.get_weather_data
            )
            return weather.WeatherForecast(
                forecast_time=datetime.now() + timedelta(hours=1),
                temperature_c=weather_report.temperature_c,
                relative_humidity=weather_report.relative_humidity,
                wind_speed_kmh=weather_report.wind_speed_kmh,
                weather_code=weather_report.weather_code,
                weather_description=weather_report.weather_description,
                precipitation_mm=getattr(weather_report, 'precipitation_mm', None),
                cloud_cover=getattr(weather_report, 'cloud_cover', None),
                probability_precipitation=None,
            )
        except Exception:
            return None


def _extract_weather_value(
    station_weather: weather.WeatherForecast | None,
    forecast: weather.WeatherForecast | None,
    row: pd.Series,
    attr_name: str,
) -> float | None:
    """Extract weather value from station-specific, forecast, or row data."""
    # Try station-specific weather first
    if station_weather:
        value = getattr(station_weather, attr_name, None)
        if value is not None:
            return float(value)
    
    # Fallback to forecast
    if forecast:
        value = getattr(forecast, attr_name, None)
        if value is not None:
            return float(value)
    
    # Finally try row data
    row_key_map = {
        "temperature_c": "weather_temperature_c",
        "wind_speed_kmh": "weather_wind_speed_kmh",
        "relative_humidity": "weather_humidity",
    }
    row_key = row_key_map.get(attr_name, f"weather_{attr_name}")
    row_value = row.get(row_key)
    
    return float(row_value) if pd.notna(row_value) else None


def _create_station_dict(
    station_code: str,
    station_info: dict[str, Any],
    row: pd.Series,
    station_weather: weather.WeatherForecast | None = None,
    forecast: weather.WeatherForecast | None = None,
) -> dict[str, Any]:
    """Create station dictionary from station info and data row."""
    return {
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
        "temperature": _extract_weather_value(station_weather, forecast, row, "temperature_c"),
        "wind_speed": _extract_weather_value(station_weather, forecast, row, "wind_speed_kmh"),
        "humidity": _extract_weather_value(station_weather, forecast, row, "relative_humidity"),
    }


def _generate_explanation(
    *,
    aqi: float,
    no2: float | None,
    o3: float | None,
    pm10: float | None,
    pm25: float | None,
    temperature: float | None,
    wind_speed: float | None,
    humidity: float | None,
    is_good: bool,
    prob_good: float | None,
    station_type: str,
    running_score: float = 0,
) -> dict[str, Any]:
    """Generate explanation for why a station is good/bad for running.
    
    Returns a dictionary with:
    - reasons: List of positive factors
    - warnings: List of negative factors
    - summary: Overall explanation
    """
    reasons = []
    warnings = []
    
    # Analizar calidad del aire
    if aqi < 25:
        reasons.append(f"Calidad del aire excelente (AQI: {aqi:.1f})")
    elif aqi < 50:
        reasons.append(f"Calidad del aire buena (AQI: {aqi:.1f})")
    elif aqi >= 50:
        warnings.append(f"Calidad del aire moderada (AQI: {aqi:.1f})")
    
    # Analizar contaminantes específicos
    if no2 is not None:
        if no2 < 50:
            reasons.append(f"Bajo NO₂ ({no2:.1f} µg/m³)")
        elif no2 > 100:
            warnings.append(f"Alto NO₂ ({no2:.1f} µg/m³)")
    
    if o3 is not None:
        if o3 < 50:
            reasons.append(f"Bajo O₃ ({o3:.1f} µg/m³)")
        elif o3 > 120:
            warnings.append(f"Alto O₃ ({o3:.1f} µg/m³)")
    
    if pm10 is not None:
        if pm10 < 25:
            reasons.append(f"Bajo PM10 ({pm10:.1f} µg/m³)")
        elif pm10 > 50:
            warnings.append(f"Alto PM10 ({pm10:.1f} µg/m³)")
    
    if pm25 is not None:
        if pm25 < 15:
            reasons.append(f"Bajo PM2.5 ({pm25:.1f} µg/m³)")
        elif pm25 > 25:
            warnings.append(f"Alto PM2.5 ({pm25:.1f} µg/m³)")
    
    # Analizar condiciones meteorológicas
    if wind_speed is not None:
        if wind_speed > 20:
            reasons.append(f"Viento fuerte ({wind_speed:.1f} km/h) - dispersa contaminación")
        elif wind_speed < 5:
            warnings.append(f"Poco viento ({wind_speed:.1f} km/h) - contaminación se acumula")
        else:
            reasons.append(f"Viento moderado ({wind_speed:.1f} km/h)")
    
    if temperature is not None:
        if 15 <= temperature <= 25:
            reasons.append(f"Temperatura ideal ({temperature:.1f}°C)")
        elif temperature > 30:
            warnings.append(f"Temperatura alta ({temperature:.1f}°C)")
        elif temperature < 5:
            warnings.append(f"Temperatura baja ({temperature:.1f}°C)")
    
    if humidity is not None:
        if 40 <= humidity <= 70:
            reasons.append(f"Humedad cómoda ({humidity:.0f}%)")
        elif humidity > 80:
            warnings.append(f"Humedad alta ({humidity:.0f}%)")
    
    # Tipo de estación
    if station_type == "Suburbana":
        reasons.append("Zona suburbana - menos tráfico")
    elif station_type == "Tráfico":
        warnings.append("Zona de tráfico - más contaminación")
    
    # Generar resumen basado en score numérico
    if running_score >= 80:
        summary = f"Excelente momento para correr (Score: {running_score:.0f}/100). Condiciones ideales."
    elif running_score >= 60:
        summary = f"Buen momento para correr (Score: {running_score:.0f}/100). Condiciones favorables."
    elif running_score >= 40:
        summary = f"Condiciones moderadas (Score: {running_score:.0f}/100). Considera esperar o usar mascarilla."
    elif running_score >= 20:
        summary = f"Condiciones regulares (Score: {running_score:.0f}/100). No muy recomendado."
    else:
        summary = f"Condiciones no recomendadas (Score: {running_score:.0f}/100). Mejor esperar."
    
    return {
        "reasons": reasons[:5],  # Máximo 5 razones
        "warnings": warnings[:3],  # Máximo 3 advertencias
        "summary": summary,
    }


class PredictionRequest(BaseModel):
    use_realtime: bool = True  # Default to real-time predictions




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
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        collector = DataCollector()
        
        # Get air quality data first (fast)
        air_quality_df = collector.collect_air_quality_data()
        
        if air_quality_df.empty:
            raise HTTPException(status_code=503, detail="No air quality data available")
        
        # Get weather forecast with fallback
        weather_forecast = await _get_weather_forecast_with_fallback(collector)
        
        ml_df = await loop.run_in_executor(
            None,
            lambda: collector.create_ml_dataset(
                air_quality_df=air_quality_df,
                weather_forecast=weather_forecast,
                use_forecast=weather_forecast is not None,
                min_records=0,
            )
        )
        
        if ml_df.empty:
            raise HTTPException(status_code=503, detail="No processed data available")
        
        # Convert to stations data
        latest_data = ml_df.sort_values("measurement_time", ascending=False).groupby("station_code").first()
        
        # Cache for station-specific weather to avoid too many API calls
        station_weather_cache: dict[str, weather.WeatherForecast | None] = {}
        
        stations = []
        for station_code, row in latest_data.iterrows():
            station_info = get_station_info(station_code)
            if not station_info:
                continue
            
            station_lat = station_info.get("latitude")
            station_lon = station_info.get("longitude")
            
            # Try to get station-specific weather if coordinates are available
            station_weather = None
            if station_lat and station_lon:
                cache_key = f"{station_lat:.2f},{station_lon:.2f}"
                if cache_key not in station_weather_cache:
                    try:
                        station_weather_cache[cache_key] = await loop.run_in_executor(
                            None,
                            lambda lat=station_lat, lon=station_lon: collector._weather_client.fetch_weather_for_location(lat, lon)
                        )
                    except Exception:
                        station_weather_cache[cache_key] = weather_forecast
                station_weather = station_weather_cache[cache_key]
            
            stations.append(_create_station_dict(
                station_code, station_info, row, station_weather, weather_forecast
            ))
        
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
            
            stations.append(_create_station_dict(station_code, station_info, row))
        
        return {"stations": stations, "count": len(stations)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict")
async def predict(request: PredictionRequest):
    """Run ML model predictions based on current conditions.
    
    By default, uses real-time data to make predictions. This ensures
    predictions are based on the most current air quality and weather conditions.
    """
    try:
        # Load model
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=404, detail="Model not found. Train the model first.")
        
        with open(MODEL_PATH, "rb") as f:
            model: RunningSuitabilityModel = pickle.load(f)
        
        # Always use real-time data for predictions (current conditions)
        if request.use_realtime:
            loop = asyncio.get_event_loop()
            collector = DataCollector()
            
            # Get real-time air quality data
            air_quality_df = await loop.run_in_executor(
                None, collector.collect_air_quality_data
            )
            
            if air_quality_df.empty:
                raise HTTPException(status_code=503, detail="No air quality data available")
            
            # Get weather forecast for predictions (1 hour ahead)
            weather_forecast = await _get_weather_forecast_with_fallback(collector)
            
            # Create ML dataset with current conditions
            df = await loop.run_in_executor(
                None,
                lambda: collector.create_ml_dataset(
                    air_quality_df=air_quality_df,
                    weather_forecast=weather_forecast,
                    use_forecast=True,
                    min_records=0,
                )
            )
        else:
            # Use historical data if explicitly requested
            df = load_accumulated_dataset(DATASET_PATH)
        
        if df.empty:
            raise HTTPException(status_code=503, detail="No data available")
        
        # Make predictions using ML algorithm
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)
        scores = model.predict_score(df)  # Score numérico 0-100
        
        # Add predictions to dataframe
        df["prediction"] = predictions
        df["prob_good"] = probabilities["prob_good"]
        df["prob_not_good"] = probabilities["prob_not_good"]
        df["running_score"] = scores
        
        # Get latest data per station (most recent measurements)
        latest_data = df.sort_values("measurement_time", ascending=False).groupby("station_code").first()
        
        results = []
        for station_code, row in latest_data.iterrows():
            station_info = get_station_info(station_code)
            if not station_info:
                continue
            
            # Get station-specific weather if available
            station_lat = station_info.get("latitude")
            station_lon = station_info.get("longitude")
            
            station_weather = None
            if station_lat and station_lon and request.use_realtime:
                try:
                    station_weather = await loop.run_in_executor(
                        None,
                        lambda lat=station_lat, lon=station_lon: collector._weather_client.fetch_weather_for_location(lat, lon)
                    )
                except Exception:
                    pass
            
            # Get weather forecast for fallback
            forecast = weather_forecast if request.use_realtime else None
            
            # Obtener running_score ANTES de generar la explicación
            running_score = float(row.get("running_score", 0)) if pd.notna(row.get("running_score")) else 0
            
            # Extract weather values for explanation
            temp_value = _extract_weather_value(station_weather, forecast, row, "temperature_c")
            wind_value = _extract_weather_value(station_weather, forecast, row, "wind_speed_kmh")
            humidity_value = _extract_weather_value(station_weather, forecast, row, "relative_humidity")
            
            # Generar explicación del porqué (con el score correcto)
            explanation = _generate_explanation(
                aqi=float(row.get("air_quality_index", 0)) if pd.notna(row.get("air_quality_index")) else 0,
                no2=float(row.get("no2")) if pd.notna(row.get("no2")) else None,
                o3=float(row.get("o3")) if pd.notna(row.get("o3")) else None,
                pm10=float(row.get("pm10")) if pd.notna(row.get("pm10")) else None,
                pm25=float(row.get("pm25")) if pd.notna(row.get("pm25")) else None,
                temperature=temp_value,
                wind_speed=wind_value,
                humidity=humidity_value,
                is_good=bool(row.get("prediction") == 1),
                prob_good=float(row.get("prob_good")) if pd.notna(row.get("prob_good")) else None,
                station_type=station_info.get("type", "Unknown"),
                running_score=running_score,  # Pasar el score correcto
            )
            
            station_dict = _create_station_dict(
                station_code, station_info, row, station_weather, forecast
            )
            station_dict.update({
                "is_good_to_run": bool(row.get("prediction") == 1),
                "prob_good": float(row.get("prob_good")) if pd.notna(row.get("prob_good")) else None,
                "prob_not_good": float(row.get("prob_not_good")) if pd.notna(row.get("prob_not_good")) else None,
                "running_score": running_score,
                "explanation": explanation,
            })
            results.append(station_dict)
        
        # Sort by running score (best conditions first)
        results.sort(key=lambda x: x.get("running_score", 0) or 0, reverse=True)
        
        return {
            "predictions": results,
            "total": len(results),
            "good_count": sum(1 for r in results if r["is_good_to_run"]),
            "timestamp": datetime.now().isoformat(),
            "data_source": "realtime" if request.use_realtime else "historical",
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

