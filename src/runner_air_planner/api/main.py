"""FastAPI backend for Runner Air Planner."""

from __future__ import annotations

import asyncio
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    import asyncio
    try:
        collector = DataCollector()
        
        # Get air quality data first (fast)
        air_quality_df = collector.collect_air_quality_data()
        
        if air_quality_df.empty:
            raise HTTPException(status_code=503, detail="No air quality data available")
        
        # Try to get forecast with timeout, but don't fail if it doesn't work
        weather_forecast = None
        try:
            # Run forecast in executor with timeout
            loop = asyncio.get_event_loop()
            weather_forecast = await asyncio.wait_for(
                loop.run_in_executor(None, collector.get_weather_forecast),
                timeout=10.0  # 10 second timeout for forecast
            )
        except (asyncio.TimeoutError, Exception) as e:
            # Fallback to current weather if forecast fails or times out
            try:
                weather_report = await loop.run_in_executor(
                    None, collector.get_weather_data
                )
                # Convert to forecast format
                from datetime import timedelta
                weather_forecast = weather.WeatherForecast(
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
                weather_forecast = None
        
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
        station_weather_cache = {}
        
        stations = []
        for station_code, row in latest_data.iterrows():
            station_info = get_station_info(station_code)
            if not station_info:
                continue
            
            station_lat = station_info.get("latitude")
            station_lon = station_info.get("longitude")
            
            # Try to get station-specific weather if coordinates are available
            temp_value = None
            wind_value = None
            humidity_value = None
            
            if station_lat and station_lon:
                # Use station coordinates as cache key
                cache_key = f"{station_lat:.2f},{station_lon:.2f}"
                
                if cache_key not in station_weather_cache:
                    try:
                        # Get weather for this specific station location
                        station_forecast = await loop.run_in_executor(
                            None,
                            lambda: collector._weather_client.fetch_weather_for_location(
                                station_lat, station_lon
                            )
                        )
                        station_weather_cache[cache_key] = station_forecast
                    except Exception as e:
                        # Fallback to city-wide forecast
                        station_weather_cache[cache_key] = weather_forecast
                
                station_weather = station_weather_cache[cache_key]
                
                if station_weather:
                    temp_value = station_weather.temperature_c
                    wind_value = station_weather.wind_speed_kmh
                    humidity_value = station_weather.relative_humidity
            
            # Fallback to city-wide forecast if station-specific failed
            if temp_value is None:
                if weather_forecast and weather_forecast.temperature_c is not None:
                    temp_value = float(weather_forecast.temperature_c)
                elif pd.notna(row.get("weather_temperature_c")):
                    temp_value = float(row.get("weather_temperature_c"))
            
            if wind_value is None:
                if weather_forecast and weather_forecast.wind_speed_kmh is not None:
                    wind_value = float(weather_forecast.wind_speed_kmh)
                elif pd.notna(row.get("weather_wind_speed_kmh")):
                    wind_value = float(row.get("weather_wind_speed_kmh"))
            
            if humidity_value is None:
                if weather_forecast and weather_forecast.relative_humidity is not None:
                    humidity_value = float(weather_forecast.relative_humidity)
                elif pd.notna(row.get("weather_humidity")):
                    humidity_value = float(row.get("weather_humidity"))
            
            stations.append({
                "station_code": str(station_code),
                "station_name": station_info.get("name", f"Estación {station_code}"),
                "latitude": station_lat or 0,
                "longitude": station_lon or 0,
                "station_type": station_info.get("type", "Unknown"),
                "aqi": float(row.get("air_quality_index", 0)) if pd.notna(row.get("air_quality_index")) else 0,
                "no2": float(row.get("no2")) if pd.notna(row.get("no2")) else None,
                "o3": float(row.get("o3")) if pd.notna(row.get("o3")) else None,
                "pm10": float(row.get("pm10")) if pd.notna(row.get("pm10")) else None,
                "pm25": float(row.get("pm25")) if pd.notna(row.get("pm25")) else None,
                "temperature": float(temp_value) if temp_value is not None else None,
                "wind_speed": float(wind_value) if wind_value is not None else None,
                "humidity": float(humidity_value) if humidity_value is not None else None,
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
            try:
                weather_forecast = await asyncio.wait_for(
                    loop.run_in_executor(None, collector.get_weather_forecast),
                    timeout=10.0
                )
            except (asyncio.TimeoutError, Exception):
                # Fallback to current weather
                weather_report = await loop.run_in_executor(
                    None, collector.get_weather_data
                )
                weather_forecast = weather.WeatherForecast(
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
            
            temp_value = None
            wind_value = None
            humidity_value = None
            
            if station_lat and station_lon and request.use_realtime:
                try:
                    station_weather = await loop.run_in_executor(
                        None,
                        lambda: collector._weather_client.fetch_weather_for_location(
                            station_lat, station_lon
                        )
                    )
                    if station_weather:
                        temp_value = station_weather.temperature_c
                        wind_value = station_weather.wind_speed_kmh
                        humidity_value = station_weather.relative_humidity
                except Exception:
                    pass
            
            # Fallback to row data
            if temp_value is None:
                temp_value = row.get("weather_temperature_c") if pd.notna(row.get("weather_temperature_c")) else None
            if wind_value is None:
                wind_value = row.get("weather_wind_speed_kmh") if pd.notna(row.get("weather_wind_speed_kmh")) else None
            if humidity_value is None:
                humidity_value = row.get("weather_humidity") if pd.notna(row.get("weather_humidity")) else None
            
            # Obtener running_score ANTES de generar la explicación
            running_score = float(row.get("running_score", 0)) if pd.notna(row.get("running_score")) else 0
            
            # Generar explicación del porqué (con el score correcto)
            explanation = _generate_explanation(
                aqi=float(row.get("air_quality_index", 0)) if pd.notna(row.get("air_quality_index")) else 0,
                no2=float(row.get("no2")) if pd.notna(row.get("no2")) else None,
                o3=float(row.get("o3")) if pd.notna(row.get("o3")) else None,
                pm10=float(row.get("pm10")) if pd.notna(row.get("pm10")) else None,
                pm25=float(row.get("pm25")) if pd.notna(row.get("pm25")) else None,
                temperature=float(temp_value) if temp_value is not None else None,
                wind_speed=float(wind_value) if wind_value is not None else None,
                humidity=float(humidity_value) if humidity_value is not None else None,
                is_good=bool(row.get("prediction") == 1),
                prob_good=float(row.get("prob_good")) if pd.notna(row.get("prob_good")) else None,
                station_type=station_info.get("type", "Unknown"),
                running_score=running_score,  # Pasar el score correcto
            )
            
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
                "temperature": float(temp_value) if temp_value is not None else None,
                "wind_speed": float(wind_value) if wind_value is not None else None,
                "humidity": float(humidity_value) if humidity_value is not None else None,
                "is_good_to_run": bool(row.get("prediction") == 1),
                "prob_good": float(row.get("prob_good")) if pd.notna(row.get("prob_good")) else None,
                "prob_not_good": float(row.get("prob_not_good")) if pd.notna(row.get("prob_not_good")) else None,
                "running_score": running_score,  # Score numérico 0-100
                "explanation": explanation,
            })
        
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

