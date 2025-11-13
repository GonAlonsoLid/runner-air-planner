"""Streamlit dashboard moderno y user-friendly para Runner Air Planner."""

from __future__ import annotations

import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from runner_air_planner.data_pipeline import (
    DataCollector,
    get_station_info,
    load_accumulated_dataset,
)
from runner_air_planner.ml.model import MODELS_DIR, RunningSuitabilityModel


# Configuración de página
st.set_page_config(
    page_title="Runner Air Planner",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS moderno y profesional
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 30px 30px;
    }
    
    .main-header {
        font-size: 4rem;
        font-weight: 900;
        color: white;
        margin: 0;
        text-align: center;
        text-shadow: 0 4px 20px rgba(0,0,0,0.2);
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        border: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.75rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1e293b;
        line-height: 1;
        margin: 0;
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    
    .status-excellent {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .status-good {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: white;
    }
    
    .status-moderate {
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
        color: white;
    }
    
    .status-poor {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    .action-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 15px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .action-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    .realtime-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 50px;
        font-size: 0.875rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.9; transform: scale(1.02); }
    }
    
    .section-title {
        font-size: 2rem;
        font-weight: 800;
        color: #1e293b;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 1rem;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #667eea 0%, #764ba2 100%) 1;
    }
    
    .info-box {
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        padding: 2rem;
        border-radius: 20px;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 1.5rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: #f8fafc;
        padding: 0.5rem;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Ocultar sidebar y elementos de configuración */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stDeployButton {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)


# Configuración automática (sin mostrar al usuario)
DATASET_PATH = Path("data/ml_dataset_accumulated.csv")
MODEL_PATH = MODELS_DIR / "running_model.pkl"


@st.cache_data(ttl=300)
def load_historical_dataset() -> pd.DataFrame:
    """Load historical dataset."""
    try:
        return load_accumulated_dataset(DATASET_PATH)
    except Exception:
        return pd.DataFrame()


@st.cache_resource
def load_model() -> RunningSuitabilityModel | None:
    """Load trained model."""
    if not MODEL_PATH.exists():
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def collect_realtime_data() -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Collect real-time data from APIs."""
    try:
        collector = DataCollector()
        
        # Obtener datos de calidad del aire
        air_quality_df = collector.collect_air_quality_data()
        
        # Obtener datos meteorológicos
        weather_report = collector.get_weather_data()
        
        if air_quality_df.empty:
            return pd.DataFrame(), None
        
        # Crear dataset ML con datos en tiempo real
        ml_df = collector.create_ml_dataset(
            air_quality_df=air_quality_df,
            weather_report=weather_report,
            min_records=0,
        )
        
        # Convert weather report to dict
        weather_dict = None
        if weather_report:
            weather_dict = {
                "temperature": weather_report.temperature_c,
                "humidity": weather_report.relative_humidity,
                "wind_speed": weather_report.wind_speed_kmh,
                "weather_code": weather_report.weather_code,
                "description": weather_report.weather_description,
                "observed_at": weather_report.observed_at.isoformat() if weather_report.observed_at else None,
            }
        
        return ml_df, weather_dict
    except Exception as e:
        st.error(f"Error recopilando datos: {e}")
        return pd.DataFrame(), None


def create_air_quality_map(df: pd.DataFrame, predictions: pd.Series | None = None) -> folium.Map:
    """Create interactive map with air quality data."""
    madrid_center = [40.4168, -3.7038]
    
    m = folium.Map(
        location=madrid_center,
        zoom_start=11,
        tiles="CartoDB positron",
    )
    
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", overlay=False, control=True).add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Modo Oscuro", overlay=False, control=True).add_to(m)
    
    if not df.empty and "station_code" in df.columns:
        latest_data = df.sort_values("measurement_time", ascending=False).groupby("station_code").first()
        
        for station_code, row in latest_data.iterrows():
            station_info = get_station_info(station_code)
            if not station_info:
                continue
            
            lat = station_info.get("latitude")
            lon = station_info.get("longitude")
            station_name = station_info.get("name", f"Estación {station_code}")
            
            if lat is None or lon is None:
                continue
            
            # Obtener valores, manejando None y NaN correctamente
            aqi = row.get("air_quality_index")
            no2 = row.get("no2")
            o3 = row.get("o3")
            pm10 = row.get("pm10")
            pm25 = row.get("pm25")
            temp = row.get("weather_temperature_c")
            wind = row.get("weather_wind_speed_kmh")
            humidity = row.get("weather_humidity")
            
            # Convertir None/NaN a 0 solo para visualización
            aqi = float(aqi) if pd.notna(aqi) and aqi is not None else 0
            no2 = float(no2) if pd.notna(no2) and no2 is not None else 0
            o3 = float(o3) if pd.notna(o3) and o3 is not None else 0
            pm10 = float(pm10) if pd.notna(pm10) and pm10 is not None else 0
            pm25 = float(pm25) if pd.notna(pm25) and pm25 is not None else 0
            temp = float(temp) if pd.notna(temp) and temp is not None else 0
            wind = float(wind) if pd.notna(wind) and wind is not None else 0
            humidity = float(humidity) if pd.notna(humidity) and humidity is not None else 0
            
            if aqi <= 25:
                color = "#10b981"
                quality = "Excelente"
            elif aqi <= 50:
                color = "#fbbf24"
                quality = "Buena"
            elif aqi <= 75:
                color = "#f97316"
                quality = "Moderada"
            else:
                color = "#ef4444"
                quality = "Mala"
            
            radius = max(10, min(25, int(temp / 1.5))) if temp > 0 else 12
            
            prediction_html = ""
            if predictions is not None and station_code in predictions.index:
                is_good = predictions[station_code] == 1
                badge_color = "#10b981" if is_good else "#ef4444"
                badge_text = "✅ Bueno para correr" if is_good else "❌ No recomendado"
                prediction_html = f'<div style="margin-top: 0.5rem; padding: 0.5rem; background: {badge_color}20; border-radius: 6px; text-align: center; font-weight: 600; color: {badge_color};">{badge_text}</div>'
            
            popup_html = f"""
            <div style="width: 280px; font-family: 'Poppins', sans-serif;">
                <h3 style="margin: 0 0 0.5rem 0; color: #1e293b; font-size: 1.1rem; font-weight: 700;">{station_name}</h3>
                <p style="margin: 0.25rem 0; color: #64748b; font-size: 0.85rem;"><strong>Tipo:</strong> {station_info.get('type', 'N/A')}</p>
                <hr style="margin: 0.75rem 0; border: none; border-top: 1px solid #e2e8f0;">
                <div style="background: {color}20; padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0;">
                    <p style="margin: 0; color: {color}; font-weight: 700; font-size: 1.1rem;">Calidad: {quality}</p>
                    <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.9rem;">AQI: {aqi:.1f}</p>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin: 0.75rem 0;">
                    <div><p style="margin: 0; font-size: 0.75rem; color: #64748b;">NO₂</p><p style="margin: 0; font-weight: 600; color: #1e293b;">{no2:.1f} µg/m³</p></div>
                    <div><p style="margin: 0; font-size: 0.75rem; color: #64748b;">O₃</p><p style="margin: 0; font-weight: 600; color: #1e293b;">{o3:.1f} µg/m³</p></div>
                    <div><p style="margin: 0; font-size: 0.75rem; color: #64748b;">PM10</p><p style="margin: 0; font-weight: 600; color: #1e293b;">{pm10:.1f} µg/m³</p></div>
                    <div><p style="margin: 0; font-size: 0.75rem; color: #64748b;">PM2.5</p><p style="margin: 0; font-weight: 600; color: #1e293b;">{pm25:.1f} µg/m³</p></div>
                </div>
                <div style="background: #f1f5f9; padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0;">
                    <p style="margin: 0; font-size: 0.85rem; color: #64748b; font-weight: 600;">🌡️ Condiciones</p>
                    <p style="margin: 0.25rem 0 0 0; color: #1e293b; font-weight: 700; font-size: 1.2rem;">{temp:.1f}°C</p>
                    <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.85rem;">💨 {wind:.1f} km/h | 💧 {humidity:.0f}%</p>
                </div>
                {prediction_html}
            </div>
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{station_name} - {quality} ({temp:.1f}°C)",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=3,
            ).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    return m


def main() -> None:
    """Main Streamlit app."""
    # Header moderno
    st.markdown(
        """
        <div class="main-container">
            <h1 class="main-header">🏃 Runner Air Planner</h1>
            <p class="sub-header">Encuentra el mejor momento para correr en Madrid</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Inicializar estado
    if "realtime_data" not in st.session_state:
        st.session_state.realtime_data = None
    if "realtime_weather" not in st.session_state:
        st.session_state.realtime_weather = None
    if "last_update" not in st.session_state:
        st.session_state.last_update = None
    if "predictions" not in st.session_state:
        st.session_state.predictions = None
    
    # Botones de acción principales
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        if st.button("🔄 Actualizar Datos en Tiempo Real", type="primary", use_container_width=True):
            with st.spinner("🔄 Obteniendo datos actuales..."):
                ml_df, weather = collect_realtime_data()
                if not ml_df.empty:
                    st.session_state.realtime_data = ml_df
                    st.session_state.realtime_weather = weather
                    st.session_state.last_update = datetime.now()
                    st.success("✅ Datos actualizados correctamente")
                    st.rerun()
                else:
                    st.error("❌ No se pudieron obtener datos")
    
    with col2:
        model = load_model()
        if model:
            if st.button("🤖 Ejecutar Predicción ML", type="primary", use_container_width=True):
                df_to_use = st.session_state.realtime_data if st.session_state.realtime_data is not None else load_historical_dataset()
                if not df_to_use.empty:
                    with st.spinner("🤖 Ejecutando modelo..."):
                        try:
                            predictions = model.predict(df_to_use)
                            probabilities = model.predict_proba(df_to_use)
                            st.session_state.predictions = predictions
                            st.session_state.probabilities = probabilities
                            st.success("✅ Predicción completada")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
        else:
            st.info("💡 Entrena el modelo primero")
    
    with col3:
        if st.button("🔄", help="Recargar página"):
            st.cache_data.clear()
            st.rerun()
    
    # Indicador de tiempo real
    if st.session_state.realtime_data is not None:
        update_time = st.session_state.last_update.strftime("%H:%M:%S") if st.session_state.last_update else "Ahora"
        st.markdown(
            f'<div class="realtime-indicator">🟢 Datos en Tiempo Real • Actualizado: {update_time}</div>',
            unsafe_allow_html=True,
        )
    
    # Seleccionar datos a usar
    df = st.session_state.realtime_data if st.session_state.realtime_data is not None else load_historical_dataset()
    
    if df.empty:
        st.warning("⚠️ No hay datos disponibles. Haz clic en 'Actualizar Datos' para comenzar.")
        return
    
    # Métricas principales
    st.markdown('<div class="section-title">📊 Resumen</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_aqi = df["air_quality_index"].mean() if "air_quality_index" in df.columns else 0
        aqi_status = "status-excellent" if avg_aqi <= 25 else "status-good" if avg_aqi <= 50 else "status-moderate" if avg_aqi <= 75 else "status-poor"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">🌬️</div>
                <div class="metric-label">Calidad del Aire</div>
                <div class="metric-value">{avg_aqi:.1f}</div>
                <div class="status-badge {aqi_status}">AQI Promedio</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        avg_temp = df["weather_temperature_c"].mean() if "weather_temperature_c" in df.columns else 0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">🌡️</div>
                <div class="metric-label">Temperatura</div>
                <div class="metric-value">{avg_temp:.1f}°C</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col3:
        stations = df["station_code"].nunique() if "station_code" in df.columns else 0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">📍</div>
                <div class="metric-label">Estaciones</div>
                <div class="metric-value">{stations}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col4:
        if st.session_state.predictions is not None:
            good_count = int(st.session_state.predictions.sum())
            total = len(st.session_state.predictions)
            good_pct = (good_count / total * 100) if total > 0 else 0
        else:
            good_count = 0
            total = len(df)
            good_pct = 0
        
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">✅</div>
                <div class="metric-label">Buenas para Correr</div>
                <div class="metric-value">{good_count}</div>
                <div style="margin-top: 0.5rem; color: #64748b; font-size: 0.875rem;">de {total} ({good_pct:.0f}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # Información meteorológica en tiempo real
    if st.session_state.realtime_weather:
        weather = st.session_state.realtime_weather
        st.markdown('<div class="section-title">🌤️ Condiciones Actuales</div>', unsafe_allow_html=True)
        
        wcol1, wcol2, wcol3, wcol4 = st.columns(4)
        with wcol1:
            st.metric("🌡️ Temperatura", f"{weather.get('temperature', 0):.1f}°C")
        with wcol2:
            st.metric("💧 Humedad", f"{weather.get('humidity', 0):.0f}%")
        with wcol3:
            st.metric("💨 Viento", f"{weather.get('wind_speed', 0):.1f} km/h")
        with wcol4:
            st.metric("☁️ Estado", weather.get('description', 'N/A'))
    
    # Mapa interactivo
    st.markdown('<div class="section-title">🗺️ Mapa Interactivo</div>', unsafe_allow_html=True)
    
    predictions = st.session_state.predictions
    air_map = create_air_quality_map(df, predictions)
    st_folium(air_map, width=1200, height=600)
    
    # Resultados del modelo
    if st.session_state.predictions is not None and st.session_state.probabilities is not None:
        st.markdown('<div class="section-title">🎯 Mejores Lugares para Correr</div>', unsafe_allow_html=True)
        
        df_results = df.copy()
        df_results["prob_good"] = st.session_state.probabilities["prob_good"]
        best_stations = df_results.nlargest(10, "prob_good")[
            ["station_name", "station_type", "prob_good", "air_quality_index", 
             "weather_temperature_c", "weather_wind_speed_kmh"]
        ]
        
        st.dataframe(
            best_stations.style.format({
                "prob_good": "{:.1%}",
                "air_quality_index": "{:.1f}",
                "weather_temperature_c": "{:.1f}°C",
                "weather_wind_speed_kmh": "{:.1f} km/h",
            }),
            width='stretch',
            hide_index=True,
        )


if __name__ == "__main__":
    main()
