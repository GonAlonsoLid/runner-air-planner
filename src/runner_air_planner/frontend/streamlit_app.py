"""Streamlit dashboard profesional para Runner Air Planner con mapa interactivo."""

from __future__ import annotations

import pickle
import sys
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
    page_title="Runner Air Planner - Madrid",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado para una interfaz más profesional
st.markdown(
    """
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.75rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300)  # Cache por 5 minutos
def load_dataset(path: Path) -> pd.DataFrame:
    """Load ML dataset."""
    try:
        return load_accumulated_dataset(path)
    except Exception:
        return pd.DataFrame()


@st.cache_resource
def load_model(model_path: Path) -> RunningSuitabilityModel | None:
    """Load trained model."""
    if not model_path.exists():
        return None
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


@st.cache_data(ttl=60)  # Cache por 1 minuto
def collect_realtime_data() -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Collect real-time data from APIs."""
    try:
        collector = DataCollector()
        air_quality_df = collector.collect_air_quality_data()
        weather_report = collector.get_weather_data()
        
        # Convert weather report to dict
        weather_dict = None
        if weather_report:
            weather_dict = {
                "temperature": weather_report.temperature_c,
                "humidity": weather_report.relative_humidity,
                "wind_speed": weather_report.wind_speed_kmh,
                "weather_code": weather_report.weather_code,
                "description": weather_report.weather_description,
            }
        
        return air_quality_df, weather_dict
    except Exception as e:
        st.error(f"Error recopilando datos en tiempo real: {e}")
        return pd.DataFrame(), None


def create_air_quality_map(df: pd.DataFrame, predictions: pd.Series | None = None) -> folium.Map:
    """Create interactive map with air quality data."""
    # Centro de Madrid
    madrid_center = [40.4168, -3.7038]
    
    # Crear mapa base
    m = folium.Map(
        location=madrid_center,
        zoom_start=11,
        tiles="OpenStreetMap",
    )
    
    # Añadir capa de satélite
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satélite",
        overlay=False,
        control=True,
    ).add_to(m)
    
    # Agrupar por estación (última medición)
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
            
            # Obtener valores de calidad del aire
            aqi = row.get("air_quality_index", 0)
            no2 = row.get("no2", 0)
            o3 = row.get("o3", 0)
            pm10 = row.get("pm10", 0)
            temp = row.get("weather_temperature_c", 0)
            
            # Color según calidad del aire
            if aqi <= 25:
                color = "green"
                quality = "Excelente"
            elif aqi <= 50:
                color = "yellow"
                quality = "Buena"
            elif aqi <= 75:
                color = "orange"
                quality = "Moderada"
            else:
                color = "red"
                quality = "Mala"
            
            # Tamaño del marcador según temperatura
            if temp > 0:
                radius = max(8, min(20, int(temp / 2)))
            else:
                radius = 10
            
            # Predicción del modelo
            prediction_text = ""
            if predictions is not None and station_code in predictions.index:
                is_good = predictions[station_code] == 1
                prediction_text = f"<br><strong>🏃 {'✅ Bueno para correr' if is_good else '❌ No recomendado'}</strong>"
            
            # Popup con información
            popup_html = f"""
            <div style="width: 250px;">
                <h4>{station_name}</h4>
                <p><strong>Tipo:</strong> {station_info.get('type', 'N/A')}</p>
                <hr>
                <p><strong>Calidad del Aire:</strong> {quality} (AQI: {aqi:.1f})</p>
                <p><strong>NO₂:</strong> {no2:.1f} µg/m³</p>
                <p><strong>O₃:</strong> {o3:.1f} µg/m³</p>
                <p><strong>PM10:</strong> {pm10:.1f} µg/m³</p>
                <p><strong>🌡️ Temperatura:</strong> {temp:.1f}°C</p>
                {prediction_text}
            </div>
            """
            
            # Icono personalizado
            icon = folium.Icon(
                color=color,
                icon="info-sign",
                prefix="glyphicon",
            )
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{station_name} - {quality}",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6,
                weight=2,
            ).add_to(m)
            
            folium.Marker(
                location=[lat, lon],
                icon=icon,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=station_name,
            ).add_to(m)
    
    # Añadir leyenda
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 150px; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
    <h4>Leyenda</h4>
    <p><span style="color:green">●</span> Excelente (AQI ≤ 25)</p>
    <p><span style="color:yellow">●</span> Buena (AQI ≤ 50)</p>
    <p><span style="color:orange">●</span> Moderada (AQI ≤ 75)</p>
    <p><span style="color:red">●</span> Mala (AQI > 75)</p>
    <p><small>🌡️ Tamaño = Temperatura</small></p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m


def create_temperature_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create temperature heatmap."""
    if df.empty or "weather_temperature_c" not in df.columns:
        return go.Figure()
    
    # Agrupar por estación
    station_data = df.groupby("station_code").agg({
        "weather_temperature_c": "mean",
        "station_name": "first",
    }).reset_index()
    
    # Obtener coordenadas
    coords = []
    temps = []
    names = []
    
    for _, row in station_data.iterrows():
        station_info = get_station_info(row["station_code"])
        if station_info and station_info.get("latitude"):
            coords.append([station_info["latitude"], station_info["longitude"]])
            temps.append(row["weather_temperature_c"])
            names.append(row["station_name"])
    
    if not coords:
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scattermapbox(
            lat=[c[0] for c in coords],
            lon=[c[1] for c in coords],
            mode="markers",
            marker=dict(
                size=15,
                color=temps,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Temperatura (°C)"),
                cmin=min(temps) if temps else 0,
                cmax=max(temps) if temps else 30,
            ),
            text=names,
            hovertemplate="<b>%{text}</b><br>Temperatura: %{marker.color:.1f}°C<extra></extra>",
        )
    )
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=40.4168, lon=-3.7038),
            zoom=10,
        ),
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    
    return fig


def main() -> None:
    """Main Streamlit app."""
    # Header
    st.markdown('<h1 class="main-header">🏃 Runner Air Planner</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Predicciones ML en tiempo real para el mejor momento de correr en Madrid</p>',
        unsafe_allow_html=True,
    )
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        dataset_path = st.text_input(
            "📊 Dataset",
            value="data/ml_dataset_accumulated.csv",
            help="Ruta al dataset acumulado",
        )
        
        model_path = st.text_input(
            "🤖 Modelo ML",
            value=str(MODELS_DIR / "running_model.pkl"),
            help="Ruta al modelo entrenado",
        )
        
        st.divider()
        
        # Botón para recopilar datos en tiempo real
        if st.button("🔄 Actualizar Datos en Tiempo Real", type="primary"):
            with st.spinner("Recopilando datos..."):
                air_df, weather = collect_realtime_data()
                if not air_df.empty:
                    st.success("✅ Datos actualizados")
                    st.session_state["realtime_air"] = air_df
                    st.session_state["realtime_weather"] = weather
                else:
                    st.error("❌ No se pudieron obtener datos")
        
        st.divider()
        
        # Botón para ejecutar modelo
        run_model = st.button("🚀 Ejecutar Modelo ML", type="primary", use_container_width=True)
        
        if st.button("🔄 Limpiar Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    
    # Load data
    df = load_dataset(Path(dataset_path))
    
    if df.empty:
        st.warning("⚠️ Dataset vacío. Recopila datos primero.")
        if st.button("📥 Recopilar Datos"):
            with st.spinner("Recopilando..."):
                collector = DataCollector()
                air_df = collector.collect_air_quality_data()
                weather = collector.get_weather_data()
                ml_df = collector.create_ml_dataset(air_quality_df=air_df, weather_report=weather)
                collector.save_ml_dataset(Path(dataset_path), air_quality_df=air_df, weather_report=weather)
                st.success("✅ Datos recopilados")
                st.rerun()
        return
    
    # Load model
    model = load_model(Path(model_path))
    
    # Main content - Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Registros", f"{len(df):,}")
    
    with col2:
        st.metric("📍 Estaciones", df["station_code"].nunique() if "station_code" in df.columns else 0)
    
    with col3:
        avg_aqi = df["air_quality_index"].mean() if "air_quality_index" in df.columns else 0
        st.metric("🌬️ AQI Promedio", f"{avg_aqi:.1f}")
    
    with col4:
        avg_temp = df["weather_temperature_c"].mean() if "weather_temperature_c" in df.columns else 0
        st.metric("🌡️ Temp. Promedio", f"{avg_temp:.1f}°C")
    
    st.divider()
    
    # Ejecutar modelo si se presionó el botón
    predictions = None
    probabilities = None
    
    if run_model and model:
        with st.spinner("🤖 Ejecutando modelo ML..."):
            try:
                predictions = model.predict(df)
                probabilities = model.predict_proba(df)
                df["prediction"] = predictions
                df["prob_good"] = probabilities["prob_good"]
                st.session_state["predictions"] = predictions
                st.session_state["probabilities"] = probabilities
                st.success("✅ Modelo ejecutado correctamente")
            except Exception as e:
                st.error(f"❌ Error ejecutando modelo: {e}")
    elif "predictions" in st.session_state:
        predictions = st.session_state["predictions"]
        probabilities = st.session_state.get("probabilities")
    
    # Mapa interactivo
    st.header("🗺️ Mapa Interactivo - Calidad del Aire y Temperatura")
    
    tab1, tab2 = st.tabs(["📍 Mapa Folium", "🌡️ Mapa de Temperatura"])
    
    with tab1:
        # Usar datos en tiempo real si están disponibles
        map_df = df
        if "realtime_air" in st.session_state:
            map_df = st.session_state["realtime_air"]
        
        air_map = create_air_quality_map(map_df, predictions)
        map_data = st_folium(air_map, width=1200, height=600)
    
    with tab2:
        temp_fig = create_temperature_heatmap(df)
        if temp_fig.data:
            st.plotly_chart(temp_fig, use_container_width=True)
        else:
            st.info("No hay datos de temperatura disponibles")
    
    # Resultados del modelo
    if predictions is not None:
        st.divider()
        st.header("🎯 Resultados del Modelo ML")
        
        good_count = int(predictions.sum())
        total_count = len(predictions)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("✅ Buen momento para correr", f"{good_count} ({good_count/total_count:.1%})")
        
        with col2:
            st.metric("❌ No recomendado", f"{total_count - good_count} ({(total_count-good_count)/total_count:.1%})")
        
        # Tabla de mejores estaciones
        if probabilities is not None:
            df_results = df.copy()
            df_results["prob_good"] = probabilities["prob_good"]
            best_stations = df_results.nlargest(10, "prob_good")[
                ["station_name", "station_type", "prob_good", "air_quality_index", 
                 "weather_temperature_c", "weather_wind_speed_kmh"]
            ]
            
            st.subheader("🏆 Top 10 Mejores Estaciones para Correr")
            st.dataframe(best_stations, use_container_width=True, hide_index=True)
    
    # Gráficos adicionales
    st.divider()
    st.header("📈 Análisis de Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "air_quality_index" in df.columns:
            fig_aqi = px.histogram(
                df,
                x="air_quality_index",
                nbins=30,
                title="Distribución del Índice de Calidad del Aire",
                labels={"air_quality_index": "AQI", "count": "Frecuencia"},
            )
            st.plotly_chart(fig_aqi, use_container_width=True)
    
    with col2:
        if "weather_temperature_c" in df.columns:
            fig_temp = px.histogram(
                df,
                x="weather_temperature_c",
                nbins=30,
                title="Distribución de Temperatura",
                labels={"weather_temperature_c": "Temperatura (°C)", "count": "Frecuencia"},
            )
            st.plotly_chart(fig_temp, use_container_width=True)
    
    # Información del modelo
    if not model:
        st.info(
            "💡 Modelo no encontrado. Entrena el modelo primero:\n\n"
            "```bash\n"
            "docker-compose exec app poetry run train\n"
            "```"
        )


if __name__ == "__main__":
    main()
