"""Streamlit dashboard for Runner Air Planner predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from runner_air_planner.data_pipeline import DataCollector, load_accumulated_dataset
from runner_air_planner.ml.model import MODELS_DIR, RunningSuitabilityModel


@st.cache_data
def load_dataset(path: Path) -> pd.DataFrame:
    """Load ML dataset."""
    return load_accumulated_dataset(path)


@st.cache_resource
def load_model(model_path: Path) -> RunningSuitabilityModel | None:
    """Load trained model."""
    if not model_path.exists():
        return None
    import pickle
    with open(model_path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Runner Air Planner",
        page_icon="🏃",
        layout="wide",
    )
    
    st.title("🏃‍♀️ Runner Air Planner")
    st.markdown(
        "Predicciones de Machine Learning para determinar el mejor momento "
        "para salir a correr en Madrid basándose en calidad del aire y condiciones meteorológicas."
    )
    
    # Sidebar
    st.sidebar.header("Configuración")
    
    dataset_path = st.sidebar.text_input(
        "Ruta del dataset",
        value="data/ml_dataset_accumulated.csv",
    )
    
    model_path = st.sidebar.text_input(
        "Ruta del modelo",
        value=str(MODELS_DIR / "running_model.pkl"),
    )
    
    if st.sidebar.button("🔄 Actualizar datos"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    # Load data
    try:
        df = load_dataset(Path(dataset_path))
        if df.empty:
            st.error("Dataset vacío. Ejecuta primero el pipeline de datos.")
            return
    except Exception as e:
        st.error(f"Error cargando dataset: {e}")
        return
    
    # Load model
    model = load_model(Path(model_path))
    
    # Main content
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total registros", len(df))
    with col2:
        st.metric("Estaciones", df["station_code"].nunique())
    with col3:
        st.metric("Modelo cargado", "✅" if model else "❌")
    
    # Predictions if model available
    if model:
        st.subheader("🎯 Predicciones")
        
        try:
            predictions = model.predict(df)
            probabilities = model.predict_proba(df)
            
            df["prediction"] = predictions
            df["prob_good"] = probabilities["prob_good"]
            df["is_good_to_run"] = df["prediction"] == 1
            
            # Summary
            good_count = predictions.sum()
            st.success(f"✅ {good_count} de {len(df)} estaciones/momentos son buenos para correr ({good_count/len(df):.1%})")
            
            # Filter good predictions
            good_df = df[df["is_good_to_run"]].copy()
            
            if not good_df.empty:
                st.subheader("🏃 Mejores momentos para correr")
                
                # Sort by probability
                good_df = good_df.sort_values("prob_good", ascending=False)
                
                # Display table
                display_cols = [
                    "station_name",
                    "station_type",
                    "measurement_time",
                    "prob_good",
                    "air_quality_index",
                    "no2",
                    "o3",
                    "pm10",
                    "weather_wind_speed_kmh",
                    "weather_temperature_c",
                ]
                available_cols = [c for c in display_cols if c in good_df.columns]
                st.dataframe(
                    good_df[available_cols].head(20),
                    use_container_width=True,
                )
            else:
                st.warning("No hay momentos recomendados para correr según el modelo.")
        
        except Exception as e:
            st.error(f"Error haciendo predicciones: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    else:
        st.info(
            "💡 Modelo no encontrado. Entrena el modelo primero:\n\n"
            "```bash\n"
            "PYTHONPATH=src python -m runner_air_planner.ml.train\n"
            "```"
        )
    
    # Raw data view
    with st.expander("📊 Ver datos completos"):
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()

