"""Simple Streamlit dashboard to visualise Madrid air quality predictions."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_DEFAULT = "http://localhost:8000"


@st.cache_data(ttl=300)
def fetch_predictions(api_url: str) -> list[dict[str, Any]]:
    response = requests.get(f"{api_url.rstrip('/')}/predictions", timeout=20)
    response.raise_for_status()
    return response.json()


def build_predictions_table(predictions: list[dict[str, Any]]) -> pd.DataFrame:
    if not predictions:
        return pd.DataFrame()
    frame = pd.DataFrame(predictions)
    frame["measurement_time"] = pd.to_datetime(frame["measurement_time"], errors="coerce")
    pollutant_frame = frame["pollutants"].apply(pd.Series)
    merged = pd.concat([frame.drop(columns=["pollutants"]), pollutant_frame], axis=1)
    return merged.sort_values("measurement_time", ascending=False)


def main() -> None:
    st.set_page_config(page_title="Runner Air Planner", page_icon="🏃", layout="wide")
    st.title("🏃‍♀️ Runner Air Planner")
    st.write(
        "Consulta la calidad del aire en Madrid en tiempo real y obtén una recomendación rápida "
        "sobre si es buena idea salir a correr."
    )

    api_url = st.sidebar.text_input("URL del backend", value=API_DEFAULT)
    if st.sidebar.button("Actualizar"):
        st.cache_data.clear()

    try:
        predictions = fetch_predictions(api_url)
    except requests.RequestException as error:
        st.error(
            "No se pudieron obtener predicciones del backend. Asegúrate de que el servicio FastAPI está en marcha."
        )
        st.exception(error)
        return

    if not predictions:
        st.info("Todavía no hay predicciones generadas. Ejecuta el pipeline de datos primero.")
        return

    table = build_predictions_table(predictions)
    st.subheader("Recomendaciones actuales")
    st.dataframe(table, use_container_width=True)

    st.subheader("Detalle por estación")
    for _, row in table.iterrows():
        station = row.get("station_code") or "Desconocido"
        label = row.get("air_quality_label")
        measurement_time = row.get("measurement_time")
        if isinstance(measurement_time, pd.Timestamp):
            measurement_time = measurement_time.to_pydatetime()
        measurement_text = measurement_time.strftime("%Y-%m-%d %H:%M") if isinstance(measurement_time, datetime) else "N/A"
        pollutants = {
            key: value
            for key, value in row.items()
            if key not in {"station_code", "measurement_time", "cluster", "air_quality_label"}
            and pd.notna(value)
        }
        with st.expander(f"Estación {station} - {label}"):
            st.write(f"Medición: {measurement_text}")
            st.metric("Cluster", int(row.get("cluster", 0)), help="Índice agrupado por el modelo KMeans")
            if pollutants:
                st.write(pd.DataFrame([pollutants]).T.rename(columns={0: "Concentración (µg/m³)"}))
            else:
                st.write("Sin datos de contaminantes disponibles para esta estación.")


if __name__ == "__main__":  # pragma: no cover - Streamlit entry point
    main()
