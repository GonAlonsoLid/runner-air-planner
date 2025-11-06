"""Quick Streamlit UI to explore and score Madrid air quality data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from backend.app.model import AirQualityRiskModel
from backend.app.storage import DATA_PATH, load_dataset

st.set_page_config(page_title="Runner Air Planner", layout="wide")
st.title("Runner Air Planner")
st.caption("Plan your next training session in Madrid with up-to-date air quality insights.")


@st.cache_data(show_spinner=False)
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    frame = load_dataset(path=path)
    if frame.empty:
        st.warning("Aún no hay datos descargados. Ejecuta el pipeline de ingesta primero.")
    return frame


def main() -> None:
    frame = load_data()
    if frame.empty:
        return

    st.subheader("Últimas mediciones")
    st.dataframe(frame.tail(200), use_container_width=True)

    st.subheader("Evaluación rápida")
    value = st.number_input("Introduce un valor de medición", min_value=0.0, value=float(frame["value"].median()))
    model = AirQualityRiskModel()
    model.fit(frame)
    prediction = model.predict(value)
    col1, col2 = st.columns(2)
    col1.metric("Calidad estimada", "Buena" if prediction.label == "good" else "Mala")
    col2.metric("Probabilidad de mala calidad", f"{prediction.probability:.2%}")


if __name__ == "__main__":
    main()
