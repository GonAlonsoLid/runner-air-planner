# Runner Air Planner

Aplicación integral para descargar datos abiertos de calidad del aire en Madrid, entrenar un modelo de riesgo sencillo y exponerlo mediante un backend FastAPI y una interfaz Streamlit.

## Estructura del repositorio

```
runner-air-planner/
├─ README.md
├─ requirements.txt
├─ data/                   # salidas CSV (se crea en runtime)
│  └─ .gitkeep
├─ data_pipeline/
│  ├─ __init__.py
│  └─ ingest_madrid_air.py   # script de ingesta
├─ backend/
│  └─ app/
│     ├─ __init__.py
│     ├─ main.py             # FastAPI (endpoints)
│     ├─ model.py            # Modelo ML ligero
│     └─ storage.py          # helpers carga CSV
├─ frontend/
│  ├─ __init__.py
│  └─ streamlit_app.py       # UI rápida
└─ .github/
   └─ workflows/
      └─ ci.yml              # pipeline de CI (pytest)
```

## Ingesta de datos

El script `data_pipeline/ingest_madrid_air.py` descarga el dataset de **calidad del aire en tiempo real** publicado por el Ayuntamiento de Madrid, lo normaliza y lo almacena en `data/madrid_air_quality.csv`.

```bash
python -m data_pipeline.ingest_madrid_air
```

## Backend FastAPI

El backend expone tres endpoints principales:

- `GET /health`: comprobación rápida.
- `GET /measurements?limit=100`: últimas mediciones del CSV.
- `POST /predict`: recibe `{ "value": <float> }` y devuelve la probabilidad de que la calidad sea "poor" junto a la etiqueta.

Para ejecutarlo de forma local:

```bash
uvicorn backend.app.main:app --reload
```

## Modelo de machine learning

Se entrena automáticamente al iniciar el backend utilizando las mediciones disponibles. Implementa una regresión logística unidimensional sobre el valor de la medición para estimar la probabilidad de calidad del aire "mala" (`poor`). Si el dataset aún no contiene suficientes ejemplos, el modelo aplica una regla de umbral configurable.

## Frontend en Streamlit

La aplicación de Streamlit (`frontend/streamlit_app.py`) carga el CSV generado por la ingesta, muestra las últimas observaciones y permite introducir un valor para obtener una predicción instantánea.

```bash
streamlit run frontend/streamlit_app.py
```

## Dependencias y entorno

Instala las dependencias de desarrollo con:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pruebas

```bash
PYTHONPATH=. pytest
```
