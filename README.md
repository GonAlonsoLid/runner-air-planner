# Runner Air Planner

Aplicación de ejemplo que combina datos abiertos de calidad del aire en Madrid con un modelo de *machine learning* ligero para recomendar cuándo salir a correr. El proyecto está dividido en tres capas: un pipeline de ingesta de datos, un backend en FastAPI que entrena el modelo y expone endpoints, y un panel en Streamlit que consume dichas predicciones.

## Estructura del repositorio

```
runner-air-planner/
├── README.md
├── requirements.txt
├── data/
│   └── .gitkeep
├── data_pipeline/
│   ├── __init__.py
│   └── ingest_madrid_air.py      # Descarga y normalización de datos abiertos
├── backend/
│   └── app/
│       ├── __init__.py
│       ├── main.py               # API FastAPI con el modelo KMeans
│       └── storage.py            # Utilidades de carga y pivoteado de CSV
├── frontend/
│   └── streamlit_app.py          # Panel interactivo
└── .github/
    └── workflows/
        └── ci.yml                # Workflow de integración continua
```

## Requisitos

- Python 3.11 o superior
- Las dependencias listadas en `requirements.txt`

Instalación rápida:

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Si prefieres Poetry, el repositorio incluye un `pyproject.toml` con `package-mode = false` para evitar el error
`No file/folder found for package runner-air-planner`. Puedes instalar las dependencias con:

```bash
poetry install
poetry run uvicorn backend.app.main:app --reload
```

## 1. Pipeline de datos

El script `data_pipeline/ingest_madrid_air.py` descarga el dataset «Calidad del aire. Datos en tiempo real» del portal de datos abiertos de Madrid y lo normaliza a CSV.

```bash
python data_pipeline/ingest_madrid_air.py --output data/madrid_air_quality_raw.csv
```

- El fichero resultante contiene columnas `station_code`, `pollutant`, `measurement_time`, `value`, `unit` e `is_valid`.
- Puedes ejecutar el script periódicamente (cron, Airflow, etc.) para mantener actualizado el dataset.

## 2. Modelo de *machine learning*

El backend entrena automáticamente un modelo de clustering KMeans a partir del CSV generado en el paso anterior. El modelo agrupa las mediciones por estación usando contaminantes como NO₂, O₃ o PM₂.₅ y asigna etiquetas cualitativas (`Excelente`, `Precaución moderada`, etc.) según la severidad del cluster.

Características clave:

- Los valores faltantes se rellenan con medias por contaminante.
- El número máximo de clusters es 3 para mantener interpretabilidad (se reduce automáticamente si hay menos muestras).
- Cada cluster se etiqueta de manera ordenada por la suma de los centroides (cuanto mayor concentración, más restrictivo).

## 3. Backend FastAPI

Arranca la API después de generar el CSV:

```bash
uvicorn backend.app.main:app --reload
```

Endpoints principales:

- `GET /health`: estado del modelo (número de muestras, features disponibles).
- `GET /predictions`: lista de estaciones con el cluster asignado, etiqueta y valores de contaminantes.
- `GET /stations`: estaciones disponibles y fecha de la última medición.

## 4. Frontend en Streamlit

El panel Streamlit consume el endpoint `/predictions` y muestra las recomendaciones en una tabla interactiva y paneles desplegables.

```bash
streamlit run frontend/streamlit_app.py
```

En la barra lateral puedes indicar la URL del backend (por defecto `http://localhost:8000`). El botón «Actualizar» fuerza la recarga de datos almacenados en caché.

## Integración continua

El workflow `.github/workflows/ci.yml` instala las dependencias y ejecuta `python -m compileall` sobre los módulos principales para garantizar que no hay errores de sintaxis.

## Próximos pasos sugeridos

- Persistir históricos de predicciones y generar visualizaciones temporales.
- Añadir meteorología como segunda fuente de datos y enriquecer el modelo.
- Publicar la API y el panel en un servicio gestionado (Railway, Render, Streamlit Cloud, etc.).
