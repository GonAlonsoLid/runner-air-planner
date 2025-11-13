# Runner Air Planner - ML Model Data Pipeline

Pipeline de datos y modelo de Machine Learning que predice el mejor momento para salir a correr en Madrid basándose en la calidad del aire y condiciones meteorológicas.

## 🎯 Objetivo

Crear un dataset estructurado con **mínimo 1000 registros** y entrenar un modelo ML que combine:
- **Calidad del aire** por estación (NO₂, O₃, PM10, PM2.5, etc.)
- **Condiciones meteorológicas** (temperatura, humedad, viento)
- **Features temporales** (hora, día semana, mes)
- **Features de sinergia** (interacciones entre variables)

## 🚀 Inicio Rápido con Docker

### Levantar la aplicación

```bash
# Construir y levantar (primera vez)
docker-compose up -d --build

# Ver logs
docker-compose logs -f

# La app estará disponible en http://localhost:8501
```

### Comandos útiles

```bash
# Recopilar datos
docker-compose exec app poetry run collect --accumulate

# Entrenar modelo (cuando tengas 1000+ registros)
docker-compose exec app poetry run train

# Hacer predicciones
docker-compose exec app poetry run predict

# Abrir shell en el contenedor
docker-compose exec app bash

# Detener
docker-compose down
```

### Desarrollo con hot-reload

```bash
docker-compose -f docker-compose.dev.yml up
```

## 📦 Instalación con Poetry (Local)

```bash
# Instalar Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Instalar dependencias
poetry install

# Activar entorno
poetry shell
```

## 🏗️ Estructura del Proyecto

```
runner-air-planner/
├── src/runner_air_planner/
│   ├── data_pipeline/          # Pipeline de datos para ML
│   │   ├── ingest_madrid_air.py    # Descarga datos calidad aire
│   │   ├── weather.py              # Cliente Open-Meteo
│   │   ├── master_data.py          # Datos maestros
│   │   ├── data_collector.py       # Clase principal que integra todo
│   │   ├── accumulate_data.py      # Acumulación de datos históricos
│   │   └── cli_collect.py          # CLI para recopilar datos
│   ├── ml/                      # Modelos de Machine Learning
│   │   ├── model.py                # Definición del modelo
│   │   ├── train.py                # Entrenamiento
│   │   └── predict.py              # Predicciones
│   └── frontend/                # Interfaz de usuario
│       └── streamlit_app.py        # Dashboard Streamlit
├── scripts/
│   └── collect_multiple_days.py
├── data/                           # Datasets y modelos
│   ├── ml_dataset_accumulated.csv
│   └── models/
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml                  # Gestión con Poetry
```

## 📊 Uso

### 1. Recopilar datos

```bash
# Con Docker
docker-compose exec app poetry run collect --accumulate

# Con Poetry local
poetry run collect --accumulate
```

### 2. Acumular hasta 1000+ registros

La API de Madrid solo devuelve datos del día actual (~300-400 registros). Para alcanzar 1000+:

```bash
# Ejecutar varias veces (cada 1-2 horas)
docker-compose exec app poetry run collect --accumulate

# O automático
python scripts/collect_multiple_days.py --min-records 1000 --interval-hours 1
```

### 3. Entrenar modelo

```bash
docker-compose exec app poetry run train
```

### 4. Hacer predicciones

```bash
docker-compose exec app poetry run predict
```

### 5. Visualizar en Frontend

Abre http://localhost:8501 en tu navegador (si usas Docker) o:

```bash
poetry run streamlit run src/runner_air_planner/frontend/streamlit_app.py
```

## 📈 Dataset para ML

El dataset final (`ml_dataset_accumulated.csv`) contiene **~42 features**:

- **Contaminantes**: `no2`, `o3`, `pm10`, `pm25`, `no`, `nox`, `so2`, `co`
- **Estación**: código, nombre, tipo (Tráfico/Suburbana), coordenadas
- **Temporales**: hora, día semana, mes, `is_weekend`, `is_rush_hour`
- **Meteorológicas**: temperatura, humedad, viento, código tiempo
- **Sinergias**: `wind_*_synergy`, `temp_o3_synergy`, `air_quality_index`, etc.

## 🔌 APIs Utilizadas

- **Calidad del Aire Madrid**: `https://datos.madrid.es/egob/catalogo/212531-12751102-calidad-aire-tiempo-real.json`
- **Open-Meteo**: `https://api.open-meteo.com/v1/forecast` (gratuita, sin API key)

## 📋 Requisitos

- **Docker & Docker Compose** (recomendado)
- O **Python 3.11+** y **Poetry** (para desarrollo local)

## 📝 Notas

- Los datos se acumulan automáticamente, eliminando duplicados
- Por defecto se mantienen últimos 30 días de historial
- El dataset se actualiza incrementalmente con cada ejecución
- Los modelos entrenados se guardan en `data/models/`

## 🛠️ Comandos Makefile

```bash
make up          # Levantar app
make collect     # Recopilar datos
make train       # Entrenar modelo
make predict     # Hacer predicciones
make logs        # Ver logs
make shell       # Abrir shell
make down        # Detener
```

Ver `QUICKSTART.md` para más detalles.
