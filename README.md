# Runner Air Planner - ML Model Data Pipeline

Pipeline de datos y modelo de Machine Learning que predice el mejor momento para salir a correr en Madrid basándose en la calidad del aire y condiciones meteorológicas.

## 🎯 Objetivo

Crear un dataset estructurado con **mínimo 1000 registros** y entrenar un modelo ML que combine:
- **Calidad del aire** por estación (NO₂, O₃, PM10, PM2.5, etc.)
- **Condiciones meteorológicas** (temperatura, humedad, viento)
- **Features temporales** (hora, día semana, mes)
- **Features de sinergia** (interacciones entre variables)

## 🚀 Inicio Rápido con Docker (Recomendado)

### Primera vez

```bash
# Construir y levantar la aplicación
docker-compose up -d --build

# Ver los logs para verificar que todo funciona
docker-compose logs -f
```

### Uso normal

```bash
# Levantar la aplicación
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener la aplicación
docker-compose down
```

**La aplicación estará disponible en:**
- **Frontend + API**: http://localhost:8080
- **API Health Check**: http://localhost:8080/api/health

### Comandos útiles con Docker

```bash
# Recopilar datos
docker-compose exec api poetry run collect --accumulate

# Entrenar modelo (cuando tengas 1000+ registros)
docker-compose exec api poetry run train

# Hacer predicciones
docker-compose exec api poetry run predict

# Abrir shell en el contenedor
docker-compose exec api bash

# Detener
docker-compose down
```

## 📦 Instalación Local (Sin Docker)

### Requisitos

- Python 3.11 o superior
- Poetry instalado

### Pasos

1. **Instalar Poetry** (si no lo tienes):
```bash
# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Linux/Mac
curl -sSL https://install.python-poetry.org | python3 -
```

2. **Instalar dependencias:**
```bash
poetry install
```

3. **Activar el entorno virtual:**
```bash
poetry shell
```

4. **Ejecutar la API:**
```bash
# Opción 1: Con uvicorn directamente
uvicorn runner_air_planner.api.main:app --host 0.0.0.0 --port 8001 --reload

# Opción 2: Con Python
python -m runner_air_planner.api.main
```

5. **Servir el frontend:**
```bash
# Con Python simple server
cd frontend
python -m http.server 8080

# O con cualquier servidor web estático
# El frontend está en la carpeta frontend/
```

**La aplicación estará disponible en:**
- **Frontend + API**: http://localhost:8000 (o el puerto que especifiques)

## 📊 Uso

### 1. Recopilar datos

```bash
# Con Docker
docker-compose exec api poetry run collect --accumulate

# Localmente
poetry run collect --accumulate
```

### 2. Acumular hasta 1000+ registros

La API de Madrid solo devuelve datos del día actual (~300-400 registros). Para alcanzar 1000+:

```bash
# Ejecutar varias veces (cada 1-2 horas)
docker-compose exec api poetry run collect --accumulate

# O automático con el script (corre en background cada 30 min hasta 2000 registros)
PYTHONUNBUFFERED=1 nohup python scripts/collect_multiple_days.py \
  --min-records 2000 \
  --interval-hours 0.5 \
  --max-iterations 200 > data/collection.log 2>&1 &

# Ver progreso
tail -f data/collection.log

# Ver procesos de recolección activos
ps aux | grep collect_multiple_days | grep -v grep
```

### 3. Entrenar modelo

```bash
# Con Docker
docker-compose exec api poetry run train

# Localmente
poetry run train
```

### 4. Hacer predicciones

```bash
# Con Docker
docker-compose exec api poetry run predict

# Localmente
poetry run predict
```

### 5. Visualizar en Frontend

Abre http://localhost:8080 en tu navegador (si usas Docker) o http://localhost:8000 si ejecutas localmente con uvicorn.

El frontend y la API se sirven desde el mismo servidor (FastAPI sirve los archivos estáticos).

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
│   └── api/                      # API Backend
│       └── main.py                # FastAPI application
├── frontend/                     # Frontend estático (HTML/JS/CSS)
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── scripts/
│   └── collect_multiple_days.py  # Script para recopilar datos automáticamente
├── data/                         # Datasets y modelos
│   ├── ml_dataset_accumulated.csv
│   └── models/
│       └── running_model.pkl
├── Dockerfile                    # Imagen Docker para la API
├── docker-compose.yml            # Configuración Docker Compose
├── pyproject.toml                # Gestión con Poetry
└── .pre-commit-config.yaml       # Configuración de pre-commit hooks
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

- La primera vez que ejecutes la app, necesitarás recopilar datos y entrenar el modelo
- Los datos se guardan en `data/ml_dataset_accumulated.csv`
- El modelo entrenado se guarda en `data/models/running_model.pkl`
- Los datos se acumulan automáticamente, eliminando duplicados
- Por defecto se mantienen últimos 30 días de historial
- El dataset se actualiza incrementalmente con cada ejecución

## 🔧 API Endpoints

La API FastAPI proporciona los siguientes endpoints:

- `GET /` - Health check básico
- `GET /api/health` - Health check detallado
- `GET /api/data/realtime` - Obtener datos de calidad del aire en tiempo real
- `GET /api/data/historical` - Obtener datos históricos acumulados
- `POST /api/predict` - Ejecutar predicciones ML con el modelo entrenado

### Ejemplo de uso de la API

```bash
# Obtener datos en tiempo real
curl http://localhost:8080/api/data/realtime

# Hacer predicciones
curl -X POST http://localhost:8080/api/predict \
  -H "Content-Type: application/json" \
  -d '{"use_realtime": true}'
```

## 🐛 Troubleshooting

### Docker no inicia
```bash
# Ver logs
docker-compose logs

# Reconstruir imagen
docker-compose build --no-cache
docker-compose up -d
```

### Poetry no encuentra dependencias
```bash
# Reinstalar
poetry install

# Limpiar cache
poetry cache clear pypi --all
poetry install
```

### Puerto ocupado
Si el puerto 8080 está ocupado, puedes cambiarlo en `docker-compose.yml`:
```yaml
ports:
  - "9000:8000"  # Cambia 8080 a 9000
```

### El frontend no carga
Verifica que:
1. La API esté corriendo (Docker o uvicorn)
2. Limpia el caché del navegador: `Cmd + Shift + R` (Mac) o `Ctrl + Shift + R` (Windows/Linux)

## 🛠️ Desarrollo

### Pre-commit hooks

El proyecto usa pre-commit para mantener la calidad del código:

```bash
# Instalar pre-commit
pip install pre-commit
pre-commit install

# Ejecutar en todos los archivos
pre-commit run --all-files
```

Hooks configurados:
- `trailing-whitespace` - elimina espacios al final
- `end-of-file-fixer` - asegura newline al final
- `check-yaml` - valida archivos YAML
- `ruff` - linting con autofix
- `ruff-format` - formato de código

## ☁️ Despliegue en Render con Docker

El proyecto está configurado para desplegarse en Render usando **Docker**.

### Configuración en Render

1. **Crea un nuevo servicio Web Service** en Render
2. **Conecta tu repositorio de GitHub**
3. **Configura el servicio**:
   - **Environment**: `Docker` (o "Dockerfile")
   - Render detectará automáticamente el `Dockerfile` en la raíz
   - **Health Check Path**: `/api/health` (opcional)

4. **Variables de Entorno** (opcional, en Settings → Environment):
   - `PYTHONUNBUFFERED`: `1`

### Cómo funciona

- Render construye la imagen Docker usando el `Dockerfile`
- El `Dockerfile` instala Poetry y todas las dependencias desde `pyproject.toml`
- La aplicación se inicia automáticamente con uvicorn
- Render asigna automáticamente el puerto usando la variable `PORT`

### Notas importantes para Render

- ✅ El `Dockerfile` ya está configurado para usar la variable `PORT` de Render
- ✅ No necesitas `requirements.txt` ni `render.yaml` (el Dockerfile usa Poetry directamente)
- ⚠️ Los datos se almacenan en el sistema de archivos del contenedor (no persisten entre reinicios)
- 💡 Para datos persistentes, considera usar un servicio de base de datos o almacenamiento externo
- 🌐 El frontend estático necesita desplegarse por separado o integrarse con la API

### Solución de problemas

**El servicio no inicia:**
- Verifica que el tipo de servicio sea **"Web Service"** con **"Docker"** como environment
- Revisa los logs en Render para ver errores específicos

**Error de puerto:**
- El `Dockerfile` ya está configurado para usar `${PORT}` automáticamente
- No definas la variable `PORT` manualmente en Render
