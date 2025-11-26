#!/bin/bash
# Script de inicio para Render

# Crear directorios de datos si no existen
mkdir -p data/models data/raw data/interim data/processed

# Configurar PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Iniciar la aplicación
poetry run uvicorn runner_air_planner.api.main:app --host 0.0.0.0 --port ${PORT:-8000}

