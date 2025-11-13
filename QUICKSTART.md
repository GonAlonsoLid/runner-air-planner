# Quick Start - Runner Air Planner

## Opción 1: Docker (Recomendado - Más fácil)

### Levantar la aplicación

```bash
# Construir y levantar
docker-compose up -d

# Ver logs
docker-compose logs -f

# La app estará en http://localhost:8501
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
# Modo desarrollo
docker-compose -f docker-compose.dev.yml up
```

## Opción 2: Poetry (Local)

### Instalación

```bash
# Instalar Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Instalar dependencias
poetry install

# Activar entorno
poetry shell
```

### Uso

```bash
# Recopilar datos
poetry run collect --accumulate

# Entrenar modelo
poetry run train

# Hacer predicciones
poetry run predict

# Levantar frontend
poetry run streamlit run src/runner_air_planner/frontend/streamlit_app.py
```

## Flujo Completo

1. **Recopilar datos** (hasta 1000+ registros)
   ```bash
   docker-compose exec app poetry run collect --accumulate
   ```

2. **Entrenar modelo**
   ```bash
   docker-compose exec app poetry run train
   ```

3. **Hacer predicciones**
   ```bash
   docker-compose exec app poetry run predict
   ```

4. **Ver en frontend**
   - Abre http://localhost:8501 en tu navegador

## Troubleshooting

### Docker no inicia
```bash
# Ver logs
docker-compose logs

# Reconstruir imagen
docker-compose build --no-cache
```

### Poetry no encuentra dependencias
```bash
# Reinstalar
poetry install
```

### Puerto 8501 ocupado
```bash
# Cambiar puerto en docker-compose.yml
ports:
  - "8502:8501"  # Usa 8502 en lugar de 8501
```

