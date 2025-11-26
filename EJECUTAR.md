# Cómo Ejecutar la Aplicación

## Opción 1: Con Docker (Más Fácil) 🐳

### Primera vez:
```bash
# Construir y levantar la aplicación
docker-compose up -d --build

# Ver los logs para verificar que todo funciona
docker-compose logs -f
```

### Uso normal:
```bash
# Levantar la aplicación
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener la aplicación
docker-compose down
```

**La app estará disponible en:** http://localhost:8501

---

## Opción 2: Localmente (Sin Docker) 💻

### Requisitos:
- Python 3.11 o superior
- Poetry instalado

### Pasos:

1. **Instalar dependencias:**
```bash
poetry install
```

2. **Activar el entorno virtual:**
```bash
poetry shell
```

3. **Ejecutar la app Streamlit:**
```bash
streamlit run src/runner_air_planner/frontend/streamlit_app.py
```

**La app estará disponible en:** http://localhost:8501

---

## Comandos Útiles

### Recopilar datos:
```bash
# Con Docker
docker-compose exec app poetry run collect --accumulate

# Localmente
poetry run collect --accumulate
```

### Entrenar el modelo:
```bash
# Con Docker
docker-compose exec app poetry run train

# Localmente
poetry run train
```

### Hacer predicciones:
```bash
# Con Docker
docker-compose exec app poetry run predict

# Localmente
poetry run predict
```

---

## Notas

- La primera vez que ejecutes la app, necesitarás recopilar datos y entrenar el modelo
- Los datos se guardan en `data/ml_dataset_accumulated.csv`
- El modelo entrenado se guarda en `data/models/running_model.pkl`

