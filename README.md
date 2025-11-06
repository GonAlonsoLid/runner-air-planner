# Runner’s Clean Air Planner

## Descripción
**Runner’s Clean Air Planner** es una aplicación web que ayuda a corredores urbanos y personas activas a elegir las mejores horas para entrenar al aire libre en la ciudad.  
La herramienta combina datos abiertos de **calidad del aire** y **meteorología** con un modelo de *machine learning* que predice cómo evolucionarán estas variables en las próximas horas.  
A partir de esa predicción, la aplicación recomienda de forma personalizada las franjas horarias más adecuadas para correr, teniendo en cuenta preferencias del usuario como duración del entreno, temperatura máxima aceptable, viento o lluvia.

---

## Objetivos principales
1. Reunir y almacenar datos abiertos de calidad del aire y meteorología.  
2. Desarrollar un modelo de *machine learning* que prediga la evolución del índice de calidad del aire (AQI) en un horizonte de 1 a 6 horas.  
3. Implementar un sistema de recomendación que combine predicciones y preferencias del usuario.  
4. Construir una interfaz web sencilla e intuitiva que muestre la información de forma clara y práctica.  

## Público objetivo
- Corredores urbanos y deportistas amateurs.  
- Ciudadanos que quieran elegir el mejor momento para pasear, ir en bici o hacer actividades al aire libre.  
- Estudiantes y profesionales que busquen un caso práctico de uso de datos abiertos y *machine learning*.  

---

## Plan inicial de trabajo

### Fase 1: Preparación
- Crear el repositorio en GitHub.  
- Configurar el entorno de desarrollo y dependencias básicas.  

### Fase 2: Ingesta de datos
- Conectar con APIs de calidad del aire (red municipal de Madrid u OpenAQ).  
- Incorporar datos meteorológicos (Open-Meteo).  
- Guardar la información en una base de datos ligera (SQLite).  

### Fase 3: Análisis y features
- Explorar el comportamiento histórico de la calidad del aire.  
- Construir variables (lags, medias móviles, interacciones con meteorología).  

### Fase 4: Modelado
- Entrenar un modelo de predicción para anticipar la calidad del aire a corto plazo.  
- Validar el modelo con backtesting.  

### Fase 5: Backend
- Implementar un servidor con FastAPI.  
- Crear endpoints para exponer datos, predicciones y recomendaciones.  

### Fase 6: Frontend
- Construir un prototipo con Streamlit.  
- Mostrar un mapa con estaciones, predicciones y recomendaciones.  

### Fase 7: Documentación y despliegue
- Mejorar README y documentación técnica.  
- Desplegar la aplicación en un servicio en la nube gratutito.  

---

## Estado actual
📌 Proyecto en fase inicial. Este repositorio servirá como base para organizar el desarrollo en las próximas semanas.

---

## Estructura del proyecto

La primera iteración del proyecto ya incluye una estructura mínima en Python para descargar y almacenar los datos de calidad del aire de Madrid.

```
runner-air-planner/
├── configs/                     # Plantillas de configuración (Toml)
├── data/
│   ├── raw/                     # Descargas en bruto desde las APIs
│   ├── interim/
│   └── processed/
├── scripts/                     # Scripts ejecutables desde la línea de comandos
├── src/runner_air_planner/      # Código fuente del paquete principal
│   ├── config.py                # Gestión centralizada de configuración
│   ├── data_sources/madrid_air.py
│   ├── storage/local.py
│   └── workflows/fetch_latest_air_quality.py
├── tests/                       # Pruebas automatizadas (pytest)
└── pyproject.toml               # Dependencias y metadatos del paquete
```

### Dependencias principales

La base del proyecto utiliza únicamente la biblioteca estándar de Python, por lo que no es necesario instalar paquetes externos para ejecutar el flujo de descarga o las pruebas unitarias. Basta con tener Python 3.11 (o superior) disponible y exportar el `PYTHONPATH` al directorio `src` cuando se ejecuten comandos manualmente:

```bash
export PYTHONPATH="$(pwd)/src"
```

### Configuración

1. Copia el archivo de ejemplo `configs/settings.example.toml` a un nuevo `configs/settings.toml` y ajusta los parámetros si lo necesitas (por ejemplo para trabajar con otro conjunto de datos o cambiar la carpeta de descargas).
2. Opcionalmente, crea un archivo `.env` en la raíz para sobreescribir variables puntuales. Todas las claves utilizan el prefijo `RAP_`.

### Descarga de datos en bruto

El script `scripts/fetch_air_quality.py` coordina la descarga y almacenamiento de los datos en bruto del portal de datos abiertos de Madrid.

```bash
python scripts/fetch_air_quality.py --params station=28079004 magnitud=NO2
```

El comando anterior guardará un archivo JSON con marca temporal en `data/raw/` y mostrará la ruta en pantalla. Si el portal ofreciera filtros compatibles (estación, magnitud, etc.), pueden añadirse mediante `--params` con la sintaxis `clave=valor`.

> ⚠️ Algunos recursos del portal de datos de Madrid requieren cabeceras o credenciales específicas y pueden devolver `403 Forbidden` desde entornos sin navegador. El cliente incorporado implementa manejadores de error y registrará el mensaje en caso de fallo para ayudar al diagnóstico.

### Próximos pasos sugeridos

1. Automatizar la ingesta periódica y almacenar históricos.
2. Integrar una segunda fuente meteorológica (Open-Meteo) y unificar los esquemas.
3. Definir un pipeline de features y experimentación para el modelo de predicción.
4. Levantar la API (FastAPI) y el prototipo de interfaz (Streamlit) descritos en el plan inicial.
