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
