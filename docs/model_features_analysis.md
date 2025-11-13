# Análisis de Features para Modelo de ML - Runner Air Planner

## Datos disponibles de la API

### Estructura de datos en bruto:
```json
{
  "PROVINCIA": "28",
  "MUNICIPIO": "079", 
  "ESTACION": "11",
  "MAGNITUD": "12",
  "PUNTO_MUESTREO": "28079011_12_8",
  "ANO": "2025",
  "MES": "11",
  "DIA": "13",
  "H01": "83", "V01": "V",
  "H02": "37", "V02": "V",
  ...
  "H24": "0", "V24": "N"
}
```

### Datos procesados (CSV):
- `station_code`: Código de estación (4, 11, 16, 17, 18, 24, 27, 35, 36, 38, 39, 40, 47, 48, 49, 50, 54, 56, 57)
- `pollutant`: Código de contaminante (1, 6, 7, 8, 9, 10, 12, 14, 20, 30, 35)
- `measurement_time`: Timestamp de la medición
- `value`: Valor numérico de la concentración
- `is_valid`: Si la medición es válida (V/N)

## Contaminantes disponibles y su relevancia para correr

| Código | Nombre | Relevancia para correr | ¿Necesita master? |
|--------|--------|------------------------|-------------------|
| 1 | SO₂ | Baja (poco común en Madrid) | ✅ Para interpretación |
| 6 | CO | Media (tráfico) | ✅ Para interpretación |
| 7 | NO | Media (tráfico) | ✅ Para interpretación |
| 8 | NO₂ | **ALTA** (irritante respiratorio) | ✅ **CRÍTICO** |
| 9 | PM2.5 | **MUY ALTA** (penetra pulmones) | ✅ **CRÍTICO** |
| 10 | PM10 | **ALTA** (afecta respiración) | ✅ **CRÍTICO** |
| 12 | NOx | Media (suma NO+NO2) | ✅ Para interpretación |
| 14 | O₃ | **ALTA** (irritante, peor en verano) | ✅ **CRÍTICO** |
| 20 | Tolueno | Baja (COV, menos relevante) | ✅ Para interpretación |
| 30 | Benceno | Media (carcinógeno, pero bajos niveles) | ✅ Para interpretación |
| 35 | Etilbenceno | Baja (COV, menos relevante) | ✅ Para interpretación |

## Features necesarias para el modelo

### 1. Features de contaminantes (NUMÉRICAS) - **CRÍTICAS**
Estas son las más importantes para predecir si es bueno correr:

- `no2`: Dióxido de nitrógeno (µg/m³)
- `o3`: Ozono (µg/m³) 
- `pm10`: Partículas PM10 (µg/m³)
- `pm25`: Partículas PM2.5 (µg/m³)
- `no`: Óxido de nitrógeno (µg/m³) - menos crítico pero útil
- `nox`: Óxidos de nitrógeno totales (µg/m³)

**¿Necesita master?** NO para el modelo (solo valores numéricos), SÍ para interpretación y visualización

### 2. Features temporales - **IMPORTANTES**
El mejor momento para correr varía según la hora del día:

- `hour`: Hora del día (0-23) - **CRÍTICO** (contaminación varía por hora)
- `day_of_week`: Día de la semana (0-6) - Importante (tráfico varía)
- `month`: Mes (1-12) - Importante (O3 peor en verano)
- `is_weekend`: Boolean - Útil (menos tráfico)
- `is_rush_hour`: Boolean (7-9, 18-20) - **CRÍTICO**

**¿Necesita master?** NO, se extrae de timestamps

### 3. Features de estación - **ÚTILES**
El tipo de estación puede indicar el contexto:

- `station_type`: Tráfico vs Suburbana - **ÚTIL** (afecta niveles base)
- `station_code`: Código numérico - Puede ser feature categórica
- `district`: Distrito - Menos relevante para el modelo

**¿Necesita master?** SÍ, para obtener el tipo de estación

### 4. Features meteorológicas - **IMPORTANTES**
Ya integradas vía Open-Meteo:

- `temperature_c`: Temperatura - Importante (afecta O3)
- `relative_humidity`: Humedad - Útil
- `wind_speed_kmh`: Velocidad del viento - **CRÍTICO** (dispersa contaminación)
- `weather_code`: Código del tiempo - Útil

**¿Necesita master?** NO, ya viene de la API

### 5. Features derivadas - **ÚTILES**
Calculadas a partir de las anteriores:

- `air_quality_index`: Índice combinado de calidad del aire
- `worst_pollutant`: El contaminante con peor valor
- `pollutant_count`: Número de contaminantes medidos
- `valid_measurements_ratio`: Ratio de mediciones válidas

**¿Necesita master?** NO, se calculan

## Conclusión: ¿Necesitamos master data?

### ✅ SÍ, necesitamos master data para:

1. **Interpretación de códigos de contaminantes**
   - El modelo trabaja con números, pero necesitamos saber que "8" = NO₂
   - Para mostrar nombres legibles en el frontend
   - Para entender qué contaminante es más problemático

2. **Tipo de estación (feature del modelo)**
   - "Tráfico" vs "Suburbana" puede ser una feature categórica útil
   - Las estaciones de tráfico tienen niveles base más altos
   - Puede ayudar al modelo a contextualizar los valores

3. **Visualización y UX**
   - Mostrar nombres de estaciones en lugar de códigos
   - Mostrar nombres de contaminantes con símbolos (NO₂, O₃, PM₂.₅)
   - Ayudar al usuario a entender las recomendaciones

### ❌ NO necesitamos master data para:

1. **Valores numéricos de contaminantes** - Ya vienen en los datos
2. **Features temporales** - Se extraen de timestamps
3. **Features meteorológicas** - Ya vienen de Open-Meteo
4. **Cálculos del modelo** - El modelo solo necesita números

## Recomendación final

**Mantener el master data** porque:
- Es necesario para crear la feature `station_type` (tráfico/suburbana)
- Es esencial para interpretación y visualización
- Es pequeño y fácil de mantener
- Mejora significativamente la UX

**Features prioritarias para el modelo:**
1. Valores de contaminantes críticos (NO₂, O₃, PM10, PM2.5)
2. Hora del día (hour)
3. Tipo de estación (de master data)
4. Velocidad del viento (meteorología)
5. Día de la semana
6. Is rush hour

