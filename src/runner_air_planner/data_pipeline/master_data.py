"""Master data (lookup tables) for Madrid air quality stations and pollutants."""

from __future__ import annotations

from typing import Any

# Códigos de magnitudes (contaminantes) según estándar español
POLLUTANT_MASTER: dict[int | str, dict[str, Any]] = {
    1: {
        "code": "1",
        "name": "SO₂",
        "full_name": "Dióxido de azufre",
        "english_name": "Sulfur dioxide",
        "unit": "µg/m³",
        "description": "Gas incoloro con olor irritante, emitido principalmente por la quema de combustibles fósiles",
    },
    6: {
        "code": "6",
        "name": "CO",
        "full_name": "Monóxido de carbono",
        "english_name": "Carbon monoxide",
        "unit": "mg/m³",
        "description": "Gas incoloro e inodoro, producto de la combustión incompleta",
    },
    7: {
        "code": "7",
        "name": "NO",
        "full_name": "Óxido de nitrógeno",
        "english_name": "Nitrogen monoxide",
        "unit": "µg/m³",
        "description": "Gas tóxico formado en procesos de combustión a alta temperatura",
    },
    8: {
        "code": "8",
        "name": "NO₂",
        "full_name": "Dióxido de nitrógeno",
        "english_name": "Nitrogen dioxide",
        "unit": "µg/m³",
        "description": "Gas pardo-rojizo, irritante para las vías respiratorias",
    },
    9: {
        "code": "9",
        "name": "PM2.5",
        "full_name": "Partículas PM2.5",
        "english_name": "Particulate matter 2.5",
        "unit": "µg/m³",
        "description": "Partículas en suspensión con diámetro menor a 2.5 micrómetros",
    },
    10: {
        "code": "10",
        "name": "PM10",
        "full_name": "Partículas PM10",
        "english_name": "Particulate matter 10",
        "unit": "µg/m³",
        "description": "Partículas en suspensión con diámetro menor a 10 micrómetros",
    },
    12: {
        "code": "12",
        "name": "NOx",
        "full_name": "Óxidos de nitrógeno",
        "english_name": "Nitrogen oxides",
        "unit": "µg/m³",
        "description": "Suma de NO y NO₂, principales precursores del ozono troposférico",
    },
    14: {
        "code": "14",
        "name": "O₃",
        "full_name": "Ozono",
        "english_name": "Ozone",
        "unit": "µg/m³",
        "description": "Gas formado por reacciones fotoquímicas, puede ser perjudicial a nivel del suelo",
    },
    20: {
        "code": "20",
        "name": "Tolueno",
        "full_name": "Tolueno",
        "english_name": "Toluene",
        "unit": "µg/m³",
        "description": "Compuesto orgánico volátil (COV), usado como disolvente",
    },
    30: {
        "code": "30",
        "name": "Benceno",
        "full_name": "Benceno",
        "english_name": "Benzene",
        "unit": "µg/m³",
        "description": "Compuesto orgánico volátil (COV), carcinógeno conocido",
    },
    35: {
        "code": "35",
        "name": "Etilbenceno",
        "full_name": "Etilbenceno",
        "english_name": "Ethylbenzene",
        "unit": "µg/m³",
        "description": "Compuesto orgánico volátil (COV), usado en la producción de estireno",
    },
    # Aliases en minúsculas para compatibilidad
    "1": {
        "code": "1",
        "name": "SO₂",
        "full_name": "Dióxido de azufre",
        "english_name": "Sulfur dioxide",
        "unit": "µg/m³",
        "description": "Gas incoloro con olor irritante, emitido principalmente por la quema de combustibles fósiles",
    },
    "so2": {
        "code": "1",
        "name": "SO₂",
        "full_name": "Dióxido de azufre",
        "english_name": "Sulfur dioxide",
        "unit": "µg/m³",
        "description": "Gas incoloro con olor irritante, emitido principalmente por la quema de combustibles fósiles",
    },
    "no2": {
        "code": "8",
        "name": "NO₂",
        "full_name": "Dióxido de nitrógeno",
        "english_name": "Nitrogen dioxide",
        "unit": "µg/m³",
        "description": "Gas pardo-rojizo, irritante para las vías respiratorias",
    },
    "no": {
        "code": "7",
        "name": "NO",
        "full_name": "Óxido de nitrógeno",
        "english_name": "Nitrogen monoxide",
        "unit": "µg/m³",
        "description": "Gas tóxico formado en procesos de combustión a alta temperatura",
    },
    "nox": {
        "code": "12",
        "name": "NOx",
        "full_name": "Óxidos de nitrógeno",
        "english_name": "Nitrogen oxides",
        "unit": "µg/m³",
        "description": "Suma de NO y NO₂, principales precursores del ozono troposférico",
    },
    "o3": {
        "code": "14",
        "name": "O₃",
        "full_name": "Ozono",
        "english_name": "Ozone",
        "unit": "µg/m³",
        "description": "Gas formado por reacciones fotoquímicas, puede ser perjudicial a nivel del suelo",
    },
    "pm10": {
        "code": "10",
        "name": "PM10",
        "full_name": "Partículas PM10",
        "english_name": "Particulate matter 10",
        "unit": "µg/m³",
        "description": "Partículas en suspensión con diámetro menor a 10 micrómetros",
    },
    "pm25": {
        "code": "9",
        "name": "PM2.5",
        "full_name": "Partículas PM2.5",
        "english_name": "Particulate matter 2.5",
        "unit": "µg/m³",
        "description": "Partículas en suspensión con diámetro menor a 2.5 micrómetros",
    },
    "pm2.5": {
        "code": "9",
        "name": "PM2.5",
        "full_name": "Partículas PM2.5",
        "english_name": "Particulate matter 2.5",
        "unit": "µg/m³",
        "description": "Partículas en suspensión con diámetro menor a 2.5 micrómetros",
    },
}

# Información básica de estaciones de calidad del aire de Madrid
# Nota: Los códigos pueden variar, esta es una lista aproximada basada en datos comunes
STATION_MASTER: dict[int | str, dict[str, Any]] = {
    4: {
        "code": "4",
        "name": "Escuelas Aguirre",
        "district": "Retiro",
        "type": "Tráfico",
        "latitude": 40.4140,
        "longitude": -3.6800,
    },
    8: {
        "code": "8",
        "name": "Plaza del Carmen",
        "district": "Centro",
        "type": "Tráfico",
        "latitude": 40.4200,
        "longitude": -3.7050,
    },
    11: {
        "code": "11",
        "name": "Plaza de España",
        "district": "Centro",
        "type": "Tráfico",
        "latitude": 40.4240,
        "longitude": -3.7120,
    },
    16: {
        "code": "16",
        "name": "Arturo Soria",
        "district": "Chamartín",
        "type": "Tráfico",
        "latitude": 40.4400,
        "longitude": -3.6600,
    },
    17: {
        "code": "17",
        "name": "Villaverde",
        "district": "Villaverde",
        "type": "Tráfico",
        "latitude": 40.3500,
        "longitude": -3.7000,
    },
    18: {
        "code": "18",
        "name": "Farolillo",
        "district": "Villaverde",
        "type": "Tráfico",
        "latitude": 40.3400,
        "longitude": -3.7100,
    },
    24: {
        "code": "24",
        "name": "Casa de Campo",
        "district": "Moncloa-Aravaca",
        "type": "Suburbana",
        "latitude": 40.4100,
        "longitude": -3.7500,
    },
    27: {
        "code": "27",
        "name": "Barajas",
        "district": "Barajas",
        "type": "Suburbana",
        "latitude": 40.4800,
        "longitude": -3.5700,
    },
    35: {
        "code": "35",
        "name": "Plaza del Carmen",
        "district": "Centro",
        "type": "Tráfico",
        "latitude": 40.4200,
        "longitude": -3.7050,
    },
    36: {
        "code": "36",
        "name": "Moratalaz",
        "district": "Moratalaz",
        "type": "Tráfico",
        "latitude": 40.4000,
        "longitude": -3.6500,
    },
    38: {
        "code": "38",
        "name": "Cuatro Caminos",
        "district": "Tetúan",
        "type": "Tráfico",
        "latitude": 40.4400,
        "longitude": -3.7000,
    },
    39: {
        "code": "39",
        "name": "Barrio del Pilar",
        "district": "Fuencarral-El Pardo",
        "type": "Tráfico",
        "latitude": 40.4800,
        "longitude": -3.7100,
    },
    40: {
        "code": "40",
        "name": "Vallecas",
        "district": "Villa de Vallecas",
        "type": "Tráfico",
        "latitude": 40.3700,
        "longitude": -3.6200,
    },
    47: {
        "code": "47",
        "name": "Mendez Alvaro",
        "district": "Arganzuela",
        "type": "Tráfico",
        "latitude": 40.4000,
        "longitude": -3.6800,
    },
    48: {
        "code": "48",
        "name": "Castellana",
        "district": "Chamartín",
        "type": "Tráfico",
        "latitude": 40.4500,
        "longitude": -3.6900,
    },
    49: {
        "code": "49",
        "name": "Parque del Retiro",
        "district": "Retiro",
        "type": "Suburbana",
        "latitude": 40.4150,
        "longitude": -3.6800,
    },
    50: {
        "code": "50",
        "name": "Plaza de Castilla",
        "district": "Chamartín",
        "type": "Tráfico",
        "latitude": 40.4660,
        "longitude": -3.6900,
    },
    54: {
        "code": "54",
        "name": "Ensanche de Vallecas",
        "district": "Villa de Vallecas",
        "type": "Tráfico",
        "latitude": 40.3600,
        "longitude": -3.6100,
    },
    56: {
        "code": "56",
        "name": "Plaza Elíptica",
        "district": "Usera",
        "type": "Tráfico",
        "latitude": 40.3900,
        "longitude": -3.7200,
    },
    57: {
        "code": "57",
        "name": "Sanchinarro",
        "district": "Hortaleza",
        "type": "Tráfico",
        "latitude": 40.4900,
        "longitude": -3.6500,
    },
}


def get_pollutant_info(pollutant_code: int | str) -> dict[str, Any] | None:
    """Get information about a pollutant by its code.
    
    Args:
        pollutant_code: Code of the pollutant (int or str, e.g., 1, "1", "so2", "no2")
    
    Returns:
        Dictionary with pollutant information or None if not found
    """
    # Normalizar el código
    if isinstance(pollutant_code, str):
        pollutant_code = pollutant_code.lower().strip()
    
    return POLLUTANT_MASTER.get(pollutant_code)


def get_pollutant_name(pollutant_code: int | str) -> str:
    """Get the short name of a pollutant.
    
    Args:
        pollutant_code: Code of the pollutant
    
    Returns:
        Short name (e.g., "NO₂", "PM10") or the code as string if not found
    """
    info = get_pollutant_info(pollutant_code)
    if info:
        return info["name"]
    return str(pollutant_code)


def get_pollutant_full_name(pollutant_code: int | str) -> str:
    """Get the full name of a pollutant.
    
    Args:
        pollutant_code: Code of the pollutant
    
    Returns:
        Full name (e.g., "Dióxido de nitrógeno") or the code as string if not found
    """
    info = get_pollutant_info(pollutant_code)
    if info:
        return info["full_name"]
    return str(pollutant_code)


def get_station_info(station_code: int | str) -> dict[str, Any] | None:
    """Get information about a station by its code.
    
    Args:
        station_code: Code of the station (int or str)
    
    Returns:
        Dictionary with station information or None if not found
    """
    # Normalizar el código
    if isinstance(station_code, str):
        try:
            station_code = int(station_code)
        except ValueError:
            return None
    
    return STATION_MASTER.get(station_code)


def get_station_name(station_code: int | str) -> str:
    """Get the name of a station.
    
    Args:
        station_code: Code of the station
    
    Returns:
        Station name or the code as string if not found
    """
    info = get_station_info(station_code)
    if info:
        return info["name"]
    return f"Estación {station_code}"


def normalize_pollutant_code(pollutant_code: int | str) -> str:
    """Normalize a pollutant code to the standard format used in the model.
    
    Maps numeric codes to lowercase string names used by the model.
    
    Args:
        pollutant_code: Code of the pollutant
    
    Returns:
        Normalized code (e.g., "no2", "pm10", "o3")
    """
    info = get_pollutant_info(pollutant_code)
    if info:
        code = info["code"]
        # Mapear a nombres usados en el modelo
        mapping = {
            "1": "so2",
            "6": "co",
            "7": "no",
            "8": "no2",
            "9": "pm25",
            "10": "pm10",
            "12": "nox",
            "14": "o3",
            "20": "tolueno",
            "30": "benceno",
            "35": "etilbenceno",
        }
        return mapping.get(code, code)
    
    # Si no se encuentra, intentar normalizar directamente
    if isinstance(pollutant_code, str):
        return pollutant_code.lower().strip()
    return str(pollutant_code).lower()


__all__ = [
    "POLLUTANT_MASTER",
    "STATION_MASTER",
    "get_pollutant_info",
    "get_pollutant_name",
    "get_pollutant_full_name",
    "get_station_info",
    "get_station_name",
    "normalize_pollutant_code",
]

