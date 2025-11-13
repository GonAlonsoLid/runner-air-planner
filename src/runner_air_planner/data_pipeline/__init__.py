"""Data pipeline modules for Runner Air Planner ML model."""

from .accumulate_data import (
    accumulate_ml_dataset,
    load_accumulated_dataset,
    save_accumulated_dataset,
)
from .data_collector import DataCollector
from .ingest_madrid_air import (
    DATASET_URL,
    DEFAULT_OUTPUT,
    MadridAirDownloadError,
    download_latest_measurements,
    persist_measurements,
)
from .master_data import (
    POLLUTANT_MASTER,
    STATION_MASTER,
    get_pollutant_info,
    get_pollutant_name,
    get_station_info,
    get_station_name,
    normalize_pollutant_code,
)
from .weather import (
    WeatherClient,
    WeatherForecast,
    WeatherReport,
    WeatherServiceError,
)

__all__ = [
    # Main collector
    "DataCollector",
    # Data accumulation
    "accumulate_ml_dataset",
    "load_accumulated_dataset",
    "save_accumulated_dataset",
    # Air quality ingestion
    "DATASET_URL",
    "DEFAULT_OUTPUT",
    "MadridAirDownloadError",
    "download_latest_measurements",
    "persist_measurements",
    # Master data
    "POLLUTANT_MASTER",
    "STATION_MASTER",
    "get_pollutant_info",
    "get_pollutant_name",
    "get_station_info",
    "get_station_name",
    "normalize_pollutant_code",
    # Weather
    "WeatherClient",
    "WeatherForecast",
    "WeatherReport",
    "WeatherServiceError",
]
