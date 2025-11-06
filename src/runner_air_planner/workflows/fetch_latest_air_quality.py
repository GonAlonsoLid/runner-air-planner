"""Workflow that downloads the latest Madrid air quality snapshot."""

from __future__ import annotations

from typing import Mapping

from runner_air_planner.config import get_settings
from runner_air_planner.data_sources.madrid_air import MadridAirQualityClient
from runner_air_planner.storage.local import LocalDataStorage


def run(params: Mapping[str, str] | None = None) -> str:
    """Fetch Madrid air quality data and persist it locally.

    Parameters
    ----------
    params:
        Optional query-string parameters forwarded to the Madrid API. They can be
        used to filter by station or pollutant if the provider supports it.

    Returns
    -------
    str
        The path to the stored JSON payload.
    """

    settings = get_settings()
    client = MadridAirQualityClient(settings=settings.air_quality)
    storage = LocalDataStorage(settings.storage.raw_data_dir)

    raw_payload = client.fetch_raw_payload(params=params)
    stored_path = storage.write_json(raw_payload, prefix="madrid_air_quality")
    return str(stored_path)


__all__ = ["run"]
