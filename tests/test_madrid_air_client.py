from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pytest

from runner_air_planner.config import AirQualityConfig
from runner_air_planner.data_sources.madrid_air import (
    MadridAirQualityClient,
    MadridAirQualityClientError,
)


class DummyHeaders:
    def __init__(self, charset: str = "utf-8") -> None:
        self._charset = charset

    def get_content_charset(self) -> str:
        return self._charset


class DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.headers = DummyHeaders()

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - nothing special to clean up
        return None


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> MadridAirQualityClient:
    payload = {
        "@graph": [
            {
                "estacion": "28079004",
                "punto_muestreo": "28079004-1",
                "magnitud": "NO2",
                "fecha": "20240131",
                "hora": "13",
                "valor": "12.5",
                "unidad": "µg/m3",
                "valido": "V",
            }
        ]
    }

    def fake_urlopen(request, timeout):  # type: ignore[override]
        return DummyResponse(payload)

    monkeypatch.setattr("runner_air_planner.data_sources.madrid_air.urlopen", fake_urlopen)
    return MadridAirQualityClient(settings=AirQualityConfig())


def test_fetch_measurements_normalises_payload(client: MadridAirQualityClient) -> None:
    measurements = client.fetch_measurements()

    assert len(measurements) == 1
    measurement = measurements[0]
    assert measurement.station_code == "28079004"
    assert measurement.pollutant == "NO2"
    assert measurement.value == pytest.approx(12.5)
    assert measurement.is_valid is True
    assert measurement.measurement_time == datetime(2024, 1, 31, 13, 0)


def test_fetch_raw_payload_returns_original_payload(client: MadridAirQualityClient) -> None:
    payload = client.fetch_raw_payload()
    assert "@graph" in payload


def test_extract_records_raises_for_unknown_structure(client: MadridAirQualityClient) -> None:
    with pytest.raises(MadridAirQualityClientError):
        client._extract_records({"unexpected": []})
