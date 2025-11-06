from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app


def prepare_dataset(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "station": ["28079004", "28079004", "28079008"],
            "pollutant": ["NO2", "NO2", "O3"],
            "value": [35.0, 80.0, 20.0],
            "unit": ["µg/m3", "µg/m3", "µg/m3"],
            "is_valid": [True, True, True],
            "timestamp": pd.to_datetime([
                "2024-04-01T08:00:00",
                "2024-04-01T09:00:00",
                "2024-04-01T10:00:00",
            ]),
        }
    )
    dataset_path = tmp_path / "dataset.csv"
    frame.to_csv(dataset_path, index=False)
    return dataset_path


def test_api_endpoints(tmp_path, monkeypatch):
    dataset_path = prepare_dataset(tmp_path)

    monkeypatch.setattr("backend.app.storage.DATA_PATH", dataset_path)
    monkeypatch.setattr("backend.app.main.DATA_PATH", dataset_path)

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

        response = client.get("/measurements", params={"limit": 2})
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["pollutant"] in {"NO2", "O3"}

        response = client.post("/predict", json={"value": 55.0})
        assert response.status_code == 200
        payload = response.json()
        assert payload["label"] in {"good", "poor"}
        assert 0 <= payload["probability"] <= 1
