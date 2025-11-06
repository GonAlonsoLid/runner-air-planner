from datetime import datetime

import pandas as pd
import pytest

from data_pipeline.ingest_madrid_air import (
    AirQualityIngestionError,
    build_dataframe,
    persist_dataframe,
    run_pipeline,
)


@pytest.fixture()
def sample_payload():
    return {
        "@graph": [
            {
                "estacion": "28079004",
                "magnitud": "NO2",
                "valor": "42",
                "unidad": "µg/m3",
                "fecha": "2024-04-01T08:00:00",
                "valido": "S",
            },
            {
                "estacion": "28079004",
                "magnitud": "NO2",
                "valor": "55",
                "unidad": "µg/m3",
                "fecha": "2024-04-01T09:00:00",
                "valido": "N",
            },
        ]
    }


def test_build_dataframe_orders_rows(sample_payload):
    frame = build_dataframe(sample_payload)
    assert list(frame["value"]) == [42.0, 55.0]
    assert frame.loc[0, "timestamp"] == datetime(2024, 4, 1, 8)


def test_persist_dataframe(tmp_path, sample_payload):
    frame = build_dataframe(sample_payload)
    path = persist_dataframe(frame, output_path=tmp_path / "dataset.csv")
    assert path.exists()
    loaded = pd.read_csv(path)
    assert len(loaded) == 2


def test_run_pipeline_empty_payload(tmp_path):
    class EmptySession:
        def get(self, *_args, **_kwargs):
            class Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"@graph": []}

            return Response()

    with pytest.raises(AirQualityIngestionError):
        run_pipeline(output_path=tmp_path / "dataset.csv", session=EmptySession())
