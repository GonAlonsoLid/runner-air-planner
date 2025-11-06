"""FastAPI application exposing access to Madrid air quality data."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .model import AirQualityRiskModel
from .storage import DATA_PATH, load_dataset, recent_measurements

app = FastAPI(title="Runner Air Planner", version="0.1.0")


class PredictionRequest(BaseModel):
    value: float = Field(..., description="Measurement value in the dataset's units")


class PredictionResponse(BaseModel):
    label: str
    probability: float


class DatasetInfo(BaseModel):
    records: int
    source_path: Path | None


class AppState:
    def __init__(self) -> None:
        self.model = AirQualityRiskModel()
        self.dataset_path = DATA_PATH
        self._fit_model()

    def _fit_model(self) -> None:
        frame = load_dataset(path=self.dataset_path)
        if frame.empty:
            self.model.fit(frame)
        else:
            self.model.fit(frame)


def get_state() -> AppState:
    state = getattr(app.state, "instance", None)
    if state is None:
        state = AppState()
        app.state.instance = state
    return state


@app.on_event("startup")
def setup_state() -> None:
    app.state.instance = AppState()


@app.get("/health", tags=["meta"])
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/dataset", response_model=DatasetInfo, tags=["dataset"])
def dataset_info(state: Annotated[AppState, Depends(get_state)]) -> DatasetInfo:
    frame = load_dataset(path=state.dataset_path)
    return DatasetInfo(records=len(frame), source_path=state.dataset_path if state.dataset_path.exists() else None)


@app.get("/measurements", tags=["dataset"])
def measurements(
    state: Annotated[AppState, Depends(get_state)],
    limit: Annotated[int | None, Query(ge=1, le=500, description="Maximum number of rows to return")] = 100,
) -> list[dict[str, object]]:
    records = list(recent_measurements(limit=limit, path=state.dataset_path))
    return records


@app.post("/predict", response_model=PredictionResponse, tags=["ml"])
def predict(
    payload: PredictionRequest,
    state: Annotated[AppState, Depends(get_state)],
) -> PredictionResponse:
    if payload.value is None:  # pragma: no cover - validation already enforces this
        raise HTTPException(status_code=400, detail="Missing value")
    result = state.model.predict(payload.value)
    return PredictionResponse(label=result.label, probability=result.probability)
