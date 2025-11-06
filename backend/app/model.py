"""Simple machine-learning helper to assess air quality risk levels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DEFAULT_THRESHOLD = 50.0


@dataclass
class PredictionResult:
    label: str
    probability: float


class AirQualityRiskModel:
    """Tiny wrapper around a one-dimensional logistic regression model."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold
        self._model: LogisticRegression | None = LogisticRegression()
        self._fallback_positive_rate: float | None = None

    def fit(self, frame: pd.DataFrame) -> None:
        cleaned = frame.dropna(subset=["value"]).copy()
        if cleaned.empty:
            self._model = None
            self._fallback_positive_rate = 0.0
            return
        cleaned["target"] = (cleaned["value"] >= self.threshold).astype(int)
        features = cleaned[["value"]].to_numpy(dtype=float)
        target = cleaned["target"].to_numpy(dtype=int)
        if len(np.unique(target)) < 2:
            self._model = None
            self._fallback_positive_rate = float(target[0])
            return
        model = LogisticRegression()
        model.fit(features, target)
        self._model = model
        self._fallback_positive_rate = None

    def fit_from_records(self, records: Iterable[dict[str, object]]) -> None:
        frame = pd.DataFrame(records)
        self.fit(frame)

    def predict(self, value: float) -> PredictionResult:
        if self._model is not None:
            probability = float(self._model.predict_proba(np.array([[value]], dtype=float))[0][1])
        else:
            if value >= self.threshold:
                probability = 1.0
            else:
                probability = self._fallback_positive_rate or 0.0
        label = "poor" if probability >= 0.5 else "good"
        return PredictionResult(label=label, probability=probability)
