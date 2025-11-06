import pandas as pd

from backend.app.model import AirQualityRiskModel


def test_model_fits_and_predicts():
    frame = pd.DataFrame(
        {
            "value": [20, 30, 45, 60, 80, 100],
        }
    )
    model = AirQualityRiskModel(threshold=50)
    model.fit(frame)
    low = model.predict(25)
    high = model.predict(90)
    assert low.label == "good"
    assert high.label == "poor"
    assert 0 <= low.probability <= 1
    assert 0 <= high.probability <= 1


def test_model_fallback_when_single_class():
    frame = pd.DataFrame({"value": [10, 20, 30]})
    model = AirQualityRiskModel(threshold=5)
    model.fit(frame)
    result = model.predict(4)
    assert result.label == "good"
    assert result.probability == 0.0
