"""Machine Learning model definition for Runner Air Planner."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Path donde se guardan los modelos entrenados
MODELS_DIR = Path(__file__).resolve().parents[3] / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class RunningSuitabilityModel:
    """Modelo de ML para predecir si es buen momento para correr.
    
    Usa Random Forest para clasificar condiciones de calidad del aire
    y meteorología como "bueno para correr" (1) o "no recomendado" (0).
    """

    def __init__(self, *, model_path: Path | None = None) -> None:
        """Initialize the model.
        
        Args:
            model_path: Path to saved model (loads if exists)
        """
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        self.model_path = model_path or (MODELS_DIR / "running_model.pkl")
        self.feature_columns: list[str] | None = None
        self.is_trained = False

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature columns for training/prediction.
        
        Args:
            df: DataFrame with all columns
            
        Returns:
            DataFrame with only feature columns
        """
        # Features numéricas de contaminantes
        pollutant_features = ["no2", "o3", "pm10", "pm25", "no", "nox"]
        
        # Features temporales
        temporal_features = ["hour", "day_of_week", "month", "is_weekend", "is_rush_hour"]
        
        # Features meteorológicas
        weather_features = [
            "weather_temperature_c",
            "weather_humidity",
            "weather_wind_speed_kmh",
        ]
        
        # Features de sinergia
        synergy_features = [
            "wind_no2_synergy",
            "wind_o3_synergy",
            "wind_pm10_synergy",
            "wind_pm25_synergy",
            "wind_strong",
            "wind_weak",
            "temp_o3_synergy",
            "air_quality_index",
        ]
        
        # Features de estación
        station_features = ["is_traffic_station", "is_suburban_station"]
        
        all_features = (
            pollutant_features
            + temporal_features
            + weather_features
            + synergy_features
            + station_features
        )
        
        # Filtrar solo las que existen en el DataFrame
        available_features = [f for f in all_features if f in df.columns]
        
        # Guardar para uso futuro
        if self.feature_columns is None:
            self.feature_columns = available_features
        
        return df[available_features].fillna(0)

    def create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create target variable: is_good_to_run (0/1).
        
        Basado en umbrales de calidad del aire y condiciones meteorológicas.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Series with target values (1 = good to run, 0 = not recommended)
        """
        target = pd.Series(1, index=df.index)  # Start with all good
        
        # Penalizar alto AQI
        if "air_quality_index" in df.columns:
            target[df["air_quality_index"] > 50] = 0
        
        # Penalizar contaminantes altos
        for pollutant in ["no2", "o3", "pm10", "pm25"]:
            if pollutant in df.columns:
                # Umbrales basados en estándares de calidad del aire
                thresholds = {"no2": 100, "o3": 120, "pm10": 50, "pm25": 25}
                threshold = thresholds.get(pollutant, 100)
                target[df[pollutant] > threshold] = 0
        
        # Penalizar mal tiempo
        if "weather_code" in df.columns:
            bad_weather = [61, 63, 65, 71, 73, 75, 80, 81, 82, 95, 96, 99]
            target[df["weather_code"].isin(bad_weather)] = 0
        
        # Bonus por viento fuerte (dispersa contaminación)
        if "wind_strong" in df.columns:
            target[(df["wind_strong"] == 1) & (target == 0)] = 1
        
        return target

    def train(self, df: pd.DataFrame) -> dict[str, Any]:
        """Train the model.
        
        Args:
            df: DataFrame with features and target
            
        Returns:
            Dictionary with training metrics
        """
        X = self.prepare_features(df)
        y = self.create_target(df)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        return {
            "train_accuracy": train_score,
            "test_accuracy": test_score,
            "n_samples": len(df),
            "n_features": len(X.columns),
        }

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict if it's good to run.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Series with predictions (1 = good, 0 = not recommended)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        X = self.prepare_features(df)
        return pd.Series(self.model.predict(X), index=df.index)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict probabilities for each class.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with probabilities for each class
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        X = self.prepare_features(df)
        proba = self.model.predict_proba(X)
        return pd.DataFrame(
            proba,
            index=df.index,
            columns=["prob_not_good", "prob_good"],
        )


__all__ = ["RunningSuitabilityModel", "MODELS_DIR"]

