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
        # Si el modelo ya está entrenado, usar SOLO las features que conoce
        # Verificar si el modelo tiene feature_names_ (scikit-learn)
        has_fitted_features = hasattr(self.model, 'feature_names_in_') and self.model.feature_names_in_ is not None
        
        if has_fitted_features:
            # El modelo tiene features definidas (fue entrenado)
            # Usar exactamente las features que scikit-learn conoce
            required_features = list(self.model.feature_names_in_)
            
            # Crear DataFrame con las features requeridas
            # Añadir features faltantes con valor 0, ignorar features extra
            feature_df = pd.DataFrame(index=df.index)
            for feature in required_features:
                if feature in df.columns:
                    feature_df[feature] = df[feature].fillna(0)
                else:
                    # Feature no existe en el DataFrame, usar 0
                    feature_df[feature] = 0
            
            # Asegurar el orden correcto de las columnas (mismo orden que durante entrenamiento)
            feature_df = feature_df[required_features]
            return feature_df
        elif self.is_trained and self.feature_columns is not None:
            # Fallback: usar feature_columns si está disponible
            required_features = self.feature_columns.copy()
            
            # Crear DataFrame con las features requeridas
            feature_df = pd.DataFrame(index=df.index)
            for feature in required_features:
                if feature in df.columns:
                    feature_df[feature] = df[feature].fillna(0)
                else:
                    # Feature no existe en el DataFrame, usar 0
                    feature_df[feature] = 0
            
            # Asegurar el orden correcto de las columnas
            feature_df = feature_df[required_features]
            return feature_df
        
        # Si no está entrenado, usar la lógica normal para definir features
        # Features numéricas de contaminantes
        pollutant_features = ["no2", "o3", "pm10", "pm25", "no", "nox"]
        
        # Features temporales
        temporal_features = ["hour", "day_of_week", "month", "is_weekend", "is_rush_hour"]
        
        # Features meteorológicas (usando predicción de 1 hora)
        weather_features = [
            "weather_temperature_c",
            "weather_humidity",
            "weather_wind_speed_kmh",
            "weather_precipitation_mm",
            "weather_cloud_cover",
            "weather_precipitation_probability",
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
            "precipitation_risk",  # Nueva feature basada en probabilidad de lluvia
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
        
        # Crear DataFrame con las features disponibles, rellenando NaN con 0
        feature_df = df[available_features].fillna(0)
        return feature_df

    def calculate_running_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate running suitability score (0-100) based on all factors.
        
        Score más suave que permite comparar estaciones.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Series with scores (0-100, higher is better)
        """
        score = pd.Series(100.0, index=df.index)  # Start with perfect score
        
        # Penalizar AQI de forma gradual (más suave)
        if "air_quality_index" in df.columns:
            aqi = df["air_quality_index"].fillna(0)
            # Penalización gradual: -1 punto por cada punto de AQI sobre 20
            score -= (aqi - 20).clip(lower=0) * 1.5
            # Penalización fuerte solo si AQI > 70
            score[aqi > 70] -= 30
        
        # Penalizar contaminantes de forma gradual
        pollutant_penalties = {
            "no2": {"threshold": 50, "severe": 100, "penalty_per_unit": 0.3, "severe_penalty": 25},
            "o3": {"threshold": 60, "severe": 120, "penalty_per_unit": 0.25, "severe_penalty": 30},
            "pm10": {"threshold": 30, "severe": 50, "penalty_per_unit": 0.4, "severe_penalty": 20},
            "pm25": {"threshold": 15, "severe": 25, "penalty_per_unit": 0.5, "severe_penalty": 25},
        }
        
        for pollutant, config in pollutant_penalties.items():
            if pollutant in df.columns:
                values = df[pollutant].fillna(0)
                # Penalización gradual desde el threshold
                excess = (values - config["threshold"]).clip(lower=0)
                score -= excess * config["penalty_per_unit"]
                # Penalización severa si supera el umbral crítico
                score[values > config["severe"]] -= config["severe_penalty"]
        
        # Penalizar mal tiempo (pero menos drástico)
        if "weather_code" in df.columns:
            bad_weather = [61, 63, 65, 71, 73, 75, 80, 81, 82, 95, 96, 99]
            is_bad = df["weather_code"].isin(bad_weather)
            score[is_bad] -= 40  # Penalización fuerte pero no total
        
        # Penalizar probabilidad de lluvia de forma gradual
        if "weather_precipitation_probability" in df.columns:
            prob = df["weather_precipitation_probability"].fillna(0)
            # Penalización gradual: -0.5 puntos por cada % sobre 30%
            score -= (prob - 30).clip(lower=0) * 0.5
            # Penalización fuerte si > 80%
            score[prob > 80] -= 20
        
        # Penalizar precipitación actual
        if "weather_precipitation_mm" in df.columns:
            precip = df["weather_precipitation_mm"].fillna(0)
            score -= precip * 5  # -5 puntos por cada mm
            score[precip > 2] -= 15  # Penalización extra si > 2mm
        
        # Bonus por viento fuerte (dispersa contaminación)
        if "wind_strong" in df.columns:
            score[df["wind_strong"] == 1] += 15
        
        # Penalizar viento débil (pero menos)
        if "wind_weak" in df.columns:
            score[df["wind_weak"] == 1] -= 8
        
        # Bonus por viento moderado
        if "weather_wind_speed_kmh" in df.columns:
            wind = df["weather_wind_speed_kmh"].fillna(0)
            # Bonus si viento está entre 10-20 km/h (ideal)
            ideal_wind = (wind >= 10) & (wind <= 20)
            score[ideal_wind] += 5
        
        # Bonus/penalización por temperatura
        if "weather_temperature_c" in df.columns:
            temp = df["weather_temperature_c"].fillna(20)
            # Temperatura ideal: 15-25°C
            ideal_temp = (temp >= 15) & (temp <= 25)
            score[ideal_temp] += 5
            # Penalizar temperaturas extremas
            score[temp > 30] -= 10
            score[temp < 5] -= 15
        
        # Bonus por zona suburbana
        if "is_suburban_station" in df.columns:
            score[df["is_suburban_station"] == 1] += 5
        
        # Penalizar zona de tráfico (pero menos)
        if "is_traffic_station" in df.columns:
            score[df["is_traffic_station"] == 1] -= 5
        
        # Asegurar que el score esté entre 0 y 100
        score = score.clip(0, 100)
        
        return score

    def create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create target variable: is_good_to_run (0/1).
        
        Basado en el score numérico: score >= 50 = bueno (1), score < 50 = no recomendado (0).
        Esto hace el modelo menos estricto.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Series with target values (1 = good to run, 0 = not recommended)
        """
        score = self.calculate_running_score(df)
        # Usar umbral más suave: score >= 50 es bueno
        target = (score >= 50).astype(int)
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
        
        # Guardar las features usadas (importante para predicciones futuras)
        # Asegurar que feature_columns esté definido con las features exactas usadas
        if self.feature_columns is None:
            self.feature_columns = list(X.columns)
        else:
            # Asegurar que coincidan
            self.feature_columns = list(X.columns)
        
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
    
    def predict_score(self, df: pd.DataFrame) -> pd.Series:
        """Predict running suitability score (0-100).
        
        Combina la predicción del modelo ML con el score calculado para dar
        una recomendación numérica más precisa.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Series with scores (0-100, higher is better)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Calcular score base
        base_score = self.calculate_running_score(df)
        
        # Obtener probabilidad del modelo ML
        proba = self.predict_proba(df)
        prob_good = proba["prob_good"]
        
        # Combinar: usar el score base pero ajustarlo con la confianza del modelo
        # Si el modelo tiene alta confianza, ajustar más el score
        ml_adjustment = (prob_good - 0.5) * 20  # Ajuste de -10 a +10 puntos
        
        final_score = base_score + ml_adjustment
        
        # Asegurar que esté entre 0 y 100
        final_score = final_score.clip(0, 100)
        
        return final_score

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
        
        # proba debería ser un array numpy con shape (n_samples, n_classes)
        # Para RandomForestClassifier con 2 clases, debería ser (n_samples, 2)
        import numpy as np
        
        # Convertir a numpy array si no lo es
        if not isinstance(proba, np.ndarray):
            proba = np.array(proba)
        
        # Asegurar que es 2D
        if proba.ndim == 1:
            # Si es 1D, podría ser que solo hay una clase, crear segunda columna
            proba = proba.reshape(-1, 1)
            if proba.shape[1] == 1:
                # Solo una clase, crear segunda columna complementaria
                proba = np.column_stack([1 - proba, proba])
        
        # Asegurar que tiene 2 columnas
        if proba.shape[1] == 1:
            # Solo una columna, crear la segunda
            proba = np.column_stack([1 - proba, proba])
        elif proba.shape[1] > 2:
            # Más de 2 columnas, tomar solo las primeras 2
            proba = proba[:, :2]
        
        # Crear DataFrame con el índice del df original
        result_df = pd.DataFrame(
            {
                "prob_not_good": proba[:, 0],
                "prob_good": proba[:, 1],
            },
            index=df.index,  # Usar el índice del df original
        )
        
        return result_df

    def predict_current_conditions(
        self,
        *,
        no2: float | None = None,
        o3: float | None = None,
        pm10: float | None = None,
        pm25: float | None = None,
        temperature: float | None = None,
        humidity: float | None = None,
        wind_speed: float | None = None,
        hour: int | None = None,
        is_weekend: bool | None = None,
        station_type: str | None = None,
    ) -> dict[str, Any]:
        """Predict running suitability for current conditions.
        
        This method creates a single-row DataFrame with the provided conditions
        and makes a prediction. Useful for real-time predictions.
        
        Args:
            no2: NO2 concentration (µg/m³)
            o3: O3 concentration (µg/m³)
            pm10: PM10 concentration (µg/m³)
            pm25: PM25 concentration (µg/m³)
            temperature: Temperature in Celsius
            humidity: Relative humidity (%)
            wind_speed: Wind speed (km/h)
            hour: Current hour (0-23)
            is_weekend: Whether it's weekend
            station_type: Station type ("Tráfico" or "Suburbana")
            
        Returns:
            Dictionary with prediction, probability, and recommendation
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Get current time if not provided
        from datetime import datetime
        now = datetime.now()
        if hour is None:
            hour = now.hour
        if is_weekend is None:
            is_weekend = now.weekday() >= 5
        
        # Create a single-row DataFrame with all features
        data = {
            "no2": [no2 or 0],
            "o3": [o3 or 0],
            "pm10": [pm10 or 0],
            "pm25": [pm25 or 0],
            "no": [0],
            "nox": [0],
            "hour": [hour],
            "day_of_week": [now.weekday()],
            "month": [now.month],
            "is_weekend": [1 if is_weekend else 0],
            "is_rush_hour": [1 if hour in [7, 8, 9, 18, 19, 20] else 0],
            "weather_temperature_c": [temperature or 0],
            "weather_humidity": [humidity or 0],
            "weather_wind_speed_kmh": [wind_speed or 0],
            "weather_precipitation_mm": [0],
            "weather_cloud_cover": [0],
            "weather_precipitation_probability": [0],
            "is_traffic_station": [1 if station_type == "Tráfico" else 0],
            "is_suburban_station": [1 if station_type == "Suburbana" else 0],
        }
        
        # Create synergy features
        wind = wind_speed or 0
        data["wind_no2_synergy"] = [wind / (no2 + 1) if no2 else 0]
        data["wind_o3_synergy"] = [wind / (o3 + 1) if o3 else 0]
        data["wind_pm10_synergy"] = [wind / (pm10 + 1) if pm10 else 0]
        data["wind_pm25_synergy"] = [wind / (pm25 + 1) if pm25 else 0]
        data["wind_strong"] = [1 if wind > 20 else 0]
        data["wind_weak"] = [1 if wind < 5 else 0]
        
        temp = temperature or 0
        data["temp_o3_synergy"] = [(temp * (o3 or 0)) / 100]
        
        # Calculate AQI
        pollutant_weights = {"no2": 0.25, "o3": 0.25, "pm10": 0.20, "pm25": 0.20}
        aqi = 0
        for pollutant, weight in pollutant_weights.items():
            value = data[pollutant][0]
            normalized = min((value / 100), 1) * 100
            aqi += normalized * weight
        data["air_quality_index"] = [aqi]
        data["precipitation_risk"] = [0]
        
        df = pd.DataFrame(data)
        
        # Make prediction
        prediction = self.predict(df).iloc[0]
        proba = self.predict_proba(df).iloc[0]
        
        is_good = bool(prediction == 1)
        prob_good = float(proba["prob_good"])
        
        # Generate recommendation
        if prob_good >= 0.7:
            recommendation = "Excelente momento para correr"
        elif prob_good >= 0.5:
            recommendation = "Buen momento para correr"
        elif prob_good >= 0.3:
            recommendation = "Condiciones moderadas, considera esperar"
        else:
            recommendation = "No recomendado para correr"
        
        return {
            "is_good_to_run": is_good,
            "prob_good": prob_good,
            "prob_not_good": float(proba["prob_not_good"]),
            "recommendation": recommendation,
            "air_quality_index": aqi,
        }


__all__ = ["RunningSuitabilityModel", "MODELS_DIR"]

