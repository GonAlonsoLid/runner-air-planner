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
        if self.is_trained:
            # Preferir feature_names_in_ de scikit-learn si está disponible
            if hasattr(self.model, 'feature_names_in_') and self.model.feature_names_in_ is not None:
                required_features = list(self.model.feature_names_in_)
            elif self.feature_columns is not None:
                required_features = self.feature_columns.copy()
            else:
                raise RuntimeError("Model is trained but feature names are not available")
            
            # Crear DataFrame con las features requeridas, rellenando faltantes con 0
            feature_df = pd.DataFrame(index=df.index)
            for feature in required_features:
                feature_df[feature] = df[feature].fillna(0) if feature in df.columns else 0
            
            return feature_df[required_features]  # Asegurar orden correcto
        
        # Si no está entrenado, definir features desde cero
        feature_groups = {
            "pollutant": ["no2", "o3", "pm10", "pm25", "no", "nox"],
            "temporal": ["hour", "day_of_week", "month", "is_weekend", "is_rush_hour"],
            "weather": [
                "weather_temperature_c",
                "weather_humidity",
                "weather_wind_speed_kmh",
                "weather_precipitation_mm",
                "weather_cloud_cover",
                "weather_precipitation_probability",
            ],
            "synergy": [
                "wind_no2_synergy",
                "wind_o3_synergy",
                "wind_pm10_synergy",
                "wind_pm25_synergy",
                "wind_strong",
                "wind_weak",
                "temp_o3_synergy",
                "air_quality_index",
                "precipitation_risk",
            ],
            "station": ["is_traffic_station", "is_suburban_station"],
        }
        
        all_features = [f for group in feature_groups.values() for f in group]
        available_features = [f for f in all_features if f in df.columns]
        
        # Guardar para uso futuro
        self.feature_columns = available_features
        
        return df[available_features].fillna(0)

    def calculate_running_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate running suitability score (0-100) based on all factors.
        
        Score más granular que discrimina mejor entre estaciones, produciendo
        resultados más variados incluso con buenas condiciones.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Series with scores (0-100, higher is better)
        """
        import numpy as np
        
        # Empezamos con un score base de 70 (permite subir a ~95 máx y bajar a 0)
        score = pd.Series(70.0, index=df.index)
        
        # === CALIDAD DEL AIRE (factor principal, ~40% del score) ===
        if "air_quality_index" in df.columns:
            aqi = df["air_quality_index"].fillna(25)
            
            # Bonus/penalización por rangos de AQI
            score.loc[aqi < 10] += 15  # Excelente
            score.loc[(aqi >= 10) & (aqi < 20)] += 8  # Muy bueno
            score.loc[(aqi >= 20) & (aqi < 30)] += 3  # Bueno
            # AQI 30-40: neutro
            score.loc[(aqi >= 40) & (aqi < 50)] -= 8  # Regular
            score.loc[(aqi >= 50) & (aqi < 70)] -= 20  # Malo
            score.loc[aqi >= 70] -= 35  # Muy malo
            
            # Penalización proporcional continua (para granularidad)
            score -= aqi * 0.25
        
        # === CONTAMINANTES ESPECÍFICOS ===
        pollutant_config = {
            "no2": {"ideal": 20, "moderate": 50, "high": 80, "penalty": 0.15},
            "o3": {"ideal": 30, "moderate": 60, "high": 100, "penalty": 0.12},
            "pm10": {"ideal": 15, "moderate": 35, "high": 50, "penalty": 0.2},
            "pm25": {"ideal": 8, "moderate": 18, "high": 25, "penalty": 0.25},
        }
        
        for pollutant, cfg in pollutant_config.items():
            if pollutant in df.columns:
                val = df[pollutant].fillna(cfg["moderate"] * 0.8)
                
                # Bonus por niveles ideales
                score.loc[val < cfg["ideal"]] += 3
                
                # Penalización gradual desde nivel moderado
                excess = (val - cfg["moderate"]).clip(lower=0)
                score -= excess * cfg["penalty"]
                
                # Penalización extra por nivel alto
                score.loc[val >= cfg["high"]] -= 8
        
        # === TEMPERATURA ===
        if "weather_temperature_c" in df.columns:
            temp = df["weather_temperature_c"].fillna(18)
            
            # Temperatura ideal para correr: 12-20°C
            score.loc[(temp >= 12) & (temp <= 20)] += 6
            # Aceptable: 8-12°C o 20-25°C
            score.loc[((temp >= 8) & (temp < 12)) | ((temp > 20) & (temp <= 25))] += 2
            # Frío: 3-8°C
            score.loc[(temp >= 3) & (temp < 8)] -= 4
            # Muy frío: < 3°C
            score.loc[temp < 3] -= 12
            # Calor: 25-30°C
            score.loc[(temp > 25) & (temp <= 30)] -= 6
            # Mucho calor: > 30°C
            score.loc[temp > 30] -= 15
        
        # === VIENTO ===
        if "weather_wind_speed_kmh" in df.columns:
            wind = df["weather_wind_speed_kmh"].fillna(10)
            
            # Viento ideal: 8-15 km/h (dispersa contaminación sin ser incómodo)
            score.loc[(wind >= 8) & (wind <= 15)] += 5
            # Viento moderado: 15-22 km/h
            score.loc[(wind > 15) & (wind <= 22)] += 2
            # Poco viento: < 5 km/h (contaminación se acumula)
            score.loc[wind < 5] -= 8
            # Viento fuerte: > 25 km/h (incómodo)
            score.loc[wind > 25] -= 4
        
        # === HUMEDAD ===
        if "weather_humidity" in df.columns:
            humidity = df["weather_humidity"].fillna(60)
            
            # Humedad ideal: 40-60%
            score.loc[(humidity >= 40) & (humidity <= 60)] += 3
            # Muy seco: < 30%
            score.loc[humidity < 30] -= 2
            # Muy húmedo: > 80%
            score.loc[humidity > 80] -= 6
        
        # === PRECIPITACIÓN ===
        if "weather_precipitation_probability" in df.columns:
            precip_prob = df["weather_precipitation_probability"].fillna(0)
            
            # Sin lluvia
            score.loc[precip_prob <= 10] += 2
            # Probabilidad media: 30-60%
            score.loc[(precip_prob > 30) & (precip_prob <= 60)] -= 6
            # Alta probabilidad: > 60%
            score.loc[precip_prob > 60] -= 15
        
        if "weather_precipitation_mm" in df.columns:
            precip = df["weather_precipitation_mm"].fillna(0)
            score -= precip * 6
            score.loc[precip > 1] -= 10
        
        # Mal tiempo (códigos específicos)
        if "weather_code" in df.columns:
            bad_weather = [61, 63, 65, 71, 73, 75, 80, 81, 82, 95, 96, 99]
            score.loc[df["weather_code"].isin(bad_weather)] -= 25
        
        # === TIPO DE ESTACIÓN ===
        if "is_suburban_station" in df.columns:
            score.loc[df["is_suburban_station"] == 1] += 6
        
        if "is_traffic_station" in df.columns:
            score.loc[df["is_traffic_station"] == 1] -= 8
        
        # === FACTORES DE SINERGIA ===
        if "wind_strong" in df.columns:
            score.loc[df["wind_strong"] == 1] += 3
        
        if "wind_weak" in df.columns:
            score.loc[df["wind_weak"] == 1] -= 4
        
        # === HORA DEL DÍA ===
        if "hour" in df.columns:
            hour = df["hour"].fillna(12)
            
            # Horas óptimas: 7-10 (mañana) y 18-20 (tarde)
            optimal = ((hour >= 7) & (hour <= 10)) | ((hour >= 18) & (hour <= 20))
            score.loc[optimal] += 3
            
            # Rush hour en días laborables
            if "is_weekend" in df.columns:
                rush = ((hour >= 8) & (hour <= 9)) | ((hour >= 17) & (hour <= 19))
                score.loc[rush & (df["is_weekend"] == 0)] -= 2
            
            # Noche/madrugada
            score.loc[(hour < 6) | (hour > 22)] -= 4
        
        # Asegurar rango 5-98 para evitar extremos absolutos
        # (solo condiciones verdaderamente perfectas/terribles llegan a 98/5)
        score = score.clip(5, 98).round(1)
        
        return score

    def create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create target variable: is_good_to_run (0/1).
        
        Basado en el score numérico: score >= 60 = bueno (1), score < 60 = no recomendado (0).
        El umbral de 60 asegura que solo condiciones genuinamente buenas se marquen como positivas.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Series with target values (1 = good to run, 0 = not recommended)
        """
        score = self.calculate_running_score(df)
        # Umbral de 60: selectivo pero no demasiado estricto
        target = (score >= 60).astype(int)
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
        una recomendación numérica más precisa y variada.
        
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
        
        # Combinar: ponderar 70% score base + 30% confianza del modelo ML
        # Esto da más peso al cálculo basado en reglas pero permite que el ML
        # ajuste significativamente el resultado
        ml_score = prob_good * 100  # Convertir probabilidad a escala 0-100
        
        final_score = (base_score * 0.7) + (ml_score * 0.3)
        
        # Asegurar que esté entre 0 y 100
        final_score = final_score.clip(0, 100)
        
        # Redondear a 1 decimal
        final_score = final_score.round(1)
        
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


__all__ = ["RunningSuitabilityModel", "MODELS_DIR"]

