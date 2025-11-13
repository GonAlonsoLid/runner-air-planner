"""Script to make predictions with the trained ML model."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

from .model import MODELS_DIR, RunningSuitabilityModel

# Configurar encoding UTF-8 para Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def main() -> None:
    """Make predictions with trained model."""
    parser = argparse.ArgumentParser(description="Predecir mejor momento para correr")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODELS_DIR / "running_model.pkl",
        help="Ruta al modelo entrenado (default: data/models/running_model.pkl)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/ml_dataset_accumulated.csv"),
        help="Dataset para predecir (default: data/ml_dataset_accumulated.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta donde guardar predicciones (opcional)",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PREDICCIÓN - MEJOR MOMENTO PARA CORRER")
    print("=" * 80)
    print()
    
    # Load model
    if not args.model.exists():
        print(f"❌ Error: Modelo no encontrado en {args.model}")
        print("   Entrena primero: PYTHONPATH=src python -m runner_air_planner.ml.train")
        sys.exit(1)
    
    print(f"🤖 Cargando modelo: {args.model}")
    with open(args.model, "rb") as f:
        model: RunningSuitabilityModel = pickle.load(f)
    
    print("   ✓ Modelo cargado")
    
    # Load data
    if not args.dataset.exists():
        print(f"❌ Error: Dataset no encontrado en {args.dataset}")
        sys.exit(1)
    
    print(f"📊 Cargando datos: {args.dataset}")
    df = pd.read_csv(args.dataset)
    
    if "measurement_time" in df.columns:
        df["measurement_time"] = pd.to_datetime(df["measurement_time"])
    
    print(f"   ✓ {len(df)} registros")
    
    # Make predictions
    print()
    print("🔮 Generando predicciones...")
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)
    
    # Add predictions to dataframe
    df["prediction"] = predictions
    df["prob_good"] = probabilities["prob_good"]
    df["prob_not_good"] = probabilities["prob_not_good"]
    
    # Summary
    good_count = predictions.sum()
    total_count = len(predictions)
    
    print()
    print("=" * 80)
    print("RESULTADOS")
    print("=" * 80)
    print(f"✅ Buen momento para correr: {good_count} de {total_count} ({good_count/total_count:.1%})")
    print(f"❌ No recomendado: {total_count - good_count} de {total_count} ({(total_count-good_count)/total_count:.1%})")
    print()
    
    # Show best stations
    good_stations = df[df["prediction"] == 1].copy()
    if not good_stations.empty:
        print("🏃 MEJORES ESTACIONES PARA CORRER:")
        print("-" * 80)
        best = good_stations.nlargest(10, "prob_good")[
            ["station_name", "station_type", "prob_good", "air_quality_index", "weather_wind_speed_kmh"]
        ]
        for _, row in best.iterrows():
            print(f"  • {row['station_name']} ({row['station_type']})")
            print(f"    Probabilidad: {row['prob_good']:.1%} | AQI: {row['air_quality_index']:.1f} | Viento: {row['weather_wind_speed_kmh']:.1f} km/h")
        print()
    
    # Save if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"💾 Predicciones guardadas en: {args.output}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

