"""Script to train the ML model."""

from __future__ import annotations

import argparse
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
    """Train the ML model."""
    parser = argparse.ArgumentParser(
        description="Entrenar modelo de ML para Runner Air Planner"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/ml_dataset_accumulated.csv"),
        help="Ruta al dataset para entrenar (default: data/ml_dataset_accumulated.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODELS_DIR / "running_model.pkl",
        help="Ruta donde guardar el modelo entrenado",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1000,
        help="Mínimo de muestras requeridas para entrenar (default: 1000)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("ENTRENAMIENTO DEL MODELO DE ML")
    print("=" * 80)
    print()

    # Load dataset
    if not args.dataset.exists():
        print(f"❌ Error: Dataset no encontrado en {args.dataset}")
        print(
            "   Ejecuta primero: PYTHONPATH=src python -m runner_air_planner.data_pipeline.cli_collect --accumulate"
        )
        sys.exit(1)

    print(f"📊 Cargando dataset: {args.dataset}")
    df = pd.read_csv(args.dataset)

    if "measurement_time" in df.columns:
        df["measurement_time"] = pd.to_datetime(df["measurement_time"])

    print(f"   ✓ {len(df)} registros cargados")
    print(f"   ✓ {len(df.columns)} features")

    # Check minimum samples
    if len(df) < args.min_samples:
        print()
        print(
            f"⚠️  ADVERTENCIA: Solo {len(df)} registros (mínimo recomendado: {args.min_samples})"
        )
        print("   El modelo puede tener bajo rendimiento con pocos datos.")
        response = input("   ¿Continuar de todas formas? (s/n): ")
        if response.lower() != "s":
            print("   Entrenamiento cancelado.")
            sys.exit(0)

    # Create and train model
    print()
    print("🤖 Entrenando modelo...")
    model = RunningSuitabilityModel(model_path=args.output)

    try:
        metrics = model.train(df)

        print()
        print("✅ Modelo entrenado exitosamente!")
        print(f"   Precisión en entrenamiento: {metrics['train_accuracy']:.2%}")
        print(f"   Precisión en test: {metrics['test_accuracy']:.2%}")
        print(f"   Muestras: {metrics['n_samples']}")
        print(f"   Features: {metrics['n_features']}")

        # Save model
        import pickle

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "wb") as f:
            pickle.dump(model, f)

        print()
        print(f"💾 Modelo guardado en: {args.output}")
        print()
        print("=" * 80)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
