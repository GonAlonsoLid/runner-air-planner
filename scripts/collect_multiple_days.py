"""Script to collect data over multiple days to reach minimum 1000 records."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from runner_air_planner.data_pipeline.accumulate_data import (
    load_accumulated_dataset,
    save_accumulated_dataset,
)
from runner_air_planner.data_pipeline.data_collector import DataCollector


def collect_until_minimum(
    min_records: int = 1000,
    accumulated_path: Path = Path("data/ml_dataset_accumulated.csv"),
    max_days: int = 30,
    interval_hours: float = 1,
    max_iterations: int = 100,
) -> None:
    """Collect data repeatedly until minimum records are reached.

    Args:
        min_records: Minimum number of records required
        accumulated_path: Path to accumulated dataset
        max_days: Maximum days of history to keep
        interval_hours: Hours to wait between collections
        max_iterations: Maximum number of collection attempts
    """
    print("=" * 80)
    print("RECOPILACIÓN CONTINUA DE DATOS")
    print("=" * 80)
    print(f"Objetivo: {min_records} registros mínimos")
    interval_str = (
        f"{int(interval_hours * 60)} minutos"
        if interval_hours < 1
        else f"{interval_hours} horas"
    )
    print(f"Intervalo: {interval_str}")
    print(f"Máximo de intentos: {max_iterations}")
    print()

    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*80}")
        print(f"ITERACIÓN {iteration}/{max_iterations}")
        print(f"Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")

        try:
            # Collect new data
            from runner_air_planner.data_pipeline.accumulate_data import (
                accumulate_ml_dataset,
            )

            collector = DataCollector()
            air_quality_df = collector.collect_air_quality_data()
            weather_report = collector.get_weather_data()

            ml_df = collector.create_ml_dataset(
                air_quality_df=air_quality_df,
                weather_report=weather_report,
                min_records=0,  # Don't warn during accumulation
            )

            # Accumulate
            accumulated_df = accumulate_ml_dataset(
                ml_df,
                accumulated_path,
                max_days=max_days,
            )

            # Save
            save_accumulated_dataset(accumulated_df, accumulated_path)

            current_records = len(accumulated_df)
            print(f"\n✅ Registros acumulados: {current_records}")

            if current_records >= min_records:
                print(f"\n🎉 OBJETIVO ALCANZADO: {current_records} >= {min_records}")
                print(f"Dataset guardado en: {accumulated_path}")
                break

            remaining = min_records - current_records
            print(f"⏳ Faltan {remaining} registros para alcanzar el mínimo")

            if iteration < max_iterations:
                wait_str = (
                    f"{int(interval_hours * 60)} minutos"
                    if interval_hours < 1
                    else f"{interval_hours} hora(s)"
                )
                print(f"\n⏸️  Esperando {wait_str} antes de la próxima recopilación...")
                time.sleep(interval_hours * 3600)

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrumpido por el usuario")
            break
        except Exception as e:
            print(f"\n❌ Error en iteración {iteration}: {e}")
            import traceback

            traceback.print_exc()
            if iteration < max_iterations:
                retry_str = (
                    f"{int(interval_hours * 60)} minutos"
                    if interval_hours < 1
                    else f"{interval_hours} hora(s)"
                )
                print(f"\n⏸️  Esperando {retry_str} antes de reintentar...")
                time.sleep(interval_hours * 3600)

    # Final summary
    if accumulated_path.exists():
        final_df = load_accumulated_dataset(accumulated_path)
        print(f"\n{'='*80}")
        print("RESUMEN FINAL")
        print(f"{'='*80}")
        print(f"Total registros: {len(final_df)}")
        print(f"Estaciones: {final_df['station_code'].nunique()}")
        print(
            f"Rango temporal: {final_df['measurement_time'].min()} a {final_df['measurement_time'].max()}"
        )
        print(f"Dataset: {accumulated_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recopilar datos durante múltiples días hasta alcanzar mínimo de registros."
    )
    parser.add_argument(
        "--min-records",
        type=int,
        default=1000,
        help="Mínimo de registros requeridos (default: 1000)",
    )
    parser.add_argument(
        "--accumulated-path",
        type=Path,
        default=Path("data/ml_dataset_accumulated.csv"),
        help="Ruta del archivo acumulado",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=30,
        help="Máximo de días de historial a mantener (default: 30)",
    )
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=1,
        help="Horas de espera entre recopilaciones (default: 1, usa 0.5 para media hora)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Máximo número de intentos (default: 100)",
    )

    args = parser.parse_args()

    collect_until_minimum(
        min_records=args.min_records,
        accumulated_path=args.accumulated_path,
        max_days=args.max_days,
        interval_hours=args.interval_hours,
        max_iterations=args.max_iterations,
    )


if __name__ == "__main__":
    main()
