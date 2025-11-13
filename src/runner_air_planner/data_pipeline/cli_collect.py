"""CLI tool to collect and prepare ML-ready dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Configurar encoding UTF-8 para Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from .accumulate_data import accumulate_ml_dataset, save_accumulated_dataset
from .data_collector import DataCollector


def main() -> None:
    """Collect data from both APIs and create ML-ready dataset."""
    parser = argparse.ArgumentParser(
        description="Recopilar datos de calidad del aire y meteorología para el modelo de ML."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ml_dataset.csv"),
        help="Ruta donde guardar el dataset para ML (default: data/ml_dataset.csv)",
    )
    parser.add_argument(
        "--air-quality-csv",
        type=Path,
        default=None,
        help="Usar CSV existente de calidad del aire en lugar de descargar",
    )
    parser.add_argument(
        "--force-weather-refresh",
        action="store_true",
        help="Forzar actualización de datos meteorológicos (ignorar caché)",
    )
    parser.add_argument(
        "--accumulate",
        action="store_true",
        help="Acumular con datos históricos existentes",
    )
    parser.add_argument(
        "--accumulated-path",
        type=Path,
        default=Path("data/ml_dataset_accumulated.csv"),
        help="Ruta del archivo acumulado (default: data/ml_dataset_accumulated.csv)",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=30,
        help="Máximo de días de historial a mantener (default: 30)",
    )
    parser.add_argument(
        "--min-records",
        type=int,
        default=1000,
        help="Mínimo de registros requeridos (default: 1000)",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("RECOPILACIÓN DE DATOS PARA MODELO DE ML")
    print("=" * 80)
    print()
    
    collector = DataCollector()
    
    # Collect air quality data
    print("📊 Recopilando datos de calidad del aire...")
    try:
        if args.air_quality_csv and args.air_quality_csv.exists():
            import pandas as pd
            air_quality_df = pd.read_csv(args.air_quality_csv)
            air_quality_df["measurement_time"] = pd.to_datetime(air_quality_df["measurement_time"])
            print(f"   ✓ Cargado desde: {args.air_quality_csv}")
        else:
            air_quality_df = collector.collect_air_quality_data()
            print(f"   ✓ Descargadas {len(air_quality_df)} mediciones")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        sys.exit(1)
    
    # Collect weather data
    print("🌤️  Recopilando datos meteorológicos...")
    try:
        weather_report = collector.get_weather_data(force_refresh=args.force_weather_refresh)
        print(f"   ✓ Temperatura: {weather_report.temperature_c:.1f}°C")
        print(f"   ✓ Humedad: {weather_report.relative_humidity:.0f}%")
        print(f"   ✓ Viento: {weather_report.wind_speed_kmh:.1f} km/h")
        print(f"   ✓ Condiciones: {weather_report.weather_description}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        sys.exit(1)
    
    # Create ML dataset
    print()
    print("🔧 Creando dataset para modelo de ML...")
    try:
        # Create new dataset
        ml_df = collector.create_ml_dataset(
            air_quality_df=air_quality_df,
            weather_report=weather_report,
            min_records=args.min_records,
        )
        
        # Accumulate if requested
        if args.accumulate:
            print("   📦 Acumulando con datos históricos...")
            ml_df = accumulate_ml_dataset(
                ml_df,
                args.accumulated_path,
                max_days=args.max_days,
            )
            output_path = save_accumulated_dataset(ml_df, args.accumulated_path)
            print(f"   ✓ Datos acumulados en: {output_path}")
        else:
            # Save single dataset
            output_path = collector.save_ml_dataset(
                args.output,
                air_quality_df=air_quality_df,
                weather_report=weather_report,
            )
        
        # Show summary
        print(f"   ✓ Dataset: {output_path}")
        print(f"   ✓ Estaciones: {ml_df['station_code'].nunique()}")
        print(f"   ✓ Features: {len(ml_df.columns)}")
        print(f"   ✓ Filas: {len(ml_df)}")
        
        # Check minimum records
        if len(ml_df) < args.min_records:
            print()
            print(f"   ⚠️  ADVERTENCIA: Solo {len(ml_df)} registros (mínimo requerido: {args.min_records})")
            print(f"   💡 Usa --accumulate para acumular datos de múltiples ejecuciones")
        print()
        print("📋 Columnas principales:")
        key_columns = [
            "station_code", "station_name", "station_type",
            "no2", "o3", "pm10", "pm25",
            "weather_wind_speed_kmh", "weather_temperature_c",
            "hour", "is_rush_hour",
            "air_quality_index", "running_suitability_preliminary",
        ]
        for col in key_columns:
            if col in ml_df.columns:
                print(f"   - {col}")
        
        print()
        print("=" * 80)
        print("✅ DATASET LISTO PARA EL MODELO DE ML")
        print("=" * 80)
        print(f"\nArchivo guardado en: {output_path}")
        print("\nPuedes usar este dataset para entrenar tu modelo de ML.")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

