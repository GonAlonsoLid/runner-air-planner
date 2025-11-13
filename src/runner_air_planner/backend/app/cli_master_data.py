"""CLI tool to query master data for stations and pollutants."""

from __future__ import annotations

import argparse
import sys

# Configurar encoding UTF-8 para Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from . import master_data


def list_pollutants() -> None:
    """List all available pollutants."""
    print("=" * 80)
    print("CONTAMINANTES (MAGNITUDES) DISPONIBLES")
    print("=" * 80)
    print()
    
    for code in sorted([k for k in master_data.POLLUTANT_MASTER.keys() if isinstance(k, int)]):
        info = master_data.POLLUTANT_MASTER[code]
        print(f"Código: {info['code']}")
        print(f"  Nombre: {info['name']} ({info['full_name']})")
        print(f"  Unidad: {info['unit']}")
        print(f"  Descripción: {info['description']}")
        print()


def list_stations() -> None:
    """List all available stations."""
    print("=" * 80)
    print("ESTACIONES DE CALIDAD DEL AIRE DISPONIBLES")
    print("=" * 80)
    print()
    
    for code in sorted([k for k in master_data.STATION_MASTER.keys() if isinstance(k, int)]):
        info = master_data.STATION_MASTER[code]
        print(f"Código: {info['code']}")
        print(f"  Nombre: {info['name']}")
        print(f"  Distrito: {info['district']}")
        print(f"  Tipo: {info['type']}")
        print(f"  Coordenadas: {info['latitude']}, {info['longitude']}")
        print()


def query_pollutant(code: str) -> None:
    """Query information about a specific pollutant."""
    # Intentar convertir a int si es posible
    try:
        code_int = int(code)
        info = master_data.get_pollutant_info(code_int)
    except ValueError:
        info = master_data.get_pollutant_info(code)
    
    if not info:
        print(f"❌ No se encontró información para el contaminante con código: {code}")
        print("\nCódigos disponibles:")
        for c in sorted([k for k in master_data.POLLUTANT_MASTER.keys() if isinstance(k, int)]):
            print(f"  - {c}")
        sys.exit(1)
    
    print("=" * 80)
    print(f"INFORMACIÓN DEL CONTAMINANTE: {code}")
    print("=" * 80)
    print()
    print(f"Código: {info['code']}")
    print(f"Nombre corto: {info['name']}")
    print(f"Nombre completo: {info['full_name']}")
    print(f"Nombre en inglés: {info['english_name']}")
    print(f"Unidad: {info['unit']}")
    print(f"Descripción: {info['description']}")
    print()


def query_station(code: str) -> None:
    """Query information about a specific station."""
    info = master_data.get_station_info(code)
    if not info:
        print(f"❌ No se encontró información para la estación con código: {code}")
        print("\nCódigos disponibles:")
        for c in sorted([k for k in master_data.STATION_MASTER.keys() if isinstance(k, int)]):
            print(f"  - {c}")
        sys.exit(1)
    
    print("=" * 80)
    print(f"INFORMACIÓN DE LA ESTACIÓN: {code}")
    print("=" * 80)
    print()
    print(f"Código: {info['code']}")
    print(f"Nombre: {info['name']}")
    print(f"Distrito: {info['district']}")
    print(f"Tipo: {info['type']}")
    print(f"Latitud: {info['latitude']}")
    print(f"Longitud: {info['longitude']}")
    print()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Consulta datos maestros de estaciones y contaminantes de calidad del aire de Madrid."
    )
    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")
    
    # Listar contaminantes
    subparsers.add_parser("pollutants", help="Listar todos los contaminantes disponibles")
    
    # Listar estaciones
    subparsers.add_parser("stations", help="Listar todas las estaciones disponibles")
    
    # Consultar contaminante
    pollutant_parser = subparsers.add_parser("pollutant", help="Consultar información de un contaminante")
    pollutant_parser.add_argument("code", help="Código del contaminante (ej: 1, 8, 'no2', 'pm10')")
    
    # Consultar estación
    station_parser = subparsers.add_parser("station", help="Consultar información de una estación")
    station_parser.add_argument("code", type=int, help="Código de la estación (ej: 11, 39, 57)")
    
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "pollutants":
        list_pollutants()
    elif args.command == "stations":
        list_stations()
    elif args.command == "pollutant":
        query_pollutant(args.code)
    elif args.command == "station":
        query_station(args.code)


if __name__ == "__main__":
    main()

