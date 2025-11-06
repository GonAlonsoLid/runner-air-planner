"""Command line entry-point to download Madrid air quality data."""

from __future__ import annotations

import argparse
import json
import sys

from runner_air_planner.workflows.fetch_latest_air_quality import run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params",
        metavar="KEY=VALUE",
        nargs="*",
        help="Optional query parameters to send to the Madrid API (e.g. station=28079004)",
    )
    parser.add_argument(
        "--print",
        dest="should_print",
        action="store_true",
        help="Print the stored payload path to stdout (default behaviour).",
    )
    parser.add_argument(
        "--no-print",
        dest="should_print",
        action="store_false",
        help="Do not emit the stored payload path to stdout.",
    )
    parser.set_defaults(should_print=True)
    return parser.parse_args(argv)


def parse_params(pairs: list[str] | None) -> dict[str, str]:
    if not pairs:
        return {}
    parsed: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid parameter '{pair}'. Expected the form key=value")
        key, value = pair.split("=", 1)
        parsed[key] = value
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        params = parse_params(args.params)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    output_path = run(params=params)

    if args.should_print:
        print(json.dumps({"stored_path": output_path}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
