from __future__ import annotations

import csv
from pathlib import Path

from runner_air_planner.storage.local import LocalDataStorage


def test_write_json(tmp_path: Path) -> None:
    storage = LocalDataStorage(root=tmp_path)
    payload = {"foo": "bar"}

    written_path = storage.write_json(payload, prefix="test")

    assert written_path.exists()
    assert written_path.suffix == ".json"
    assert written_path.read_text(encoding="utf-8").strip().startswith("{")


def test_write_csv(tmp_path: Path) -> None:
    storage = LocalDataStorage(root=tmp_path)
    rows = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]

    written_path = storage.write_csv(rows, prefix="df")

    assert written_path.exists()
    with written_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        loaded = list(reader)
    assert loaded[0]["a"] == "1"
    assert loaded[1]["b"] == "4"
