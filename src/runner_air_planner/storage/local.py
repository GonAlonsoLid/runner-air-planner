"""Helpers to persist raw datasets on the local filesystem."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class LocalDataStorage:
    """Utility class that writes API payloads to timestamped files."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, data: object, prefix: str) -> Path:
        """Persist the provided mapping or list as a JSON document."""

        self.ensure_root()
        file_path = self.root / f"{prefix}_{self._timestamp_suffix()}.json"
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        return file_path

    def write_csv(
        self,
        rows: Iterable[Mapping[str, object]],
        prefix: str,
        fieldnames: Sequence[str] | None = None,
    ) -> Path:
        """Persist a series of dictionaries as a CSV file."""

        rows = list(rows)
        if not rows:
            raise ValueError("No rows provided to write_csv")

        self.ensure_root()
        file_path = self.root / f"{prefix}_{self._timestamp_suffix()}.csv"
        headers = list(fieldnames) if fieldnames else list(rows[0].keys())
        with file_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in headers})
        return file_path

    @staticmethod
    def _timestamp_suffix() -> str:
        return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


__all__ = ["LocalDataStorage"]
