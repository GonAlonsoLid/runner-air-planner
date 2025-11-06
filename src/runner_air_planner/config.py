"""Application configuration models and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Mapping

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for very old interpreters
    tomllib = None  # type: ignore[assignment]


@dataclass
class AirQualityConfig:
    """Settings required to connect to Madrid's open data API."""

    base_url: str = "https://datos.madrid.es/egob/catalogo/"
    dataset_id: str = "210227-0-calidad-aire-tiempo-real"
    response_format: str = "json"
    request_timeout_seconds: float = 30.0


@dataclass
class StorageConfig:
    """Settings describing where to persist downloaded assets."""

    raw_data_dir: Path = Path("data/raw")


@dataclass
class Settings:
    """Top level application settings object."""

    air_quality: AirQualityConfig = field(default_factory=AirQualityConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    def normalise(self) -> None:
        self.storage.raw_data_dir = self.storage.raw_data_dir.expanduser().resolve()


def _apply_mapping(target: Any, updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, Mapping):
            _apply_mapping(current, value)
        else:
            target_field = next((f for f in fields(type(target)) if f.name == key), None)
            if target_field is not None and target_field.type is Path:
                setattr(target, key, Path(value))
            else:
                setattr(target, key, value)


def _load_toml_settings(path: Path) -> Mapping[str, Any]:
    if tomllib is None or not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def _apply_env_overrides(settings: Settings, env: Mapping[str, str]) -> None:
    mapping: dict[str, tuple[str, str]] = {
        "RAP_AIR_BASE_URL": ("air_quality", "base_url"),
        "RAP_AIR_DATASET_ID": ("air_quality", "dataset_id"),
        "RAP_AIR_RESPONSE_FORMAT": ("air_quality", "response_format"),
        "RAP_AIR_REQUEST_TIMEOUT_SECONDS": ("air_quality", "request_timeout_seconds"),
        "RAP_STORAGE_RAW_DATA_DIR": ("storage", "raw_data_dir"),
    }
    for env_key, (section_name, attribute) in mapping.items():
        if env_key not in env:
            continue
        section = getattr(settings, section_name)
        raw_value = env[env_key]
        if attribute == "request_timeout_seconds":
            value: Any = float(raw_value)
        elif attribute.endswith("dir"):
            value = Path(raw_value)
        else:
            value = raw_value
        setattr(section, attribute, value)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""

    settings = Settings()

    project_root = Path(__file__).resolve().parents[2]
    toml_path = project_root / "configs" / "settings.toml"
    toml_overrides = _load_toml_settings(toml_path)
    if toml_overrides:
        _apply_mapping(settings, toml_overrides)

    env_values = dict(_load_env_file(project_root / ".env"))
    env_values.update(os.environ)
    _apply_env_overrides(settings, env_values)

    settings.normalise()
    return settings


__all__ = ["AirQualityConfig", "StorageConfig", "Settings", "get_settings"]
