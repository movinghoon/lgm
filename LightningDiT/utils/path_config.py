from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).resolve().parents[1] / "path.yaml"


def load_path_config(config_path: str | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else CONFIG_PATH
    if not path.exists():
        return {}
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _get(config: dict[str, Any], section: str, key: str) -> Any:
    value = config.get(section, {})
    if not isinstance(value, dict):
        return None
    return value.get(key)


def path_value(section: str, key: str, default: Any = None, config_path: str | None = None) -> Any:
    value = _get(load_path_config(config_path), section, key)
    return default if value is None else value
