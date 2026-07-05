"""YAML configuration loading, deep-merging, and validation.

All scripts load configuration through :func:`load_config`, which merges
``configs/default.yaml`` with any number of additional YAML files (later
files win on key conflicts) and, optionally, a dotted-key override dict
parsed from the CLI (``--set model.adapters.bottleneck_dim=64``).
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


class ConfigError(ValueError):
    """Raised when a configuration file is missing required keys."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a top-level mapping")
    return data


def set_by_dotted_key(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``config[a][b][c] = value`` given ``dotted_key == "a.b.c"``."""
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _coerce_scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if "." in raw or "e" in lowered:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def parse_cli_overrides(overrides: list[str] | None) -> dict[str, Any]:
    """Parse ``["a.b.c=1", "x.y=foo"]`` CLI overrides into a nested dict."""
    result: dict[str, Any] = {}
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"Invalid override '{item}', expected key=value")
        key, raw_value = item.split("=", 1)
        set_by_dotted_key(result, key.strip(), _coerce_scalar(raw_value.strip()))
    return result


def load_config(
    *extra_files: str | Path,
    base: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and deep-merge one or more YAML config files.

    Parameters
    ----------
    extra_files:
        Additional YAML files merged on top of ``base``, in order.
    base:
        Base config file, defaults to ``configs/default.yaml``.
    overrides:
        A nested dict merged on top of everything else (highest priority),
        typically produced by :func:`parse_cli_overrides`.
    """
    base_path = Path(base) if base is not None else CONFIG_DIR / "default.yaml"
    merged = _load_yaml(base_path)
    for extra in extra_files:
        merged = _deep_merge(merged, _load_yaml(extra))
    if overrides:
        merged = _deep_merge(merged, overrides)
    return merged


def load_full_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convenience loader that merges all standard config files.

    Merges default -> data -> model -> training -> api, which is the
    combination most training/eval scripts need.
    """
    return load_config(
        CONFIG_DIR / "data.yaml",
        CONFIG_DIR / "model.yaml",
        CONFIG_DIR / "training.yaml",
        CONFIG_DIR / "api.yaml",
        overrides=overrides,
    )


def resolve_path(relative_or_absolute: str | os.PathLike) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(relative_or_absolute)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def resolve_device(device_setting: str) -> str:
    """Resolve the "auto" device setting to "cuda" or "cpu"."""
    if device_setting != "auto":
        return device_setting
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
