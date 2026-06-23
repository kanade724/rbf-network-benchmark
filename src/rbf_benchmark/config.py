from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml


DATASETS = (
    "iris",
    "wine",
    "breast_cancer",
    "fashion_mnist",
    "optical_digits",
    "pen_digits",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_config(path: Path | None = None) -> dict[str, Any]:
    path = path or project_root() / "config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def workspace_path(config: dict[str, Any], key: str) -> Path:
    paths = config["paths"]
    root = Path(paths.get("workspace_root", "."))
    root = root if root.is_absolute() else (project_root() / root).resolve()
    value = Path(paths[key])
    return value if value.is_absolute() else root / value


def merged_dataset_config(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    result = deepcopy(config)
    for key, value in config.get("dataset_overrides", {}).get(dataset, {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key].update(value)
        else:
            result[key] = value
    return result


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("runtime", {}).get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("runtime.device must be one of: auto, cpu, cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device is cuda, but CUDA is unavailable.")
    return torch.device(requested)
