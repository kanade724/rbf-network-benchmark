from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import DATASETS, resolve_device


CLASSICAL_LOADERS = {"iris": load_iris, "wine": load_wine, "breast_cancer": load_breast_cancer}
OPENML_DATASETS = {
    "fashion_mnist": {"data_id": 40996, "name": "Fashion-MNIST"},
    "optical_digits": {"data_id": 28, "name": "Optical Recognition of Handwritten Digits"},
    "pen_digits": {"data_id": 32, "name": "Pen-Based Recognition of Handwritten Digits"},
}


def fetch_dataset(name: str, data_dir: Path, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    # Reuse a legacy/raw cache when present. Processed CSV files remain the only training input.
    raw_cache = data_dir / f"{name}.npz"
    if raw_cache.exists():
        cached = np.load(raw_cache, allow_pickle=False)
        return cached["x"], cached["y"].astype(np.int64), cached["target_names"].astype(str).tolist()
    if name in CLASSICAL_LOADERS:
        data = CLASSICAL_LOADERS[name]()
        return data.data, data.target, [str(item) for item in data.target_names]
    if name not in OPENML_DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Expected one of {DATASETS}.")
    settings = config.get("download", {}).get(name, {})
    source = OPENML_DATASETS[name]
    data = fetch_openml(data_id=int(settings.get("data_id", source["data_id"])), as_frame=False, parser="auto",
                         data_home=data_dir / "openml_cache")
    encoder = LabelEncoder().fit(data.target)
    return data.data, encoder.transform(data.target), encoder.classes_.astype(str).tolist()


def _standardize(train: torch.Tensor, test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False).clamp_min(1e-12)
    return (train - mean) / std, (test - mean) / std


def _pca(train: torch.Tensor, test: torch.Tensor, components: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 < components <= min(train.shape):
        raise ValueError("fashion_mnist_pipeline.pca_components must not exceed the training matrix rank.")
    torch.manual_seed(seed)
    # q oversampling makes the truncated PCA more stable while keeping GPU execution practical.
    _, _, vectors = torch.pca_lowrank(train, q=min(components + 10, min(train.shape)), center=False)
    basis = vectors[:, :components]
    return train @ basis, test @ basis


def preprocess_split(x: np.ndarray, y: np.ndarray, name: str, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    experiment = config["experiment"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=float(experiment["test_size"]), stratify=y, random_state=int(experiment["random_state"])
    )
    device = resolve_device(config)
    train = torch.as_tensor(x_train, dtype=torch.float32, device=device).flatten(1)
    test = torch.as_tensor(x_test, dtype=torch.float32, device=device).flatten(1)
    normalize = name == "fashion_mnist" and bool(experiment.get("fashion_mnist_pipeline", {}).get("normalize_pixels", True))
    if normalize:
        train, test = train / 255.0, test / 255.0
    standardize = bool(experiment.get("standardize", True))
    if standardize:
        train, test = _standardize(train, test)
    pca_components = None
    if name == "fashion_mnist":
        pca_components = experiment.get("fashion_mnist_pipeline", {}).get("pca_components", 100)
        if pca_components is not None:
            train, test = _pca(train, test, int(pca_components), int(experiment["random_state"]))
    metadata = {"device": str(device), "normalize_pixels": normalize, "standardize": standardize,
                "pca_components": pca_components, "features": int(train.shape[1])}
    return train.cpu().numpy(), test.cpu().numpy(), y_train.astype(np.int64), y_test.astype(np.int64), metadata


def _write_csv(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join([*(f"feature_{i}" for i in range(x.shape[1])), "label"])
    np.savetxt(path, np.column_stack((x, y)), delimiter=",", header=header, comments="", fmt="%.8g")


def write_preprocessed_dataset(name: str, data_dir: Path, config: dict[str, Any]) -> tuple[Path, Path]:
    x, y, target_names = fetch_dataset(name, data_dir, config)
    x_train, x_test, y_train, y_test, metadata = preprocess_split(x, y, name, config)
    train_path, test_path = data_dir / f"{name}_train.csv", data_dir / f"{name}_test.csv"
    _write_csv(train_path, x_train, y_train)
    _write_csv(test_path, x_test, y_test)
    (data_dir / f"{name}_metadata.json").write_text(json.dumps({"dataset": name, "target_names": target_names, **metadata}, indent=2), encoding="utf-8")
    return train_path, test_path


def load_preprocessed_dataset(name: str, data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    train_path, test_path = data_dir / f"{name}_train.csv", data_dir / f"{name}_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing processed CSV files for {name}. Run: python src/prepare_datasets.py --dataset {name}")
    train, test = np.loadtxt(train_path, delimiter=",", skiprows=1, dtype=np.float32), np.loadtxt(test_path, delimiter=",", skiprows=1, dtype=np.float32)
    metadata = json.loads((data_dir / f"{name}_metadata.json").read_text(encoding="utf-8"))
    return train[:, :-1], train[:, -1].astype(np.int64), test[:, :-1], test[:, -1].astype(np.int64), metadata
