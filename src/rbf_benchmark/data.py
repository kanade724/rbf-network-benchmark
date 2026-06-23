from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .config import DATASETS, resolve_device


CLASSICAL_LOADERS = {"iris": load_iris, "wine": load_wine, "breast_cancer": load_breast_cancer}
OPENML_DATASETS = {
    "fashion_mnist": {"data_id": 40996, "name": "Fashion-MNIST"},
    "optical_digits": {"data_id": 28, "name": "Optical Recognition of Handwritten Digits"},
    "pen_digits": {"data_id": 32, "name": "Pen-Based Recognition of Handwritten Digits"},
}


def dataset_dir(data_dir: Path, name: str) -> Path:
    """Return the directory that contains every artifact for one dataset."""
    return data_dir / name


def raw_dataset_path(download_dir: Path, name: str) -> Path:
    return dataset_dir(download_dir, name) / "raw.npz"


def _save_raw_dataset(path: Path, x: np.ndarray, y: np.ndarray, target_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x, y=y.astype(np.int64), target_names=np.asarray(target_names, dtype=str))


def fetch_dataset(name: str, download_dir: Path, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load raw data from download/<dataset>, downloading only when absent."""
    root = dataset_dir(download_dir, name)
    raw_cache = raw_dataset_path(download_dir, name)
    if raw_cache.exists():
        cached = np.load(raw_cache, allow_pickle=False)
        return cached["x"], cached["y"].astype(np.int64), cached["target_names"].astype(str).tolist()
    if name in CLASSICAL_LOADERS:
        data = CLASSICAL_LOADERS[name]()
        target_names = [str(item) for item in data.target_names]
        _save_raw_dataset(raw_cache, data.data, data.target, target_names)
        return data.data, data.target, target_names
    if name not in OPENML_DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Expected one of {DATASETS}.")
    settings = config.get("download", {}).get(name, {})
    source = OPENML_DATASETS[name]
    data = fetch_openml(data_id=int(settings.get("data_id", source["data_id"])), as_frame=False, parser="auto",
                         data_home=root / "openml_cache")
    encoder = LabelEncoder().fit(data.target)
    y = encoder.transform(data.target)
    target_names = encoder.classes_.astype(str).tolist()
    _save_raw_dataset(raw_cache, data.data, y, target_names)
    return data.data, y, target_names


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


def _quantize_for_hardware(
    train: torch.Tensor,
    test: torch.Tensor,
    levels: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit per-feature bounds on train, then map both splits to hardware states."""
    if levels < 2:
        raise ValueError("experiment.hardware_quantization.levels must be at least 2.")
    lower = train.amin(dim=0)
    span = (train.amax(dim=0) - lower).clamp_min(1e-12)
    state_table = torch.linspace(0.0, 1.0, steps=levels, dtype=train.dtype, device=train.device)

    def quantize(values: torch.Tensor) -> torch.Tensor:
        normalized = ((values - lower) / span).clamp(0.0, 1.0)
        indices = torch.round(normalized * (levels - 1)).to(torch.long)
        return state_table[indices]

    return quantize(train), quantize(test), state_table


def preprocess_split(
    x: np.ndarray,
    y: np.ndarray,
    name: str,
    config: dict[str, Any],
    status_callback: Callable[[str, bool], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    def report(stage: str, completed: bool = False) -> None:
        if status_callback is not None:
            status_callback(stage, completed)

    experiment = config["experiment"]
    report("splitting train/test")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=float(experiment["test_size"]), stratify=y, random_state=int(experiment["random_state"])
    )
    report("split complete", completed=True)
    device = resolve_device(config)
    train = torch.as_tensor(x_train, dtype=torch.float32, device=device).flatten(1)
    test = torch.as_tensor(x_test, dtype=torch.float32, device=device).flatten(1)
    normalize = name == "fashion_mnist" and bool(experiment.get("fashion_mnist_pipeline", {}).get("normalize_pixels", True))
    report("normalizing and standardizing")
    if normalize:
        train, test = train / 255.0, test / 255.0
    standardize = bool(experiment.get("standardize", True))
    if standardize:
        train, test = _standardize(train, test)
    report("normalization/standardization complete", completed=True)
    pca_components = None
    if name == "fashion_mnist":
        pca_components = experiment.get("fashion_mnist_pipeline", {}).get("pca_components", 100)
        if pca_components is not None:
            report(f"PCA dimensionality reduction to {pca_components} features")
            train, test = _pca(train, test, int(pca_components), int(experiment["random_state"]))
    report("PCA complete" if pca_components is not None else "PCA skipped", completed=True)
    hardware_levels = int(experiment.get("hardware_quantization", {}).get("levels", 200))
    report(f"quantizing to {hardware_levels} hardware levels")
    train, test, state_table = _quantize_for_hardware(train, test, hardware_levels)
    report("hardware quantization complete", completed=True)
    metadata = {
        "device": str(device),
        "normalize_pixels": normalize,
        "standardize": standardize,
        "pca_components": pca_components,
        "features": int(train.shape[1]),
        "hardware_quantization": {
            "levels": hardware_levels,
            "range": [0.0, 1.0],
            "fit_split": "train",
            "test_out_of_range": "clip_to_train_range",
        },
    }
    return (train.cpu().numpy(), test.cpu().numpy(), y_train.astype(np.int64), y_test.astype(np.int64),
            metadata, state_table.cpu().numpy())


def _write_csv(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join([*(f"feature_{i}" for i in range(x.shape[1])), "label"])
    np.savetxt(path, np.column_stack((x, y)), delimiter=",", header=header, comments="", fmt="%.8g")


def write_preprocessed_dataset(
    name: str,
    download_dir: Path,
    data_dir: Path,
    config: dict[str, Any],
) -> tuple[Path, Path]:
    cached = raw_dataset_path(download_dir, name).exists()
    with tqdm(total=6, desc=f"{name}: {'using raw cache' if cached else 'downloading raw data'}", unit="stage", leave=False) as progress:
        x, y, target_names = fetch_dataset(name, download_dir, config)
        progress.set_description(f"{name}: raw {'cache loaded' if cached else 'download complete'}")
        progress.update()

        def update_stage(stage: str, completed: bool) -> None:
            progress.set_description(f"{name}: {stage}")
            if completed:
                progress.update()

        x_train, x_test, y_train, y_test, metadata, state_table = preprocess_split(
            x, y, name, config, status_callback=update_stage
        )
        root = dataset_dir(data_dir, name)
        train_path, test_path = root / "train.csv", root / "test.csv"
        progress.set_description(f"{name}: writing CSV and metadata")
        _write_csv(train_path, x_train, y_train)
        _write_csv(test_path, x_test, y_test)
        np.savetxt(root / "differential_levels.csv", state_table, delimiter=",",
                   header="differential_level", comments="", fmt="%.8g")
        (root / "metadata.json").write_text(json.dumps({"dataset": name, "target_names": target_names, **metadata}, indent=2), encoding="utf-8")
        progress.update()
        progress.set_description(f"{name}: complete")
    return train_path, test_path


def load_preprocessed_dataset(name: str, data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    root = dataset_dir(data_dir, name)
    train_path, test_path = root / "train.csv", root / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing processed CSV files for {name}. Run: python src/prepare_datasets.py --dataset {name}")
    train, test = np.loadtxt(train_path, delimiter=",", skiprows=1, dtype=np.float32), np.loadtxt(test_path, delimiter=",", skiprows=1, dtype=np.float32)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    return train[:, :-1], train[:, -1].astype(np.int64), test[:, :-1], test[:, -1].astype(np.int64), metadata
