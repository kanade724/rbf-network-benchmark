from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris, load_wine
from sklearn.preprocessing import LabelEncoder


CLASSICAL_LOADERS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}


def read_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_workspace_path(config: dict[str, Any], key: str) -> Path:
    paths = config.get("paths", {})
    root_value = Path(paths.get("workspace_root", "."))
    workspace_root = root_value if root_value.is_absolute() else (project_root() / root_value).resolve()

    value = Path(paths[key])
    return value if value.is_absolute() else (workspace_root / value).resolve()


def save_npz(
    output_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    target_names: list[str],
    feature_names: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x=x.astype(np.float64),
        y=y.astype(np.int64),
        target_names=np.array(target_names, dtype=str),
        feature_names=np.array(feature_names, dtype=str),
    )


def download_classical_dataset(name: str, data_dir: Path) -> Path:
    dataset = CLASSICAL_LOADERS[name]()
    output_path = data_dir / f"{name}.npz"
    save_npz(
        output_path,
        dataset.data,
        dataset.target,
        list(dataset.target_names),
        list(dataset.feature_names),
    )
    return output_path


def download_fashion_mnist(config: dict[str, Any], data_dir: Path) -> Path:
    settings = config.get("download", {}).get("fashion_mnist", {})
    cache_dir = data_dir / "openml_cache"
    dataset = fetch_openml(
        settings.get("openml_name", "Fashion-MNIST"),
        version=int(settings.get("version", 1)),
        as_frame=False,
        parser="auto",
        data_home=cache_dir,
    )

    x = dataset.data.astype(np.float64) / 255.0
    y = LabelEncoder().fit_transform(dataset.target)
    target_names = [str(i) for i in range(10)]
    feature_names = [f"pixel_{i}" for i in range(x.shape[1])]

    output_path = data_dir / "fashion_mnist.npz"
    save_npz(output_path, x, y, target_names, feature_names)
    return output_path


def download_dataset(name: str, config: dict[str, Any], data_dir: Path) -> Path:
    if name in CLASSICAL_LOADERS:
        return download_classical_dataset(name, data_dir)
    if name == "fashion_mnist":
        return download_fashion_mnist(config, data_dir)
    raise ValueError(f"Unknown dataset: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download benchmark datasets into the workspace data folder.")
    parser.add_argument("--config", type=Path, default=project_root() / "config.yaml")
    parser.add_argument("--dataset", choices=["iris", "wine", "breast_cancer", "fashion_mnist"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    data_dir = resolve_workspace_path(config, "data_dir")
    datasets = [args.dataset] if args.dataset else config.get("download", {}).get("datasets", [])

    if not datasets:
        raise ValueError("No datasets selected. Set download.datasets in config.yaml or pass --dataset.")

    for dataset in datasets:
        output_path = download_dataset(dataset, config, data_dir)
        print(f"saved {dataset}: {output_path}")


if __name__ == "__main__":
    main()
