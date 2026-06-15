from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

default_matplotlib_dir = Path(__file__).resolve().parents[3] / "output" / ".matplotlib_cache"
default_matplotlib_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(default_matplotlib_dir))

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


@dataclass
class TorchRBFNetwork:
    n_centers: int
    center_init: str
    sigma: str | float
    sigma_scale: float
    epochs: int
    learning_rate: float
    l2_alpha: float
    random_state: int
    kmeans_max_iter: int
    device: torch.device
    batch_size: int
    eval_batch_size: int
    history_interval: int

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_eval: np.ndarray | None = None,
        y_eval: np.ndarray | None = None,
        progress_label: str = "training",
    ) -> "TorchRBFNetwork":
        if self.n_centers <= 0:
            raise ValueError("rbf.n_centers must be positive.")
        if self.n_centers > len(x):
            raise ValueError("rbf.n_centers cannot exceed the number of training samples.")
        if self.epochs <= 0:
            raise ValueError("rbf.epochs must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("rbf.learning_rate must be positive.")

        torch.manual_seed(self.random_state)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_state)

        self.classes_ = np.unique(y)
        centers = self._select_centers(x)
        self.sigma_ = self._resolve_sigma(centers)
        self.centers_ = torch.as_tensor(centers, dtype=torch.float32, device=self.device)

        n_classes = len(self.classes_)
        self.weights_ = torch.empty(self.n_centers + 1, n_classes, dtype=torch.float32, device=self.device)
        torch.nn.init.normal_(self.weights_, mean=0.0, std=0.01)
        self.weights_.requires_grad_(True)
        optimizer = torch.optim.SGD([self.weights_], lr=self.learning_rate)

        x_train_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        y_train_t = torch.as_tensor(np.searchsorted(self.classes_, y), dtype=torch.long, device=self.device)
        x_eval_t = torch.as_tensor(x_eval, dtype=torch.float32, device=self.device) if x_eval is not None else None
        y_eval_t = (
            torch.as_tensor(np.searchsorted(self.classes_, y_eval), dtype=torch.long, device=self.device)
            if y_eval is not None
            else None
        )

        self.history_ = [self._history_row(0, x_train_t, y_train_t, x_eval_t, y_eval_t)]
        progress = tqdm(
            range(1, self.epochs + 1),
            desc=f"{progress_label} ({self.device})",
            unit="epoch",
            mininterval=0.5,
            leave=False,
            disable=not sys.stderr.isatty(),
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.random_state)

        for epoch in progress:
            permutation = torch.randperm(len(x_train_t), generator=generator, device=self.device)
            for start in range(0, len(x_train_t), self.batch_size):
                indices = permutation[start : start + self.batch_size]
                phi = self._gaussian_rbf_features(x_train_t[indices])
                logits = phi @ self.weights_
                loss = F.cross_entropy(logits, y_train_t[indices])
                loss = loss + 0.5 * self.l2_alpha * torch.sum(self.weights_[1:] * self.weights_[1:])

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            if epoch % self.history_interval == 0 or epoch == self.epochs:
                row = self._history_row(epoch, x_train_t, y_train_t, x_eval_t, y_eval_t)
                self.history_.append(row)
                progress.set_postfix(
                    train_loss=f"{row['train_loss']:.4f}",
                    train_acc=f"{row['train_accuracy']:.4f}",
                )

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        predictions = []
        with torch.no_grad():
            for start in range(0, len(x_t), self.eval_batch_size):
                phi = self._gaussian_rbf_features(x_t[start : start + self.eval_batch_size])
                predictions.append(torch.argmax(phi @ self.weights_, dim=1).cpu().numpy())
        return self.classes_[np.concatenate(predictions)]

    def _select_centers(self, x: np.ndarray) -> np.ndarray:
        if self.center_init == "kmeans":
            model = KMeans(
                n_clusters=self.n_centers,
                n_init="auto",
                max_iter=self.kmeans_max_iter,
                random_state=self.random_state,
            )
            model.fit(x)
            return model.cluster_centers_.astype(np.float32)

        if self.center_init == "random":
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(len(x), size=self.n_centers, replace=False)
            return x[indices].astype(np.float32)

        raise ValueError("rbf.center_init must be 'kmeans' or 'random'.")

    def _resolve_sigma(self, centers: np.ndarray) -> float:
        if isinstance(self.sigma, (int, float)):
            if self.sigma <= 0:
                raise ValueError("rbf.sigma must be positive.")
            return float(self.sigma)

        if self.sigma != "auto":
            raise ValueError("rbf.sigma must be 'auto' or a positive number.")

        squared = pairwise_squared_distances_np(centers, centers)
        distances = np.sqrt(squared[np.triu_indices_from(squared, k=1)])
        distances = distances[distances > 0]
        if len(distances) == 0:
            return 1.0
        return float(np.median(distances) * self.sigma_scale)

    def _gaussian_rbf_features(self, x: torch.Tensor) -> torch.Tensor:
        squared_distances = torch.cdist(x, self.centers_).pow(2)
        hidden = torch.exp(-squared_distances / (2.0 * self.sigma_**2))
        return torch.cat([torch.ones((len(x), 1), dtype=torch.float32, device=self.device), hidden], dim=1)

    def _history_row(
        self,
        epoch: int,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_eval: torch.Tensor | None,
        y_eval: torch.Tensor | None,
    ) -> dict[str, float]:
        train_loss, train_accuracy = self._evaluate_tensor_split(x_train, y_train)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
        }
        if x_eval is not None and y_eval is not None:
            test_loss, test_accuracy = self._evaluate_tensor_split(x_eval, y_eval)
            row["test_loss"] = test_loss
            row["test_accuracy"] = test_accuracy
        return row

    def _evaluate_tensor_split(self, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
        total_loss = 0.0
        total_correct = 0
        total = 0
        with torch.no_grad():
            for start in range(0, len(x), self.eval_batch_size):
                x_batch = x[start : start + self.eval_batch_size]
                y_batch = y[start : start + self.eval_batch_size]
                phi = self._gaussian_rbf_features(x_batch)
                logits = phi @ self.weights_
                total_loss += F.cross_entropy(logits, y_batch, reduction="sum").item()
                total_correct += (torch.argmax(logits, dim=1) == y_batch).sum().item()
                total += len(y_batch)

            penalty = 0.5 * self.l2_alpha * torch.sum(self.weights_[1:] * self.weights_[1:]).item()
        return float(total_loss / total + penalty), float(total_correct / total)


def pairwise_squared_distances_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    a_norm = np.sum(a * a, axis=1, keepdims=True)
    b_norm = np.sum(b * b, axis=1, keepdims=True).T
    return np.maximum(a_norm + b_norm - 2.0 * a @ b.T, 0.0)


def read_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def dataset_config(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    overrides = config.get("dataset_overrides", {}).get(dataset, {})
    return deep_merge(config, overrides)


def resolve_workspace_path(config: dict[str, Any], key: str) -> Path:
    paths = config.get("paths", {})
    root_value = Path(paths.get("workspace_root", "."))
    workspace_root = root_value if root_value.is_absolute() else (project_root() / root_value).resolve()

    value = Path(paths[key])
    return value if value.is_absolute() else (workspace_root / value).resolve()


def load_saved_dataset(data_dir: Path, dataset: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    path = data_dir / f"{dataset}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}. Run src/download_datasets.py first.")

    data = np.load(path, allow_pickle=False)
    return data["x"], data["y"], data["target_names"].astype(str).tolist()


def standardize_split(
    x_train: np.ndarray,
    x_test: np.ndarray,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not enabled:
        return x_train.astype(np.float32), x_test.astype(np.float32)

    scaler = StandardScaler()
    return scaler.fit_transform(x_train).astype(np.float32), scaler.transform(x_test).astype(np.float32)


def prepare_fashion_mnist_features(
    x_train: np.ndarray,
    x_test: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pipeline = config["experiment"].get("fashion_mnist_pipeline", {})
    normalize_pixels = bool(pipeline.get("normalize_pixels", True))
    pca_components = pipeline.get("pca_components", 100)

    x_train = x_train.astype(np.float32).reshape(len(x_train), -1)
    x_test = x_test.astype(np.float32).reshape(len(x_test), -1)

    if normalize_pixels:
        x_train = x_train / 255.0
        x_test = x_test / 255.0

    x_train, x_test = standardize_split(x_train, x_test, enabled=True)

    if pca_components is not None:
        pca = PCA(
            n_components=int(pca_components),
            random_state=config["experiment"].get("random_state", 42),
        )
        x_train = pca.fit_transform(x_train).astype(np.float32)
        x_test = pca.transform(x_test).astype(np.float32)

    return x_train, x_test, {
        "normalize_pixels": normalize_pixels,
        "standardize": True,
        "pca_components": pca_components,
    }


def make_model(config: dict[str, Any], device: torch.device) -> TorchRBFNetwork:
    rbf = config["rbf"]
    gpu = config.get("gpu", {})
    return TorchRBFNetwork(
        n_centers=int(rbf["n_centers"]),
        center_init=rbf.get("center_init", "kmeans"),
        sigma=rbf.get("sigma", "auto"),
        sigma_scale=float(rbf.get("sigma_scale", 1.0)),
        epochs=int(rbf.get("epochs", 100)),
        learning_rate=float(rbf.get("learning_rate", 0.05)),
        l2_alpha=float(rbf.get("l2_alpha", 1e-3)),
        random_state=int(config["experiment"].get("random_state", 42)),
        kmeans_max_iter=int(rbf.get("kmeans_max_iter", 300)),
        device=device,
        batch_size=int(gpu.get("batch_size", 2048)),
        eval_batch_size=int(gpu.get("eval_batch_size", 4096)),
        history_interval=int(gpu.get("history_interval", 1)),
    )


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("gpu", {}).get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("gpu.device is 'cuda', but torch.cuda.is_available() is false.")
    return torch.device(requested)


def write_json(path: Path, content: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(content, file, indent=2)


def write_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_index", "y_true", "y_pred"])
        for index, (true_value, pred_value) in enumerate(zip(y_true, y_pred)):
            writer.writerow([index, int(true_value), int(pred_value)])


def write_confusion_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(matrix.astype(int).tolist())


def write_training_curves(path: Path, history: list[dict[str, float]]) -> None:
    fields = ["epoch", "train_loss", "test_loss", "train_accuracy", "test_accuracy"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_named_curve(path: Path, history: list[dict[str, float]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot_training_curves(dataset_dir: Path, history: list[dict[str, float]]) -> None:
    epochs = [int(row["epoch"]) for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="Train Loss")
    if "test_loss" in history[0]:
        plt.plot(epochs, [row["test_loss"] for row in history], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dataset_dir / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_accuracy"] for row in history], label="Train Accuracy")
    if "test_accuracy" in history[0]:
        plt.plot(epochs, [row["test_accuracy"] for row in history], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dataset_dir / "accuracy_curve.png", dpi=150)
    plt.close()


def plot_confusion_matrix(path: Path, matrix: np.ndarray, target_names: list[str]) -> None:
    fig_size = max(6, min(12, len(target_names) * 0.75))
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45, ha="right")
    plt.yticks(tick_marks, target_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            plt.text(col, row, int(matrix[row, col]), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def evaluate_dataset(config: dict[str, Any], dataset: str, run_dir: Path, device: torch.device) -> dict[str, Any]:
    config = dataset_config(config, dataset)
    data_dir = resolve_workspace_path(config, "data_dir")
    x, y, target_names = load_saved_dataset(data_dir, dataset)

    experiment = config["experiment"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=float(experiment["test_size"]),
        stratify=y,
        random_state=int(experiment["random_state"]),
    )

    if dataset == "fashion_mnist":
        x_train, x_test, preprocessing = prepare_fashion_mnist_features(x_train, x_test, config)
    else:
        standardize = bool(experiment.get("standardize", True))
        x_train, x_test = standardize_split(x_train, x_test, standardize)
        preprocessing = {
            "normalize_pixels": False,
            "standardize": standardize,
            "pca_components": None,
        }

    model = make_model(config, device)
    model.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, progress_label=dataset)
    y_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)

    dataset_dir = run_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    labels = np.arange(len(target_names))
    metrics = {
        "dataset": dataset,
        "device": str(model.device),
        "samples": int(len(x)),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "classes": int(len(target_names)),
        "preprocessing": preprocessing,
        "features": int(x_train.shape[1]),
        "rbf": {
            "n_centers": int(model.n_centers),
            "center_init": model.center_init,
            "output_layer": "softmax",
            "optimizer": "mini_batch_gradient_descent",
            "batch_size": int(model.batch_size),
            "epochs": int(model.epochs),
            "learning_rate": float(model.learning_rate),
            "sigma": float(model.sigma_),
            "l2_alpha": float(model.l2_alpha),
        },
        "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
        "train_macro_f1": float(f1_score(y_train, y_train_pred, average="macro")),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "final_train_loss": float(model.history_[-1]["train_loss"]),
        "final_test_loss": float(model.history_[-1].get("test_loss", 0.0)),
        "final_train_accuracy": float(model.history_[-1]["train_accuracy"]),
        "final_test_accuracy": float(model.history_[-1].get("test_accuracy", 0.0)),
    }

    write_json(dataset_dir / "metrics.json", metrics)
    report = classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0)
    (dataset_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    if experiment.get("save_predictions", True):
        write_predictions(dataset_dir / "predictions.csv", y_test, y_pred)

    write_training_curves(dataset_dir / "training_curves.csv", model.history_)
    write_named_curve(dataset_dir / "loss_curve.csv", model.history_, ["epoch", "train_loss", "test_loss"])
    write_named_curve(
        dataset_dir / "accuracy_curve.csv",
        model.history_,
        ["epoch", "train_accuracy", "test_accuracy"],
    )

    if experiment.get("save_confusion_matrix", True):
        matrix = confusion_matrix(y_test, y_pred, labels=labels)
        write_confusion_matrix(dataset_dir / "confusion_matrix.csv", matrix)
        if experiment.get("save_plots", True):
            plot_confusion_matrix(dataset_dir / "confusion_matrix.png", matrix, target_names)

    if experiment.get("save_plots", True):
        plot_training_curves(dataset_dir, model.history_)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Gaussian RBF networks with PyTorch.")
    parser.add_argument("--config", type=Path, default=project_root() / "config.yaml")
    parser.add_argument("--dataset", choices=["iris", "wine", "breast_cancer", "fashion_mnist"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    device = resolve_device(config)
    output_dir = resolve_workspace_path(config, "output_dir")
    run_dir = output_dir / datetime.now().strftime("rbf_gpu_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=True)

    datasets = [args.dataset] if args.dataset else config.get("experiment", {}).get("datasets", [])
    if not datasets:
        raise ValueError("No datasets selected. Set experiment.datasets in config.yaml or pass --dataset.")

    print(f"device: {device}")
    summary = [evaluate_dataset(config, dataset, run_dir, device) for dataset in datasets]
    write_json(run_dir / "summary.json", {"device": str(device), "results": summary})

    for item in summary:
        print(f"{item['dataset']:15s} accuracy={item['accuracy']:.4f} macro_f1={item['macro_f1']:.4f}")
    print(f"outputs: {run_dir}")


if __name__ == "__main__":
    main()
