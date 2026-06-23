from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from .config import merged_dataset_config, resolve_device, workspace_path
from .data import load_preprocessed_dataset
from .model import RBFRidgeClassifier


def run_dataset(dataset: str, config: dict[str, Any], run_dir: Path | None = None) -> dict[str, Any]:
    config = merged_dataset_config(config, dataset)
    progress = tqdm(total=4, desc=f"Training {dataset}", unit="stage", leave=False)
    try:
        x_train, y_train, x_test, y_test, metadata = load_preprocessed_dataset(dataset, workspace_path(config, "data_dir"))
        device = resolve_device(config)
        progress.set_postfix_str("data loaded")
        progress.update()
        rbf = config["rbf"]
        model = RBFRidgeClassifier(n_centers=int(rbf["n_centers"]), center_init=rbf.get("center_init", "kmeans"),
            sigma=rbf.get("sigma", "auto"), sigma_scale=float(rbf.get("sigma_scale", 1)), ridge_alpha=float(rbf["ridge_alpha"]),
            random_state=int(config["experiment"]["random_state"]), kmeans_max_iter=int(rbf.get("kmeans_max_iter", 300)), device=device)
        train_t, test_t = torch.as_tensor(x_train, device=device), torch.as_tensor(x_test, device=device)
        progress.set_postfix_str("fitting RBF ridge model")
        model.fit(train_t, torch.as_tensor(y_train, device=device))
        progress.update()
        progress.set_postfix_str("predicting")
        pred = model.predict(test_t).cpu().numpy()
        train_pred = model.predict(train_t).cpu().numpy()
        progress.update()
        result = {"dataset": dataset, "device": str(device), "train_samples": len(y_train), "test_samples": len(y_test),
            "features": x_train.shape[1], "classes": len(metadata["target_names"]), "preprocessing": metadata,
            "rbf": {"n_centers": model.n_centers, "center_init": model.center_init, "sigma": model.sigma_, "classifier": "ridge_regression", "ridge_alpha": model.ridge_alpha},
            "train_accuracy": float(accuracy_score(y_train, train_pred)), "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro"))}
        if run_dir is None:
            run_dir = workspace_path(config, "output_dir") / datetime.now().strftime("rbf_ridge_%Y%m%d_%H%M%S_%f")
        output = run_dir / dataset
        output.mkdir(parents=True, exist_ok=True)
        progress.set_postfix_str("writing results")
        (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        np.savetxt(output / "predictions.csv", np.column_stack((y_test, pred)), delimiter=",", header="label,prediction", comments="", fmt="%d")
        np.savetxt(output / "confusion_matrix.csv", confusion_matrix(y_test, pred), delimiter=",", fmt="%d")
        (output / "classification_report.txt").write_text(classification_report(y_test, pred, target_names=metadata["target_names"], zero_division=0), encoding="utf-8")
        progress.update()
        tqdm.write(f"{dataset}: device={device}, accuracy={result['accuracy']:.4f}, macro_f1={result['macro_f1']:.4f}")
        return result
    finally:
        progress.close()
