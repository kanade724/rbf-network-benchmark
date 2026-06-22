"""Dataset loading and preprocessing helpers."""

import gzip
from pathlib import Path

import numpy as np
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.model_selection import train_test_split


SKLEARN_LOADERS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
}


def load_pen_based_digits(data_dir):
    """Load Pendigits from local UCI/OpenML cache, downloading through OpenML if needed."""
    data_dir = Path(data_dir)
    candidates = [
        (data_dir / "pendigits" / "pendigits.tra", data_dir / "pendigits" / "pendigits.tes"),
        (Path("pendigits.tra"), Path("pendigits.tes")),
    ]
    for train_path, test_path in candidates:
        if train_path.exists() and test_path.exists():
            data = np.vstack((np.loadtxt(train_path, delimiter=","), np.loadtxt(test_path, delimiter=",")))
            return data[:, :-1], data[:, -1].astype(int), "UCI local files"

    try:
        dataset = fetch_openml(data_id=32, as_frame=False, parser="auto", data_home=data_dir / "openml_cache")
        return dataset.data.astype(float), dataset.target.astype(int), "OpenML pendigits data_id=32"
    except ImportError:
        cache_path = data_dir / "openml_cache/openml/openml.org/data/v1/download/32/pendigits.arff.gz"
        if not cache_path.exists():
            raise
        with gzip.open(cache_path, "rt") as cache_file:
            for line in cache_file:
                if line.strip().lower() == "@data":
                    break
            data = np.loadtxt(cache_file, delimiter=",")
        return data[:, :-1], data[:, -1].astype(int), "OpenML pendigits cache"


def load_dataset(name, data_dir, random_state, train_size=None, test_size=0.2):
    """Return train/test arrays, class labels, and a source description."""
    if name in SKLEARN_LOADERS:
        dataset = SKLEARN_LOADERS[name]()
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.data, dataset.target, test_size=test_size, random_state=random_state, stratify=dataset.target
        )
        return X_train, X_test, y_train, y_test, dataset.target_names, f"scikit-learn {name}"

    if name == "pendigits":
        X, y, source = load_pen_based_digits(data_dir)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test, np.unique(y), source

    if name == "fashion_mnist":
        from torchvision import datasets

        train = datasets.FashionMNIST(root=data_dir, train=True, download=True)
        test = datasets.FashionMNIST(root=data_dir, train=False, download=True)
        X_train, y_train = train.data.numpy().reshape(-1, 784) / 255.0, train.targets.numpy()
        X_test, y_test = test.data.numpy().reshape(-1, 784) / 255.0, test.targets.numpy()
        if train_size:
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train, train_size=train_size, stratify=y_train, random_state=random_state
            )
        return X_train, X_test, y_train, y_test, np.unique(y_train), "torchvision FashionMNIST"

    raise ValueError(f"Unsupported dataset: {name}")
