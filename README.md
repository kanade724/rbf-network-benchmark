# rbf-network-benchmark

[English](README.md) | [中文](README_cn.md)

A configurable Gaussian RBF network benchmark for classification tasks. The project keeps dataset download, preprocessing, RBF feature generation, output-layer training, and result export in one pipeline so different datasets and parameter settings can be compared consistently.

## 1. Project Layout

The repository stores code and configuration only. Datasets and experiment outputs are written to the outer `SURF2026` workspace so generated files do not pollute the source tree.

```text
SURF2026/
  data/                               # downloaded .npz datasets
  output/                             # per-run experiment outputs
  src/
    rbf-network-benchmark/
      config.yaml                     # global config and per-dataset overrides
      requirements.txt
      README.md
      README_cn.md
      src/
        download_datasets.py          # download and save datasets
        train_evaluate.py             # NumPy / scikit-learn train + eval entry
        train_evaluate_gpu.py         # PyTorch train + eval entry
```

The default path settings come from `config.yaml`:

```yaml
paths:
  workspace_root: ../..
  data_dir: data
  output_dir: output
```

Because the repository lives under `SURF2026/src/rbf-network-benchmark`, `../..` resolves to the `SURF2026` root.

## 2. Pipeline Overview

The codebase is organized into 4 layers:

1. `download_datasets.py`
   Downloads classical datasets and OpenML datasets, then saves everything as `.npz`.
2. `train_evaluate.py`
   Handles CPU training, evaluation, plotting, and metric export.
3. `train_evaluate_gpu.py`
   Keeps the same input/output contract while using PyTorch for faster RBF feature computation and softmax training.
4. `config.yaml`
   Controls dataset selection, preprocessing, RBF parameters, and per-dataset overrides.

The dataset flow in `train_evaluate.py` is:

```text
load .npz data
-> train_test_split(stratify=y)
-> preprocessing
-> build RBFNetwork
-> train output layer
-> predict test split
-> save metrics / report / curves / confusion matrix
```

## 3. Data Types and Storage Format

### 3.1 Supported datasets

The current code supports 6 datasets:

- `iris`
- `wine`
- `breast_cancer`
- `fashion_mnist`
- `optdigits`
- `pendigits`

Source split:

- `iris`, `wine`, and `breast_cancer` come from `sklearn.datasets`.
- `fashion_mnist`, `optdigits`, and `pendigits` are downloaded from OpenML.

### 3.2 Saved dataset format

Every dataset is stored as `SURF2026/data/<dataset>.npz` with a fixed schema:

- `x`: feature matrix, `float64`
- `y`: labels, `int64`
- `target_names`: class names
- `feature_names`: feature names

This gives the training code one uniform interface regardless of the original data source.

### 3.3 Current dataset shapes

The following shapes come from the current files in `SURF2026/data`:

| Dataset | `x` shape | `y` shape | Raw feature dimension | Classes | Notes |
|---|---:|---:|---:|---:|---|
| Iris | `(150, 4)` | `(150,)` | 4 | 3 | small tabular dataset |
| Wine | `(178, 13)` | `(178,)` | 13 | 3 | low-dimensional tabular data |
| Breast Cancer | `(569, 30)` | `(569,)` | 30 | 2 | binary tabular classification |
| Fashion-MNIST | `(70000, 784)` | `(70000,)` | 784 | 10 | flattened grayscale images |
| Optdigits | `(5620, 64)` | `(5620,)` | 64 | 10 | optical handwritten digit recognition |
| Pendigits | `(10992, 16)` | `(10992,)` | 16 | 10 | pen-trajectory digit features |

## 4. RBF Network Design

### 4.1 Network definition

The main model is implemented in [src/train_evaluate.py](/home/ziyan/SURF2026/src/rbf-network-benchmark/src/train_evaluate.py:22).

Its structure is:

```text
input features -> Gaussian RBF hidden layer -> linear output layer -> softmax classification
```

The Gaussian RBF is:

```text
phi(x, c) = exp(-||x - c||^2 / (2 * sigma^2))
```

In code:

```python
hidden = np.exp(-squared_distances / (2.0 * self.sigma_**2))
```

The implementation prepends a bias column of ones before the output layer, so the actual output-layer input dimension is:

```text
n_centers + 1
```

### 4.2 Center selection and sigma

RBF centers can be initialized with:

- `kmeans`
- `random`

`sigma` can be:

- a fixed positive number
- `auto`

When `sigma: auto`, the code computes the median pairwise distance between centers and multiplies it by `sigma_scale`. This gives each dataset a more reasonable kernel width without hard-coding one global scale.

### 4.3 Output-layer solvers

The current code supports two output-layer solvers:

- `solver: gd`
  Uses the handwritten softmax + cross-entropy + L2 regularization + gradient descent path.
- `solver: lbfgs`
  Uses `sklearn.linear_model.LogisticRegression` with `lbfgs`, then maps the learned coefficients back into the unified `weights_` interface.

This solver split is one of the key reasons the digit datasets improved sharply in the latest round.

## 5. Preprocessing

### 5.1 Tabular-style datasets

`iris`, `wine`, `breast_cancer`, `optdigits`, and `pendigits` follow the same base path:

```text
train_test_split
-> StandardScaler(fit on train only)
-> RBF features
-> output-layer training
```

This is controlled by:

```yaml
experiment:
  standardize: true
```

### 5.2 Fashion-MNIST pipeline

`fashion_mnist` uses a separate image pipeline:

```text
784-d pixel vectors
-> float32
-> divide by 255
-> StandardScaler(fit on train only)
-> PCA to 200 dimensions
-> RBF features
-> softmax classification
```

The corresponding config is:

```yaml
experiment:
  fashion_mnist_pipeline:
    normalize_pixels: true
    pca_components: 200
```

The main benefit is that the model does not have to run RBF feature generation directly on 784-dimensional image vectors.

## 6. Current Configuration Snapshot

The important settings in the current `config.yaml` are:

| Dataset | centers | center_init | solver | epochs | lr | sigma_scale | Notes |
|---|---:|---|---|---:|---:|---:|---|
| Iris | 12 | kmeans | gd | 150 | 0.05 | 1.0 | small baseline |
| Wine | 30 | kmeans | gd | 200 | 0.05 | 1.0 | 3-class tabular |
| Breast Cancer | 40 | kmeans | gd | 200 | 0.05 | 1.2 | binary classification |
| Fashion-MNIST | 6000 | random | gd | 400 | 0.02 | 0.35 | trained after PCA |
| Optdigits | 20 | kmeans | lbfgs | 2000 | 0.02 | 0.8 | optimized digit config |
| Pendigits | 20 | kmeans | lbfgs | 2000 | 0.02 | 0.6 | optimized digit config |

## 7. How To Run

### 7.1 Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 7.2 Download datasets

Download every dataset listed in config:

```powershell
python src\download_datasets.py
```

Download a single dataset:

```powershell
python src\download_datasets.py --dataset iris
python src\download_datasets.py --dataset wine
python src\download_datasets.py --dataset breast_cancer
python src\download_datasets.py --dataset fashion_mnist
python src\download_datasets.py --dataset optdigits
python src\download_datasets.py --dataset pendigits
```

### 7.3 Train and evaluate

Run the datasets listed in `experiment.datasets`:

```powershell
python src\train_evaluate.py
```

Run a single dataset:

```powershell
python src\train_evaluate.py --dataset iris
python src\train_evaluate.py --dataset optdigits
python src\train_evaluate.py --dataset pendigits
```

GPU version:

```powershell
python src\train_evaluate_gpu.py --dataset fashion_mnist
```

Outputs are saved under:

```text
SURF2026/output/rbf_YYYYMMDD_HHMMSS_ffffff/
```

Each dataset directory contains:

- `metrics.json`
- `classification_report.txt`
- `predictions.csv`
- `training_curves.csv`
- `loss_curve.csv`
- `accuracy_curve.csv`
- `loss_curve.png`
- `accuracy_curve.png`
- `confusion_matrix.csv`
- `confusion_matrix.png`

Each full run also writes `summary.json`.

## 8. Benchmark Results and Test History

### 8.1 Classical dataset baseline

The most complete classical-dataset run is:

- `SURF2026/output/rbf_20260616_132819_188901/`

Results:

| Dataset | Train / Test split | Feature dimension seen by the model | Test Accuracy | Test Macro F1 |
|---|---:|---:|---:|---:|
| Iris | 112 / 38 | 4 | 0.8421 | 0.8462 |
| Wine | 133 / 45 | 13 | 1.0000 | 1.0000 |
| Breast Cancer | 426 / 143 | 30 | 0.9231 | 0.9133 |
| Fashion-MNIST | 52500 / 17500 | 200 | 0.7091 | 0.7035 |

Notes:

- `Iris`, `Wine`, and `Breast Cancer` go straight from standardized features into the RBF layer.
- `Fashion-MNIST` is evaluated on 200-dimensional PCA features rather than raw 784-dimensional pixels.

### 8.2 Digit dataset tuning history

The digit experiments focused on getting strong accuracy while keeping `n_centers` constrained.

Early runs:

| Time | Dataset | n_centers | solver | epochs | lr | Test Accuracy | Test Macro F1 |
|---|---|---:|---|---:|---:|---:|---:|
| 2026-06-16 14:21 | Optdigits | 20 | gd | 150 | 0.05 | 0.4228 | 0.3476 |
| 2026-06-16 14:21 | Pendigits | 20 | gd | 150 | 0.05 | 0.5360 | 0.4132 |
| 2026-06-16 14:39 | Optdigits | 80 | gd | 300 | 0.02 | 0.6698 | 0.6503 |
| 2026-06-16 14:39 | Pendigits | 80 | gd | 300 | 0.02 | 0.6234 | 0.5587 |
| 2026-06-16 14:40 | Optdigits | 120 | gd | 300 | 0.02 | 0.7402 | 0.7295 |
| 2026-06-16 14:40 | Pendigits | 120 | gd | 300 | 0.02 | 0.6641 | 0.6117 |

This phase showed two things:

- Increasing the number of RBF centers improves representational capacity.
- But under a small-center budget, the handwritten gradient-descent output layer is not strong enough for high digit accuracy.

### 8.3 Latest key change and final result

The most important recent update happened on 2026-06-22, corresponding to:

- `SURF2026/output/rbf_20260622_103017_013676/`

Why the change was made:

- The goal was to keep `n_centers = 20` while still pushing `optdigits` and `pendigits` to much higher accuracy.
- With `solver: gd`, the model tended to underfit when the hidden layer width stayed small.

What changed:

1. Added a `solver` option to `RBFNetwork`.
2. Kept the original `gd` path for backward compatibility.
3. Added an `lbfgs` path backed by `LogisticRegression`.
4. Tuned `sigma_scale` separately for `optdigits` and `pendigits`.
5. Kept `n_centers: 20` fixed and improved the quality of the output-layer optimizer instead of widening the hidden layer further.

Latest results:

| Time | Dataset | n_centers | solver | epochs | sigma_scale | Train Accuracy | Test Accuracy | Test Macro F1 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| 2026-06-22 10:30 | Optdigits | 20 | lbfgs | 2000 | 0.8 | 0.9511 | 0.9580 | 0.9581 |
| 2026-06-22 10:30 | Pendigits | 20 | lbfgs | 2000 | 0.6 | 0.9773 | 0.9687 | 0.9689 |

Compared with the earliest `20 centers + gd` run, the improvement is large:

| Dataset | Early Test Accuracy | Latest Test Accuracy | Accuracy Gain |
|---|---:|---:|---:|
| Optdigits | 0.4228 | 0.9580 | +0.5352 |
| Pendigits | 0.5360 | 0.9687 | +0.4327 |

Why this change matters:

- It delivers a large accuracy gain without increasing the number of RBF centers.
- It suggests that for these digit datasets, the main bottleneck was not only hidden-layer width but also the quality of the output-layer solver.
- The project still keeps one unified `RBFNetwork` interface, so future comparisons across CPU and GPU pipelines remain straightforward.

## 9. Natural Next Steps

Good extensions from here would be:

1. Save a fuller config snapshot into `metrics.json` for easier README updates.
2. Add a fixed benchmark table for digit experiments so run history is easier to maintain.
3. Add a paired discussion of GPU runs and the current `lbfgs` CPU path.
4. If hardware-aware RBF simulation is added later, keep the same data / feature / output / reporting layers and swap only the middle feature computation block.
