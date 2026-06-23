# RBF Ridge Classification Benchmark

The project evaluates Iris, Wine, Breast Cancer, Fashion-MNIST, Optical Digits, and Pen Digits with one shared Gaussian RBF feature layer and a ridge-regression classifier. `runtime.device` in `config.yaml` selects `auto`, `cuda`, or `cpu`; there are no separate CPU and GPU training implementations.

Run the three standard entry points:

```powershell
python src\prepare_datasets.py
python src\train_dataset.py --dataset iris
python src\train_all.py
```

The download program performs the complete pre-RBF pipeline: stratified split, Fashion-MNIST pixel normalization, train-fitted standardization, and Fashion-MNIST PCA. It writes `data/<dataset>_train.csv` and `data/<dataset>_test.csv`; the final `label` column is the class target. Training uses those CSV files directly.

Dataset wrappers may also be executed directly from `src/dataset_runners/`.

The model builds KMeans or random centers, applies Gaussian RBF features with an intercept, then solves multi-output ridge regression. Outputs are written to `SURF2026/output/rbf_ridge_*/<dataset>/`.
