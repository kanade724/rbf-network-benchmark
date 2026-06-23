# RBF 岭回归分类基准

该工程对 Iris、Wine、Breast Cancer、Fashion-MNIST、Optical Recognition of Handwritten Digits 与 Pen-Based Recognition of Handwritten Digits 使用同一套 Gaussian RBF 特征层与岭回归分类器。计算设备由 `config.yaml` 的 `runtime.device` 统一控制：`auto`、`cuda` 或 `cpu`；不再存在独立的 CPU / GPU 训练实现。

## 数据流

`prepare_datasets.py` 会下载数据、按标签分层切分训练集和测试集，并完成全部 RBF 前预处理后再落盘：

```text
原始数据 → 分层 train/test 切分 → 像素归一化（仅 Fashion-MNIST）
→ 基于训练集的标准化 → PCA（仅 Fashion-MNIST） → data/<dataset>_{train,test}.csv
```

每个 CSV 的最后一列是整数 `label`，其余列为已处理的 `feature_*`。同目录的 `<dataset>_metadata.json` 保存类别名与预处理信息。标准化和 PCA 只在训练集拟合，因此测试集不会泄漏到预处理参数中。

## 程序入口

有三个标准入口（均带 `main()`）：

```powershell
# 下载、切分、预处理并生成全部 CSV
python src\prepare_datasets.py

# 训练一个数据集
python src\train_dataset.py --dataset iris

# 按 config.yaml 的 experiment.datasets 训练全部数据集
python src\train_all.py
```

也可单独运行固定数据集包装脚本，例如 `dataset_runners/iris.py`、`dataset_runners/optical_digits.py`、`dataset_runners/pen_digits.py`。

## 模型

训练脚本从 CSV 读取数据，在配置的设备上完成：

```text
训练特征 → KMeans 或随机 RBF 中心 → Gaussian RBF 特征 + 偏置
→ 多输出 one-hot 岭回归 → 类别分数 argmax
```

岭回归闭式解为 `W = (ΦᵀΦ + αI)⁻¹ΦᵀY`，偏置项不正则化。Fashion-MNIST 的中心数默认降为 1000，因为岭回归需要求解中心数平方规模的线性系统；原先的 6000 中心不适合作为此求解器的默认配置。

结果保存到 `SURF2026/output/rbf_ridge_*/<dataset>/`，包含 `metrics.json`、`predictions.csv`、`confusion_matrix.csv` 和 `classification_report.txt`。
