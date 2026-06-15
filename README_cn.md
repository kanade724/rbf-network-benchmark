# rbf-network-benchmark

[English](README.md) | [中文](README_cn.md)

一个可配置的高斯 RBF 网络评估框架，用于经典数据集分类实验，并为后续加入硬件感知模拟保留扩展空间。

## 项目目标

本项目使用经典径向基函数网络评估四个数据集：

- Wine dataset
- Breast Cancer Wisconsin dataset
- Iris dataset
- Fashion-MNIST

当前版本重点是软件侧评估。后续迭代可以在不改变基本项目结构的前提下加入硬件感知 RBF 模拟。

## 目录设计

仓库中只保存代码和配置。数据集和实验输出会写入外层的 SURF2026 工作空间。

```text
SURF2026/
  data/                         # 下载后的数据集，由 download_datasets.py 生成
  output/                       # 训练和测试输出
  src/
    rbf-network-benchmark/
      config.yaml               # 所有可配置参数
      requirements.txt
      src/
        download_datasets.py    # 将数据集下载到 SURF2026/data
        train_evaluate.py       # 训练和评估 RBF 网络
```

这样可以避免数据集和生成结果污染代码仓库。

## RBF 网络结构

模型实现在 `src/train_evaluate.py` 中。

隐藏层使用经典高斯 RBF：

```text
phi(x, c) = exp(-||x - c||^2 / (2 * sigma^2))
```

代码中对应 `RBFNetwork._gaussian_rbf_features`：

```python
hidden = np.exp(-squared_distances / (2.0 * self.sigma_**2))
```

网络结构是：

```text
输入特征 -> 高斯 RBF 隐藏层 -> softmax 输出层
```

RBF 中心可以通过 KMeans 或随机训练样本选择。默认情况下，输出层使用 softmax 梯度下降训练，并记录每个 epoch 的 loss 和 accuracy。训练循环使用 `tqdm` 显示每个数据集的训练进度。

## 使用 venv 配置环境

在仓库根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 配置说明

所有可配置参数都在 `config.yaml` 中。

重要配置项：

- `paths`：控制工作空间、数据目录和输出目录。
- `download.datasets`：控制需要下载哪些数据集。
- `experiment.datasets`：控制需要训练和测试哪些数据集。
- `preprocess`：控制特征提取和标准化。
- `rbf`：控制默认 RBF 网络参数。
- `dataset_overrides`：控制每个数据集的专属参数。

每个数据集都可以通过 `dataset_overrides` 单独覆盖 `epochs`、`learning_rate`、`l2_alpha` 等 RBF 训练参数：

```yaml
dataset_overrides:
  iris:
    rbf:
      epochs: 150
      learning_rate: 0.05
      l2_alpha: 0.001
  fashion_mnist:
    rbf:
      epochs: 150
      learning_rate: 0.05
      l2_alpha: 0.01
```

默认路径配置是：

```yaml
paths:
  workspace_root: ../..
  data_dir: data
  output_dir: output
```

因为本仓库位于 `SURF2026/src/rbf-network-benchmark`，所以 `../..` 会解析到 `SURF2026` 工作空间根目录。

## 下载数据集

下载所有配置的数据集：

```powershell
python src\download_datasets.py
```

只下载一个数据集：

```powershell
python src\download_datasets.py --dataset iris
python src\download_datasets.py --dataset wine
python src\download_datasets.py --dataset breast_cancer
python src\download_datasets.py --dataset fashion_mnist
```

Fashion-MNIST 从 OpenML 获取，因此第一次下载需要网络连接。

## 训练和测试

训练并评估所有配置的数据集：

```powershell
python src\train_evaluate.py
```

训练并评估单个数据集：

```powershell
python src\train_evaluate.py --dataset iris
```

输出会写入：

```text
SURF2026/output/rbf_YYYYMMDD_HHMMSS/
```

每个数据集会生成：

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

完整运行还会生成：

- `summary.json`

## Fashion-MNIST 说明

Fashion-MNIST 是图像数据，因此默认配置会在 RBF 层前使用 HOG 特征。HOG 能保留边缘和形状信息，比直接使用原始像素距离更适合图像任务：

```yaml
dataset_overrides:
  fashion_mnist:
    preprocess:
      feature_extractor: hog
```

这样分类器主体仍然是 RBF 网络，但输入特征更适合图像任务。
