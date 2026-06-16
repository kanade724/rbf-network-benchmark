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
        train_evaluate.py       # CPU 训练和评估入口
        train_evaluate_gpu.py   # PyTorch 设备训练和评估入口
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

## 数据集维度

下表使用符号表示每个数据集在当前网络中的结构维度。具体的 `n_centers_*` 数值以 `config.yaml` 中的配置为准。

| 数据集 | 输入维度 | RBF hidden 维度 | softmax 输入维度 | softmax 输出维度 |
|---|---:|---:|---:|---:|
| Iris | 4 | `n_centers_iris` | `n_centers_iris + 1` | 3 |
| Wine | 13 | `n_centers_wine` | `n_centers_wine + 1` | 3 |
| Breast Cancer Wisconsin | 30 | `n_centers_breast_cancer` | `n_centers_breast_cancer + 1` | 2 |
| Fashion-MNIST | 原始 784 维像素经 PCA 后为 200 维 | `n_centers_fashion_mnist` | `n_centers_fashion_mnist + 1` | 10 |

其中 `+ 1` 是进入 softmax 输出层前拼接的 bias 列。

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
- `experiment.standardize`：控制是否使用只在训练集上拟合的特征标准化。
- `gpu`：控制 `train_evaluate_gpu.py` 使用的 PyTorch 设备、mini-batch 大小和曲线记录间隔。
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
      epochs: 800
      learning_rate: 0.01
      l2_alpha: 0.00001
```

Fashion-MNIST 的 RBF 特征层比表格数据集宽很多，因此使用更小的学习率。

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

GPU/PyTorch 入口：

```powershell
python src\train_evaluate_gpu.py --dataset fashion_mnist
```

`train_evaluate_gpu.py` 保留相同的 RBF 网络定义和输出文件格式，但使用 PyTorch 计算 Gaussian RBF 特征和 softmax 梯度下降。如果 `gpu.device` 设置为 `auto`，当 `torch.cuda.is_available()` 为 true 时会使用 CUDA，否则自动回退到 CPU。

```yaml
gpu:
  device: auto
  batch_size: 2048
  eval_batch_size: 4096
  history_interval: 1
```

特征准备、PCA 和 KMeans 中心选择仍然使用 scikit-learn 在 CPU 上完成。为了让 Fashion-MNIST 训练更快，当前配置使用随机训练样本作为 RBF centers。

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

## 当前参考结果

最新一次完整运行保存在 `SURF2026/output/rbf_20260616_132819_188901/`。下面的表格记录了这次运行的主要指标，方便在 README 中保留一个明确的基准快照。

| 数据集 | 训练准确率 | 测试准确率 | 训练 Macro F1 | 测试 Macro F1 |
|---|---:|---:|---:|---:|
| Iris | 0.8839 | 0.8421 | 0.8827 | 0.8462 |
| Wine | 0.9699 | 1.0000 | 0.9714 | 1.0000 |
| Breast Cancer Wisconsin | 0.8967 | 0.9231 | 0.8842 | 0.9133 |
| Fashion-MNIST | 0.7085 | 0.7091 | 0.7032 | 0.7035 |

这些结果对应当前 `config.yaml` 的配置，包括各数据集的 RBF 参数和 Fashion-MNIST 的 PCA 管线。

## Fashion-MNIST 说明

Fashion-MNIST 在进入 RBF 层前使用专门的图像处理管线：

```text
原始 Fashion-MNIST 图片
-> 转 float32
-> 除以 255，归一化到 0~1
-> reshape 成 784 维向量
-> stratified train/test split
-> StandardScaler 标准化，并且只在训练集上拟合
-> PCA 降到 200 维
-> 按照 config 选择 RBF centers，当前 Fashion-MNIST 使用随机训练样本
-> Gaussian RBF 升维
-> softmax 梯度下降分类
```

PCA 维度由 `config.yaml` 中的 `experiment.fashion_mnist_pipeline.pca_components` 控制。
