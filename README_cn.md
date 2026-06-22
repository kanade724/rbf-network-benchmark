# rbf-network-benchmark

[English](README.md) | [中文](README_cn.md)

一个面向分类任务的高斯 RBF 神经网络实验框架。项目重点是把数据下载、预处理、RBF 特征生成、输出层训练和结果落盘整合到同一套可配置流程里，方便比较不同数据集和不同参数设置的表现。

## 1. 项目结构

仓库本身只保存代码和配置，数据与实验输出统一写到外层 `SURF2026` 工作区，避免结果文件污染源码目录。

```text
SURF2026/
  data/                               # 下载后的 .npz 数据集
  output/                             # 每次运行的实验结果
  src/
    rbf-network-benchmark/
      config.yaml                     # 全局配置与数据集覆盖配置
      requirements.txt
      README.md
      README_cn.md
      src/
        download_datasets.py          # 下载并保存数据集
        train_evaluate.py             # NumPy / scikit-learn 训练与评估入口
        train_evaluate_gpu.py         # PyTorch 训练与评估入口
```

当前默认路径来自 `config.yaml`：

```yaml
paths:
  workspace_root: ../..
  data_dir: data
  output_dir: output
```

因为仓库位于 `SURF2026/src/rbf-network-benchmark`，所以 `../..` 会解析到 `SURF2026` 根目录。

## 2. 代码框架

整个流程分成 4 层：

1. `download_datasets.py`
   负责下载经典数据集和 OpenML 数据集，并统一保存为 `.npz`。
2. `train_evaluate.py`
   负责 CPU 版本训练、测试、绘图和指标导出。
3. `train_evaluate_gpu.py`
   保持相同输入输出格式，用 PyTorch 加速 RBF 特征和 softmax 训练。
4. `config.yaml`
   负责控制数据集选择、预处理方式、RBF 参数和单数据集覆盖参数。

`train_evaluate.py` 中单个数据集的主流程如下：

```text
读取 .npz 数据
-> train_test_split(stratify=y)
-> 预处理
-> 构造 RBFNetwork
-> 训练输出层
-> 测试集预测
-> 保存 metrics / report / curves / confusion matrix
```

## 3. 数据类型与存储格式

### 3.1 支持的数据集

当前代码实际支持 6 个数据集：

- `iris`
- `wine`
- `breast_cancer`
- `fashion_mnist`
- `optdigits`
- `pendigits`

其中：

- `iris`、`wine`、`breast_cancer` 使用 `sklearn.datasets` 直接加载。
- `fashion_mnist`、`optdigits`、`pendigits` 使用 OpenML 下载。

### 3.2 落盘格式

所有数据都会被保存为 `SURF2026/data/<dataset>.npz`，字段固定为：

- `x`：特征矩阵，`float64`
- `y`：类别标签，`int64`
- `target_names`：类别名称
- `feature_names`：特征名称

这让训练脚本只需要面对统一的数据接口，不需要关心原始数据来源。

### 3.3 当前数据形状

以下形状来自当前 `SURF2026/data` 中的数据文件：

| 数据集 | `x` 形状 | `y` 形状 | 原始特征维度 | 类别数 | 数据特点 |
|---|---:|---:|---:|---:|---|
| Iris | `(150, 4)` | `(150,)` | 4 | 3 | 小规模表格数据 |
| Wine | `(178, 13)` | `(178,)` | 13 | 3 | 中低维表格数据 |
| Breast Cancer | `(569, 30)` | `(569,)` | 30 | 2 | 二分类表格数据 |
| Fashion-MNIST | `(70000, 784)` | `(70000,)` | 784 | 10 | 灰度图像展开向量 |
| Optdigits | `(5620, 64)` | `(5620,)` | 64 | 10 | 手写数字光学识别 |
| Pendigits | `(10992, 16)` | `(10992,)` | 16 | 10 | 手写数字笔迹轨迹特征 |

## 4. RBF 神经网络结构

### 4.1 网络定义

当前模型定义在 [src/train_evaluate.py](/home/ziyan/SURF2026/src/rbf-network-benchmark/src/train_evaluate.py:22)。

网络结构是：

```text
输入特征 -> 高斯 RBF 隐层 -> 线性输出层 -> softmax 分类
```

高斯 RBF 的形式为：

```text
phi(x, c) = exp(-||x - c||^2 / (2 * sigma^2))
```

在代码中对应：

```python
hidden = np.exp(-squared_distances / (2.0 * self.sigma_**2))
```

然后会在隐层前拼接一列常数 1 作为 bias，因此实际输出层输入维度为：

```text
n_centers + 1
```

### 4.2 中心与 sigma

RBF 中心支持两种初始化方式：

- `kmeans`
- `random`

`sigma` 支持：

- 直接指定数值
- `auto`

当 `sigma: auto` 时，代码会取各中心两两距离的中位数，再乘以 `sigma_scale`，这样可以让不同数据集在不同尺度下仍有一个可比较的核宽度。

### 4.3 输出层求解方式

当前输出层支持两种训练方式：

- `solver: gd`
  使用手写 softmax + 交叉熵 + L2 正则 + 梯度下降。
- `solver: lbfgs`
  使用 `sklearn.linear_model.LogisticRegression` 的 `lbfgs` 求解器，再把系数映射回统一的 `weights_` 接口。

这也是 2026-06-22 数字数据集精度提升的关键改动之一。

## 5. 预处理流程

### 5.1 表格类数据

`iris`、`wine`、`breast_cancer`、`optdigits`、`pendigits` 走同一套基础流程：

```text
train_test_split
-> StandardScaler(仅在训练集拟合)
-> RBF 特征
-> 输出层训练
```

是否标准化由：

```yaml
experiment:
  standardize: true
```

控制。

### 5.2 Fashion-MNIST

`fashion_mnist` 有单独的数据管线：

```text
784 维像素
-> 转 float32
-> /255 归一化
-> StandardScaler(训练集拟合)
-> PCA 降到 200 维
-> RBF 特征
-> softmax 分类
```

对应配置为：

```yaml
experiment:
  fashion_mnist_pipeline:
    normalize_pixels: true
    pca_components: 200
```

这样做的主要目的是降低 784 维图像直接进入 RBF 时的计算和存储压力。

## 6. 当前配置摘要

当前 `config.yaml` 的重点设置如下：

| 数据集 | centers | center_init | solver | epochs | lr | sigma_scale | 备注 |
|---|---:|---|---|---:|---:|---:|---|
| Iris | 12 | kmeans | gd | 150 | 0.05 | 1.0 | 小数据集基线 |
| Wine | 30 | kmeans | gd | 200 | 0.05 | 1.0 | 三分类表格 |
| Breast Cancer | 40 | kmeans | gd | 200 | 0.05 | 1.2 | 二分类 |
| Fashion-MNIST | 6000 | random | gd | 400 | 0.02 | 0.35 | PCA 后训练 |
| Optdigits | 20 | kmeans | lbfgs | 2000 | 0.02 | 0.8 | 数字数据集优化配置 |
| Pendigits | 20 | kmeans | lbfgs | 2000 | 0.02 | 0.6 | 数字数据集优化配置 |

## 7. 如何运行

### 7.1 环境安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 7.2 下载数据集

下载配置文件里列出的数据集：

```powershell
python src\download_datasets.py
```

下载单个数据集：

```powershell
python src\download_datasets.py --dataset iris
python src\download_datasets.py --dataset wine
python src\download_datasets.py --dataset breast_cancer
python src\download_datasets.py --dataset fashion_mnist
python src\download_datasets.py --dataset optdigits
python src\download_datasets.py --dataset pendigits
```

### 7.3 训练与评估

按 `config.yaml` 中 `experiment.datasets` 运行：

```powershell
python src\train_evaluate.py
```

只跑一个数据集：

```powershell
python src\train_evaluate.py --dataset iris
python src\train_evaluate.py --dataset optdigits
python src\train_evaluate.py --dataset pendigits
```

GPU 版本：

```powershell
python src\train_evaluate_gpu.py --dataset fashion_mnist
```

输出目录格式为：

```text
SURF2026/output/rbf_YYYYMMDD_HHMMSS_ffffff/
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

整次运行还会额外生成 `summary.json`。

## 8. 测试结果与基线记录

### 8.1 经典数据集基线

当前最完整的一次经典数据集运行是：

- `SURF2026/output/rbf_20260616_132819_188901/`

结果如下：

| 数据集 | 训练集 / 测试集 | 预处理后特征维度 | 测试准确率 | 测试 Macro F1 |
|---|---:|---:|---:|---:|
| Iris | 112 / 38 | 4 | 0.8421 | 0.8462 |
| Wine | 133 / 45 | 13 | 1.0000 | 1.0000 |
| Breast Cancer | 426 / 143 | 30 | 0.9231 | 0.9133 |
| Fashion-MNIST | 52500 / 17500 | 200 | 0.7091 | 0.7035 |

说明：

- `Iris`、`Wine`、`Breast Cancer` 都是标准化后直接进入 RBF。
- `Fashion-MNIST` 在测试时看到的特征维度是 PCA 后的 200 维，不是原始 784 维。

### 8.2 数字数据集调参过程

数字数据集的实验重点是：在 `n_centers` 受限的情况下，尽可能提高识别精度。

早期几轮结果如下：

| 时间 | 数据集 | n_centers | solver | epochs | lr | 测试准确率 | 测试 Macro F1 |
|---|---|---:|---|---:|---:|---:|---:|
| 2026-06-16 14:21 | Optdigits | 20 | gd | 150 | 0.05 | 0.4228 | 0.3476 |
| 2026-06-16 14:21 | Pendigits | 20 | gd | 150 | 0.05 | 0.5360 | 0.4132 |
| 2026-06-16 14:39 | Optdigits | 80 | gd | 300 | 0.02 | 0.6698 | 0.6503 |
| 2026-06-16 14:39 | Pendigits | 80 | gd | 300 | 0.02 | 0.6234 | 0.5587 |
| 2026-06-16 14:40 | Optdigits | 120 | gd | 300 | 0.02 | 0.7402 | 0.7295 |
| 2026-06-16 14:40 | Pendigits | 120 | gd | 300 | 0.02 | 0.6641 | 0.6117 |

这一阶段说明了两件事：

- 单纯增加 RBF centers，确实能提高表达能力。
- 但如果 centers 数量被限制在较小范围，单靠手写梯度下降不容易把数字数据集做到很高精度。

### 8.3 最后一次关键修改与最新结果

当前最值得记录的一次修改发生在 2026-06-22，对应最新数字数据集运行：

- `SURF2026/output/rbf_20260622_103017_013676/`

这次修改的原因是：

- 希望在 `n_centers = 20` 的限制下，仍然让 `optdigits` 和 `pendigits` 达到更高准确率。
- 之前 `solver: gd` 在小 center 数配置下容易欠拟合，输出层优化不充分。

这次具体改了什么：

1. 在 `RBFNetwork` 中新增 `solver` 选项。
2. 保留原有 `gd` 路径，兼容原本实验。
3. 增加 `lbfgs` 路径，调用 `LogisticRegression` 训练输出层。
4. 给 `optdigits` 和 `pendigits` 单独设置更合适的 `sigma_scale`。
5. 保持 `n_centers: 20` 不变，重点优化“输出层求解质量”而不是继续堆宽度。

最新结果如下：

| 时间 | 数据集 | n_centers | solver | epochs | sigma_scale | 训练准确率 | 测试准确率 | 测试 Macro F1 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| 2026-06-22 10:30 | Optdigits | 20 | lbfgs | 2000 | 0.8 | 0.9511 | 0.9580 | 0.9581 |
| 2026-06-22 10:30 | Pendigits | 20 | lbfgs | 2000 | 0.6 | 0.9773 | 0.9687 | 0.9689 |

相对于最早的 `20 centers + gd` 版本，提升非常明显：

| 数据集 | 早期测试准确率 | 最新测试准确率 | 准确率提升 |
|---|---:|---:|---:|
| Optdigits | 0.4228 | 0.9580 | +0.5352 |
| Pendigits | 0.5360 | 0.9687 | +0.4327 |

这样修改的好处：

- 在 centers 数量不增加的前提下，大幅提升分类性能。
- 说明瓶颈主要不再是 RBF 特征“太少”，而是输出层求解方式不够强。
- 保留统一的 `RBFNetwork` 接口后，CPU 版和 GPU 版输出格式仍然一致，后续继续做对比实验会更方便。

## 9. 适合如何继续扩展

如果后续继续做实验，比较自然的扩展方向有：

1. 给 `metrics.json` 增加更详细的配置快照，方便 README 自动引用。
2. 给数字数据集增加固定实验表，减少手工整理历史结果的成本。
3. 在 GPU 版本里补上与 `solver: lbfgs` 对应的对照实验说明。
4. 如果未来加入硬件感知 RBF 模拟，可以保持现有“数据层 / 特征层 / 输出层 / 输出结果层”的框架不变，只替换中间特征计算部分。
