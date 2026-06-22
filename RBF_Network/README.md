# RBF Network

一个使用高斯径向基函数（RBF）和 Softmax 分类器实现的多分类项目。模型核心保留原有的 KMeans++ 中心选取、自动 `gamma` 估计和 Adam 训练逻辑；工程结构已统一为可配置的训练入口。

## 项目结构

```text
.
├── src/
│   ├── download_datasets.py    # 下载或准备数据集
│   ├── train_evaluate.py       # CPU 训练与评估入口
│   ├── train_evaluate_gpu.py   # 预留 GPU 训练入口
│   ├── datasets.py             # 数据集加载与划分
│   └── rbf_network.py          # RBF 网络核心实现
├── config.yaml                 # 各数据集的训练参数
├── requirements.txt
├── README.md
├── README_cn.md
└── LICENSE
```

## 安装

```bash
python -m pip install -r requirements.txt
```

## 使用

训练 Iris 数据集：

```bash
python src/train_evaluate.py --dataset iris
```

可选数据集：`iris`、`wine`、`breast_cancer`、`digits`、`pendigits`、`fashion_mnist`。

```bash
python src/train_evaluate.py --dataset digits
python src/train_evaluate.py --dataset fashion_mnist
```

下载并缓存需要联网的数据集：

```bash
python src/download_datasets.py --dataset all
```

通过 `config.yaml` 调整数据集划分、标准化、PCA 和 RBF 网络超参数。`train_evaluate_gpu.py` 当前只是兼容的预留入口：模型核心仍依赖 NumPy、SciPy 和 scikit-learn，因此实际训练在 CPU 上执行。
