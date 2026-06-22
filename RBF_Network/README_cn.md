# RBF 网络（中文说明）

这是一个基于高斯径向基函数和 Softmax 分类器的多分类项目。代码已按标准工程目录组织：训练参数集中在 `config.yaml`，数据准备、训练评估和核心模型均位于 `src/`。

```bash
python -m pip install -r requirements.txt
python src/train_evaluate.py --dataset iris
```

支持 `iris`、`wine`、`breast_cancer`、`digits`、`pendigits` 和 `fashion_mnist`。详细使用方式见 [README.md](README.md)。
