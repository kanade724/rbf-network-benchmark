"""
RBF Network for Regression (Curve Fitting)

使用流程：
  1. 从 NI 组拿到测量数据 (x, y)
  2. 调 K 找到最佳拟合
  3. 用 predict() 在新的 x 上生成 y
"""

import numpy as np
from sklearn.cluster import KMeans


class RBFRegressor:
    """RBF 回归网络，用高斯核拟合曲线。

    用法
    ----
    model = RBFRegressor(n_centers=10)
    model.fit(x, y)            # x, y 是一维数组
    y_pred = model.predict(x_new)  # x_new 是要预测的点
    """

    def __init__(self, n_centers: int = 10, sigma: float | None = None,
                 ridge_alpha: float = 1e-6, random_state: int = 42):
        self.n_centers = n_centers
        self.sigma = sigma
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state

        self.centers_ = None   # RBF 中心位置
        self.sigma_ = None     # 高斯核宽度
        self.weights_ = None   # 输出层权重

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RBFRegressor":
        """训练 RBF 网络。

        步骤：
        1. k-means 聚类确定 RBF 中心
        2. 自动计算 sigma
        3. 用伪逆（ridge regression）算输出权重
        """
        X = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(y, dtype=np.float64)

        # ---- 1. k-means 选中心 ----
        kmeans = KMeans(n_clusters=self.n_centers,
                        random_state=self.random_state,
                        n_init='auto')
        kmeans.fit(X)
        self.centers_ = kmeans.cluster_centers_  # shape (K, 1)

        # ---- 2. 自动算 sigma ----
        if self.sigma is None:
            dists = np.zeros((self.n_centers, self.n_centers))
            for i in range(self.n_centers):
                d = self.centers_ - self.centers_[i]
                dists[i] = np.sqrt(np.sum(d ** 2, axis=1))
            d_max = dists.max()
            self.sigma_ = d_max / np.sqrt(2.0 * self.n_centers) or 1.0
        else:
            self.sigma_ = self.sigma

        # ---- 3. 算高斯响应矩阵 Phi ----
        Phi = self._gaussian(X)

        # ---- 4. 伪逆求权重 ----
        A = Phi.T @ Phi + self.ridge_alpha * np.eye(self.n_centers)
        self.weights_ = np.linalg.solve(A, Phi.T @ y)

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """在新 x 点上预测 y 值。"""
        X = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        Phi = self._gaussian(X)
        return Phi @ self.weights_

    def _gaussian(self, X: np.ndarray) -> np.ndarray:
        """计算高斯核响应矩阵，返回 shape (n_samples, n_centers)。"""
        phi = np.zeros((X.shape[0], self.n_centers))
        for j in range(self.n_centers):
            d = X - self.centers_[j]
            dist_sq = np.sum(d ** 2, axis=1)
            phi[:, j] = np.exp(-dist_sq / (2.0 * self.sigma_ ** 2))
        return phi


def find_best_k(x, y, k_range=range(3, 51)):
    """自动找最佳 K 值（误差最小的 K）。

    参数
    ----
    x, y : 测量数据
    k_range : 要尝试的 K 值范围

    返回
    ----
    best_k, errors
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42)

    n_train = len(x_train)
    best_k = None
    best_err = float('inf')
    errors = []

    for k in k_range:
        if k >= n_train:   # k-means 要求 n_samples >= n_clusters
            break
        model = RBFRegressor(n_centers=k, random_state=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        err = mean_squared_error(y_val, y_pred)
        errors.append((k, err))
        if err < best_err:
            best_err = err
            best_k = k

    return best_k, errors


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore')

    # 模拟一组高斯形状测量数据
    np.random.seed(42)
    x = np.linspace(-5, 5, 50)
    y_true = np.exp(-x**2 / 2)
    y_meas = y_true + np.random.normal(0, 0.05, 50)

    # 找最佳 K
    best_k, errors = find_best_k(x, y_meas)
    print(f"最佳 K = {best_k}")

    # 用最佳 K 训练最终模型
    model = RBFRegressor(n_centers=best_k, random_state=42)
    model.fit(x, y_meas)

    # 生成新数据
    x_new = np.linspace(-5, 5, 200)
    y_new = model.predict(x_new)

    print(f"生成了 {len(x_new)} 个新数据点")
    print(f"前 5 个点: x={x_new[:5].round(3)}, y={y_new[:5].round(3)}")

    # 画图
    plt.scatter(x, y_meas, s=15, label='测量数据')
    plt.plot(x_new, y_new, 'r-', lw=2, label=f'RBF 拟合 (K={best_k})')
    plt.plot(x_new, np.exp(-x_new**2/2), 'k--', lw=1, alpha=0.5, label='真实曲线')
    plt.legend()
    plt.title(f'RBF 回归演示 (最佳 K={best_k})')
    plt.savefig('/Users/yijiacao/Documents/纳米所/rbf_demo.png', dpi=120)
    plt.close()
    print("已保存 rbf_demo.png")
