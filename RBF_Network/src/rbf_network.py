import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


class RBFNetwork:
    """Gaussian radial-basis-function network with a softmax output layer."""

    def __init__(
        self,
        n_centers=100,
        gamma=None,
        lr=0.01,
        epochs=50,
        reg=1e-4,
        verbose=True,
        print_every=1,
        random_state=42,
    ):
        self.n_centers = n_centers
        self.gamma = gamma
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.verbose = verbose
        self.print_every = print_every
        self.random_state = random_state

        self.centers = None
        self.W = None
        self.b = None
        self.encoder = OneHotEncoder(sparse_output=False)

    def _rbf(self, X):
        dist = np.linalg.norm(X[:, None, :] - self.centers[None, :, :], axis=2)
        return np.exp(-self.gamma * (dist**2))

    @staticmethod
    def _softmax(Z):
        Z = Z - np.max(Z, axis=1, keepdims=True)
        exp_z = np.exp(Z)
        return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-12)

    @staticmethod
    def _cross_entropy(Y, P):
        return -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_centers = min(self.n_centers, len(X))

        kmeans = KMeans(n_clusters=n_centers, random_state=self.random_state, n_init=10)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_
        self.n_centers = n_centers

        if self.gamma is None:
            distances = cdist(self.centers, self.centers)
            if len(self.centers) > 1:
                np.fill_diagonal(distances, np.inf)
                sigma = np.median(np.min(distances, axis=1))
            else:
                sigma = 1.0
            self.gamma = 1 / (2 * sigma**2 + 1e-8)

        labels = self.encoder.fit_transform(y.reshape(-1, 1))
        features = self._rbf(X)
        rng = np.random.default_rng(self.random_state)
        self.W = rng.normal(0, 0.01, size=(self.n_centers, labels.shape[1]))
        self.b = np.zeros(labels.shape[1])

        m_w, v_w = np.zeros_like(self.W), np.zeros_like(self.W)
        m_b, v_b = np.zeros_like(self.b), np.zeros_like(self.b)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for epoch in range(self.epochs):
            probabilities = self._softmax(features @ self.W + self.b)
            loss = self._cross_entropy(labels, probabilities) + 0.5 * self.reg * np.sum(self.W**2)
            grad_w = features.T @ (probabilities - labels) / len(X) + self.reg * self.W
            grad_b = np.mean(probabilities - labels, axis=0)

            step = epoch + 1
            m_w, v_w = beta1 * m_w + (1 - beta1) * grad_w, beta2 * v_w + (1 - beta2) * grad_w**2
            m_b, v_b = beta1 * m_b + (1 - beta1) * grad_b, beta2 * v_b + (1 - beta2) * grad_b**2
            self.W -= self.lr * (m_w / (1 - beta1**step)) / (np.sqrt(v_w / (1 - beta2**step)) + eps)
            self.b -= self.lr * (m_b / (1 - beta1**step)) / (np.sqrt(v_b / (1 - beta2**step)) + eps)

            prediction = self.encoder.categories_[0][np.argmax(probabilities, axis=1)]
            accuracy = accuracy_score(y, prediction)
            if self.verbose and (epoch == 0 or step % self.print_every == 0 or step == self.epochs):
                print(f"Epoch {step}/{self.epochs} | Loss {loss:.4f} | Acc {accuracy:.4f}")

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        probabilities = self._softmax(self._rbf(X) @ self.W + self.b)
        return self.encoder.categories_[0][np.argmax(probabilities, axis=1)]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
