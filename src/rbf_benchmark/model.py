from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RBFRidgeClassifier:
    n_centers: int
    center_init: str
    sigma: str | float
    sigma_scale: float
    ridge_alpha: float
    random_state: int
    kmeans_max_iter: int
    device: torch.device

    def _kmeans(self, x: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator(device=self.device).manual_seed(self.random_state)
        centers = x[torch.randperm(len(x), generator=generator, device=self.device)[:self.n_centers]].clone()
        for _ in range(self.kmeans_max_iter):
            labels = torch.cdist(x, centers).argmin(dim=1)
            counts = torch.bincount(labels, minlength=self.n_centers)
            updated = torch.zeros_like(centers).index_add_(0, labels, x)
            updated = updated / counts.clamp_min(1).unsqueeze(1)
            updated[counts == 0] = centers[counts == 0]
            if torch.allclose(updated, centers, atol=1e-5, rtol=0):
                return updated
            centers = updated
        return centers

    def _select_centers(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_centers > len(x):
            raise ValueError("rbf.n_centers cannot exceed the number of training samples.")
        if self.center_init == "kmeans":
            return self._kmeans(x)
        if self.center_init == "random":
            generator = torch.Generator(device=self.device).manual_seed(self.random_state)
            return x[torch.randperm(len(x), generator=generator, device=self.device)[:self.n_centers]]
        raise ValueError("rbf.center_init must be 'kmeans' or 'random'.")

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        rbf = torch.exp(-torch.cdist(x, self.centers_).square() / (2 * self.sigma_ ** 2))
        return torch.cat((torch.ones((len(x), 1), device=self.device), rbf), dim=1)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "RBFRidgeClassifier":
        self.classes_ = torch.unique(y, sorted=True)
        self.centers_ = self._select_centers(x)
        if isinstance(self.sigma, (int, float)):
            self.sigma_ = float(self.sigma)
        elif self.sigma == "auto":
            distances = torch.pdist(self.centers_)
            self.sigma_ = float((distances[distances > 0].median() * self.sigma_scale).item()) if len(distances) else 1.0
        else:
            raise ValueError("rbf.sigma must be 'auto' or a positive number.")
        if self.sigma_ <= 0:
            raise ValueError("rbf.sigma must be positive.")
        phi = self._features(x)
        targets = torch.nn.functional.one_hot(torch.searchsorted(self.classes_, y), len(self.classes_)).to(phi.dtype)
        regularizer = torch.eye(phi.shape[1], device=self.device, dtype=phi.dtype) * self.ridge_alpha
        regularizer[0, 0] = 0  # Do not penalize the intercept.
        self.weights_ = torch.linalg.solve(phi.T @ phi + regularizer, phi.T @ targets)
        return self

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.classes_[(self._features(x) @ self.weights_).argmax(dim=1)]
