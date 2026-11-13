"""GRU encoder trained with a Deep SVDD style hypersphere objective."""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class GRUEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, features]
        _, h_n = self.gru(x)
        h = h_n[-1]
        return self.fc(h)


class GRUDeepSVDDDetector:
    def __init__(
        self,
        window: int = 30,
        hidden_dim: int = 64,
        n_layers: int = 2,
        epochs: int = 4,
        batch_size: int = 64,
        lr: float = 1e-3,
        lr_center: float = 1e-3,
        device: str | None = None,
        seed: int = 42,
    ):
        self.window = window
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lr_center = lr_center
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.model: GRUEncoder | None = None
        self.center: torch.Tensor | None = None
        self.train_seconds = 0.0

    def _windows(self, x: np.ndarray) -> np.ndarray:
        from paper_eval.data import sliding_windows

        return sliding_windows(x, window=self.window, step=1)

    def fit(self, x_train: np.ndarray) -> "GRUDeepSVDDDetector":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        t0 = time.perf_counter()
        windows = self._windows(x_train)
        n_features = windows.shape[-1]
        self.model = GRUEncoder(
            n_features=n_features,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
        ).to(self.device)
        loader = DataLoader(
            TensorDataset(torch.tensor(windows, dtype=torch.float32)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.center = torch.zeros(self.hidden_dim, device=self.device)

        self.model.train()
        for _ in range(self.epochs):
            for (batch,) in loader:
                batch = batch.to(self.device)
                emb = self.model(batch)
                # Update center toward batch mean, then pull embeddings inward.
                with torch.no_grad():
                    batch_mean = emb.mean(dim=0)
                    self.center = (
                        (1.0 - self.lr_center) * self.center
                        + self.lr_center * batch_mean
                    )
                loss = torch.mean(torch.sum((emb - self.center) ** 2, dim=1))
                opt.zero_grad()
                loss.backward()
                opt.step()

        self.train_seconds = time.perf_counter() - t0
        return self

    def score(self, x_test: np.ndarray) -> Tuple[np.ndarray, float]:
        assert self.model is not None and self.center is not None
        t0 = time.perf_counter()
        windows = self._windows(x_test)
        loader = DataLoader(
            TensorDataset(torch.tensor(windows, dtype=torch.float32)),
            batch_size=self.batch_size,
            shuffle=False,
        )
        scores = []
        self.model.eval()
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                emb = self.model(batch)
                dist = torch.sum((emb - self.center) ** 2, dim=1)
                scores.append(dist.cpu().numpy())
        infer_seconds = time.perf_counter() - t0
        return np.concatenate(scores, axis=0), infer_seconds
