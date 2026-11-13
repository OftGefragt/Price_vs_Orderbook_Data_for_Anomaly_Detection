"""Compact 1D f-AnoGAN style detector for multivariate windows.

The architecture follows the repository design (encoder, generator,
discriminator) with reduced width so the comparison notebook finishes in a
reasonable time on CPU or a free Colab GPU.
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Encoder1D(nn.Module):
    def __init__(self, n_features: int, seq_len: int, z_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, n_features, seq_len)
            flat = self.net(dummy).shape[1]
        self.fc = nn.Linear(flat, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x))


class Generator1D(nn.Module):
    def __init__(self, n_features: int, seq_len: int, z_dim: int = 32):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.fc = nn.Linear(z_dim, 64 * max(seq_len // 2, 1))
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, n_features, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(z.size(0), 64, -1)
        out = self.deconv(h)
        if out.size(2) != self.seq_len:
            out = nn.functional.interpolate(out, size=self.seq_len, mode="linear", align_corners=False)
        return out


class Discriminator1D(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor):
        feat = self.features(x)
        pooled = torch.mean(feat, dim=2)
        logit = self.classifier(feat)
        return pooled, logit


class FanoganDetector:
    def __init__(
        self,
        window: int = 30,
        z_dim: int = 32,
        epochs_gan: int = 5,
        epochs_enc: int = 5,
        batch_size: int = 64,
        device: str | None = None,
        seed: int = 42,
    ):
        self.window = window
        self.z_dim = z_dim
        self.epochs_gan = epochs_gan
        self.epochs_enc = epochs_enc
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.G: Generator1D | None = None
        self.D: Discriminator1D | None = None
        self.E: Encoder1D | None = None
        self.train_seconds = 0.0

    def _windows_nct(self, x: np.ndarray) -> np.ndarray:
        from paper_eval.data import sliding_windows

        # [N, T, C] -> [N, C, T]
        w = sliding_windows(x, window=self.window, step=1)
        return np.transpose(w, (0, 2, 1))

    def fit(self, x_train: np.ndarray) -> "FanoganDetector":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        t0 = time.perf_counter()
        windows = self._windows_nct(x_train)
        # Map [0,1] scaled inputs to [-1,1] for tanh generator.
        windows = windows * 2.0 - 1.0
        n_features = windows.shape[1]
        self.G = Generator1D(n_features, self.window, self.z_dim).to(self.device)
        self.D = Discriminator1D(n_features).to(self.device)
        self.E = Encoder1D(n_features, self.window, self.z_dim).to(self.device)

        data = torch.tensor(windows, dtype=torch.float32)
        loader = DataLoader(TensorDataset(data), batch_size=self.batch_size, shuffle=True)
        opt_g = torch.optim.Adam(self.G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.D.parameters(), lr=1e-4, betas=(0.5, 0.999))
        opt_e = torch.optim.Adam(self.E.parameters(), lr=1e-4, betas=(0.5, 0.999))
        bce = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()

        for _ in range(self.epochs_gan):
            for (batch,) in loader:
                batch = batch.to(self.device)
                bsz = batch.size(0)
                real = torch.ones(bsz, 1, device=self.device)
                fake = torch.zeros(bsz, 1, device=self.device)

                z = torch.randn(bsz, self.z_dim, device=self.device)
                fake_x = self.G(z)
                _, d_real = self.D(batch)
                _, d_fake = self.D(fake_x.detach())
                loss_d = bce(d_real, real) + bce(d_fake, fake)
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()

                z = torch.randn(bsz, self.z_dim, device=self.device)
                fake_x = self.G(z)
                _, d_fake = self.D(fake_x)
                loss_g = bce(d_fake, real)
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()

        for _ in range(self.epochs_enc):
            for (batch,) in loader:
                batch = batch.to(self.device)
                z = self.E(batch)
                rec = self.G(z)
                loss_e = mse(rec, batch)
                opt_e.zero_grad()
                loss_e.backward()
                opt_e.step()

        self.train_seconds = time.perf_counter() - t0
        return self

    def score(self, x_test: np.ndarray) -> Tuple[np.ndarray, float]:
        assert self.G is not None and self.D is not None and self.E is not None
        t0 = time.perf_counter()
        windows = self._windows_nct(x_test) * 2.0 - 1.0
        loader = DataLoader(
            TensorDataset(torch.tensor(windows, dtype=torch.float32)),
            batch_size=self.batch_size,
            shuffle=False,
        )
        mse = nn.MSELoss(reduction="none")
        scores = []
        self.G.eval()
        self.D.eval()
        self.E.eval()
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                z = self.E(batch)
                rec = self.G(z)
                loss_rec = mse(rec, batch).mean(dim=(1, 2))
                f_real, _ = self.D(batch)
                f_rec, _ = self.D(rec)
                loss_feat = mse(f_rec, f_real).mean(dim=1)
                score = 1.0 * loss_rec + 0.1 * loss_feat
                scores.append(score.cpu().numpy())
        infer_seconds = time.perf_counter() - t0
        return np.concatenate(scores, axis=0), infer_seconds
