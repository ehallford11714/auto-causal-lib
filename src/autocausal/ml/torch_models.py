"""Optional PyTorch MLP imputer + tabular regressor (soft import).

Install: ``pip install -e ".[torch]"``
Prefer in loops: ``AUTOCAUSAL_TORCH=1``
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from autocausal.ml.construct import torch_available


@dataclass
class TorchFitStats:
    backend: str
    epochs: int = 0
    train_mae: float = float("nan")
    heldout_mask_mae: float = float("nan")
    n_features: int = 0
    n_rows: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_numeric_matrix(df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in columns if c in df.columns]
    if not cols:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    mat = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    return mat, cols


def _standardize(
    x: np.ndarray, mu: Optional[np.ndarray] = None, sigma: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mu is None:
        mu = np.nanmean(x, axis=0)
    if sigma is None:
        sigma = np.nanstd(x, axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    # fill nan with 0 in z-space after replacing with mu
    filled = np.where(np.isnan(x), mu, x)
    z = (filled - mu) / sigma
    return z, mu, sigma


class TorchMLPImputer:
    """
    Mask → reconstruct MLP for tabular KPI imputation.

    Soft-fails construction if torch is missing; callers should check
    ``torch_available()`` first.
    """

    def __init__(
        self,
        *,
        hidden: int = 32,
        epochs: int = 40,
        lr: float = 1e-2,
        mask_rate: float = 0.25,
        seed: int = 0,
    ) -> None:
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.mask_rate = mask_rate
        self.seed = seed
        self.columns: list[str] = []
        self.mu: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None
        self._model: Any = None
        self.stats = TorchFitStats(backend="torch_mlp")

    def fit(self, df: pd.DataFrame, columns: Optional[list[str]] = None) -> "TorchMLPImputer":
        if not torch_available():
            self.stats.notes.append("torch unavailable")
            return self
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        x_raw, cols = _to_numeric_matrix(df, columns or list(df.columns))
        self.columns = cols
        n, d = x_raw.shape
        self.stats.n_rows = int(n)
        self.stats.n_features = int(d)
        if n < 4 or d < 1:
            self.stats.notes.append("insufficient rows/features for torch imputer")
            return self

        z, mu, sigma = _standardize(x_raw)
        self.mu, self.sigma = mu, sigma
        observed = ~np.isnan(x_raw)

        class MLP(nn.Module):
            def __init__(self, dim: int, hidden: int) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim * 2, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, dim),
                )

            def forward(self, x: Any, mask: Any) -> Any:
                return self.net(torch.cat([x, mask], dim=-1))

        model = MLP(d, self.hidden)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        # held-out artificial masks on originally observed cells
        rng = np.random.default_rng(self.seed)
        hold = (rng.random(z.shape) < self.mask_rate) & observed
        train_mask = observed & ~hold

        x_t = torch.tensor(np.where(train_mask, z, 0.0), dtype=torch.float32)
        m_t = torch.tensor(train_mask.astype(np.float32))
        y_t = torch.tensor(z, dtype=torch.float32)

        model.train()
        last_mae = float("nan")
        for _ in range(self.epochs):
            opt.zero_grad()
            pred = model(x_t, m_t)
            # only supervise originally observed cells that we kept in train
            if train_mask.any():
                loss = loss_fn(pred[torch.tensor(train_mask)], y_t[torch.tensor(train_mask)])
            else:
                loss = loss_fn(pred, y_t)
            loss.backward()
            opt.step()
            last_mae = float(torch.mean(torch.abs(pred.detach() - y_t)).item())

        model.eval()
        with torch.no_grad():
            pred_full = model(x_t, m_t).numpy()
        if hold.any():
            held_mae = float(np.mean(np.abs(pred_full[hold] - z[hold])))
        else:
            held_mae = float("nan")

        self._model = model
        self.stats.epochs = self.epochs
        self.stats.train_mae = last_mae
        self.stats.heldout_mask_mae = held_mae
        self.stats.notes.append("trained mask→reconstruct MLP on numeric KPI columns")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self._model is None or not self.columns or self.mu is None or self.sigma is None:
            # median fallback
            for c in self.columns or []:
                if c in out.columns:
                    out[c] = pd.to_numeric(out[c], errors="coerce")
                    out[c] = out[c].fillna(out[c].median())
            return out

        import torch

        x_raw, _ = _to_numeric_matrix(out, self.columns)
        observed = ~np.isnan(x_raw)
        z, _, _ = _standardize(x_raw, self.mu, self.sigma)
        x_in = np.where(observed, z, 0.0)
        m_in = observed.astype(np.float32)
        self._model.eval()
        with torch.no_grad():
            pred = self._model(
                torch.tensor(x_in, dtype=torch.float32),
                torch.tensor(m_in, dtype=torch.float32),
            ).numpy()
        recon = pred * self.sigma + self.mu
        filled = np.where(observed, x_raw, recon)
        for i, c in enumerate(self.columns):
            out[c] = filled[:, i]
        return out

    def fit_transform(
        self, df: pd.DataFrame, columns: Optional[list[str]] = None
    ) -> pd.DataFrame:
        return self.fit(df, columns=columns).transform(df)


class TorchMLPRegressor:
    """Small tabular MLP for outcome KPI regression (optional)."""

    def __init__(
        self,
        *,
        hidden: int = 32,
        epochs: int = 40,
        lr: float = 1e-2,
        seed: int = 0,
    ) -> None:
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.feature_columns: list[str] = []
        self.outcome: Optional[str] = None
        self.mu: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None
        self.y_mu: float = 0.0
        self.y_sigma: float = 1.0
        self._model: Any = None
        self.stats = TorchFitStats(backend="torch_mlp_regressor")

    def fit(
        self,
        df: pd.DataFrame,
        *,
        outcome: str,
        features: Optional[list[str]] = None,
    ) -> "TorchMLPRegressor":
        if not torch_available():
            self.stats.notes.append("torch unavailable")
            return self
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        self.outcome = outcome
        feats = [c for c in (features or list(df.columns)) if c != outcome and c in df.columns]
        # numeric only
        feats = [c for c in feats if pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))]
        self.feature_columns = feats
        if not feats or outcome not in df.columns:
            self.stats.notes.append("missing features/outcome")
            return self

        x_raw, _ = _to_numeric_matrix(df, feats)
        y = pd.to_numeric(df[outcome], errors="coerce").to_numpy(dtype=np.float64)
        mask = ~np.isnan(y) & ~np.isnan(x_raw).any(axis=1)
        if mask.sum() < 4:
            self.stats.notes.append("insufficient complete rows")
            return self
        x_raw, y = x_raw[mask], y[mask]
        z, mu, sigma = _standardize(x_raw)
        self.mu, self.sigma = mu, sigma
        self.y_mu = float(np.mean(y))
        self.y_sigma = float(np.std(y) or 1.0)
        yz = (y - self.y_mu) / self.y_sigma

        d = z.shape[1]

        class Reg(nn.Module):
            def __init__(self, dim: int, hidden: int) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                )

            def forward(self, x: Any) -> Any:
                return self.net(x).squeeze(-1)

        model = Reg(d, self.hidden)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        xt = torch.tensor(z, dtype=torch.float32)
        yt = torch.tensor(yz, dtype=torch.float32)

        # simple holdout
        n = len(yt)
        n_hold = max(1, n // 5)
        model.train()
        last = float("nan")
        for _ in range(self.epochs):
            opt.zero_grad()
            pred = model(xt)
            loss = loss_fn(pred, yt)
            loss.backward()
            opt.step()
            last = float(torch.mean(torch.abs(pred.detach() - yt)).item())

        model.eval()
        with torch.no_grad():
            pred = model(xt).numpy()
        hold_mae = float(np.mean(np.abs(pred[-n_hold:] - yz[-n_hold:]))) * self.y_sigma
        self._model = model
        self.stats.epochs = self.epochs
        self.stats.train_mae = last * self.y_sigma
        self.stats.heldout_mask_mae = hold_mae
        self.stats.n_rows = int(n)
        self.stats.n_features = int(d)
        self.stats.notes.append(f"trained MLP regressor for `{outcome}`")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self._model is None or not self.feature_columns or self.mu is None:
            return np.full(len(df), np.nan)
        import torch

        x_raw, _ = _to_numeric_matrix(df, self.feature_columns)
        z, _, _ = _standardize(x_raw, self.mu, self.sigma)
        self._model.eval()
        with torch.no_grad():
            pred_z = self._model(torch.tensor(z, dtype=torch.float32)).numpy()
        return pred_z * self.y_sigma + self.y_mu

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "features": self.feature_columns,
            "stats": self.stats.to_dict(),
        }
