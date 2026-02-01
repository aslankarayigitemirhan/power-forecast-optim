from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class GDHistory:
    train_loss: List[float]
    val_loss: List[float]

@dataclass
class RidgeGDModel:
    w: np.ndarray
    feature_names: List[str]

def ridge_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2_lambda: float) -> float:
    n = X.shape[0]
    r = y - X @ w
    return float((r @ r) / n + l2_lambda * (w @ w))

def ridge_grad(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2_lambda: float) -> np.ndarray:
    n = X.shape[0]
    # grad = -(2/n) X^T(y - Xw) + 2 lambda w
    return (-2.0 / n) * (X.T @ (y - X @ w)) + 2.0 * l2_lambda * w

def train_ridge_minibatch_gd(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    feature_names: List[str],
    lr: float,
    batch_size: int,
    epochs: int,
    l2_lambda: float,
    seed: int = 42,
) -> Tuple[RidgeGDModel, GDHistory]:
    rng = np.random.default_rng(seed)
    n, d = X_train.shape
    w = np.zeros(d, dtype=np.float64)

    hist = GDHistory(train_loss=[], val_loss=[])

    for ep in range(epochs):
        # shuffle indices
        idx = np.arange(n)
        rng.shuffle(idx)

        for start in range(0, n, batch_size):
            batch_idx = idx[start : start + batch_size]
            Xb = X_train[batch_idx]
            yb = y_train[batch_idx]
            g = ridge_grad(Xb, yb, w, l2_lambda)
            w = w - lr * g

        tr = ridge_loss(X_train, y_train, w, l2_lambda)
        hist.train_loss.append(tr)
        if X_val is not None and y_val is not None and len(y_val) > 0:
            vl = ridge_loss(X_val, y_val, w, l2_lambda)
        else:
            vl = float("nan")
        hist.val_loss.append(vl)

    return RidgeGDModel(w=w, feature_names=feature_names), hist
