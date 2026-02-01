from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

@dataclass
class RollingFold:
    train_idx: np.ndarray
    val_idx: np.ndarray

def blocked_split(n: int, train_ratio: float, val_ratio: float) -> Split:
    if not (0.0 < train_ratio < 1.0) or not (0.0 < val_ratio < 1.0):
        raise ValueError("train_ratio and val_ratio must be in (0,1).")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.")

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n)

    return Split(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

def rolling_folds(n: int, folds: int, val_steps: int, min_train_steps: int) -> List[RollingFold]:
    """
    Walk-forward folds:
    last part of the series is used for rolling validations.
    """
    if val_steps <= 0:
        raise ValueError("val_steps must be positive.")
    if min_train_steps <= 0:
        raise ValueError("min_train_steps must be positive.")
    if folds <= 0:
        raise ValueError("folds must be positive.")

    folds_out: List[RollingFold] = []
    # We carve out folds from the end
    end = n
    for _ in range(folds):
        val_end = end
        val_start = val_end - val_steps
        train_end = val_start
        train_start = 0
        if train_end - train_start < min_train_steps:
            break
        folds_out.append(
            RollingFold(
                train_idx=np.arange(train_start, train_end),
                val_idx=np.arange(val_start, val_end),
            )
        )
        end = val_start  # move left

    folds_out.reverse()  # chronological order
    return folds_out
