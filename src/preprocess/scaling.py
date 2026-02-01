from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class StandardScaler:
    mean_: np.ndarray
    std_: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def inverse_transform(self, Xs: np.ndarray) -> np.ndarray:
        return Xs * self.std_ + self.mean_

def fit_standard_scaler(X: np.ndarray, eps: float = 1e-12) -> StandardScaler:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return StandardScaler(mean_=mean, std_=std)
