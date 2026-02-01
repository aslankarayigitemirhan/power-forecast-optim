from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

@dataclass
class SupervisedData:
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    index: pd.DatetimeIndex  # aligned with y (target timestamp)
    horizon_steps: int
    lookback_steps: int

def _calendar_features(index: pd.DatetimeIndex, granularity_min: int) -> Tuple[np.ndarray, List[str]]:
    # Calendar features adapt to granularity:
    # - If granularity < 1440: include hour-of-day sin/cos
    # - Always include day-of-week sin/cos
    dow = index.dayofweek.to_numpy()
    feat = []
    names = []

    # dow cyclic
    feat.append(np.sin(2 * np.pi * dow / 7.0))
    feat.append(np.cos(2 * np.pi * dow / 7.0))
    names += ["dow_sin", "dow_cos"]

    if granularity_min < 1440:
        hour = index.hour.to_numpy()
        feat.append(np.sin(2 * np.pi * hour / 24.0))
        feat.append(np.cos(2 * np.pi * hour / 24.0))
        names += ["hour_sin", "hour_cos"]

    month = index.month.to_numpy()
    feat.append(np.sin(2 * np.pi * (month - 1) / 12.0))
    feat.append(np.cos(2 * np.pi * (month - 1) / 12.0))
    names += ["month_sin", "month_cos"]

    return np.column_stack(feat), names

def make_supervised_energy(
    energy_kwh: pd.Series,
    df_bucket: pd.DataFrame,
    granularity_min: int,
    horizon_steps: int,
    lookback_steps: int,
    add_bucket_exog: bool = False,
    exog_cols: Optional[List[str]] = None,
) -> SupervisedData:
    """
    Build supervised dataset:
      y_k = Energy_kWh at time k+horizon_steps
      x_k = [1, lag(E_k, 0..lookback_steps-1), calendar features (at time k), optional exog summaries at time k]
    """
    E = energy_kwh.dropna().copy()

    # Align bucket dataframe on energy index (keep timestamps that exist in both)
    dfb = df_bucket.reindex(E.index)

    # Build lag matrix
    lag_feats = []
    lag_names = []
    for i in range(lookback_steps):
        lag_feats.append(E.shift(i))
        lag_names.append(f"E_lag{i}")
    lag_df = pd.concat(lag_feats, axis=1)
    lag_df.columns = lag_names

    # Calendar features at time k
    cal_X, cal_names = _calendar_features(E.index, granularity_min)

    # Optional exogenous bucket means at time k
    exog_X = None
    exog_names = []
    if add_bucket_exog:
        if exog_cols is None:
            # a reasonable default subset if present
            default_cols = [
                "Voltage",
                "Global_reactive_power",
                "Global_intensity",
                "Sub_metering_1",
                "Sub_metering_2",
                "Sub_metering_3",
            ]
            exog_cols = [c for c in default_cols if c in dfb.columns]
        ex = dfb[exog_cols].copy()
        exog_X = ex.to_numpy()
        exog_names = [f"exog_{c}" for c in exog_cols]

    # Target y at k+h
    y = E.shift(-horizon_steps)

    # Combine into design matrix at time k
    X_parts = [lag_df]
    # intercept
    intercept = pd.Series(1.0, index=E.index, name="bias")
    X_parts.insert(0, intercept)

    X_df = pd.concat(X_parts, axis=1)

    # drop rows with NaNs due to shifting
    valid = (~X_df.isna().any(axis=1)) & (~y.isna())

    X_df = X_df.loc[valid]
    y = y.loc[valid]
    idx = X_df.index

    # calendar aligned to idx
    cal_X = cal_X[valid.to_numpy()]

    X_list = [X_df.to_numpy(), cal_X]
    names = ["bias"] + lag_names + cal_names

    if exog_X is not None:
        exog_X = exog_X[valid.to_numpy()]
        X_list.append(exog_X)
        names += exog_names

    X = np.column_stack(X_list).astype(np.float64)
    y_arr = y.to_numpy(dtype=np.float64)

    return SupervisedData(
        X=X,
        y=y_arr,
        feature_names=names,
        index=idx,
        horizon_steps=horizon_steps,
        lookback_steps=lookback_steps,
    )
