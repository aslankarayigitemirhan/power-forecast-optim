from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

@dataclass
class MissingReport:
    column: str
    total_nans_before: int
    total_nans_after: int
    interpolated_points: int
    calendar_filled_points: int

def _gap_groups(is_nan: pd.Series) -> pd.Series:
    # groups contiguous NaN segments with an id
    # non-NaN => 0, NaN segments => 1..K
    nan = is_nan.astype(int)
    change = nan.diff().fillna(0).abs()
    grp = (change.cumsum())  # unique id changes at each transition
    # keep only NaN positions
    grp = grp.where(is_nan, 0)
    # compress ids so they start from 1
    ids = grp[grp > 0].unique()
    mapping = {old: i + 1 for i, old in enumerate(ids)}
    return grp.replace(mapping)

def fill_missing_two_stage(
    df: pd.DataFrame,
    column: str,
    gap_max_minutes: int,
    calendar_key: Tuple[str, ...] = ("dow", "hour", "minute"),
    use_median: bool = True,
) -> Tuple[pd.DataFrame, MissingReport]:
    """
    Stage-1: interpolate only for NaN segments whose duration <= gap_max_minutes.
    Stage-2: remaining NaNs -> fill with calendar-conditioned median/mean based on (dow,hour,minute).

    Assumes df index is a proper DateTimeIndex at 1-min frequency (or close).
    """
    out = df.copy()
    s = out[column].copy()
    n0 = int(s.isna().sum())

    # Add calendar keys
    cal = pd.DataFrame(index=out.index)
    cal["dow"] = out.index.dayofweek
    cal["hour"] = out.index.hour
    cal["minute"] = out.index.minute

    # Identify NaN segments and their lengths in minutes
    seg_id = _gap_groups(s.isna())
    interpolated_points = 0

    if seg_id.max() > 0:
        # compute segment lengths in minutes
        seg_lengths = seg_id[seg_id > 0].groupby(seg_id).size()  # number of rows
        # We assume near 1-min sampling; treat each row as 1 minute
        short_ids = seg_lengths[seg_lengths <= gap_max_minutes].index.tolist()

        # Interpolate: but only fill those segments
        # Compute full interpolation
        s_interp = s.interpolate(method="time", limit_direction="both")

        mask_short = seg_id.isin(short_ids)
        interpolated_points = int((mask_short & s.isna()).sum())
        s = s.where(~mask_short, s_interp)

    # Stage-2: calendar-conditioned fill for remaining NaNs
    n1 = int(s.isna().sum())
    calendar_filled_points = 0
    if n1 > 0:
        tmp = pd.DataFrame({"y": s}).join(cal)

        grp_cols = list(calendar_key)
        agg = tmp.dropna(subset=["y"]).groupby(grp_cols)["y"]
        stats = agg.median() if use_median else agg.mean()

        # Map for each row
        key_df = tmp[grp_cols]
        keys = list(map(tuple, key_df.values))
        mapped = np.array([stats.get(k, np.nan) for k in keys], dtype=float)

        fill_mask = tmp["y"].isna() & ~np.isnan(mapped)
        calendar_filled_points = int(fill_mask.sum())
        s.loc[fill_mask] = mapped[fill_mask.values]

    out[column] = s
    n2 = int(out[column].isna().sum())

    rep = MissingReport(
        column=column,
        total_nans_before=n0,
        total_nans_after=n2,
        interpolated_points=interpolated_points,
        calendar_filled_points=calendar_filled_points,
    )
    return out, rep
