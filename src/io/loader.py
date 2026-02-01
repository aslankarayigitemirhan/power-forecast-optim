from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

@dataclass
class LoadedData:
    df: pd.DataFrame  # datetime-indexed
    source_path: Path

def load_household_power(path: Path) -> LoadedData:
    """
    Expected original file format:
    - Separator: ';'
    - Date: 'dd/mm/yyyy'
    - Time: 'hh:mm:ss'
    - Missing values: '?' or blank
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(
        path,
        sep=";",
        low_memory=False,
        na_values=["?", "NA", "", " "],
    )

    # Build datetime index
    dt = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df = df.drop(columns=["Date", "Time"])
    df.insert(0, "datetime", dt)
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()

    # Convert all columns to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop duplicate timestamps (rare) by mean
    if df.index.has_duplicates:
        df = df.groupby(df.index).mean(numeric_only=True)

    return LoadedData(df=df, source_path=path)