from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_energy_series(energy: pd.Series, outpath: Path, title: str) -> None:
    plt.figure()
    energy.plot()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Energy (kWh)")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_energy_distribution(energy: pd.Series, outpath: Path, title: str) -> None:
    plt.figure()
    plt.hist(energy.dropna().to_numpy(), bins=60)
    plt.title(title)
    plt.xlabel("Energy (kWh)")
    plt.ylabel("Count")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_dow_hour_heatmap(energy: pd.Series, outpath: Path, title: str) -> None:
    """
    Heatmap of mean energy by (day_of_week, hour).
    Works best when granularity < 1440.
    """
    df = pd.DataFrame({"E": energy}).dropna()
    df["dow"] = df.index.dayofweek
    df["hour"] = df.index.hour
    pivot = df.pivot_table(values="E", index="dow", columns="hour", aggfunc="mean")

    plt.figure()
    plt.imshow(pivot.to_numpy(), aspect="auto")
    plt.title(title)
    plt.xlabel("Hour of day")
    plt.ylabel("Day of week (0=Mon)")
    plt.colorbar(label="Mean Energy (kWh)")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=160)
    plt.close()
