# Resampling functions
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass
class ResampleResult:
    df_bucket: pd.DataFrame  # bucket-level
    target_energy_kwh: pd.Series  # E_k
    granularity_min: int

def build_energy_target(df_minute: pd.DataFrame, granularity_min: int) -> ResampleResult:
    """
    Input: minute-level df with column 'Global_active_power' in kW.
    Output:
      - bucket-level df
      - energy per bucket in kWh: sum(power_kW * 1/60) over minutes
    """
    if "Global_active_power" not in df_minute.columns:
        raise ValueError("Expected column 'Global_active_power' not found.")

    # Resample buckets
    rule = f"{granularity_min}min"

    # Sum of kW over minutes, then convert to kWh: (sum kW) * (1/60)
    power = df_minute["Global_active_power"]
    energy_kwh = power.resample(rule).sum(min_count=1) * (1.0 / 60.0)

    # For exogenous features at bucket-level, take means (can be extended)
    df_bucket = df_minute.resample(rule).mean(numeric_only=True)

    return ResampleResult(df_bucket=df_bucket, target_energy_kwh=energy_kwh.rename("Energy_kWh"), granularity_min=granularity_min)
