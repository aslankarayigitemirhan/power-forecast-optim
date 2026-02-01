from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    models_dir: Path = data_dir / "models"
    reports_dir: Path = data_dir / "reports"

UCI_DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
DEFAULT_RAW_FILENAME = "household_power_consumption.txt"  # inside the zip
DEFAULT_RAW_PATH = Paths().raw_dir / DEFAULT_RAW_FILENAME

@dataclass
class TrainConfig:
    # Data & preprocessing
    data_path: Path = DEFAULT_RAW_PATH
    granularity_min: int = 60       # default: 60 min buckets
    horizon_steps: int = 1          # predict 1 bucket ahead
    lookback_steps: int = 24        # 24 buckets lookback (e.g., 24 hours if granularity=60)
    gap_max_min: int = 30           # <=30 min missing => interpolate, else calendar-fill
    split_type: str = "rolling"     # "rolling" or "blocked"
    train_ratio: float = 0.70       # used if blocked
    val_ratio: float = 0.15         # used if blocked

    # Rolling split
    rolling_folds: int = 3
    rolling_val_steps: int = 24     # validation window per fold in buckets
    rolling_min_train_steps: int = 24 * 14  # min train length (e.g., 2 weeks in buckets)

    # Model / optimizer
    l2_lambda: float = 1e-3
    lr: float = 0.01
    batch_size: int = 2048
    epochs: int = 30
    seed: int = 42

    # Saving
    model_name: str = "model_latest.npz"
