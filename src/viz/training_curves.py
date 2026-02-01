from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt

def plot_training_curves(train_loss, val_loss, outpath: Path, title: str) -> None:
    plt.figure()
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Ridge Loss")
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=160)
    plt.close()
