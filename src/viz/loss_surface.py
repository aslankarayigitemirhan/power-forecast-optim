# src/viz/loss_surface.py

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from src.model.ridge_gd import ridge_loss, ridge_grad
def plot_loss_surface_2d(
    X: np.ndarray,
    y: np.ndarray,
    w_ref: np.ndarray,
    l2_lambda: float,
    i: int,
    j: int,
    grid: int,
    span: float,
    gd_path: list[np.ndarray],
    outpath: Path,
    title: str,
    view_elev: float = 25.0,
    view_azim: float = -170.0,
) -> None:
    """
    3D loss surface over (w_i, w_j), others fixed at w_ref.
    """
    wi0 = w_ref[i]
    wj0 = w_ref[j]

    wi = np.linspace(wi0 - span, wi0 + span, grid)
    wj = np.linspace(wj0 - span, wj0 + span, grid)
    WI, WJ = np.meshgrid(wi, wj)

    Z = np.zeros_like(WI, dtype=float)
    for r in range(grid):
        for c in range(grid):
            w = w_ref.copy()
            w[i] = WI[r, c]
            w[j] = WJ[r, c]
            Z[r, c] = ridge_loss(X, y, w, l2_lambda)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.plot_surface(WI, WJ, Z, alpha=0.85)

    if gd_path:
        traj = np.array([[p[i], p[j]] for p in gd_path])
        ztraj = [ridge_loss(X, y, p, l2_lambda) for p in gd_path]
        ax.plot(traj[:, 0], traj[:, 1], ztraj, marker="o", color="orange")

    ax.set_xlabel(f"w[{i}]")
    ax.set_ylabel(f"w[{j}]")
    ax.set_zlabel("Loss")
    ax.set_title(title)

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=160)
    plt.show()

def plot_loss_contours_with_gradients(
    X: np.ndarray,
    y: np.ndarray,
    w_ref: np.ndarray,
    l2_lambda: float,
    i: int,
    j: int,
    grid: int,
    span: float,
    gd_path: List[np.ndarray],
    outpath: Path,
    title: str,
    quiver_stride: int = 6,
) -> None:
    """
    2D contour plot of loss over (w_i, w_j) plane (others fixed),
    with gradient vector field and GD trajectory.
    """
    wi0 = w_ref[i]
    wj0 = w_ref[j]

    wi = np.linspace(wi0 - span, wi0 + span, grid)
    wj = np.linspace(wj0 - span, wj0 + span, grid)
    WI, WJ = np.meshgrid(wi, wj)

    Z = np.zeros_like(WI, dtype=float)
    Gi = np.zeros_like(WI, dtype=float)
    Gj = np.zeros_like(WI, dtype=float)

    # Evaluate loss + gradient on grid
    for r in range(grid):
        for c in range(grid):
            w = w_ref.copy()
            w[i] = WI[r, c]
            w[j] = WJ[r, c]
            Z[r, c] = ridge_loss(X, y, w, l2_lambda)
            g = ridge_grad(X, y, w, l2_lambda)
            Gi[r, c] = g[i]
            Gj[r, c] = g[j]

    plt.figure()
    plt.axis("equal")
    # Contours
    cs = plt.contour(WI, WJ, Z, levels=25)
    plt.clabel(cs, inline=True, fontsize=7)

    # Gradient field: show negative gradient direction (steepest descent)
    # Normalize arrows to visualize direction clearly
    s = slice(None, None, quiver_stride)
    U = -Gi[s, s]
    V = -Gj[s, s]
    norm = np.sqrt(U**2 + V**2) + 1e-12
    U = U / norm
    V = V / norm
    plt.quiver(WI[s, s], WJ[s, s], U, V, angles="xy")

    # GD trajectory
    if gd_path:
        traj = np.array([[p[i], p[j]] for p in gd_path], dtype=float)
        plt.plot(traj[:, 0], traj[:, 1], marker="o", linewidth=2)

        # Mark start / end
        plt.scatter(traj[0, 0], traj[0, 1], s=80, marker="s", label="start")
        plt.scatter(traj[-1, 0], traj[-1, 1], s=80, marker="*", label="end")

    plt.title(title)
    plt.xlabel(f"w[{i}]")
    plt.ylabel(f"w[{j}]")
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=170)
    plt.close()
