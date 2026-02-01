from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, List
import zipfile
import urllib.request

import numpy as np
import pandas as pd
from src.model.ridge_gd import ridge_loss
from src.viz.loss_surface import plot_loss_surface_2d, plot_loss_contours_with_gradients
from src.config import Paths, TrainConfig, UCI_DATASET_URL, DEFAULT_RAW_FILENAME
from src.io.loader import load_household_power
from src.preprocess.missing import fill_missing_two_stage
from src.preprocess.resample import build_energy_target
from src.preprocess.features import make_supervised_energy
from src.preprocess.scaling import fit_standard_scaler
from src.split.splitter import blocked_split, rolling_folds
from src.model.ridge_gd import train_ridge_minibatch_gd, ridge_loss
from src.model.metrics import mae, rmse
from src.viz.eda import plot_energy_series, plot_energy_distribution, plot_dow_hour_heatmap
from src.viz.training_curves import plot_training_curves


def ensure_dirs() -> Paths:
    p = Paths()
    p.raw_dir.mkdir(parents=True, exist_ok=True)
    p.processed_dir.mkdir(parents=True, exist_ok=True)
    p.models_dir.mkdir(parents=True, exist_ok=True)
    p.reports_dir.mkdir(parents=True, exist_ok=True)
    return p

def cmd_download(args: argparse.Namespace) -> None:
    paths = ensure_dirs()
    zip_path = paths.raw_dir / "household_power_consumption.zip"

    if zip_path.exists() and not args.force:
        print(f"[download] Zip already exists: {zip_path} (use --force to re-download)")
    else:
        print(f"[download] Downloading dataset zip from: {UCI_DATASET_URL}")
        urllib.request.urlretrieve(UCI_DATASET_URL, zip_path)
        print(f"[download] Saved: {zip_path}")

    # extract .txt
    out_txt = paths.raw_dir / DEFAULT_RAW_FILENAME
    if out_txt.exists() and not args.force:
        print(f"[download] Raw txt already exists: {out_txt} (use --force to overwrite)")
        return

    with zipfile.ZipFile(zip_path, "r") as z:
        # some zips contain exact filename
        members = z.namelist()
        # find the target txt file
        target = None
        for m in members:
            if m.endswith(DEFAULT_RAW_FILENAME):
                target = m
                break
        if target is None:
            raise RuntimeError(f"Expected {DEFAULT_RAW_FILENAME} not found in zip. Members: {members[:5]} ...")
        with z.open(target) as f_in, open(out_txt, "wb") as f_out:
            f_out.write(f_in.read())

    print(f"[download] Extracted: {out_txt}")

def _load_and_preprocess_minute(cfg: TrainConfig) -> pd.DataFrame:
    loaded = load_household_power(cfg.data_path)
    df = loaded.df

    # Missing handling on Global_active_power (minimum viable); can be extended to other cols
    df2, rep = fill_missing_two_stage(df, "Global_active_power", gap_max_minutes=cfg.gap_max_min)
    print(f"[missing] {rep.column}: before={rep.total_nans_before}, after={rep.total_nans_after}, "
          f"interp={rep.interpolated_points}, calendar={rep.calendar_filled_points}")

    return df2

def _build_supervised(cfg: TrainConfig):
    df_min = _load_and_preprocess_minute(cfg)
    rs = build_energy_target(df_min, granularity_min=cfg.granularity_min)

    sup = make_supervised_energy(
        energy_kwh=rs.target_energy_kwh,
        df_bucket=rs.df_bucket,
        granularity_min=cfg.granularity_min,
        horizon_steps=cfg.horizon_steps,
        lookback_steps=cfg.lookback_steps,
        add_bucket_exog=False,  # keep minimal; can be toggled later
    )
    return rs, sup

def _save_model(paths: Paths, cfg: TrainConfig, w: np.ndarray, feature_names: List[str], x_scaler, y_scaler, history, extra: dict):
    model_path = paths.models_dir / cfg.model_name

    cfg_dict = asdict(cfg)
    for k, v in list(cfg_dict.items()):
        if isinstance(v, Path):
            cfg_dict[k] = str(v)

    meta = {
        "config": cfg_dict,
        "feature_names": feature_names,
        "x_mean": x_scaler.mean_.tolist(),
        "x_std": x_scaler.std_.tolist(),
        "y_mean": float(y_scaler.mean_),
        "y_std": float(y_scaler.std_),
        "history": {"train_loss": history.train_loss, "val_loss": history.val_loss},
        "extra": extra,
    }
    np.savez_compressed(model_path, w=w, meta=json.dumps(meta).encode("utf-8"))
    print(f"[save] model -> {model_path}")


def _load_model(model_path: Path):
    z = np.load(model_path, allow_pickle=False)
    w = z["w"]
    meta = json.loads(z["meta"].tobytes().decode("utf-8"))
    return w, meta

def cmd_train(args: argparse.Namespace) -> None:
    paths = ensure_dirs()
    cfg = TrainConfig()

    # override defaults from CLI
    if args.data is not None:
        cfg.data_path = Path(args.data)
    if args.granularity_min is not None:
        cfg.granularity_min = int(args.granularity_min)
    if args.horizon_steps is not None:
        cfg.horizon_steps = int(args.horizon_steps)
    if args.lookback_steps is not None:
        cfg.lookback_steps = int(args.lookback_steps)
    if args.gap_max_min is not None:
        cfg.gap_max_min = int(args.gap_max_min)

    if args.split is not None:
        cfg.split_type = args.split

    if args.lr is not None:
        cfg.lr = float(args.lr)
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    if args.l2_lambda is not None:
        cfg.l2_lambda = float(args.l2_lambda)

    # Print effective time interpretation
    horizon_minutes = cfg.horizon_steps * cfg.granularity_min
    lookback_minutes = cfg.lookback_steps * cfg.granularity_min
    print(f"[cfg] granularity={cfg.granularity_min} min | horizon={cfg.horizon_steps} step(s) = {horizon_minutes} min | "
          f"lookback={cfg.lookback_steps} step(s) = {lookback_minutes} min")

    rs, sup = _build_supervised(cfg)
    X, y = sup.X, sup.y
    n = len(y)
    print(f"[data] supervised samples={n}, features={X.shape[1]}")

    # Scale X (all features) and y (scalar)
    x_scaler = fit_standard_scaler(X)
    Xs = x_scaler.transform(X)

    # y scaler as scalar standardization
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_std = 1.0
    ys = (y - y_mean) / y_std

    class _YScaler:
        def __init__(self, mean, std):
            self.mean_ = mean
            self.std_ = std
        def transform(self, a): return (a - self.mean_) / self.std_
        def inverse_transform(self, a): return a * self.std_ + self.mean_

    y_scaler = _YScaler(y_mean, y_std)

    # Split
    if cfg.split_type == "blocked":
        sp = blocked_split(n, cfg.train_ratio, cfg.val_ratio)
        train_idx, val_idx, test_idx = sp.train_idx, sp.val_idx, sp.test_idx

        Xtr, ytr = Xs[train_idx], ys[train_idx]
        Xva, yva = Xs[val_idx], ys[val_idx]
        Xte, yte = Xs[test_idx], ys[test_idx]

        model, hist = train_ridge_minibatch_gd(
            X_train=Xtr, y_train=ytr,
            X_val=Xva, y_val=yva,
            feature_names=sup.feature_names,
            lr=cfg.lr, batch_size=cfg.batch_size, epochs=cfg.epochs, l2_lambda=cfg.l2_lambda, seed=cfg.seed
        )

        # Evaluate on test
        yhat_te = Xte @ model.w
        yhat_te = y_scaler.inverse_transform(yhat_te)
        ytrue_te = y_scaler.inverse_transform(yte)
        print(f"[test] MAE={mae(ytrue_te, yhat_te):.4f} kWh | RMSE={rmse(ytrue_te, yhat_te):.4f} kWh")

        fig_path = paths.reports_dir / "training_curves.png"
        plot_training_curves(hist.train_loss, hist.val_loss, fig_path, "Training Curves (blocked split)")
        print(f"[report] {fig_path}")

        _save_model(paths, cfg, model.w, model.feature_names, x_scaler, y_scaler, hist, extra={"split": "blocked"})

    elif cfg.split_type == "rolling":
        folds = rolling_folds(n, folds=cfg.rolling_folds, val_steps=cfg.rolling_val_steps, min_train_steps=cfg.rolling_min_train_steps)
        if len(folds) == 0:
            raise RuntimeError("Rolling folds could not be created; reduce rolling_min_train_steps or rolling_val_steps.")

        # Train on the last fold (simple, deterministic) + report fold losses
        last = folds[-1]
        Xtr, ytr = Xs[last.train_idx], ys[last.train_idx]
        Xva, yva = Xs[last.val_idx], ys[last.val_idx]

        model, hist = train_ridge_minibatch_gd(
            X_train=Xtr, y_train=ytr,
            X_val=Xva, y_val=yva,
            feature_names=sup.feature_names,
            lr=cfg.lr, batch_size=cfg.batch_size, epochs=cfg.epochs, l2_lambda=cfg.l2_lambda, seed=cfg.seed
        )

        # Validation metrics on last fold
        yhat_va = Xva @ model.w
        yhat_va = y_scaler.inverse_transform(yhat_va)
        ytrue_va = y_scaler.inverse_transform(yva)
        print(f"[val-lastfold] MAE={mae(ytrue_va, yhat_va):.4f} kWh | RMSE={rmse(ytrue_va, yhat_va):.4f} kWh")

        fig_path = paths.reports_dir / "training_curves.png"
        plot_training_curves(hist.train_loss, hist.val_loss, fig_path, "Training Curves (rolling last fold)")
        print(f"[report] {fig_path}")

        _save_model(paths, cfg, model.w, model.feature_names, x_scaler, y_scaler, hist, extra={"split": "rolling", "folds": len(folds)})

    else:
        raise ValueError(f"Unknown split_type: {cfg.split_type}")

def cmd_predict(args: argparse.Namespace) -> None:
    paths = ensure_dirs()
    model_path = Path(args.model) if args.model else (paths.models_dir / "model_latest.npz")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train first.")

    w, meta = _load_model(model_path)
    cfg_dict = meta["config"]
    cfg = TrainConfig()
    # load back key configs
    cfg.data_path = Path(cfg_dict["data_path"])
    cfg.granularity_min = int(cfg_dict["granularity_min"])
    cfg.horizon_steps = int(cfg_dict["horizon_steps"])
    cfg.lookback_steps = int(cfg_dict["lookback_steps"])
    cfg.gap_max_min = int(cfg_dict["gap_max_min"])

    # rebuild supervised using same config
    rs, sup = _build_supervised(cfg)
    X, y = sup.X, sup.y
    idx = sup.index

    # scalers
    x_mean = np.array(meta["x_mean"], dtype=float)
    x_std = np.array(meta["x_std"], dtype=float)
    Xs = (X - x_mean) / x_std

    y_mean = float(meta["y_mean"])
    y_std = float(meta["y_std"])
    ys = (y - y_mean) / y_std

    # Predict
    yhat_s = Xs @ w
    yhat = yhat_s * y_std + y_mean

    # Filter by date-range
    start = pd.to_datetime(args.start) if args.start else idx.min()
    end = pd.to_datetime(args.end) if args.end else idx.max()

    mask = (idx >= start) & (idx <= end)
    if not mask.any():
        print("[predict] No samples in the requested interval.")
        return

    y_true = y[mask]
    y_pred = yhat[mask]
    t = idx[mask]

    print(f"[predict] interval: {t.min()} -> {t.max()} | n={len(t)}")
    print(f"[predict] MAE={mae(y_true, y_pred):.4f} kWh | RMSE={rmse(y_true, y_pred):.4f} kWh")

    # Print few samples
    show = min(10, len(t))
    print("\n[timestamp | true_kWh | pred_kWh | err]")
    for i in range(show):
        err = y_true[i] - y_pred[i]
        print(f"{t[i]} | {y_true[i]:.4f} | {y_pred[i]:.4f} | {err:+.4f}")

def cmd_analyze(args: argparse.Namespace) -> None:
    paths = ensure_dirs()
    cfg = TrainConfig()
    if args.data is not None:
        cfg.data_path = Path(args.data)
    if args.granularity_min is not None:
        cfg.granularity_min = int(args.granularity_min)
    if args.gap_max_min is not None:
        cfg.gap_max_min = int(args.gap_max_min)

    df_min = _load_and_preprocess_minute(cfg)
    rs = build_energy_target(df_min, granularity_min=cfg.granularity_min)
    energy = rs.target_energy_kwh.dropna()

    plot_energy_series(energy, paths.reports_dir / "eda_energy_series.png", f"Energy (kWh) series - {cfg.granularity_min}min")
    plot_energy_distribution(energy, paths.reports_dir / "eda_energy_hist.png", "Energy (kWh) distribution")

    if cfg.granularity_min < 1440:
        plot_dow_hour_heatmap(energy, paths.reports_dir / "eda_dow_hour_heatmap.png", "Mean Energy (kWh) by DOW x Hour")

    print(f"[eda] saved to: {paths.reports_dir}")

def cmd_surface(args: argparse.Namespace) -> None:
    paths = ensure_dirs()
    model_path = Path(args.model) if args.model else (paths.models_dir / "model_latest.npz")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train first.")

    w, meta = _load_model(model_path)
    cfg_dict = meta["config"]

    # rebuild supervised (same configs)
    cfg = TrainConfig()
    cfg.data_path = Path(cfg_dict["data_path"])
    cfg.granularity_min = int(cfg_dict["granularity_min"])
    cfg.horizon_steps = int(cfg_dict["horizon_steps"])
    cfg.lookback_steps = int(cfg_dict["lookback_steps"])
    cfg.gap_max_min = int(cfg_dict["gap_max_min"])
    cfg.l2_lambda = float(cfg_dict["l2_lambda"])

    rs, sup = _build_supervised(cfg)
    X, y = sup.X, sup.y

    # apply saved scalers (surface should match trained space)
    x_mean = np.array(meta["x_mean"], dtype=float)
    x_std = np.array(meta["x_std"], dtype=float)
    Xs = (X - x_mean) / x_std

    y_mean = float(meta["y_mean"])
    y_std = float(meta["y_std"])
    ys = (y - y_mean) / y_std

    i = int(args.param_i)
    j = int(args.param_j)
    if i < 0 or j < 0 or i >= len(w) or j >= len(w):
        raise ValueError("param indices out of range.")

    # Use a subset for speed (surface + trajectory illustration)
    n = len(ys)
    n_use = min(n, 20000)
    Xp = Xs[:n_use]
    yp = ys[:n_use]

    lr = float(args.lr) if args.lr is not None else 0.05
    steps = int(args.steps) if args.steps is not None else 40

    # Build illustrative GD path from zeros
    w0 = np.zeros_like(w, dtype=float)
    wk = w0.copy()
    path = [wk.copy()]

    for step in range(steps):
        loss_before = ridge_loss(Xp, yp, wk, cfg.l2_lambda)

        g = (-2.0 / n_use) * (Xp.T @ (yp - Xp @ wk)) + 2.0 * cfg.l2_lambda * wk
        wk2 = wk - lr * g

        loss_after = ridge_loss(Xp, yp, wk2, cfg.l2_lambda)
        print(
            f"[traj] step={step:02d} "
            f"loss={loss_before:.6f} -> {loss_after:.6f} | "
            f"||g_ij||={np.linalg.norm([g[i], g[j]]):.4f}"
        )

        wk = wk2
        path.append(wk.copy())

    # 3D surface
    out = paths.reports_dir / "loss_surface.png"
    plot_loss_surface_2d(
        X=Xp, y=yp,
        w_ref=wk,
        l2_lambda=cfg.l2_lambda,
        i=i, j=j,
        grid=int(args.grid),
        span=float(args.span),
        gd_path=path,
        outpath=out,
        title=f"Loss surface on (w[{i}], w[{j}]) + GD trajectory"
    )
    print(f"[surface] saved: {out}")

    # 2D contours + gradient field
    out2 = paths.reports_dir / "loss_contours.png"
    plot_loss_contours_with_gradients(
        X=Xp, y=yp,
        w_ref=wk,
        l2_lambda=cfg.l2_lambda,
        i=i, j=j,
        grid=int(args.grid),
        span=float(args.span),
        gd_path=path,
        outpath=out2,
        title=f"Loss contours + gradients on (w[{i}], w[{j}])"
    )
    print(f"[surface] saved: {out2}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="power-forecast", description="Energy forecasting (kWh) with Ridge + mini-batch GD")
    sub = p.add_subparsers(dest="cmd", required=True)

    # download
    d = sub.add_parser("download", help="Download and extract UCI dataset into data/raw/")
    d.add_argument("--force", action="store_true", help="Re-download and overwrite existing files")
    d.set_defaults(func=cmd_download)

    # train
    t = sub.add_parser("train", help="Train model")
    t.add_argument("--data", type=str, default=None, help="Path to raw household_power_consumption.txt")
    t.add_argument("--granularity-min", type=int, default=None, help="Bucket size in minutes (default 60)")
    t.add_argument("--horizon-steps", type=int, default=None, help="Predict Δ steps ahead (Δ*granularity minutes)")
    t.add_argument("--lookback-steps", type=int, default=None, help="Lookback window in steps (p*granularity minutes)")
    t.add_argument("--gap-max-min", type=int, default=None, help="Max missing gap in minutes for interpolation")
    t.add_argument("--split", type=str, choices=["rolling", "blocked"], default=None)
    t.add_argument("--lr", type=float, default=None)
    t.add_argument("--batch-size", type=int, default=None)
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--l2-lambda", type=float, default=None)
    t.set_defaults(func=cmd_train)

    # predict
    pr = sub.add_parser("predict", help="Predict energy on a date interval")
    pr.add_argument("--model", type=str, default=None, help="Path to model .npz (default data/models/model_latest.npz)")
    pr.add_argument("--start", type=str, default=None, help='Start datetime, e.g. "2010-10-01 00:00:00"')
    pr.add_argument("--end", type=str, default=None, help='End datetime, e.g. "2010-10-07 23:59:59"')
    pr.set_defaults(func=cmd_predict)

    # analyze
    a = sub.add_parser("analyze", help="Generate EDA figures")
    a.add_argument("--data", type=str, default=None)
    a.add_argument("--granularity-min", type=int, default=None)
    a.add_argument("--gap-max-min", type=int, default=None)
    a.set_defaults(func=cmd_analyze)

    # surface
    s = sub.add_parser("surface", help="Plot loss surface (w_i, w_j) + GD trajectory")
    s.add_argument("--model", type=str, default=None)
    s.add_argument("--param-i", type=int, required=True)
    s.add_argument("--param-j", type=int, required=True)
    s.add_argument("--grid", type=int, default=80)
    s.add_argument("--span", type=float, default=2.0)
    s.add_argument("--lr", type=float, default=0.05, help="LR for illustrative GD trajectory (surface plot)")
    s.add_argument("--steps", type=int, default=40, help="Steps for illustrative GD trajectory")
    s.set_defaults(func=cmd_surface)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
