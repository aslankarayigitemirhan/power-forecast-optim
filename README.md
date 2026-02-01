# power-forecast  
**Energy Forecasting with Gradient Descent + Ridge Regression**

This repository implements an end-to-end backend pipeline for short-term household energy forecasting, formulated as a **convex optimization problem** and solved via **gradient-based learning**.

The project emphasizes **interpretability, optimization analysis, and reproducibility**, rather than black-box accuracy.

---

## Overview

The system predicts **interval-based energy consumption (kWh)** using a regularized linear regression model trained with Gradient Descent (GD).

### Key ideas
- Forecast **energy per interval** (kWh), not instantaneous power (kW)
- Preserve **time-series integrity** (no random shuffling)
- Use **Ridge (L2) regularization** for stability and conditioning
- Visualize **optimization geometry** (loss surface, contours, GD trajectory)

---

## Backend Pipeline

1. **Dataset acquisition**
   - UCI *Individual Household Electric Power Consumption*
   - Stored under `data/raw/`

2. **Missing value handling**
   - Short gaps (≤ 30 min): linear interpolation  
   - Long gaps (> 30 min): calendar-conditioned median  
     (same hour-of-day & day-of-week)

3. **Resampling**
   - User-defined granularity (default: **60 minutes**)

4. **Target construction**
   - Energy per interval:
     ```
     E_k = ∑_{t ∈ B_k} P_t · (1/60)
     ```

5. **Feature engineering**
   - Lagged energy values (lookback window)
   - Calendar features (sin/cos encoding)
   - Bias term

6. **Time-series validation**
   - Blocked split or rolling (walk-forward)
   - No random shuffling (prevents leakage)

7. **Training**
   - Ridge regression
   - Mini-batch Gradient Descent

8. **Prediction**
   - User-defined date range
   - Interval-based forecasts

9. **Visualization**
   - Training curves
   - Loss surface (2D slice)
   - Contours + GD trajectory

---

## Project Structure

```
power-forecast/
├── data/
│   ├── raw/            # Original dataset
│   ├── processed/      # Cleaned & resampled data
│   ├── models/         # Saved model checkpoints
│   └── reports/        # Figures and plots
│
├── src/
│   ├── cli.py          # Command-line interface
│   ├── data/           # Loading, cleaning, resampling
│   ├── features/       # Feature engineering
│   ├── model/          # Ridge + GD implementation
│   └── viz/            # Visualization utilities
│
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1) Create virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 2) Download dataset

Manually download the dataset from UCI and place it under:

```
data/raw/household_power_consumption.txt
```

(Automatic download can be added as future work.)

---

### 3) Train the model

```bash
python -m src.cli train
```

**Default configuration**

* Granularity: 60 min
* Horizon: 1 step (1 hour ahead)
* Lookback: 24 steps (24 hours)
* Validation: rolling (walk-forward)

Example output:

```
[data] supervised samples=34565, features=31
[val-lastfold] MAE=0.3229 kWh | RMSE=0.4690 kWh
[save] model -> data/models/model_latest.npz
```

---

### 4) Run prediction on a date range

```bash
python -m src.cli predict \
  --start "2007-04-16 21:00:00" \
  --end   "2007-07-16 20:00:00"
```

Output includes:

* MAE / RMSE over the selected interval
* Timestamp-level prediction table

---

### 5) Visualize optimization landscape

```bash
python -m src.cli surface \
  --param-i 0 \
  --param-j 1 \
  --grid 100 \
  --span 2.0 \
  --lr 0.05 \
  --steps 40
```

Generates:

* `loss_surface.png`
* `loss_contours.png`

These plots illustrate:

* Convex loss geometry
* Gradient directions
* GD convergence trajectory

---

## Optimization Formulation

**Objective (Ridge Regression):**

```
min_w [1/N ∑_{k=1}^{N} (E_k - w^T x_k)^2 + λ ||w||_2^2]
```

- Design variables: regression weights (w)
- Soft constraint via L2 regularization
- Convex ⇒ unique global minimum

---

## Why Gradient Descent?

Although a closed-form solution exists:

* GD scales better to large datasets
* Enables mini-batch training
* Supports rolling updates
* Allows **optimization diagnostics** (loss surface, trajectories)

---

## Results Summary

* **Rolling validation**

  * MAE ≈ **0.32 kWh**
  * RMSE ≈ **0.47 kWh**

* **User-selected interval**

  * MAE ≈ **0.39 kWh**
  * RMSE ≈ **0.56 kWh**

Errors are reported **per forecast interval**, making them directly interpretable for energy planning.

---

## Future Work

* GUI with:

  * User-defined granularity / horizon
  * Train & predict buttons
  * Forecast and residual plots
* Multi-step forecasting
* Probabilistic intervals
* Online / streaming updates

---

## License

Academic / educational use.
